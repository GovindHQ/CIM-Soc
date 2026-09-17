"""
attention.py — a drop-in replacement for TinyViT's Attention module in which
the projection, QK^T and AV matmuls execute on the simulated CIM array.

Structure mirrors microsoft/Cream TinyViT `Attention.forward` exactly, so the
only difference against the reference model is *how* the three matmuls are
computed. Everything else — LayerNorm, the learned attention biases, softmax,
the output projection — stays in float, which is also where it stays in the
hardware (those are the off-CIM operators: SIMD/LUT unit, not the array).

WHAT RUNS ON THE ARRAY
    qkv  : [B*N, dim] @ [dim, 2*nh_kd + dh]     static weights  (SEngine path)
    QK^T : [N, key_dim] @ [key_dim, N]          dynamic weights (DEngine path)
    AV   : [N, N] @ [N, d]                      dynamic weights (DEngine path)

WHAT DOES NOT
    norm, attention_biases add, softmax, out-projection, all residual adds.

The out-projection can be moved onto the array with cim_out_proj=True; it is off
by default to match the chosen scope.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .array import CIMArray, CIMConfig
from .matmul import TraceLog, cim_matmul, cim_matmul_batched


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().to(torch.float64).numpy()


class CIMAttention(nn.Module):
    """TinyViT Attention with the three matmuls executed on the CIM fabric.

    Constructed from an existing float Attention module via `from_float`, so the
    pretrained weights carry over unchanged.
    """

    def __init__(self, ref: nn.Module, array: CIMArray| None = None, log: TraceLog | None = None,
                 cim_out_proj: bool = False, name: str = "attn",arrays: list[CIMArray] | None = None,):
        super().__init__()
        # ---- geometry copied from the reference module ---------------------
        self.num_heads = ref.num_heads
        self.scale = ref.scale
        self.key_dim = ref.key_dim
        self.nh_kd = ref.nh_kd
        self.d = ref.d
        self.dh = ref.dh
        self.attn_ratio = ref.attn_ratio

        # ---- parameters shared by reference (no copy, same tensors) --------
        self.norm = ref.norm
        self.qkv = ref.qkv
        self.proj = ref.proj
        self.attention_biases = ref.attention_biases
        self.register_buffer("attention_bias_idxs", ref.attention_bias_idxs,
                             persistent=False)

        # ---- simulator handles ---------------------------------------------
        if arrays is not None:
            self.arrays = list(arrays)
        elif array is not None:
            self.arrays = [array]
        else:
            raise ValueError("CIMAttention requires either array or arrays")
        self.log = log
        self.cim_out_proj = cim_out_proj
        self.name = name
        self.cim_enabled = True   # flip to False to get the float reference path
        # Which matmul sites run on the array. Drop entries to isolate the
        # accuracy contribution of one site at a time.
        self.cim_ops = {"qkv", "qk", "av"}

    # ------------------------------------------------------------------ #

    @property
    def ab(self) -> torch.Tensor:
        return self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x (B, N, C)
        B, N, C = x.shape
        x = self.norm(x)

        # ---- 1. fused QKV projection ---------------------------------- #
        if self.cim_enabled and "qkv" in self.cim_ops:
            # [B*N, C] @ [C, h]. Batch and token axes merge: the array does not
            # distinguish them, it just streams row vectors.
            xf = _np(x.reshape(B * N, C))
            Wq = _np(self.qkv.weight).T                      # [C, h]
            qkv_np = cim_matmul(xf, Wq, arrays=self.arrays,
                                tag=f"{self.name}.qkv", log=self.log)
            qkv = torch.from_numpy(qkv_np).to(x.dtype).to(x.device)
            if self.qkv.bias is not None:
                qkv = qkv + self.qkv.bias                    # bias added digitally
            qkv = qkv.reshape(B, N, -1)
        else:
            qkv = self.qkv(x)

        q, k, v = qkv.view(B, N, self.num_heads, -1).split(
            [self.key_dim, self.key_dim, self.d], dim=3)
        q = q.permute(0, 2, 1, 3)      # (B, heads, N, key_dim)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # ---- 2. QK^T -------------------------------------------------- #
        if self.cim_enabled and "qk" in self.cim_ops:
            kt = k.transpose(-2, -1)                          # (B, heads, key_dim, N)
            scores_np = cim_matmul_batched(_np(q), _np(kt), arrays=self.arrays,
                                           tag=f"{self.name}.qk_t", log=self.log)
            scores = torch.from_numpy(scores_np).to(x.dtype).to(x.device)
        else:
            scores = q @ k.transpose(-2, -1)

        attn = scores * self.scale + self.ab                  # off-CIM
        attn = attn.softmax(dim=-1)                           # off-CIM

        # ---- 3. AV ---------------------------------------------------- #
        if self.cim_enabled and "av" in self.cim_ops:
            # attn is a probability matrix in [0, 1]: unsigned quantization, so
            # the sign bit is not wasted on data that has no negative side.
            out_np = cim_matmul_batched(_np(attn), _np(v), arrays=self.arrays,
                                        tag=f"{self.name}.av", log=self.log,
                                        act_signed=False)
            out = torch.from_numpy(out_np).to(x.dtype).to(x.device)
        else:
            out = attn @ v

        out = out.transpose(1, 2).reshape(B, N, self.dh)

        # ---- 4. output projection (off-array by default) --------------- #
        if self.cim_enabled and self.cim_out_proj:
            Wp = _np(self.proj.weight).T
            o = cim_matmul(_np(out.reshape(B * N, self.dh)), Wp, arrays=self.arrays,
                           tag=f"{self.name}.proj", log=self.log)
            out = torch.from_numpy(o).to(x.dtype).to(x.device).reshape(B, N, -1)
            if self.proj.bias is not None:
                out = out + self.proj.bias
            return out
        return self.proj(out)


# --------------------------------------------------------------------------- #
# Model surgery
# --------------------------------------------------------------------------- #


def convert_attention(model: nn.Module, array: CIMArray| None = None, log: TraceLog | None = None,
                      cim_out_proj: bool = False,
                      class_name: str = "Attention",arrays: list[CIMArray] | None = None,) -> list[str]:
    """Recursively replace every `Attention` submodule with `CIMAttention`.

    Matching is by class name so this works with any TinyViT copy without
    importing it. Returns the list of replaced module paths.
    """
    replaced: list[str] = []

    def _walk(parent: nn.Module, prefix: str) -> None:
        for name, child in list(parent.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if type(child).__name__ == class_name and hasattr(child, "key_dim"):
                setattr(parent, name,
                        CIMAttention(child, array=array,arrays=arrays,log=log,
                                     cim_out_proj=cim_out_proj, name=path))
                replaced.append(path)
            else:
                _walk(child, path)

    _walk(model, "")
    return replaced


def set_cim_enabled(model: nn.Module, enabled: bool) -> None:
    """Toggle every CIMAttention between the array path and the float path."""
    for m in model.modules():
        if isinstance(m, CIMAttention):
            m.cim_enabled = enabled


def set_cim_ops(model: nn.Module, ops) -> None:
    """Choose which matmul sites run on the array: any subset of
    {"qkv", "qk", "av"}. Everything not listed falls back to float torch."""
    ops = set(ops)
    for m in model.modules():
        if isinstance(m, CIMAttention):
            m.cim_ops = set(ops)
