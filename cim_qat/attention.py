"""
cim_qat/attention.py — training-time counterpart of cim_sim.attention.CIMAttention.

The forward mirrors CIMAttention.forward OPERATION FOR OPERATION (same order,
same dtypes, same off-CIM float ops), replacing only the three NumPy
cim_matmul / cim_matmul_batched calls with cim_matmul_surrogate. That is what
makes a CPU float32 surrogate forward bit-identical to the simulator forward
(test_surrogate_vs_simulator.py, tier T3).

Off-CIM (float, unchanged): LayerNorm, qkv bias add, attention-bias add,
softmax, output projection, residuals, MLP, convolutions — exactly as in the
simulator's scope (cim_out_proj is not supported: it is off in the simulator
scope chosen for this thesis phase).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .hw_config import OPS
from .surrogate import OpSpec, OpStats, cim_matmul_surrogate


class CIMAttentionSurrogate(nn.Module):
    def __init__(self, ref: nn.Module, specs: dict, name: str = "attn"):
        super().__init__()
        self.num_heads = ref.num_heads
        self.scale = ref.scale
        self.key_dim = ref.key_dim
        self.nh_kd = ref.nh_kd
        self.d = ref.d
        self.dh = ref.dh
        self.attn_ratio = ref.attn_ratio
        # parameters shared by reference (same tensors -> same state_dict keys)
        self.norm = ref.norm
        self.qkv = ref.qkv
        self.proj = ref.proj
        self.attention_biases = ref.attention_biases
        self.register_buffer("attention_bias_idxs", ref.attention_bias_idxs,
                             persistent=False)
        self.specs: dict[str, OpSpec] = dict(specs)
        self.name = name
        self.cim_enabled = True
        self.cim_ops = set(OPS)
        self.collect_stats = False
        self.stats: dict[str, OpStats] = {op: OpStats() for op in OPS}

    @property
    def ab(self) -> torch.Tensor:
        return self.attention_biases[:, self.attention_bias_idxs]

    def _st(self, op):
        return self.stats[op] if self.collect_stats else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x (B, N, C)
        B, N, C = x.shape
        x = self.norm(x)

        # ---- 1. fused QKV projection (CIM) -------------------------------- #
        if self.cim_enabled and "qkv" in self.cim_ops:
            xf = x.reshape(B * N, C)
            Wq = self.qkv.weight.t()                              # [C, h]
            qkv = cim_matmul_surrogate(xf, Wq, self.specs["qkv"], self._st("qkv"))
            qkv = qkv.to(x.dtype)
            if self.qkv.bias is not None:
                qkv = qkv + self.qkv.bias                         # digital bias add
            qkv = qkv.reshape(B, N, -1)
        else:
            qkv = self.qkv(x)

        q, k, v = qkv.view(B, N, self.num_heads, -1).split(
            [self.key_dim, self.key_dim, self.d], dim=3)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # ---- 2. QK^T (CIM, dynamic "weights" = K^T) ----------------------- #
        if self.cim_enabled and "qk" in self.cim_ops:
            kt = k.transpose(-2, -1)
            scores = cim_matmul_surrogate(q, kt, self.specs["qk"], self._st("qk")).to(x.dtype)
        else:
            scores = q @ k.transpose(-2, -1)

        attn = scores * self.scale + self.ab                      # off-CIM
        attn = attn.softmax(dim=-1)                               # off-CIM

        # ---- 3. AV (CIM, unsigned activations, dynamic "weights" = V) ----- #
        if self.cim_enabled and "av" in self.cim_ops:
            out = cim_matmul_surrogate(attn, v, self.specs["av"], self._st("av")).to(x.dtype)
        else:
            out = attn @ v

        out = out.transpose(1, 2).reshape(B, N, self.dh)
        return self.proj(out)                                     # off-CIM


def convert_attention_surrogate(model: nn.Module, specs: dict,
                                class_name: str = "Attention") -> list[str]:
    replaced = []

    def _walk(parent, prefix):
        for name, child in list(parent.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if type(child).__name__ == class_name and hasattr(child, "key_dim"):
                setattr(parent, name, CIMAttentionSurrogate(child, specs, name=path))
                replaced.append(path)
            else:
                _walk(child, path)

    _walk(model, "")
    return replaced


def surrogate_modules(model: nn.Module):
    return [m for m in model.modules() if isinstance(m, CIMAttentionSurrogate)]


def set_collect_stats(model: nn.Module, on: bool, reset: bool = True):
    for m in surrogate_modules(model):
        m.collect_stats = on
        if reset:
            m.stats = {op: OpStats() for op in OPS}


def merged_stats(model: nn.Module) -> dict:
    out = {op: OpStats() for op in OPS}
    for m in surrogate_modules(model):
        for op in OPS:
            out[op].merge(m.stats[op])
    return out
