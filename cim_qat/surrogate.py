"""
cim_qat/surrogate.py — differentiable PyTorch SURROGATE of the cim_sim analog
CIM matmul, used ONLY to provide a gradient path for QAT.

ROLE SEPARATION (do not blur)
-----------------------------
* cim_sim (NumPy) is the REFERENCE for hardware forward behaviour. It is never
  imported into the training graph and never modified by this package.
* This module re-implements the reference's forward CONTRACT in torch so that
  autograd can run, and defines the backward rule (a straight-through
  estimator, STE) that the reference does not have.
* test_surrogate_vs_simulator.py links the two on identical inputs.

THE FORWARD CONTRACT (verified bit-exact in cim_sim/test_qat_contract.py and
again, for THIS implementation, in test_surrogate_vs_simulator.py)
------------------------------------------------------------------------------
For X[..., N, K] @ W[..., K, M] at one operation (qkv | qk | av):

  1. operand quantization (cim_sim.quant.quantize):
       s_x[n] = max(amax_k |X[n,k]|, 1e-12) / qmax_a      (unsigned AV: amax X)
       Xq     = clip(rint(X / s_x), qmin_a, qmax_a)         rint = half-to-even
       s_w[m] = max(amax_k |W[k,m]|, 1e-12) / qmax_w
       Wq     = clip(rint(W / s_w), qmin_w, qmax_w)
  2. per 32-row depth block b (the ADC sits on every column of every block):
       P_b = Xq[:, b] @ Wq[b, :]                              exact integers
       d_b = clip(floor(P_b / LSB_P + 1/2 + guard), -2^(N-1), 2^(N-1)-1)
     LSB_P = V_LSB_ADC / (G_op * K_MAC) is read from the simulator's own
     AnalogChain (never re-derived here); guard = SARADC.TIE_GUARD_REL * 2^N.
  3. digital accumulation and the single rescale (cim_sim.matmul):
       y = ((sum_b d_b) * LSB_P) * s_x * s_w                 (float64, then fp32)

NUMERICS (why bit-exactness is achievable on any device)
--------------------------------------------------------
* Steps 1, 2's division/rounding and 3 run in float64, exactly as NumPy does.
* P_b is formed by a float32 matmul of integer-valued tensors. Every product
  and every partial sum is an integer with |.| <= 32*255*127 = 1,036,320
  < 2^24, so float32 (and TF32, whose 10-bit mantissa holds operands <= 2047)
  represents every intermediate exactly, in any summation order.

BACKWARD (the STE; see QAT_DESIGN.md §5)
----------------------------------------
* operand rounding: identity (STE); scales s_x, s_w are treated as constants
  (detached), as the hardware computes them from data, not from parameters.
* ADC rounding: identity inside the conversion range; ZERO where the code
  saturated (floor(.) outside [-2^(N-1), 2^(N-1)-1]) — the derivative of the
  clamp, i.e. a "clipped STE".
  => dL/dP_b = (dL/dy * s_x * s_w) * 1[block b, element not saturated]
* gradients then flow through the exact integer matmul to Xq and Wq and
  through the operand STE to X and W:
       dL/dX = (dL/dXq) / s_x,   dL/dW = (dL/dWq) / s_w
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

__all__ = ["OpSpec", "OpStats", "cim_matmul_surrogate", "quantize_operand"]

# must equal cim_sim.analog.SARADC.TIE_GUARD_REL; asserted in hw_config.py
TIE_GUARD_REL = 1e-9


@dataclass(frozen=True)
class OpSpec:
    """Everything the surrogate needs for one operation. Built from the
    simulator by cim_qat.hw_config.op_specs(); never hand-written."""
    op: str              # "qkv" | "qk" | "av"
    weight_bits: int
    act_bits: int
    act_signed: bool
    lsb_p: float         # column-sum units per ADC code (from AnalogChain)
    adc_bits: int
    rows: int = 32       # depth-block height (= macro rows)

    @property
    def code_lo(self) -> int:
        return -(2 ** (self.adc_bits - 1))

    @property
    def code_hi(self) -> int:
        return 2 ** (self.adc_bits - 1) - 1

    @property
    def guard(self) -> float:
        return TIE_GUARD_REL * 2 ** self.adc_bits


@dataclass
class OpStats:
    """Running per-operation ADC counters (observation only)."""
    conversions: int = 0          # real (non-padded) conversions
    conversions_sim: int = 0      # simulator count incl. padded output columns
    clip_hi: int = 0              # P/LSB_P >  2^(N-1)  (simulator definition)
    clip_lo: int = 0              # P/LSB_P < -2^(N-1)
    sat_codes: int = 0            # conversions whose code saturated (grad blocked)
    abs_code_hist: torch.Tensor | None = None   # histogram of |d| (0..2^(N-1))

    def merge(self, o: "OpStats") -> None:
        self.conversions += o.conversions
        self.conversions_sim += o.conversions_sim
        self.clip_hi += o.clip_hi
        self.clip_lo += o.clip_lo
        self.sat_codes += o.sat_codes
        if o.abs_code_hist is not None:
            self.abs_code_hist = (o.abs_code_hist.clone() if self.abs_code_hist is None
                                  else self.abs_code_hist + o.abs_code_hist)

    def as_dict(self) -> dict:
        n = max(self.conversions_sim, 1)
        return dict(conversions=self.conversions, conversions_sim=self.conversions_sim,
                    clip_hi=self.clip_hi, clip_lo=self.clip_lo,
                    clip_pct_sim=100.0 * (self.clip_hi + self.clip_lo) / n,
                    sat_code_pct_real=100.0 * self.sat_codes / max(self.conversions, 1))


def _int_range(bits: int, signed: bool) -> tuple[int, int]:
    if signed:
        return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    return 0, 2 ** bits - 1


@torch.no_grad()
def quantize_operand(x: torch.Tensor, bits: int, dim: int, signed: bool,
                     eps: float = 1e-12):
    """Torch port of cim_sim.quant.quantize, in float64 (as NumPy).

    `dim` is the COLLAPSED axis (-1: one scale per token; -2: one per column).
    Returns (q as float32 integer values, scale float64 with keepdim).
    """
    qmin, qmax = _int_range(bits, signed)
    xd = x.detach().to(torch.float64)
    amax = (xd.abs() if signed else xd).amax(dim=dim, keepdim=True)
    scale = torch.clamp(amax, min=eps) / qmax
    q = torch.round(xd / scale)                 # half-to-even == np.rint
    q = torch.clamp(q, qmin, qmax)
    return q.to(torch.float32), scale


class _CIMMatmul(torch.autograd.Function):
    """Forward = simulator contract; backward = clipped STE (module docstring)."""

    @staticmethod
    def forward(ctx, X, W, spec: OpSpec, stats: OpStats | None):
        out_dtype = X.dtype
        xq, sx = quantize_operand(X, spec.act_bits, -1, spec.act_signed)
        wq, sw = quantize_operand(W, spec.weight_bits, -2, True)
        K = X.shape[-1]
        R = spec.rows
        n_db = (K + R - 1) // R
        lo, hi = spec.code_lo, spec.code_hi
        half = 2 ** (spec.adc_bits - 1)
        acc = None
        masks = []
        n_hi = n_lo = n_sat = 0
        hist = None
        for b in range(n_db):
            r0, r1 = b * R, min((b + 1) * R, K)
            P = torch.matmul(xq[..., r0:r1], wq[..., r0:r1, :])     # exact ints
            t = P.to(torch.float64) / spec.lsb_p                     # = P / LSB_P
            f = torch.floor(t + 0.5 + spec.guard)
            ok = (f >= lo) & (f <= hi)                               # not saturated
            d = torch.clamp(f, lo, hi)
            acc = d if acc is None else acc + d
            masks.append(ok)
            if stats is not None:
                n_hi += int((t > half).sum())
                n_lo += int((t < -half).sum())
                n_sat += int((~ok).sum())
                # |d| histogram via sort + searchsorted: deterministic on CUDA
                # (torch.bincount has no deterministic CUDA kernel).
                a, _ = torch.sort(d.abs().flatten())
                edges = torch.arange(half + 2, device=a.device, dtype=a.dtype) - 0.5
                cum = torch.searchsorted(a, edges)
                h = (cum[1:] - cum[:-1]).to(torch.int64)
                hist = h if hist is None else hist + h
        y = ((acc * spec.lsb_p) * sx) * sw                            # float64
        if stats is not None:
            lead = 1
            for s in P.shape[:-2]:
                lead *= s
            N, M = P.shape[-2], P.shape[-1]
            stats.conversions += lead * N * M * n_db
            stats.conversions_sim += lead * N * n_db * (-(-M // 32)) * 32
            stats.clip_hi += n_hi
            stats.clip_lo += n_lo
            stats.sat_codes += n_sat
            hist = hist.cpu()
            stats.abs_code_hist = hist if stats.abs_code_hist is None else stats.abs_code_hist + hist
        ctx.save_for_backward(xq, wq, sx.to(torch.float32), sw.to(torch.float32),
                              torch.stack(masks) if len(masks) > 1 else masks[0].unsqueeze(0))
        ctx.R, ctx.K = R, K
        ctx.x_shape, ctx.w_shape = X.shape, W.shape
        ctx.x_dtype, ctx.w_dtype = X.dtype, W.dtype
        return y.to(out_dtype)

    @staticmethod
    def backward(ctx, gy):
        xq, wq, sx, sw, masks = ctx.saved_tensors
        R, K = ctx.R, ctx.K
        g = gy.to(torch.float32) * sx * sw                 # dL/dP (before masking)
        gxq = torch.zeros_like(xq) if ctx.needs_input_grad[0] else None
        gwq = torch.zeros(xq.shape[:-2] + wq.shape[-2:], dtype=torch.float32,
                          device=wq.device) if ctx.needs_input_grad[1] else None
        for b in range(masks.shape[0]):
            r0, r1 = b * R, min((b + 1) * R, K)
            gP = g * masks[b]
            if gxq is not None:
                gxq[..., r0:r1] = torch.matmul(gP, wq[..., r0:r1, :].transpose(-1, -2))
            if gwq is not None:
                gwq[..., r0:r1, :] = torch.matmul(xq[..., r0:r1].transpose(-1, -2), gP)
        gX = (gxq / sx).to(ctx.x_dtype) if gxq is not None else None
        gW = None
        if gwq is not None:
            gW = gwq / sw
            # reduce broadcast leading dims (e.g. 2-D weight with batched X)
            while gW.dim() > len(ctx.w_shape):
                gW = gW.sum(0)
            gW = gW.to(ctx.w_dtype)
        return gX, gW, None, None


def cim_matmul_surrogate(X: torch.Tensor, W: torch.Tensor, spec: OpSpec,
                         stats: OpStats | None = None) -> torch.Tensor:
    """Differentiable X[..., N, K] @ W[..., K, M] on the modelled CIM fabric.

    Runs with autocast disabled: quantization and ADC arithmetic are defined in
    float64/float32 and are meaningless in fp16/bf16.
    """
    with torch.autocast(device_type=X.device.type, enabled=False):
        return _CIMMatmul.apply(X, W, spec, stats)
