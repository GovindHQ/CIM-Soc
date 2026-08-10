"""
quant.py — integer quantization primitives for the CIM array model.

Everything the array sees is an integer. This module is the boundary between
the floating-point world of the model and the integer world of the hardware.

The scheme is deliberately the simplest one that is *physically consistent*
with a CIM array that accumulates in the analog/digital domain across depth
blocks:

    y_int32[n, m] = sum_k  xq[n, k] * wq[k, m]          <-- integer MACs
    y_float[n, m] = y_int32[n, m] * s_x[n] * s_w[m]     <-- one rescale at the end

For that single end-of-chain rescale to be valid, the scales must be CONSTANT
along the contraction axis k. That is the real constraint the hardware imposes,
and it is why:

  * activation scale is per-token (one scale for a whole input vector, shared
    across all K depth blocks of that token), and
  * weight scale is per-output-column (one scale per column, shared across all
    K depth blocks that feed that column).

A per-depth-block scale would be finer-grained and more accurate, but it would
force a rescale-and-requantize between every depth block, which defeats the
point of the cheap integer accumulator. Flagged here so the assumption is
visible rather than buried.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Integer range helpers
# --------------------------------------------------------------------------- #


def int_range(bits: int, signed: bool = True) -> tuple[int, int]:
    """Representable integer range for a given bit width.

    Signed uses two's-complement range [-2^(b-1), 2^(b-1)-1], e.g. 4b -> [-8, 7].
    Unsigned uses [0, 2^b - 1], e.g. 4b -> [0, 15].
    """
    if signed:
        return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    return 0, 2**bits - 1


# --------------------------------------------------------------------------- #
# Quantize / dequantize
# --------------------------------------------------------------------------- #


def quantize(
    x: np.ndarray,
    bits: int = 4,
    axis: int | None = None,
    signed: bool = True,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric (zero-point-free) quantization to `bits` integer levels.

    Args:
        x:      float array to quantize.
        bits:   integer bit width (4 for the current operating point).
        axis:   axis ALONG WHICH the max is taken, i.e. the axis that is
                *collapsed*. `axis=1` on an [N, K] activation matrix gives one
                scale per token (per row). `axis=0` on a [K, M] weight matrix
                gives one scale per output column. `None` gives a single
                per-tensor scale.
        signed: True  -> levels span [-2^(b-1), 2^(b-1)-1], scale = amax / qmax_pos
                False -> levels span [0, 2^b - 1], for non-negative data such as
                         post-softmax attention probabilities, where a sign bit
                         would be wasted.

    Returns:
        (q, scale) where q is an int32 array of the same shape as x, and scale
        broadcasts against x. Dequantization is simply q * scale.
    """
    qmin, qmax = int_range(bits, signed)

    if signed:
        # Use the positive limit so the mapping is symmetric about zero; the
        # extra negative code (-8 in 4b) is left as clipping headroom.
        amax = np.max(np.abs(x), axis=axis, keepdims=True)
        scale = np.maximum(amax, eps) / qmax
    else:
        amax = np.max(x, axis=axis, keepdims=True)
        scale = np.maximum(amax, eps) / qmax

    q = np.rint(x / scale).astype(np.int32)
    np.clip(q, qmin, qmax, out=q)
    return q, scale


def dequantize(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Map integers back to float."""
    return q.astype(np.float64) * scale


def fake_quant(x: np.ndarray, bits: int = 4, axis: int | None = None,
               signed: bool = True) -> np.ndarray:
    """Quantize then immediately dequantize. Useful for isolating the error
    contributed by quantization alone, with no tiling involved."""
    q, s = quantize(x, bits=bits, axis=axis, signed=signed)
    return dequantize(q, s)


# --------------------------------------------------------------------------- #
# Error metrics
# --------------------------------------------------------------------------- #


def rel_error(approx: np.ndarray, ref: np.ndarray) -> float:
    """Relative Frobenius-norm error, as a fraction (multiply by 100 for %)."""
    denom = np.linalg.norm(ref)
    if denom == 0:
        return float(np.linalg.norm(approx))
    return float(np.linalg.norm(approx - ref) / denom)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(a @ b / (na * nb))
