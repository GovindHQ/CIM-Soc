"""
matmul.py — the scheduler that turns an arbitrary matmul into 32x32 array ops.

This is the layer that replaces torch.matmul. Given X [N, K] and W [K, M] of any
shape, it:

  1. quantizes X per token and W per output column (see quant.py for why the
     scales must be constant along K),
  2. zero-pads K up to a multiple of 32 (depth blocks) and M up to a multiple of
     32 (output blocks) — the padding is exactly the "idle rows / idle columns"
     of the real array,
  3. walks token blocks x output blocks x depth blocks, loading one weight tile
     per (depth, output) pair and streaming the token block through it,
  4. accumulates the column sums across depth blocks in a digital integer
     accumulator,
  5. applies the single end-of-chain rescale back to float.

LOOP ORDER / TOKEN BLOCKING
---------------------------
The loop nest is:

    for token_block tb in ceil(N / T):
        for output_block ob in ceil(M / 32):
            for depth_block db in ceil(K / 32):
                load tile W[db, ob]                 <-- one analog write
                acc[tb, ob] += array.mvm(Xq[tb, db])

so weight writes = ceil(N/T) * ceil(M/32) * ceil(K/32), and the partial sums
that must stay live are T * 32 words.

T is the single knob trading weight-write energy against partial-sum storage:

    T = N   -> ceil(M/32)*ceil(K/32) writes, N*32 psum words (needs SRAM)
    T = 1   -> N times more writes, 32 psum words (fits in a register file)

Both extremes produce bit-identical results. Only the counters change. That is
the point of exposing T rather than hardcoding a loop order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .array import CIMArray, CIMConfig, CIMStats
from .quant import quantize


# --------------------------------------------------------------------------- #
# Per-call trace record
# --------------------------------------------------------------------------- #


@dataclass
class MatmulTrace:
    """What one logical matmul cost on the fabric."""

    tag: str
    N: int
    K: int
    M: int
    depth_blocks: int
    output_blocks: int
    token_blocks: int
    stats: CIMStats

    @property
    def tiles(self) -> int:
        return self.depth_blocks * self.output_blocks

    @property
    def row_fill(self) -> float:
        """Fraction of array rows carrying real data, averaged over depth blocks."""
        return self.K / (self.depth_blocks * 32)

    @property
    def col_fill(self) -> float:
        return self.M / (self.output_blocks * 32)


class TraceLog:
    """Collects traces across a whole forward pass, keyed by tag."""

    def __init__(self):
        self.records: list[MatmulTrace] = []

    def add(self, rec: MatmulTrace) -> None:
        self.records.append(rec)

    def total(self) -> CIMStats:
        agg = CIMStats()
        for r in self.records:
            agg.merge(r.stats)
        return agg

    def by_tag(self) -> dict[str, CIMStats]:
        out: dict[str, CIMStats] = {}
        for r in self.records:
            out.setdefault(r.tag, CIMStats()).merge(r.stats)
        return out

    def clear(self) -> None:
        self.records.clear()


# --------------------------------------------------------------------------- #
# Core scheduler
# --------------------------------------------------------------------------- #


def cim_matmul(
    X: np.ndarray,
    W: np.ndarray,
    array: CIMArray,
    tag: str = "matmul",
    log: TraceLog | None = None,
    act_signed: bool = True,
    quantize_weights: bool = True,
) -> np.ndarray:
    """Compute X @ W on the simulated CIM fabric.

    Args:
        X: [N, K] float activations. Rows are tokens (or output pixels, for an
           im2col'd convolution — the scheduler does not care which).
        W: [K, M] float weights. Held stationary in the array.
        array: the CIMArray instance. Its stats accumulate across calls.
        tag: label for the trace record, e.g. "qkv_proj", "qk_t", "av".
        act_signed: False for non-negative activations such as post-softmax
           attention probabilities, where a sign bit would be wasted.

    Returns:
        [N, M] float result.
    """
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    assert X.ndim == 2 and W.ndim == 2, "cim_matmul takes 2-D operands"
    N, K = X.shape
    assert W.shape[0] == K, f"inner dimensions disagree: {X.shape} @ {W.shape}"
    M = W.shape[1]

    cfg = array.cfg
    R, C = cfg.rows, cfg.cols

    # -- 1. quantize -------------------------------------------------------- #
    # Activation scale: one per token, constant along K (axis=1 is collapsed).
    xq, sx = quantize(X, bits=cfg.act_bits, axis=1, signed=act_signed)
    # Weight scale: one per output column, constant along K (axis=0 collapsed).
    if quantize_weights:
        wq, sw = quantize(W, bits=cfg.weight_bits, axis=0, signed=True)
    else:
        wq, sw = W.astype(np.int32), np.ones((1, M))

    # -- 2. pad to tile boundaries ------------------------------------------ #
    n_db = (K + R - 1) // R          # depth blocks
    n_ob = (M + C - 1) // C          # output blocks
    Kp, Mp = n_db * R, n_ob * C

    xq_p = np.zeros((N, Kp), dtype=np.int32)
    xq_p[:, :K] = xq
    wq_p = np.zeros((Kp, Mp), dtype=np.int32)
    wq_p[:K, :M] = wq

    # -- 3. token blocking -------------------------------------------------- #
    T = cfg.token_block or N
    T = max(1, min(T, N))
    n_tb = (N + T - 1) // T

    acc = np.zeros((N, Mp), dtype=np.int64)

    stats_before = _snapshot(array.stats)

    # -- 4. the loop nest --------------------------------------------------- #
    for tb in range(n_tb):
        t0, t1 = tb * T, min((tb + 1) * T, N)
        for ob in range(n_ob):
            c0 = ob * C
            useful_cols = min(C, M - c0)
            for db in range(n_db):
                r0 = db * R
                useful_rows = min(R, K - r0)

                array.load_weight_tile(wq_p[r0:r0 + R, c0:c0 + C],
                                       tile_id=(id(W), db, ob))
                y = array.mvm(xq_p[t0:t1, r0:r0 + R],
                              useful_rows=useful_rows,
                              useful_cols=useful_cols)
                acc[t0:t1, c0:c0 + C] += y

    # -- 5. single end-of-chain rescale ------------------------------------- #
    out = acc[:, :M].astype(np.float64) * sx * sw

    # -- bookkeeping -------------------------------------------------------- #
    array.stats.max_abs_accum = max(array.stats.max_abs_accum,
                                    int(np.abs(acc).max()) if acc.size else 0)
    array.stats.psum_words_live = max(array.stats.psum_words_live, T * C)

    if log is not None:
        delta = _delta(stats_before, array.stats)
        delta.psum_words_live = T * C
        log.add(MatmulTrace(tag=tag, N=N, K=K, M=M,
                            depth_blocks=n_db, output_blocks=n_ob,
                            token_blocks=n_tb, stats=delta))
    return out


def cim_matmul_batched(
    X: np.ndarray,
    W: np.ndarray,
    array: CIMArray,
    tag: str = "matmul",
    log: TraceLog | None = None,
    act_signed: bool = True,
) -> np.ndarray:
    """Batched X @ W over arbitrary leading dimensions.

    X: [..., N, K], W: [..., K, M]. Leading dims must match. Each batch element
    is a separate scheduling problem on the same physical array, executed in
    sequence — which is exactly what a single-array fabric does with per-head
    attention matmuls.
    """
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    lead = X.shape[:-2]
    assert W.shape[:-2] == lead, "leading dimensions must match"

    N, K = X.shape[-2:]
    M = W.shape[-1]
    Xf = X.reshape(-1, N, K)
    Wf = W.reshape(-1, K, M)

    out = np.empty((Xf.shape[0], N, M), dtype=np.float64)
    for b in range(Xf.shape[0]):
        out[b] = cim_matmul(Xf[b], Wf[b], array, tag=tag, log=log,
                            act_signed=act_signed)
    return out.reshape(*lead, N, M)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def _snapshot(s: CIMStats) -> CIMStats:
    return CIMStats(**vars(s))


def _delta(before: CIMStats, after: CIMStats) -> CIMStats:
    d = CIMStats()
    for k in vars(before):
        if k in ("psum_words_live", "max_abs_colsum", "max_abs_accum"):
            setattr(d, k, getattr(after, k))
        else:
            setattr(d, k, getattr(after, k) - getattr(before, k))
    return d
