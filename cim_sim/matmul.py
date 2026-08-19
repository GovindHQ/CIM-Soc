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
With a single array the loop nest is:

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

MULTIPLE ARRAYS — OUTPUT-TILE PARALLELISM
------------------------------------------
Pass `arrays=[arr0, arr1, ...]` instead of a single `array` to schedule across
P identical CIMArray macros. Output tiles (not depth blocks, not tokens) are
what get spread across arrays: the M axis is divided into ceil(M/32) output
tiles, and those tiles are dealt out to arrays P at a time ("parallel groups").
Within one group, one depth block's activation tile is broadcast to every
active array simultaneously, each holding a DIFFERENT output tile's weights:

    for token_block tb:
        for group in ceil(output_tiles / P):
            for depth_block db:                      <-- one hardware timestep
                for slot, ob in enumerate(tiles in this group):
                    arrays[slot].load_weight_tile(W[db, ob])
                    acc[tb, ob] += arrays[slot].mvm(Xq[tb, db])

If P does not divide the tile count, the last group is partially populated
(some array slots idle that group) — see CIMConfig.n_arrays and the
`array_parallel_utilization` field on MatmulTrace.

This governs SCHEDULING only. The digital accumulation across depth blocks,
the ADC (owned per-array, unchanged), quantization, and token blocking are
untouched — addition is commutative, so which array produced which partial
sum does not affect the numerical result. With P=1 the loop above collapses
to exactly the single-array loop nest, in the exact same call order, so
`arrays=[array]` reproduces `array=array` bit-for-bit.

CYCLE MODEL
-----------
`CIMStats.array_ops` (and the other per-array counters) are summed across all
P arrays, same as before — they represent total physical macro work, useful
for energy accounting. They are NOT hardware wall-clock time when P > 1: four
arrays firing in the same group is one hardware timestep, not four. Wall-clock
time is tracked separately, on MatmulTrace.parallel_array_cycles, incremented
once per (group, depth_block) — i.e. once per token-block-worth of cycles,
regardless of how many of the P arrays were active in that group.
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

    # Multi-array scheduling info. num_arrays=1 (the default/single-array case)
    # gives parallel_groups == output_blocks, array_slots == output_blocks,
    # active_array_invocations == output_blocks, utilization == 1.0 — i.e. this
    # section is inert and uninformative for the single-array path.
    num_arrays: int = 1
    parallel_groups: int = 0
    active_array_invocations: int = 0
    array_slots: int = 0
    parallel_array_cycles: int = 0

    @property
    def array_parallel_utilization(self) -> float:
        """active array slots / total array slots across all parallel groups.

        Separate from CIMStats.utilization (fraction of MACs inside one 32x32
        tile that are real vs. zero-padding) — this measures how fully the P
        parallel macros are populated with distinct output tiles, not how full
        any one macro's rows/columns are.
        """
        return (self.active_array_invocations / self.array_slots
                if self.array_slots else 0.0)

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

    def total_parallel_cycles(self) -> int:
        """Wall-clock array-timestep count across every matmul in this log.

        For any record with num_arrays=1 this equals that record's own
        array_ops (see MatmulTrace docstring / TEST 1 in test_multi_array.py),
        so this sum is directly comparable to CIMStats.array_ops from a
        single-array run of the same workload — it is what "cycles" already
        meant before multi-array existed, just correctly counted when P > 1.
        """
        return sum(r.parallel_array_cycles for r in self.records)

    def parallel_utilization(self) -> float:
        """Weighted array_parallel_utilization across every matmul in this log
        (weighted by array_slots, not a plain average of per-call fractions)."""
        slots = sum(r.array_slots for r in self.records)
        active = sum(r.active_array_invocations for r in self.records)
        return active / slots if slots else 0.0

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
    array: CIMArray | None = None,
    tag: str = "matmul",
    log: TraceLog | None = None,
    act_signed: bool = True,
    quantize_weights: bool = True,
    arrays: list[CIMArray] | None = None,
) -> np.ndarray:
    """Compute X @ W on the simulated CIM fabric.

    Args:
        X: [N, K] float activations. Rows are tokens (or output pixels, for an
           im2col'd convolution — the scheduler does not care which).
        W: [K, M] float weights. Held stationary in the array(s).
        array: a single CIMArray instance (unchanged single-array path).
        arrays: OR a list of P identical CIMArray instances to schedule across
           using output-tile parallelism (see module docstring). Exactly one
           of `array` / `arrays` must be given. `arrays=[a]` (a length-1 list)
           is equivalent to `array=a`.
        tag: label for the trace record, e.g. "qkv_proj", "qk_t", "av".
        act_signed: False for non-negative activations such as post-softmax
           attention probabilities, where a sign bit would be wasted.

    Returns:
        [N, M] float result.
    """
    if arrays is not None:
        array_list = list(arrays)
        assert len(array_list) >= 1, "`arrays` must contain at least one CIMArray"
    elif array is not None:
        array_list = [array]
    else:
        raise ValueError("cim_matmul requires either `array` or `arrays`")

    cfg = array_list[0].cfg
    R, C = cfg.rows, cfg.cols
    for a in array_list[1:]:
        assert (a.cfg.rows, a.cfg.cols) == (R, C), \
            "all arrays in `arrays` must share the same rows/cols geometry"
        assert (a.cfg.weight_bits, a.cfg.act_bits) == (cfg.weight_bits, cfg.act_bits), \
            "all arrays in `arrays` must share the same operand precision"
    P = len(array_list)

    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    assert X.ndim == 2 and W.ndim == 2, "cim_matmul takes 2-D operands"
    N, K = X.shape
    assert W.shape[0] == K, f"inner dimensions disagree: {X.shape} @ {W.shape}"
    M = W.shape[1]

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
    n_ob = (M + C - 1) // C          # output blocks (== "output tiles")
    Kp, Mp = n_db * R, n_ob * C

    xq_p = np.zeros((N, Kp), dtype=np.int32)
    xq_p[:, :K] = xq
    wq_p = np.zeros((Kp, Mp), dtype=np.int32)
    wq_p[:K, :M] = wq

    # -- 3. token blocking (unchanged) --------------------------------------- #
    T = cfg.token_block or N
    T = max(1, min(T, N))
    n_tb = (N + T - 1) // T

    # -- 3b. multi-array scheduling geometry --------------------------------- #
    # Output tiles are dealt out P at a time ("parallel groups"). P=1 gives
    # parallel_groups == n_ob, i.e. one tile per group — the single-array case.
    parallel_groups = (n_ob + P - 1) // P
    array_slots = parallel_groups * P
    active_array_invocations = n_ob
    peak_active_per_group = min(P, n_ob) if n_ob else 0

    acc = np.zeros((N, Mp), dtype=np.int64)

    stats_before = [_snapshot(a.stats) for a in array_list]
    parallel_array_cycles = 0

    # -- 4. the loop nest --------------------------------------------------- #
    # Depth block is the outer loop within a group: one activation tile is
    # broadcast to every active array in that group before moving to the next
    # depth block, matching the "x_tile fans out to CIM[0..P-1] simultaneously"
    # hardware picture. With P=1 this generates the exact same (ob, db) call
    # sequence, in the exact same order, as the previous single-array nesting.
    for tb in range(n_tb):
        t0, t1 = tb * T, min((tb + 1) * T, N)
        for group in range(parallel_groups):
            ob_lo = group * P
            ob_hi = min(ob_lo + P, n_ob)
            group_obs = range(ob_lo, ob_hi)   # output tiles active this group

            for db in range(n_db):
                r0 = db * R
                useful_rows = min(R, K - r0)

                for slot, ob in enumerate(group_obs):   # one array per tile
                    c0 = ob * C
                    useful_cols = min(C, M - c0)
                    arr = array_list[slot]

                    arr.load_weight_tile(wq_p[r0:r0 + R, c0:c0 + C],
                                         tile_id=(id(W), db, ob))
                    y = arr.mvm(xq_p[t0:t1, r0:r0 + R],
                               useful_rows=useful_rows,
                               useful_cols=useful_cols)
                    acc[t0:t1, c0:c0 + C] += y

                # All arrays active in this group ran depth block `db` at the
                # same modeled hardware timestep: count it once, not P times.
                parallel_array_cycles += (t1 - t0)

    # -- 5. single end-of-chain rescale ------------------------------------- #
    out = acc[:, :M].astype(np.float64) * sx * sw

    # -- bookkeeping ---------------------------------------------------------#
    # PSUM is a single shared digital accumulator downstream of all P arrays
    # (see module docstring diagram), so its peak-usage bookkeeping is not any
    # one macro's property. array_list[0] is used as the designated location
    # for this running counter, exactly as the (only) array already was in the
    # single-array case — this is bookkeeping placement only, not a numerics
    # change, and does not affect any array's own MAC/write/ADC counters.
    psum_peak = T * C * peak_active_per_group
    array_list[0].stats.max_abs_accum = max(array_list[0].stats.max_abs_accum,
                                            int(np.abs(acc).max()) if acc.size else 0)
    array_list[0].stats.psum_words_live = max(array_list[0].stats.psum_words_live,
                                              psum_peak)

    if log is not None:
        agg = CIMStats()
        for a, before in zip(array_list, stats_before):
            agg.merge(_delta(before, a.stats))
        agg.psum_words_live = psum_peak
        log.add(MatmulTrace(tag=tag, N=N, K=K, M=M,
                            depth_blocks=n_db, output_blocks=n_ob,
                            token_blocks=n_tb, stats=agg,
                            num_arrays=P,
                            parallel_groups=parallel_groups,
                            active_array_invocations=active_array_invocations,
                            array_slots=array_slots,
                            parallel_array_cycles=parallel_array_cycles))
    return out


def cim_matmul_batched(
    X: np.ndarray,
    W: np.ndarray,
    array: CIMArray | None = None,
    tag: str = "matmul",
    log: TraceLog | None = None,
    act_signed: bool = True,
    arrays: list[CIMArray] | None = None,
) -> np.ndarray:
    """Batched X @ W over arbitrary leading dimensions.

    X: [..., N, K], W: [..., K, M]. Leading dims must match. Each batch element
    is a separate scheduling problem on the same physical array(s), executed in
    sequence — which is exactly what a fabric does with per-head attention
    matmuls. Pass `arrays=[...]` to schedule each batch element's output tiles
    across P arrays; see `cim_matmul`.
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
                            act_signed=act_signed, arrays=arrays)
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