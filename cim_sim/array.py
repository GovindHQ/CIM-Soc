"""
array.py — the 32x32 CIM array primitive and its activity counters.

This is the only place where "compute" happens. Everything above it in the
stack is scheduling.

PHYSICAL MODEL
--------------
  * The array is ROWS x COLS = 32 x 32.
  * A weight tile of 32x32 4-bit integers is written into the array and stays
    resident (weight-stationary).
  * One input vector of 32 4-bit integers is broadcast along the rows.
  * Each column produces the sum of its 32 row-wise products. 32 column sums
    come out per input vector.
  * ONE bit plane is used in this model, so every distinct weight tile costs a
    real analog write. (With four resident planes, up to four tiles would be
    selectable by a control signal at zero write cost; that is deliberately
    switched off here.)

WHAT IS AND IS NOT MODELLED
---------------------------
  Modelled: integer arithmetic at 4b x 4b (or 8b x 8b), exact tile geometry,
            zero-padding of ragged edges, weight-write counts, per-cycle input
            broadcast counts, column-sum readout counts, row/column
            utilization, and (when cfg.adc_enabled) an 8-bit unipolar ADC on
            every column output, applied once per depth block.
  Not modelled (this version): DAC non-linearity, analog noise, IR drop, write
            settle time, multiple parallel arrays, multi-plane residency,
            bit-slicing to 8b.

With 32 rows of 4b x 4b products the worst case column-sum magnitude is
32 * 8 * 8 = 2048, so the exact (pre-ADC) sum needs ~12 bits before any
cross-depth accumulation, and 12 + ceil(log2(n_depth_blocks)) bits after.

ADC MODEL (cfg.adc_enabled)
----------------------------
Every column has its own ADC (32 columns -> 32 ADCs), applied to that column's
output AFTER each single-depth-block MVM and BEFORE cross-depth digital
accumulation (that accumulation happens one level up, in matmul.py). The
per-tile analog column sum is not itself accumulated across depth blocks; only
its digitized value is.

The ADC is modelled functionally: 8-bit resolution, unipolar input range
[0, Vref], Vref = 0.6 V by default (cfg.adc_bits, cfg.adc_vref).

PHYSICAL ASSUMPTION — column sum -> ADC input voltage
------------------------------------------------------
The column sum out of the array is signed (two's-complement quantized
operands), but the ADC is unipolar. The existing code does not specify a
voltage mapping directly, but it DOES already define, in this docstring, the
configuration's worst-case column-sum magnitude:

    full_scale = rows * 2^(weight_bits - 1) * 2^(act_bits - 1)

This is data-independent (depends only on array geometry and operand bit
widths, never on the values being multiplied), which is the property a real
ADC full-scale must have — it is sized for what the array CAN produce, not for
what a particular input happens to produce.

That bound is used as the ADC's symmetric input range, mapped onto [0, Vref]
with offset-binary encoding:

    V_in(y) = Vref * (y + full_scale) / (2 * full_scale)      y = -FS -> 0 V
                                                                 y = +FS -> Vref
    code     = round(V_in / Vref * (2^adc_bits - 1)), clipped to [0, 2^adc_bits-1]

The digitized value handed back to the scheduler is the code decoded back into
the same signed column-sum units used everywhere else in the pipeline, so the
accumulator contract (`acc += y`) is unchanged; the ADC's effect is purely a
resolution loss (256 levels across the full symmetric range) plus saturation
clipping at +/- full_scale.

KNOWN LIMITATION: full_scale assumes both operands use the full signed
two's-complement range. The one place that isn't true is post-softmax AV
activations, which are quantized unsigned (0..2^act_bits-1, see
quant.act_signed=False) and so never reach the negative extreme. full_scale is
therefore conservative (slightly coarser ADC step size than strictly
necessary) for that case — never wrong, just not maximally tight. Fixing this
exactly would require threading operand signedness into CIMArray, which is out
of scope for this change.

BATCHING NOTE
-------------
`mvm` accepts a batch of input vectors [T, 32] and returns [T, 32]. That batch
axis is NOT parallel hardware — it is T CONSECUTIVE CYCLES through the same
stationary weight tile. It exists so the simulation runs at a usable speed; the
counters treat it as T separate array operations, which is what it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .quant import int_range


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class CIMConfig:
    """Hardware parameters of the simulated fabric."""

    rows: int = 32          # contraction depth held by one array
    cols: int = 32          # output width held by one array
    weight_bits: int = 4    # bits per weight cell
    act_bits: int = 4       # bits per broadcast input
    planes: int = 1         # resident weight planes (1 => every tile is a write)
    n_arrays: int = 1       # parallel arrays (1 => everything is serialized)

    # Token block size T. Weight tiles are held stationary while T tokens are
    # streamed through them. T = None means "all tokens" (maximum weight reuse,
    # maximum partial-sum storage). T = 1 is the opposite extreme.
    token_block: int | None = None

    # ADC on every column output, applied per depth block before cross-depth
    # accumulation. Disabled by default so existing exact-integer behavior is
    # unchanged unless explicitly opted in.
    adc_enabled: bool = False
    adc_bits: int = 10
    adc_vref: float = 0.6   # volts, unipolar full-scale reference

    def __post_init__(self):
        if self.planes != 1:
            raise NotImplementedError(
                "This model is deliberately single-plane. Set planes=1."
            )
        # n_arrays is descriptive here: CIMArray itself always models exactly
        # ONE 32x32 macro regardless of this field's value. Multiple macros are
        # built by the caller as `[CIMArray(cfg) for _ in range(cfg.n_arrays)]`
        # and scheduled across output tiles by matmul.py's cim_matmul(arrays=...).


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


@dataclass
class CIMStats:
    """Activity counters. All counts are physical events, not FLOPs."""

    weight_tile_writes: int = 0     # 32x32 analog tiles written
    weight_cell_writes: int = 0     # individual cells written
    array_ops: int = 0              # input vectors broadcast (= array cycles)
    column_sums_read: int = 0       # column-sum readouts (= ADC conversions)
    macs_issued: int = 0            # 32*32 per array_op, padding included
    macs_useful: int = 0            # MACs on real (non-padded) data
    psum_words_live: int = 0        # peak partial-sum words that must be held
    max_abs_colsum: int = 0         # largest column sum observed, pre-accum
    max_abs_accum: int = 0          # largest value after depth accumulation
    adc_conversions: int = 0        # ADC conversions performed (0 if disabled)
    adc_saturations: int = 0        # of those, how many clipped to +/- full_scale

    def merge(self, other: "CIMStats") -> None:
        self.weight_tile_writes += other.weight_tile_writes
        self.weight_cell_writes += other.weight_cell_writes
        self.array_ops += other.array_ops
        self.column_sums_read += other.column_sums_read
        self.macs_issued += other.macs_issued
        self.macs_useful += other.macs_useful
        self.psum_words_live = max(self.psum_words_live, other.psum_words_live)
        self.max_abs_colsum = max(self.max_abs_colsum, other.max_abs_colsum)
        self.max_abs_accum = max(self.max_abs_accum, other.max_abs_accum)
        self.adc_conversions += other.adc_conversions
        self.adc_saturations += other.adc_saturations

    @property
    def utilization(self) -> float:
        """Fraction of issued MACs that did useful work (1.0 = no padding waste)."""
        return self.macs_useful / self.macs_issued if self.macs_issued else 0.0

    @property
    def colsum_bits(self) -> int:
        """Bits needed to represent a single-tile column sum (signed)."""
        return int(np.ceil(np.log2(max(self.max_abs_colsum, 1) + 1))) + 1

    @property
    def accum_bits(self) -> int:
        """Bits needed for the cross-depth accumulator (signed)."""
        return int(np.ceil(np.log2(max(self.max_abs_accum, 1) + 1))) + 1


# --------------------------------------------------------------------------- #
# ADC helper
# --------------------------------------------------------------------------- #


def _adc_convert(y: np.ndarray, bits: int, full_scale: int
                 ) -> tuple[np.ndarray, int]:
    """Functional 10-bit (or `bits`-bit) unipolar ADC.

    Maps the signed column sum `y` onto a unipolar [0, Vref] input using
    offset-binary encoding around a fixed, data-independent `full_scale` (see
    the "PHYSICAL ASSUMPTION" note in this module's docstring), then decodes
    the resulting code back into the same signed integer domain the rest of
    the pipeline uses, so callers see only a resolution/saturation loss, not a
    representation change.

    Args:
        y: exact signed column sums, any shape.
        bits: ADC resolution.
        full_scale: symmetric input range [-full_scale, +full_scale] that maps
            onto [0, Vref]. Must be a configuration-derived constant, not
            computed from `y` itself (a data-dependent full scale would make
            the ADC input-dependent, which is not physical).

    Returns:
        (y_digitized, n_saturated) where y_digitized is the same shape as y,
        rounded to the nearest representable integer, and n_saturated counts
        how many entries were clipped to +/- full_scale before conversion.
    """
    qmax = 2 ** bits - 1
    fs = full_scale

    y_clipped = np.clip(y, -fs, fs)
    n_sat = int(np.sum(y != y_clipped))

    # V_in / Vref, in [0, 1]; Vref itself cancels out of this normalized form.
    v_norm = (y_clipped.astype(np.float64) + fs) / (2.0 * fs)
    code = np.clip(np.rint(v_norm * qmax), 0, qmax)

    # Decode the code back into signed column-sum units.
    y_digitized = np.rint(code / qmax * (2.0 * fs) - fs).astype(np.int64)
    return y_digitized, n_sat


# --------------------------------------------------------------------------- #
# The array
# --------------------------------------------------------------------------- #


class CIMArray:
    """A single 32x32 weight-stationary CIM array."""

    def __init__(self, cfg: CIMConfig | None = None):
        self.cfg = cfg or CIMConfig()
        self.stats = CIMStats()
        self._W: np.ndarray | None = None       # resident weight tile
        self._W_id: int | None = None           # identity of resident tile

        self.w_min, self.w_max = int_range(self.cfg.weight_bits, signed=True)

        # Data-independent ADC full-scale: the worst-case column sum this
        # geometry and these operand bit widths can ever produce. See the
        # "PHYSICAL ASSUMPTION" note in the module docstring.
        self.adc_full_scale = (self.cfg.rows
                               * 2 ** (self.cfg.weight_bits - 1)
                               * 2 ** (self.cfg.act_bits - 1))

    # ---- weight path ---------------------------------------------------- #

    def load_weight_tile(self, W: np.ndarray, tile_id: int | None = None) -> None:
        """Write a 32x32 integer weight tile into the array.

        With a single bit plane there is no cheap plane-select alternative, so
        this always counts as a physical write unless the exact same tile is
        already resident (which the scheduler avoids anyway).
        """
        assert W.shape == (self.cfg.rows, self.cfg.cols), \
            f"weight tile must be {self.cfg.rows}x{self.cfg.cols}, got {W.shape}"
        assert W.min() >= self.w_min and W.max() <= self.w_max, \
            "weight tile outside representable integer range"

        if tile_id is not None and tile_id == self._W_id:
            return  # already resident, no write

        self._W = W.astype(np.int32)
        self._W_id = tile_id
        self.stats.weight_tile_writes += 1
        self.stats.weight_cell_writes += self.cfg.rows * self.cfg.cols

    # ---- compute path --------------------------------------------------- #

    def mvm(self, x: np.ndarray, useful_rows: int | None = None,
            useful_cols: int | None = None) -> np.ndarray:
        """Broadcast input vectors along the rows, read out the column sums.

        Args:
            x: [T, 32] int array. T is a count of CONSECUTIVE CYCLES, not
               parallel hardware.
            useful_rows / useful_cols: how much of the 32x32 tile is real data
               rather than zero padding. Used for the utilization counters only;
               it does not change the arithmetic (padding is exact zeros).

        Returns:
            [T, 32] int32 column sums.
        """
        if self._W is None:
            raise RuntimeError("no weight tile resident; call load_weight_tile first")

        x = np.atleast_2d(x).astype(np.int32)
        assert x.shape[1] == self.cfg.rows, \
            f"input vector must have {self.cfg.rows} entries, got {x.shape[1]}"

        T = x.shape[0]
        y = x @ self._W                      # [T, 32] exact integer column sums

        ur = self.cfg.rows if useful_rows is None else useful_rows
        uc = self.cfg.cols if useful_cols is None else useful_cols

        s = self.stats
        s.array_ops += T
        s.column_sums_read += T * self.cfg.cols
        s.macs_issued += T * self.cfg.rows * self.cfg.cols
        s.macs_useful += T * ur * uc
        if y.size:
            # Recorded from the exact analog sum, before any ADC digitization,
            # since this is the number that sizes the ADC in the first place.
            s.max_abs_colsum = max(s.max_abs_colsum, int(np.abs(y).max()))

        if self.cfg.adc_enabled:
            # One ADC per column, applied to this single depth block's output,
            # before matmul.py accumulates it across depth blocks.
            y, n_sat = _adc_convert(y, self.cfg.adc_bits, self.adc_full_scale)
            s.adc_conversions += T * self.cfg.cols
            s.adc_saturations += n_sat

        return y

    # ---- housekeeping --------------------------------------------------- #

    def reset_stats(self) -> None:
        self.stats = CIMStats()