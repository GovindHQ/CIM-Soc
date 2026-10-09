"""
macro.py — analog C2C CIM macro, and the signed-weight adapter in front of it.

HARDWARE STATUS
---------------
Hardware-confirmed (from the company / Aswani):
  * 32x32 macro, one ADC per column.
  * Multiplication happens in the CHARGE domain via a C2C capacitor ladder
    whose taps are controlled by digital weights held in SRAM.
  * Contributions of the MAC units in a column share one output node and
    combine by capacitive charge sharing, so the node develops the NORMALISED
    (averaged) sum, not the raw sum.
  * First-order behavioural relation supplied to us:

        V_OA = (1/m) * sum_j ( V_IA,j * W_j / 2^B )

Explicitly NOT confirmed:
  * how a SIGNED weight is realised physically.
  * the constant of proportionality in "V_OA ∝ ..." (whether full-scale input
    with full-scale weights lands exactly at the ADC reference).

THE SIGNED-WEIGHT PROBLEM (spec 44 / 47)
----------------------------------------
The supplied relation uses `W_j / 2^B`, which is a NON-NEGATIVE binary-weighted
fraction: it is the natural description of a capacitor tap selected by a bit
pattern. Feeding a two's-complement word into it would be a category error —
the bit pattern 1011 means -5 as a signed 4-bit number but would be read as
11/16 of the input by a capacitor ladder. Those are different physical things
and the simulator must not conflate them.

So the weight is split BEFORE it reaches the capacitor equation:

        W_q  --SignedWeightAdapter-->  (sign_j = +/-1,  |W_q,j| = magnitude)
        magnitude  ->  C2C capacitor code  |W_q,j| / 2^B     (non-negative)
        sign       ->  applied at an explicit behavioural boundary

WHY SIGN-MAGNITUDE WAS CHOSEN AS THE BEHAVIOURAL ABSTRACTION
------------------------------------------------------------
Candidates considered (spec 45):
  1. sign-magnitude behavioural abstraction      <-- selected
  2. positive/negative accumulation (two physical sub-columns)
  3. differential analog-equivalent output
  4. separate positive and negative contribution paths
  5. two's-complement straight into the ladder     <-- rejected outright

(5) is wrong for the reason above. (2), (3) and (4) are all plausible REAL
circuits, but each asserts structure nobody has told us exists — a second
column, a differential reference, a subtractor — and each would bake a
different area/energy/mismatch story into results we would then quote. (1)
asserts nothing physical at all: it is a pure behavioural statement that the
product X*W has the sign of W times the magnitude |W|, which is true of every
one of (2), (3), (4). It therefore over-commits the least while keeping signed
MAC semantics exact, and it is the easiest to replace: swap this one adapter
class and the capacitor model above it never changes.

The cost is honesty about what it is: sign-magnitude here is an ABSTRACTION,
not a claim that the silicon stores a sign bit. `describe()` says so, and the
report must too.

V_OA SIGN CONVENTION
--------------------
V_OA is returned as a SIGNED analog-equivalent voltage (0 -> 0.0 V). A real
single-supply node cannot be negative; the physical offset is re-applied at the
ADC boundary (adc.py), which is where the unipolar constraint actually exists.
Keeping V_OA signed here is what lets partial sums from different depth blocks
be accumulated without a per-block offset correction. See spec 49: a unipolar
ADC range does not mean the MAC must be unsigned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# signed weight adapter
# --------------------------------------------------------------------------- #


class SignedWeightAdapter:
    """Splits a signed integer weight into (sign, magnitude).

    This is THE replaceable boundary. If the hardware team says signed MAC is
    done by differential accumulation, replace this class (and the `combine`
    hook in C2CMacro) — nothing else in the chain changes.
    """

    name = "sign_magnitude"
    is_physical_claim = False        # behavioural abstraction, not silicon

    def __init__(self, weight_bits: int):
        self.weight_bits = int(weight_bits)
        self.w_min = -(2 ** (weight_bits - 1))
        self.w_max = 2 ** (weight_bits - 1) - 1

    def split(self, w_q: np.ndarray):
        """signed W_q -> (sign in {-1,0,+1}, magnitude >= 0)."""
        w_q = np.asarray(w_q)
        if np.any(w_q < self.w_min) or np.any(w_q > self.w_max):
            raise ValueError(
                f"weight outside signed {self.weight_bits}-bit range "
                f"[{self.w_min},{self.w_max}]")
        return np.sign(w_q).astype(np.int8), np.abs(w_q).astype(np.int64)

    def capacitor_code(self, magnitude: np.ndarray) -> np.ndarray:
        """Magnitude -> the non-negative C2C tap fraction |W| / 2^B.

        This is the ONLY quantity the capacitor ladder equation may see.
        """
        return magnitude.astype(np.float64) / (2.0 ** self.weight_bits)

    def describe(self) -> dict:
        return dict(name=self.name, weight_bits=self.weight_bits,
                    physical_claim=self.is_physical_claim,
                    note="sign handled behaviourally; magnitude drives the "
                         "capacitor code. Not a claim about the silicon.")


# --------------------------------------------------------------------------- #
# the macro
# --------------------------------------------------------------------------- #


@dataclass
class MacroNonidealities:
    """Hooks only; all-zero => ideal charge sharing."""
    cap_mismatch_sigma: float = 0.0     # per-tap capacitor mismatch
    offset_v: float = 0.0               # static node offset
    seed: int = 0

    @property
    def is_ideal(self) -> bool:
        return self.cap_mismatch_sigma == 0.0 and self.offset_v == 0.0


class C2CMacro:
    """One column-parallel 32xC C2C macro evaluating the supplied relation.

        V_OA[t, c] = (1/m) * sum_j ( V_IA[t, j] * sign[j,c] * |W[j,c]| / 2^B )

    `m` is the number of MAC units sharing the output node. It is a PHYSICAL
    property of the column (how many capacitors hang on the node), so it is the
    row count of the macro — NOT the number of rows that happen to carry real
    data in a ragged tile. Zero-padded rows still hold a capacitor and still
    divide the sum. Override only if the hardware team says taps disconnect.
    """

    def __init__(self, rows: int, weight_bits: int,
                 m: int | None = None,
                 adapter: SignedWeightAdapter | None = None,
                 nonidealities: MacroNonidealities | None = None):
        self.rows = int(rows)
        self.weight_bits = int(weight_bits)
        self.m = int(m) if m is not None else int(rows)
        self.adapter = adapter or SignedWeightAdapter(weight_bits)
        self.non = nonidealities or MacroNonidealities()
        self._rng = np.random.default_rng(self.non.seed)
        self._mismatch = None

    # -- weight programming ------------------------------------------------ #

    def program(self, w_q: np.ndarray):
        """Split the signed weight tile once, at write time.

        Returns the physical representation actually held by the array:
        (sign, capacitor_code). Kept explicit so debug output can show that the
        ladder never sees a two's-complement word.
        """
        sign, mag = self.adapter.split(w_q)
        cap = self.adapter.capacitor_code(mag)
        if self.non.cap_mismatch_sigma:
            if self._mismatch is None or self._mismatch.shape != cap.shape:
                self._mismatch = self._rng.normal(
                    1.0, self.non.cap_mismatch_sigma, size=cap.shape)
            cap = cap * self._mismatch
        return sign, cap

    # -- the analog evaluation --------------------------------------------- #

    def mac(self, v_ia: np.ndarray, sign: np.ndarray, cap: np.ndarray
            ) -> np.ndarray:
        """V_IA [T, rows] + programmed weights -> V_OA [T, cols], signed volts.

        `combine` is the signed boundary: the capacitor code is non-negative and
        the sign multiplies it here. Replacing SignedWeightAdapter with a
        differential scheme means replacing this combination, not the charge
        sharing itself.
        """
        v_ia = np.asarray(v_ia, dtype=np.float64)
        effective = self._combine(sign, cap)                  # [rows, cols]
        v_oa = (v_ia @ effective) / float(self.m)             # charge sharing
        if self.non.offset_v:
            v_oa = v_oa + self.non.offset_v
        return v_oa

    @staticmethod
    def _combine(sign: np.ndarray, cap: np.ndarray) -> np.ndarray:
        """signed behavioural boundary: +/-1 times a non-negative tap code."""
        return sign.astype(np.float64) * cap

    # -- scale ------------------------------------------------------------- #

    def gain_per_unit_v(self) -> float:
        """dV_OA / d(V_IA * W_q), i.e. 1/(m * 2^B).

        Combined with the DAC's v_per_unit this yields K_MAC. Derived from the
        architecture, never from data.
        """
        return 1.0 / (self.m * (2.0 ** self.weight_bits))

    def describe(self) -> dict:
        return dict(rows=self.rows, m=self.m, weight_bits=self.weight_bits,
                    adapter=self.adapter.describe(),
                    gain_per_unit_v=self.gain_per_unit_v(),
                    ideal=self.non.is_ideal,
                    relation="V_OA = (1/m) sum_j V_IA,j * sign_j * |W_j| / 2^B")
