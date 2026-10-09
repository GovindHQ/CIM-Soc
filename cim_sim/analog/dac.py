"""
dac.py — INPUT DAC family:  signed quantized activation  ->  analog voltage V_IA.

This is the INPUT DAC of the CIM datapath. It is NOT the SAR ADC's internal
DAC (see adc.py / SARDAC) — the two are deliberately separate modules.

HARDWARE STATUS
---------------
  Hardware-confirmed : nothing. The company has not specified the input DAC
                       topology. Only the ADC side (SAR, ~0-0.6 V, 8b, one per
                       column) and the 32x32 C2C macro are described.
  Professor-directed : evaluate plausible architectures experimentally.
  Therefore          : every class here is a CANDIDATE, not "the hardware".

Each architecture derives its transfer function from its own network, and each
declares how it represents a SIGNED integer. They do NOT all use the same
signed encoding — that is one of the things being investigated.

SIGNED ANALOG-EQUIVALENT CONVENTION
-----------------------------------
`transfer()` returns V_IA as a SIGNED analog-equivalent voltage: 0 maps to
0.0 V. For architectures that are physically unipolar-with-offset (R-2R,
charge-redistribution), the physical node voltage is
    V_phys = V_IA + V_cm,     V_cm = common-mode (mid-scale) voltage
and `v_common_mode` reports V_cm. Working in the signed-equivalent domain keeps
the C2C sum and the digital accumulator mathematically honest; the offset is
re-applied at the ADC boundary (see adc.py), which is where the physical
unipolar constraint actually lives.

LINEARITY / K_MAC
-----------------
Every architecture here is linear in X_q in its ideal mode, so each exposes
`v_per_unit` = dV_IA/dX_q. That single number is what the chain uses to derive
K_MAC ONCE per configuration (see chain.py). It is a property of the
ARCHITECTURE, never of the data — no DAC here ever inspects the tensor to pick
a scale. Architectures differ in v_per_unit, which is precisely why they are
worth comparing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------- #
# nonideality container (hooks only; everything defaults to ideal)
# --------------------------------------------------------------------------- #


@dataclass
class DACNonidealities:
    """All-zero by default => ideal DAC. Hooks for later, not used unless set.

    gain_error   : fractional error on the overall transfer slope (0.01 = +1%).
    offset_v     : additive output offset in volts.
    mismatch_sigma: relative sigma of per-element (resistor/capacitor/current
                   source) mismatch. Element mismatch is drawn ONCE at
                   construction, not per sample, because physical mismatch is
                   static per die.
    """

    gain_error: float = 0.0
    offset_v: float = 0.0
    mismatch_sigma: float = 0.0
    seed: int = 0

    @property
    def is_ideal(self) -> bool:
        return (self.gain_error == 0.0 and self.offset_v == 0.0
                and self.mismatch_sigma == 0.0)


# --------------------------------------------------------------------------- #
# base
# --------------------------------------------------------------------------- #


class BaseDAC:
    """Interface every input-DAC architecture implements.

    The C2C macro must not care which subclass it is handed. It receives V_IA
    and nothing else.
    """

    name = "base"
    signed_encoding = "undefined"
    produces = "voltage"

    def __init__(self, bits: int, vref: float = 0.6,
                 nonidealities: DACNonidealities | None = None):
        self.bits = int(bits)
        self.vref = float(vref)
        self.non = nonidealities or DACNonidealities()
        # Input code domain. The NN quantizer emits SIGNED codes for most
        # tensors but UNSIGNED codes where the data has no negative side (the
        # post-softmax attention probabilities of the AV operation, see
        # quant.py act_signed=False). A DAC that assumed one domain would
        # silently reinterpret the other, so the domain is explicit and
        # settable per operation. It changes the code range and, for natively
        # bipolar architectures, the slope - so it is part of domain_key.
        self.signed_input = True
        self._set_domain(True)
        self._rng = np.random.default_rng(self.non.seed)
        self._build()

    def _set_domain(self, signed: bool) -> None:
        self.signed_input = bool(signed)
        if signed:
            self.x_min = -(2 ** (self.bits - 1))
            self.x_max = 2 ** (self.bits - 1) - 1
        else:
            self.x_min = 0
            self.x_max = 2 ** self.bits - 1
        self.x_fs = self.x_max

    def set_input_signed(self, signed: bool) -> None:
        """Select signed or unsigned input codes for the coming operation."""
        self._set_domain(signed)

    # -- to be provided by subclasses ------------------------------------- #

    def _build(self) -> None:
        """Construct physical element values (resistors/caps/currents)."""

    @property
    def v_per_unit(self) -> float:
        """dV_IA / dX_q, volts per unit signed input code. Ideal slope."""
        raise NotImplementedError

    @property
    def v_common_mode(self) -> float:
        """Physical mid-scale voltage. 0 for architectures that are natively
        bipolar; vref/2 for offset-binary ones."""
        return 0.0

    def _ideal_transfer(self, x_q: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # -- public ------------------------------------------------------------ #

    def transfer(self, x_q: np.ndarray) -> np.ndarray:
        """Signed integer codes -> signed analog-equivalent V_IA (volts)."""
        x_q = np.asarray(x_q)
        if np.any(x_q < self.x_min) or np.any(x_q > self.x_max):
            raise ValueError(
                f"{self.name}: input outside "
                f"{'signed' if self.signed_input else 'unsigned'} {self.bits}-bit range "
                f"[{self.x_min},{self.x_max}] — the DAC must be given the SAME "
                f"signed codes the quantizer produced, never a reinterpreted "
                f"unsigned word.")
        v = self._ideal_transfer(x_q)
        if not self.non.is_ideal:
            v = v * (1.0 + self.non.gain_error) + self.non.offset_v
        return v

    def describe(self) -> dict:
        return dict(name=self.name, bits=self.bits, vref=self.vref,
                    signed_encoding=self.signed_encoding,
                    signed_input=self.signed_input,
                    produces=self.produces,
                    v_per_unit=self.v_per_unit,
                    v_common_mode=self.v_common_mode,
                    full_scale_neg=self.v_per_unit * self.x_min,
                    full_scale_pos=self.v_per_unit * self.x_max,
                    ideal=self.non.is_ideal)


# --------------------------------------------------------------------------- #
# 1. reference model (validation only — NOT a hardware candidate)
# --------------------------------------------------------------------------- #


class IdealLinearSignedDAC(BaseDAC):
    """Mathematically ideal bipolar DAC:  V = Vref * X_q / X_FS.

    Exists to validate the architecture-specific models against a known-exact
    transfer. Per spec 5.12 this must NOT be used as the hardware path; it is
    the reference baseline. Physically it assumes a true bipolar output able to
    swing +/- Vref, which is a strong assumption for a 0.6 V supply.
    """

    name = "ideal"
    signed_encoding = "native bipolar (+/- Vref)"

    @property
    def v_per_unit(self) -> float:
        return self.vref / self.x_fs

    def _ideal_transfer(self, x_q):
        return x_q.astype(np.float64) * self.v_per_unit


# --------------------------------------------------------------------------- #
# 2. R-2R ladder, offset binary
# --------------------------------------------------------------------------- #


class R2RDAC(BaseDAC):
    """R-2R resistor ladder driven by an offset-binary code.

    Plausible because: R-2R needs only two resistor values regardless of
    resolution, is compact at low resolution, and is a standard way to make a
    voltage-output DAC feeding a capacitive load like a C2C ladder.

    Signed handling: OFFSET BINARY. The signed code is shifted into an unsigned
    ladder code
        D = X_q + 2^(N-1)  in [0, 2^N - 1]
    and the ladder produces, by the standard binary-weighted current summation
    of an R-2R network,
        V_phys = Vref * D / 2^N            (0 .. Vref*(2^N-1)/2^N)
    Mid-scale D = 2^(N-1) is the common mode Vref/2, so the signed-equivalent
    output is
        V_IA = V_phys - Vref/2 = Vref * X_q / 2^N
    Note the slope is Vref/2^N, NOT Vref/(2^(N-1)-1): an offset-binary ladder
    spends half its range on negative codes, so its usable swing is +/-Vref/2.
    That is a real architectural difference from the ideal bipolar model and it
    directly halves K_MAC.

    Parameters: R_unit (the "R" of R-2R). The ideal transfer is independent of
    the absolute value of R, so R_unit only matters for mismatch/loading; it is
    exposed rather than invented.
    """

    name = "r2r"
    signed_encoding = "offset binary (D = X_q + 2^(N-1)), common mode Vref/2"

    def __init__(self, bits, vref=0.6, nonidealities=None, r_unit: float = 10e3):
        self.r_unit = float(r_unit)
        super().__init__(bits, vref, nonidealities)

    def _build(self):
        # per-branch weights, nominally 2^k; mismatch perturbs them
        self._w = 2.0 ** np.arange(self.bits)
        if self.non.mismatch_sigma:
            self._w = self._w * (1.0 + self._rng.normal(
                0.0, self.non.mismatch_sigma, size=self.bits))

    @property
    def v_per_unit(self) -> float:
        return self.vref / (2 ** self.bits)

    @property
    def v_common_mode(self) -> float:
        # unsigned codes need no offset: a plain unipolar ladder already spans
        # the data's range, so mid-scale referencing would be wrong.
        return self.vref / 2.0 if self.signed_input else 0.0

    def _ideal_transfer(self, x_q):
        off = 2 ** (self.bits - 1) if self.signed_input else 0
        D = (x_q.astype(np.int64) + off)                       # offset binary
        if self.non.mismatch_sigma:
            bits = ((D[..., None] >> np.arange(self.bits)) & 1).astype(np.float64)
            v_phys = self.vref * (bits * self._w).sum(-1) / (2 ** self.bits)
        else:
            v_phys = self.vref * D.astype(np.float64) / (2 ** self.bits)
        return v_phys - self.v_common_mode          # signed-equivalent


# --------------------------------------------------------------------------- #
# 3. binary-weighted charge-redistribution (capacitive) DAC
# --------------------------------------------------------------------------- #


class CapacitiveDAC(BaseDAC):
    """Binary-weighted capacitor array, charge redistribution, offset binary.

    Plausible because: the macro is already a switched-capacitor (C2C) charge-
    domain design, so a capacitive input DAC shares the same technology, needs
    no static current, and settles into a capacitive load naturally. This is
    arguably the most likely real topology for this accelerator.

    Transfer derived from charge conservation on the summing node:
        V_phys = Vref * (sum_k b_k C_k) / C_total,
        C_k = 2^k * C_unit,  C_total = 2^N * C_unit
             => V_phys = Vref * D / 2^N
    identical ideal transfer to R-2R, but a DIFFERENT physical parameter set
    (C_unit, capacitor mismatch) and different nonideality behaviour, which is
    why it is a separate class rather than an alias.

    Signed handling: OFFSET BINARY, common mode Vref/2, as for R-2R.
    """

    name = "capacitive"
    signed_encoding = "offset binary (D = X_q + 2^(N-1)), common mode Vref/2"

    def __init__(self, bits, vref=0.6, nonidealities=None, c_unit: float = 1e-15):
        self.c_unit = float(c_unit)
        super().__init__(bits, vref, nonidealities)

    def _build(self):
        self._caps = self.c_unit * 2.0 ** np.arange(self.bits)
        if self.non.mismatch_sigma:
            self._caps = self._caps * (1.0 + self._rng.normal(
                0.0, self.non.mismatch_sigma, size=self.bits))
        self._c_total = self.c_unit * (2 ** self.bits)

    @property
    def v_per_unit(self) -> float:
        return self.vref / (2 ** self.bits)

    @property
    def v_common_mode(self) -> float:
        return self.vref / 2.0 if self.signed_input else 0.0

    def _ideal_transfer(self, x_q):
        off = 2 ** (self.bits - 1) if self.signed_input else 0
        D = (x_q.astype(np.int64) + off)
        if self.non.mismatch_sigma:
            bits = ((D[..., None] >> np.arange(self.bits)) & 1).astype(np.float64)
            v_phys = self.vref * (bits * self._caps).sum(-1) / self._c_total
        else:
            v_phys = self.vref * D.astype(np.float64) / (2 ** self.bits)
        return v_phys - self.v_common_mode


# --------------------------------------------------------------------------- #
# 4. current-steering DAC
# --------------------------------------------------------------------------- #


class CurrentSteeringDAC(BaseDAC):
    """Current-steering DAC followed by an explicit I->V conversion.

    Plausible because: current steering is the standard choice when conversion
    must be fast, and the per-row input of a CIM macro is converted very often.

    This architecture natively produces CURRENT, so the I->V step is explicit
    rather than hidden:
        I_out = I_lsb * D        (offset-binary code D)
        V_phys = I_out * R_load
        V_IA  = V_phys - I_lsb * 2^(N-1) * R_load      (signed-equivalent)
              = I_lsb * R_load * X_q

    Parameters I_lsb and R_load are NOT derivable from anything we have been
    told, so they are configurable assumptions. Their defaults are chosen so
    the full-scale swing matches the offset-binary voltage DACs (+/-Vref/2),
    making the comparison fair rather than an artefact of arbitrary scaling;
    change them to explore drive/headroom trade-offs.
    """

    name = "current_steering"
    signed_encoding = "offset-binary current code, differential-referenced I->V"
    produces = "current (converted to voltage by explicit R_load)"

    def __init__(self, bits, vref=0.6, nonidealities=None,
                 i_lsb: float | None = None, r_load: float = 10e3):
        self.r_load = float(r_load)
        # default I_lsb makes full scale == vref/2 at the given R_load
        self.i_lsb = (float(i_lsb) if i_lsb is not None
                      else (vref / (2 ** bits)) / float(r_load))
        super().__init__(bits, vref, nonidealities)

    def _build(self):
        self._srcs = 2.0 ** np.arange(self.bits)
        if self.non.mismatch_sigma:
            self._srcs = self._srcs * (1.0 + self._rng.normal(
                0.0, self.non.mismatch_sigma, size=self.bits))

    @property
    def v_per_unit(self) -> float:
        return self.i_lsb * self.r_load

    @property
    def v_common_mode(self) -> float:
        off = 2 ** (self.bits - 1) if self.signed_input else 0
        return self.i_lsb * self.r_load * off

    def current(self, x_q: np.ndarray) -> np.ndarray:
        """Exposed so the native physical quantity is inspectable."""
        off = 2 ** (self.bits - 1) if self.signed_input else 0
        D = (np.asarray(x_q).astype(np.int64) + off)
        return self.i_lsb * D.astype(np.float64)

    def _ideal_transfer(self, x_q):
        if self.non.mismatch_sigma:
            off = 2 ** (self.bits - 1) if self.signed_input else 0
            D = (x_q.astype(np.int64) + off)
            bits = ((D[..., None] >> np.arange(self.bits)) & 1).astype(np.float64)
            i_out = self.i_lsb * (bits * self._srcs).sum(-1)
        else:
            i_out = self.current(x_q)
        return i_out * self.r_load - self.v_common_mode


# --------------------------------------------------------------------------- #
# 5. differential (two unipolar paths)
# --------------------------------------------------------------------------- #


class DifferentialDAC(BaseDAC):
    """Two unipolar sub-DACs; the sign lives in WHICH path is driven.

    Plausible because: it is the most natural way to get a signed input out of
    a single-supply 0.6 V process without needing a negative rail, and it pairs
    naturally with a differential C2C column.

        X_q > 0 : V+ = Vref*|X_q|/X_FS,  V- = 0
        X_q < 0 : V+ = 0,                V- = Vref*|X_q|/X_FS
        V_IA = V+ - V-

    Unlike offset binary, no code is "wasted" on the negative half, so the
    slope is Vref/X_FS — the same as the ideal model — at the cost of two
    physical paths and sensitivity to path mismatch (exposed via
    mismatch_sigma, applied as a gain difference between the + and - paths).
    """

    name = "differential"
    signed_encoding = "two unipolar paths, V_IA = V+ - V-"

    def _build(self):
        g = 1.0
        if self.non.mismatch_sigma:
            g = 1.0 + self._rng.normal(0.0, self.non.mismatch_sigma)
        self._gain_pos, self._gain_neg = 1.0, g   # path gain mismatch

    @property
    def v_per_unit(self) -> float:
        return self.vref / self.x_fs

    def paths(self, x_q: np.ndarray):
        """Return (V+, V-) so the two physical rails are inspectable."""
        x = np.asarray(x_q).astype(np.float64)
        step = self.vref / self.x_fs
        vp = np.where(x > 0, x * step, 0.0) * self._gain_pos
        vn = np.where(x < 0, -x * step, 0.0) * self._gain_neg
        return vp, vn

    def _ideal_transfer(self, x_q):
        vp, vn = self.paths(x_q)
        return vp - vn


# --------------------------------------------------------------------------- #

DAC_TYPES = {
    "IDEAL": IdealLinearSignedDAC,
    "R2R": R2RDAC,
    "CAPACITIVE": CapacitiveDAC,
    "CURRENT_STEERING": CurrentSteeringDAC,
    "DIFFERENTIAL": DifferentialDAC,
}


def make_dac(dac_type: str, bits: int, vref: float = 0.6,
             nonidealities: DACNonidealities | None = None, **kw) -> BaseDAC:
    key = str(dac_type).upper()
    if key not in DAC_TYPES:
        raise ValueError(f"unknown dac_type {dac_type!r}; "
                         f"choose from {sorted(DAC_TYPES)}")
    return DAC_TYPES[key](bits=bits, vref=vref,
                          nonidealities=nonidealities, **kw)
