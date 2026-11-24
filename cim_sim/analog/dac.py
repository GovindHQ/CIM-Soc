"""
dac.py — the CIM INPUT DAC: quantized activation code  ->  analog voltage V_IA.

This is the INPUT DAC of the CIM datapath. It is NOT the SAR ADC's internal
DAC (see adc.py / SARDAC) — the two are deliberately separate modules.

CAPACITIVE DAC ONLY (Oct 2026)
------------------------------
The input DAC is HARDWARE-SPECIFIED: an UNSIGNED, binary-weighted CAPACITIVE
DAC (CDAC), 8 input bits b7..b0, V_REF_DAC = 0.6 V:

    V_out = V_REF * (b7/2 + b6/4 + b5/8 + b4/16 + b3/32 + b2/64 + b1/128 + b0/256)
          = V_REF * c / 2^N          (c unsigned, N = 8)

so the all-ones code gives V_REF*(1 - 2^-N), never V_REF.
`CapacitiveDAC.vout_from_unsigned_code` implements exactly this equation and
is tested bit by bit (test_analog.py Test 11). The four comparison candidates
the simulator used to carry (ideal bipolar, R-2R, current-steering,
differential) were removed once this topology was confirmed; the DAC is no
longer a configuration choice (AnalogConfig has no dac_type).

What is still NOT specified by the hardware, and therefore PROVISIONAL:
  * how a SIGNED activation is presented to this unsigned DAC. The simulator
    uses offset binary (c = X_q + 2^(N-1)) and removes the resulting V_REF/2
    common mode IDEALLY (signed-equivalent domain, below). Where that common
    mode is cancelled in silicon (reference column, differential column,
    digital correction of the weight-dependent term) is an open hardware
    question; chain.py measures the size of the term that would otherwise
    reach the ADC.
  * how an activation narrower than the DAC (e.g. A4 on the 8-bit CDAC) is
    aligned onto its bits: MSB alignment (code x 16) by default, see
    AnalogConfig.dac_input_alignment (chain.py).

SIGNED ANALOG-EQUIVALENT CONVENTION
-----------------------------------
`transfer()` returns V_IA as a SIGNED analog-equivalent voltage: code 0 maps to
0.0 V. The physical node voltage is
    V_phys = V_IA + V_cm,     V_cm = V_REF/2 for signed inputs, 0 for unsigned
and `v_common_mode` reports V_cm. Working in the signed-equivalent domain keeps
the C2C sum and the digital accumulator mathematically honest; the unipolar
offset is re-applied at the ADC boundary (see adc.py), which is where the
physical unipolar constraint of the converter actually lives.

LINEARITY / K_MAC
-----------------
The ideal transfer is linear in X_q with slope `v_per_unit` = V_REF / 2^N
volts per code, in both input domains. That single number is what the chain
uses to derive K_MAC ONCE per configuration (see chain.py). It is a property
of the DAC, never of the data — the DAC never inspects a tensor to pick a
scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# nonideality container (hooks only; everything defaults to ideal)
# --------------------------------------------------------------------------- #


@dataclass
class DACNonidealities:
    """All-zero by default => ideal DAC. Hooks for later, not used unless set.

    gain_error    : fractional error on the overall transfer slope (0.01 = +1%).
    offset_v      : additive output offset in volts.
    mismatch_sigma: relative sigma of per-capacitor mismatch. Drawn ONCE at
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
# the capacitive DAC
# --------------------------------------------------------------------------- #


class CapacitiveDAC:
    """Binary-weighted capacitor array, charge redistribution.

    Transfer derived from charge conservation on the summing node:
        V_phys = V_REF * (sum_k b_k C_k) / C_total,
        C_k = 2^k * C_unit,  C_total = 2^N * C_unit
             => V_phys = V_REF * c / 2^N
    which is the hardware equation above. The C2C macro receives V_IA and
    nothing else.

    Input code domain: the NN quantizer emits SIGNED codes for most tensors but
    UNSIGNED codes where the data has no negative side (the post-softmax
    attention probabilities of the AV operation, quant.py act_signed=False). A
    DAC that assumed one domain would silently reinterpret the other, so the
    domain is explicit and settable per operation (set_input_signed) and is
    part of AnalogChain.domain_key.
      signed   : offset binary, c = X_q + 2^(N-1), common mode V_REF/2
                 (simulator assumption, see module header)
      unsigned : c = X_q, no offset
    """

    name = "capacitive"
    signed_encoding = "offset binary (D = X_q + 2^(N-1)), common mode Vref/2"
    produces = "voltage"

    def __init__(self, bits: int = 8, vref: float = 0.6,
                 nonidealities: DACNonidealities | None = None,
                 c_unit: float = 1e-15):
        self.bits = int(bits)
        self.vref = float(vref)
        self.c_unit = float(c_unit)
        self.non = nonidealities or DACNonidealities()
        self.signed_input = True
        self._set_domain(True)
        self._rng = np.random.default_rng(self.non.seed)
        self._caps = self.c_unit * 2.0 ** np.arange(self.bits)
        if self.non.mismatch_sigma:
            self._caps = self._caps * (1.0 + self._rng.normal(
                0.0, self.non.mismatch_sigma, size=self.bits))
        self._c_total = self.c_unit * (2 ** self.bits)

    # -- input code domain ------------------------------------------------- #

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

    # -- scale ------------------------------------------------------------- #

    @property
    def v_per_unit(self) -> float:
        """dV_IA / dX_q = V_REF / 2^N volts per code (both input domains)."""
        return self.vref / (2 ** self.bits)

    @property
    def v_common_mode(self) -> float:
        """Physical mid-scale offset: V_REF/2 for signed inputs, 0 for unsigned."""
        return self.vref / 2.0 if self.signed_input else 0.0

    # -- the hardware equation ---------------------------------------------- #

    def unsigned_code(self, x_q: np.ndarray) -> np.ndarray:
        """NN integer code -> the UNSIGNED code c actually applied to the CDAC.

        Signed input domain : offset binary, c = X_q + 2^(N-1)  (assumption)
        Unsigned input domain (post-softmax AV): c = X_q        (no offset)
        """
        off = 2 ** (self.bits - 1) if self.signed_input else 0
        c = np.asarray(x_q).astype(np.int64) + off
        if np.any(c < 0) or np.any(c > 2 ** self.bits - 1):
            raise ValueError(f"{self.name}: unsigned DAC code outside "
                             f"[0, {2 ** self.bits - 1}]")
        return c

    def vout_from_unsigned_code(self, c: np.ndarray) -> np.ndarray:
        """The specified CDAC transfer: V_out = V_REF * c / 2^N (physical volts).

        With capacitor mismatch enabled, the binary weights are the perturbed
        capacitors: V_out = V_REF * sum_k b_k C_k / C_total.
        """
        c = np.asarray(c).astype(np.int64)
        if self.non.mismatch_sigma:
            bits = ((c[..., None] >> np.arange(self.bits)) & 1).astype(np.float64)
            return self.vref * (bits * self._caps).sum(-1) / self._c_total
        return self.vref * c.astype(np.float64) / (2 ** self.bits)

    # -- public ------------------------------------------------------------ #

    def transfer(self, x_q: np.ndarray) -> np.ndarray:
        """NN integer codes -> signed analog-equivalent V_IA (volts)."""
        x_q = np.asarray(x_q)
        if np.any(x_q < self.x_min) or np.any(x_q > self.x_max):
            raise ValueError(
                f"{self.name}: input outside "
                f"{'signed' if self.signed_input else 'unsigned'} {self.bits}-bit range "
                f"[{self.x_min},{self.x_max}] — the DAC must be given the SAME "
                f"codes the quantizer produced, never a reinterpreted "
                f"two's-complement word.")
        # physical node voltage from the specified equation, then the signed-
        # equivalent value (common mode removed ideally - see module header).
        v = self.vout_from_unsigned_code(self.unsigned_code(x_q)) - self.v_common_mode
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
