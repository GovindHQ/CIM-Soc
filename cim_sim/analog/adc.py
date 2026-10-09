"""
adc.py — pre-ADC analog gain stage, the SAR ADC, and the SAR ADC's internal DAC.

Three separate blocks live here because they form the analog->digital boundary:

    V_OA  --PreADCGain-->  V_ADC_IN  --SARADC(+SARDAC)-->  signed code d

The SARDAC here is the ADC's INTERNAL comparison DAC. It is NOT the input DAC
that converts activations (dac.py). They are deliberately different modules.

OFFSET BINARY AT THE ADC BOUNDARY
---------------------------------
V_OA and V_ADC_IN are carried as SIGNED analog-equivalent voltages. The
physical ADC input is unipolar, 0..V_REF_ADC (~0.6 V). The offset is applied
HERE, at the boundary, because this is the only place the unipolar constraint
physically exists:

    V_phys = V_CM + V_ADC_IN,     V_CM = V_REF_ADC / 2
    usable signed swing  |V_ADC_IN| <= V_REF_ADC / 2

The conversion produces an unsigned physical code in [0, 2^N-1]; the digital
side immediately re-centres it to a signed code

    d = code - 2^(N-1)      in [-2^(N-1), 2^(N-1)-1]

so that zero volts maps exactly to d = 0 and signed partial sums can be summed
in a plain integer accumulator with no per-block offset bookkeeping. Nothing
about this hides the clipping: a signed input beyond +/-V_REF_ADC/2 saturates
the physical code and is counted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# pre-ADC analog gain
# --------------------------------------------------------------------------- #


@dataclass
class GainNonidealities:
    gain_error: float = 0.0
    offset_v: float = 0.0
    # output swing limit of the amplifier itself, volts (None => no limit).
    # Distinct from ADC clipping: an amplifier can saturate before the ADC does.
    v_swing_limit: float | None = None

    @property
    def is_ideal(self) -> bool:
        return (self.gain_error == 0.0 and self.offset_v == 0.0
                and self.v_swing_limit is None)


class PreADCGain:
    """Ideal model:  V_ADC_IN = G * V_OA.

    G is a HARDWARE CONFIGURATION PARAMETER, per operation. It is never derived
    from the data at inference time — doing that would recreate the arbitrary
    normalisation this rewrite exists to remove (spec 10, 12). A calibration
    utility may later RECOMMEND a G from measured statistics, but it stays
    outside the inference path.

    This is NOT the ADC full scale. The ADC keeps its own fixed reference.
    """

    def __init__(self, gain: float = 1.0,
                 nonidealities: GainNonidealities | None = None):
        self.gain = float(gain)
        self.non = nonidealities or GainNonidealities()
        self.amp_saturated = 0

    def apply(self, v_oa: np.ndarray) -> np.ndarray:
        g = self.gain * (1.0 + self.non.gain_error)
        v = np.asarray(v_oa, dtype=np.float64) * g + self.non.offset_v
        lim = self.non.v_swing_limit
        if lim is not None:
            sat = np.abs(v) > lim
            self.amp_saturated += int(np.count_nonzero(sat))
            v = np.clip(v, -lim, lim)
        return v

    def describe(self) -> dict:
        return dict(gain=self.gain, ideal=self.non.is_ideal,
                    note="configuration parameter, not data-derived")


# --------------------------------------------------------------------------- #
# SAR internal DAC
# --------------------------------------------------------------------------- #


@dataclass
class SARDACNonidealities:
    cap_mismatch_sigma: float = 0.0
    seed: int = 0

    @property
    def is_ideal(self) -> bool:
        return self.cap_mismatch_sigma == 0.0


class SARDAC:
    """The comparison DAC inside the SAR ADC.

    Ideal binary-weighted: trial code k contributes weight 2^k of V_REF/2^N.
    Kept as its own object so capacitor mismatch (the dominant SAR INL/DNL
    mechanism) can be injected later without touching the search logic.
    """

    def __init__(self, bits: int, vref: float,
                 nonidealities: SARDACNonidealities | None = None):
        self.bits = int(bits)
        self.vref = float(vref)
        self.non = nonidealities or SARDACNonidealities()
        self._w = 2.0 ** np.arange(self.bits)[::-1]      # MSB first
        if self.non.cap_mismatch_sigma:
            rng = np.random.default_rng(self.non.seed)
            self._w = self._w * (1.0 + rng.normal(
                0.0, self.non.cap_mismatch_sigma, size=self.bits))

    def weight_volts(self, bit_index: int) -> float:
        """Volts contributed by trial bit `bit_index` (0 = MSB)."""
        return self.vref * self._w[bit_index] / (2.0 ** self.bits)

    def describe(self) -> dict:
        return dict(bits=self.bits, vref=self.vref, ideal=self.non.is_ideal)


# --------------------------------------------------------------------------- #
# SAR ADC
# --------------------------------------------------------------------------- #


@dataclass
class ADCNonidealities:
    comparator_offset_v: float = 0.0
    comparator_noise_v: float = 0.0     # per-decision gaussian sigma
    seed: int = 0

    @property
    def is_ideal(self) -> bool:
        return (self.comparator_offset_v == 0.0
                and self.comparator_noise_v == 0.0)


class SARADC:
    """Successive-approximation ADC, modelled as an actual bit-by-bit search.

    The search really is performed (N comparator decisions against SAR-DAC
    trial voltages), vectorised across all samples at once. In the ideal case
    the result equals round-to-nearest quantisation; that equivalence is
    asserted by the test suite rather than assumed, and it is what lets
    comparator offset/noise and SAR-DAC mismatch be added later without
    restructuring anything.

    `convert()` takes SIGNED V_ADC_IN and returns SIGNED codes d.
    """

    def __init__(self, bits: int = 8, vref: float = 0.6,
                 nonidealities: ADCNonidealities | None = None,
                 sar_dac: SARDAC | None = None):
        self.bits = int(bits)
        self.vref = float(vref)
        self.non = nonidealities or ADCNonidealities()
        self.sar_dac = sar_dac or SARDAC(bits, vref)
        self.v_cm = self.vref / 2.0                  # offset-binary mid point
        self.n_levels = 2 ** self.bits
        self.code_mid = 2 ** (self.bits - 1)
        self._rng = np.random.default_rng(self.non.seed)
        self.n_clipped_hi = 0
        self.n_clipped_lo = 0
        self.n_converted = 0

    # -- physical LSB ------------------------------------------------------ #

    @property
    def lsb_volts(self) -> float:
        """Vref / 2^N.

        This is the LSB a binary-weighted SAR DAC actually produces: its trial
        weights are Vref*2^k/2^N, so they sum to Vref*(2^N-1)/2^N and step by
        Vref/2^N. Using Vref/(2^N-1) instead - the convention of the legacy
        `_adc_convert` in array.py - would make the quantiser inconsistent with
        the SAR DAC driving it and produces a 1-code error at full scale. The
        signed swing is then exactly code_mid*LSB = Vref/2.
        """
        return self.vref / self.n_levels

    @property
    def signed_swing_volts(self) -> float:
        """Largest |V_ADC_IN| the unipolar input can represent."""
        return self.code_mid * self.lsb_volts

    # -- conversion -------------------------------------------------------- #

    def convert(self, v_adc_in: np.ndarray) -> np.ndarray:
        """Signed V_ADC_IN (volts) -> signed integer code d.

        Steps, kept explicit:
          1. sample
          2. offset binary: V_phys = V_CM + V_ADC_IN, clipped to [0, Vref]
          3. successive approximation against the SAR DAC
          4. re-centre the unsigned physical code to a signed code
        """
        v = np.asarray(v_adc_in, dtype=np.float64)

        # 2. offset binary + physical clipping (observable, never hidden)
        v_phys = self.v_cm + v
        hi = v_phys > self.vref
        lo = v_phys < 0.0
        self.n_clipped_hi += int(np.count_nonzero(hi))
        self.n_clipped_lo += int(np.count_nonzero(lo))
        self.n_converted += int(v.size)
        v_phys = np.clip(v_phys, 0.0, self.vref)

        # half-LSB offset so the ideal search rounds to nearest rather than
        # truncating; this is the standard sampling offset, not a fudge.
        v_s = v_phys + 0.5 * self.lsb_volts

        # 3. successive approximation, MSB -> LSB
        code = np.zeros(v.shape, dtype=np.int64)
        acc = np.zeros(v.shape, dtype=np.float64)     # SAR-DAC output so far
        for k in range(self.bits):
            trial = acc + self.sar_dac.weight_volts(k)
            decide = v_s - trial
            if self.non.comparator_offset_v:
                decide = decide - self.non.comparator_offset_v
            if self.non.comparator_noise_v:
                decide = decide + self._rng.normal(
                    0.0, self.non.comparator_noise_v, size=decide.shape)
            keep = decide >= 0.0
            code = code | (keep.astype(np.int64) << (self.bits - 1 - k))
            acc = np.where(keep, trial, acc)

        code = np.clip(code, 0, self.n_levels - 1)
        # 4. signed re-centring
        return code - self.code_mid

    def code_to_volts(self, d: np.ndarray) -> np.ndarray:
        """Signed code -> the signed V_ADC_IN it represents."""
        return np.asarray(d, dtype=np.float64) * self.lsb_volts

    def describe(self) -> dict:
        return dict(bits=self.bits, vref=self.vref, lsb_volts=self.lsb_volts,
                    signed_swing_volts=self.signed_swing_volts,
                    v_common_mode=self.v_cm, ideal=self.non.is_ideal,
                    sar_dac=self.sar_dac.describe())
