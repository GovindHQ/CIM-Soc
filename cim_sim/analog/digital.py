"""
digital.py — digital partial-sum accumulator and the requantizer.

These are the two blocks after the ADC:

    signed ADC codes d  --DigitalAccumulator-->  integer sum
                        --Requantizer-->         real-valued output

THE SCALE CHAIN (spec 17, 20)
-----------------------------
Two independent quantization boundaries exist and are NOT merged:

  (1) neural-network quantization
          X_real = S_X * X_q        S_X = per-token activation scale
          W_real = S_W * W_q        S_W = per-output-column weight scale

  (2) analog hardware chain
          V_IA   = v_per_unit * X_q                     (input DAC)
          V_OA   = V_IA . W / (m 2^B)                   (C2C macro)
                 = K_MAC * P,   P = sum X_q W_q
          K_MAC  = v_per_unit / (m * 2^B)
          V_ADCIN= G * V_OA                             (pre-ADC gain)
          d      = round(V_ADCIN / LSB_V)               (SAR ADC, signed code,
                   clipped to [-2^(N_ADC-1), 2^(N_ADC-1)-1])
          LSB_V  = V_REF_ADC / 2^N_ADC

LSB_V is Vref / 2^N, NOT Vref / (2^N - 1): it is the step of the binary-
weighted SAR DAC (trial weights Vref*2^k/2^N), and it is what
SARADC.lsb_volts returns (see adc.py). The usable signed swing is therefore
exactly +/- 2^(N-1) * LSB_V = +/- V_REF_ADC / 2. (The legacy functional ADC in
array.py, `_adc_convert`, uses a different (2^N - 1) normalisation; that
model is separate and intentionally left unchanged.)

Composing the analog half (ignoring clipping):

          d  ~=  (G * K_MAC / LSB_V) * P

so one ADC code step corresponds to a fixed number of MAC units:

          LSB_P  =  LSB_V / (G * K_MAC)                 [column-sum units/code]

ACCUMULATION IS INTEGER
-----------------------
Because LSB_P depends only on (DAC architecture, macro, G, ADC) and NOT on the
data, every partial sum within one accumulation domain shares it. So the codes
themselves can be summed as plain integers and rescaled ONCE at the end:

          P_hat = LSB_P * sum_blocks d_block
          y_real = P_hat * S_X * S_W
                 = (LSB_P * S_X * S_W) * sum d

This is what "K_MAC must be consistent for all partial sums in the same
accumulation domain" (spec 8) buys: a single multiply at the end instead of a
rescale between blocks. The accumulator below therefore holds int64 and knows
nothing about volts — exactly like real digital hardware.

The NN scales S_X, S_W are applied only in the Requantizer, and stay separate
from the hardware scale LSB_P so the two error sources remain separable
(spec 39.H, question 12).

RELATION TO THE PRODUCTION PATH
-------------------------------
matmul.cim_matmul does not instantiate these two classes. It implements the
same contract inline, because its accumulator spans token blocks, output
tiles and P arrays at once: `acc += d` over depth blocks (plain int64), then a
single `acc * LSB_P * S_X * S_W` — numerically identical to
Requantizer.to_real. The domain_key guard that DigitalAccumulator.add enforces
per block is enforced there once per matmul instead, by checking that every
array in the pool reports the same AnalogChain.domain_key after
set_operation(). These classes remain the reference implementation of the
contract, used directly by test_analog.py and available via
AnalogChain.accumulator() / AnalogChain.requantizer().
"""

from __future__ import annotations

import numpy as np


class DigitalAccumulator:
    """Integer partial-sum accumulator over depth blocks.

    Holds raw signed ADC codes. Refuses to mix accumulation domains: the domain
    key (which encodes DAC/macro/gain/ADC configuration) must match for every
    block added, which is the mechanical guard against spec 8's failure mode of
    adding codes produced under different analog scales.
    """

    def __init__(self, shape, domain_key, dtype=np.int64):
        self.acc = np.zeros(shape, dtype=dtype)
        self.domain_key = domain_key
        self.n_blocks = 0

    def add(self, d_codes: np.ndarray, domain_key) -> None:
        if domain_key != self.domain_key:
            raise ValueError(
                "refusing to accumulate ADC codes from a different analog "
                f"scale domain: accumulator={self.domain_key!r} "
                f"block={domain_key!r}. Codes produced under different "
                "DAC/macro/gain/ADC configurations are not commensurable.")
        self.acc += d_codes.astype(self.acc.dtype)
        self.n_blocks += 1

    def value(self) -> np.ndarray:
        return self.acc


class Requantizer:
    """Turns an accumulated integer code sum into a real-valued result.

        y_real = sum_d * LSB_P * S_X * S_W

    Keeps the hardware scale (LSB_P) and the NN scales (S_X, S_W) as separate
    factors rather than one opaque constant, so either can be inspected,
    swept, or attributed independently.
    """

    def __init__(self, lsb_p: float):
        self.lsb_p = float(lsb_p)

    def to_colsum_units(self, acc: np.ndarray) -> np.ndarray:
        """Accumulated codes -> estimated integer MAC sum P_hat."""
        return acc.astype(np.float64) * self.lsb_p

    def to_real(self, acc: np.ndarray, s_x: np.ndarray, s_w: np.ndarray
                ) -> np.ndarray:
        """Accumulated codes -> real-valued output in the NN's units."""
        return self.to_colsum_units(acc) * s_x * s_w

    def describe(self) -> dict:
        return dict(lsb_p=self.lsb_p,
                    note="hardware scale; NN scales S_X/S_W applied separately")
