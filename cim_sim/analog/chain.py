"""
chain.py — AnalogConfig, AnalogStats, AnalogChain.

The chain wires the blocks together and owns the one thing that must be global
to an accumulation domain: K_MAC, and hence LSB_P.

    X_q -> InputDAC -> V_IA -> C2CMacro -> V_OA -> PreADCGain -> V_ADC_IN
        -> SARADC -> signed code d      (accumulated as integers upstream)

K_MAC IS DERIVED ONCE, FROM ARCHITECTURE
----------------------------------------
    K_MAC = dac.v_per_unit / (m * 2^B)      [volts per unit of sum X_q W_q]
    LSB_P = adc.lsb_volts / (G * K_MAC)     [sum units per ADC code]

Both depend only on configuration. Neither ever looks at a tensor. `domain_key`
fingerprints exactly the parameters that must match for two partial sums to be
addable, and the DigitalAccumulator enforces it.

WHY THIS IS NOT A DIGITAL MAC WITH A SCALE FACTOR (spec 7, 38)
--------------------------------------------------------------
The ideal equations do collapse algebraically — that is stated openly and is
what `k_mac` reports. What matters is that the collapse is NOT the
implementation: `execute()` really does build V_IA volts from the DAC, hand
those volts to the macro, get V_OA volts back, scale them through the
amplifier, and run a bit-by-bit SAR search. Every intermediate physical
quantity exists and is observable. Vectorising those steps over a whole tile is
an implementation detail of NumPy, not a collapse of the architecture: swapping
`dac_type`, adding capacitor mismatch, or adding comparator noise changes the
result precisely because the boundaries are real.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .adc import (ADCNonidealities, GainNonidealities, PreADCGain, SARADC,
                  SARDAC, SARDACNonidealities)
from .dac import BaseDAC, DACNonidealities, make_dac
from .digital import DigitalAccumulator, Requantizer
from .macro import C2CMacro, MacroNonidealities, SignedWeightAdapter


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #


@dataclass
class AnalogConfig:
    """Everything the analog path needs. All physical values are explicit.

    Values marked ASSUMPTION are not hardware-confirmed; they are here so they
    can be swept and later replaced, not because we know them.
    """

    enabled: bool = False

    # ---- input DAC (topology UNKNOWN - under investigation) ---------------
    dac_type: str = "CAPACITIVE"
    dac_bits: int | None = None      # None => follow activation precision
    dac_vref: float = 0.6            # ASSUMPTION: same rail as the ADC
    dac_kwargs: dict = field(default_factory=dict)

    # ---- C2C macro (relation supplied; constant of proportionality UNKNOWN)
    macro_m: int | None = None       # None => number of rows

    # ---- pre-ADC amplifier (existence proposed by us, not confirmed) ------
    gain_default: float = 1.0
    gain_per_op: dict = field(default_factory=dict)   # {"qkv":.., "qk":.., "av":..}

    # ---- SAR ADC (HARDWARE-CONFIRMED: SAR, ~0-0.6 V, 8b, one per column) --
    adc_bits: int = 8
    adc_vref: float = 0.6

    # ---- nonidealities (all ideal by default) -----------------------------
    dac_non: DACNonidealities = field(default_factory=DACNonidealities)
    macro_non: MacroNonidealities = field(default_factory=MacroNonidealities)
    gain_non: GainNonidealities = field(default_factory=GainNonidealities)
    adc_non: ADCNonidealities = field(default_factory=ADCNonidealities)
    sar_dac_non: SARDACNonidealities = field(default_factory=SARDACNonidealities)

    def gain_for(self, op: str | None) -> float:
        if op is None:
            return self.gain_default
        return float(self.gain_per_op.get(op, self.gain_default))


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #


class AnalogStats:
    """Running statistics of the analog nodes, bucketed by operation label.

    Accumulates scalars plus a bounded random subsample, so memory stays flat
    over ~1e8 conversions. Per spec 12 this only OBSERVES; it never feeds back
    into the gain.
    """

    def __init__(self, keep_frac: float = 0.002, seed: int = 0, cap: int = 400_000):
        self.keep_frac, self.cap = keep_frac, cap
        self._rng = np.random.default_rng(seed)
        self.b = {}

    def _bucket(self, label):
        if label not in self.b:
            self.b[label] = dict(n=0, clip_hi=0, clip_lo=0,
                                 voa_min=np.inf, voa_max=-np.inf,
                                 voa_sum=0.0, voa_sumsq=0.0,
                                 voa=[], vadc=[], code=[])
        return self.b[label]

    def record(self, label, v_oa, v_adc, d, clip_hi=0, clip_lo=0):
        s = self._bucket(label)
        v_oa = np.asarray(v_oa).ravel()
        s["n"] += v_oa.size
        s["clip_hi"] += int(clip_hi)
        s["clip_lo"] += int(clip_lo)
        if v_oa.size:
            s["voa_min"] = min(s["voa_min"], float(v_oa.min()))
            s["voa_max"] = max(s["voa_max"], float(v_oa.max()))
            s["voa_sum"] += float(v_oa.sum())
            s["voa_sumsq"] += float((v_oa.astype(np.float64) ** 2).sum())
            if sum(a.size for a in s["voa"]) < self.cap:
                k = max(1, int(v_oa.size * self.keep_frac))
                idx = self._rng.choice(v_oa.size, size=min(k, v_oa.size),
                                       replace=False)
                s["voa"].append(v_oa[idx].astype(np.float32))
                s["vadc"].append(np.asarray(v_adc).ravel()[idx].astype(np.float32))
                s["code"].append(np.asarray(d).ravel()[idx].astype(np.int32))

    def summary(self, adc_vref=0.6, adc_bits=8) -> dict:
        out = {}
        half = adc_vref / 2.0
        for label, s in self.b.items():
            if not s["n"]:
                continue
            voa = (np.concatenate(s["voa"]) if s["voa"] else np.zeros(0, np.float32))
            vad = (np.concatenate(s["vadc"]) if s["vadc"] else np.zeros(0, np.float32))
            mean = s["voa_sum"] / s["n"]
            var = max(0.0, s["voa_sumsq"] / s["n"] - mean * mean)
            r = dict(n=s["n"], voa_min=s["voa_min"], voa_max=s["voa_max"],
                     voa_mean=mean, voa_std=var ** 0.5,
                     clip_hi=s["clip_hi"], clip_lo=s["clip_lo"],
                     clip_pct=100.0 * (s["clip_hi"] + s["clip_lo"]) / s["n"])
            if voa.size:
                a = np.abs(voa)
                r.update(voa_median=float(np.median(voa)),
                         voa_absp50=float(np.percentile(a, 50)),
                         voa_absp95=float(np.percentile(a, 95)),
                         voa_absp99=float(np.percentile(a, 99)),
                         voa_absp999=float(np.percentile(a, 99.9)),
                         voa_absmax_sampled=float(a.max()))
            if vad.size:
                av = np.abs(vad)
                r.update(vadc_absp50=float(np.percentile(av, 50)),
                         vadc_absp999=float(np.percentile(av, 99.9)),
                         # how much of the usable +/-Vref/2 swing is occupied
                         util_p999_pct=100.0 * float(np.percentile(av, 99.9)) / half,
                         util_median_pct=100.0 * float(np.percentile(av, 50)) / half)
            out[label] = r
        return out

    def reset(self):
        self.b.clear()


# --------------------------------------------------------------------------- #
# the chain
# --------------------------------------------------------------------------- #


class AnalogChain:
    """Composes DAC -> macro -> gain -> SAR ADC for one 32xC macro."""

    def __init__(self, cfg: AnalogConfig, rows: int, weight_bits: int,
                 act_bits: int, stats: AnalogStats | None = None):
        self.cfg = cfg
        self.rows = rows
        self.weight_bits = weight_bits
        self.act_bits = act_bits

        dac_bits = cfg.dac_bits if cfg.dac_bits is not None else act_bits
        self.dac: BaseDAC = make_dac(cfg.dac_type, bits=dac_bits,
                                     vref=cfg.dac_vref,
                                     nonidealities=cfg.dac_non,
                                     **cfg.dac_kwargs)
        self.macro = C2CMacro(rows=rows, weight_bits=weight_bits, m=cfg.macro_m,
                              adapter=SignedWeightAdapter(weight_bits),
                              nonidealities=cfg.macro_non)
        self.amp = PreADCGain(cfg.gain_default, cfg.gain_non)
        self.adc = SARADC(cfg.adc_bits, cfg.adc_vref, cfg.adc_non,
                          SARDAC(cfg.adc_bits, cfg.adc_vref, cfg.sar_dac_non))

        self.stats = stats if stats is not None else AnalogStats()
        self.label = None                 # current operation, for stats+gain
        self._prog = None                 # programmed (sign, cap)

    # -- scale ------------------------------------------------------------- #

    @property
    def k_mac(self) -> float:
        """Volts of V_OA per unit of sum(X_q W_q). Architecture-derived."""
        return self.dac.v_per_unit * self.macro.gain_per_unit_v()

    @property
    def gain(self) -> float:
        return self.amp.gain

    def set_operation(self, op: str | None, act_signed: bool = True) -> None:
        """Select the per-operation gain, input code domain, and stats bucket.

        Called once per operation, never per depth block, so every partial sum
        in an accumulation domain shares one gain and one input domain (spec 9).
        `act_signed` follows the NN quantizer: False for the post-softmax AV
        activations, which have no negative side.
        """
        self.label = op
        self.amp.gain = self.cfg.gain_for(op)
        self.dac.set_input_signed(act_signed)

    @property
    def lsb_p(self) -> float:
        """Column-sum units represented by one ADC code step."""
        return self.adc.lsb_volts / (self.gain * self.k_mac)

    @property
    def domain_key(self):
        """Fingerprint of everything that must match to add two code sums."""
        return (self.dac.name, self.dac.bits, self.dac.vref,
                self.dac.signed_input, self.macro.m, self.weight_bits,
                round(self.gain, 12), self.adc.bits, self.adc.vref)

    def requantizer(self) -> Requantizer:
        return Requantizer(self.lsb_p)

    def accumulator(self, shape) -> DigitalAccumulator:
        return DigitalAccumulator(shape, self.domain_key)

    # -- weight programming ------------------------------------------------ #

    def program(self, w_q: np.ndarray) -> None:
        self._prog = self.macro.program(w_q)

    # -- one analog conversion --------------------------------------------- #

    def execute(self, x_q: np.ndarray) -> np.ndarray:
        """Signed activation codes [T, rows] -> signed ADC codes [T, cols].

        Every stage is a real object boundary carrying a physical quantity.
        """
        if self._prog is None:
            raise RuntimeError("no weights programmed; call program() first")
        sign, cap = self._prog

        v_ia = self.dac.transfer(x_q)                       # 1. input DAC
        v_oa = self.macro.mac(v_ia, sign, cap)              # 2. C2C macro
        v_adc_in = self.amp.apply(v_oa)                     # 3. pre-ADC gain

        hi0, lo0 = self.adc.n_clipped_hi, self.adc.n_clipped_lo
        d = self.adc.convert(v_adc_in)                      # 4. SAR ADC
        self.stats.record(self.label, v_oa, v_adc_in, d,
                          self.adc.n_clipped_hi - hi0,
                          self.adc.n_clipped_lo - lo0)
        return d

    # -- introspection ----------------------------------------------------- #

    def describe(self) -> dict:
        return dict(dac=self.dac.describe(), macro=self.macro.describe(),
                    gain=self.amp.describe(), adc=self.adc.describe(),
                    k_mac=self.k_mac, lsb_p=self.lsb_p,
                    domain_key=self.domain_key)

    def scale_report(self) -> str:
        d = self.dac
        lines = [
            "analog scale chain",
            f"  DAC {d.name!r}: v_per_unit = {d.v_per_unit:.6e} V/code, "
            f"signed encoding = {d.signed_encoding}",
            f"  macro: 1/(m*2^B) = {self.macro.gain_per_unit_v():.6e}, m={self.macro.m}",
            f"  K_MAC = {self.k_mac:.6e} V per unit sum(Xq*Wq)",
            f"  gain G = {self.gain:g}  (operation {self.label!r})",
            f"  ADC: {self.adc.bits}b, Vref={self.adc.vref} V, "
            f"LSB = {self.adc.lsb_volts*1e3:.4f} mV, "
            f"signed swing = +/-{self.adc.signed_swing_volts*1e3:.1f} mV",
            f"  LSB_P = {self.lsb_p:.4f} column-sum units per code",
        ]
        return "\n".join(lines)
