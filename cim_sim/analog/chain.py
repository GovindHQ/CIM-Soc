"""
chain.py — AnalogConfig, AnalogStats, AnalogChain.

The chain wires the blocks together and owns the one thing that must be global
to an accumulation domain: K_MAC, and hence LSB_P.

    X_q -> InputDAC -> V_IA -> C2CMacro -> V_OA -> PreADCGain -> V_ADC_IN
        -> SARADC -> signed code d      (accumulated as integers upstream)

K_MAC IS DERIVED ONCE, FROM ARCHITECTURE
----------------------------------------
    K_MAC = dac.v_per_unit * align / (m * 2^B)  [volts per unit of sum X_q W_q]
    LSB_P = adc.lsb_volts / (G * K_MAC)         [sum units per ADC code]

`align` = 2^(dac_bits - act_bits) when an activation narrower than the DAC is
MSB-aligned onto it (AnalogConfig.dac_input_alignment = "msb"), else 1. With
dac_bits == act_bits (the historical default) align == 1 and nothing changes.

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
an implementation detail of NumPy, not a collapse of the architecture: adding
DAC or macro capacitor mismatch, or comparator offset/noise, changes the
result precisely because the boundaries are real.

The input DAC is the hardware-specified unsigned 8-bit capacitive DAC
(CapacitiveDAC, dac.py); it is not a configuration choice.

DIAGNOSTICS (observation only — never fed back into any scale or gain)
---------------------------------------------------------------------
Per operation label AnalogStats records: conversions, ADC clipping (input
outside [0, V_REF_ADC] after the V_REF_ADC/2 offset), rail-code hits,
amplifier saturation (only if a swing limit is configured), V_OA and V_ADC_IN
min/max/mean/std and weighted-sample percentiles, and the COMMON-MODE RESIDUE:
the weight-dependent term  V_CM_DAC * sum_j(sign_j |W_j| / 2^B) / m  that an
offset-binary input DAC adds to every column and that the simulator removes
ideally. It is reported as a counterfactual ("would this clip if it were not
cancelled before the amplifier?") and is NEVER applied to the outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .adc import (ADCNonidealities, GainNonidealities, PreADCGain, SARADC,
                  SARDAC, SARDACNonidealities)
from .dac import CapacitiveDAC, DACNonidealities
from .digital import DigitalAccumulator, Requantizer
from .macro import C2CMacro, MacroNonidealities, SignedWeightAdapter


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #


DAC_ALIGNMENTS = ("msb", "lsb")

# ADC resolution of the initial QAT operating point (decided Oct 2026). The ADC
# stays configurable (AnalogConfig.adc_bits) for later experiments; QAT code
# should take its ADC width from here rather than hard-coding it.
QAT_INITIAL_ADC_BITS = 8


@dataclass
class AnalogConfig:
    """Everything the analog path needs. All physical values are explicit.

    Values marked ASSUMPTION are not hardware-confirmed; they are here so they
    can be swept and later replaced, not because we know them.
    """

    enabled: bool = False

    # ---- input DAC --------------------------------------------------------
    # HARDWARE-SPECIFIED (Oct 2026): unsigned binary-weighted CAPACITIVE DAC,
    # 8 bits, V_REF_DAC = 0.6 V, V_out = V_REF * c / 2^N. This is the only
    # input DAC in the simulator (no dac_type option).
    # dac_bits defaults to the hardware's 8; narrower activations are aligned
    # per dac_input_alignment. None => follow act_bits (a DAC exactly as wide
    # as the activation; with "msb" alignment this is bit-identical to the
    # 8-bit default for A4/A8). dac_bits < act_bits is rejected.
    dac_bits: int | None = 8
    dac_vref: float = 0.6            # specified 0.6 V; kept separate from adc_vref
    # PROVISIONAL ASSUMPTION (not specified by hardware): how an act_bits-wide
    # code is placed on a wider DAC.
    #   "msb": code << (dac_bits - act_bits). Uses the full DAC swing; gives
    #          EXACTLY the transfer of an act_bits-wide DAC (default).
    #   "lsb": code applied as-is. Uses 2^act_bits / 2^dac_bits of the swing.
    dac_input_alignment: str = "msb"
    dac_kwargs: dict = field(default_factory=dict)   # CapacitiveDAC extras (c_unit)

    # ---- C2C macro (relation supplied; constant of proportionality UNKNOWN)
    macro_m: int | None = None       # None => number of rows

    # ---- pre-ADC amplifier (existence proposed by us, not confirmed) ------
    gain_default: float = 1.0
    gain_per_op: dict = field(default_factory=dict)   # {"qkv":.., "qk":.., "av":..}

    # ---- SAR ADC (HARDWARE-CONFIRMED: SAR, ~0-0.6 V, 8b, one per column) --
    # Configurable for future studies (10/12 b are hypothetical design
    # points); the initial QAT operating point is QAT_INITIAL_ADC_BITS = 8.
    adc_bits: int = QAT_INITIAL_ADC_BITS
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


_SCALARS = ("n", "clip_hi", "clip_lo", "rail_hi", "rail_lo", "amp_sat",
            "cm_would_clip", "cm_n",
            "voa_min", "voa_max", "voa_sum", "voa_sumsq",
            "vadc_min", "vadc_max", "vadc_sum", "vadc_sumsq",
            "cm_absmax", "cm_gain_absmax", "kf", "n_kept")
_SAMPLES = ("voa", "vadc", "code", "w")


def _weighted_percentile(values: np.ndarray, weights: np.ndarray,
                         q: float) -> float:
    """q in [0, 100]. Inverse of the weighted empirical CDF."""
    if values.size == 0:
        return float("nan")
    order = np.argsort(values, kind="stable")
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    target = q / 100.0 * cw[-1]
    i = int(np.searchsorted(cw, target, side="left"))
    return float(v[min(i, v.size - 1)])


class AnalogStats:
    """Running statistics of the analog nodes, bucketed by operation label.

    Exact running scalars (counts, clip/rail/amp counters, min, max, sum,
    sum of squares) plus a bounded, UNIFORM weighted subsample for
    percentiles. Each record() call keeps k = max(1, int(size * keep_frac))
    samples drawn uniformly from that call, each weighted size/k, so the
    weights always sum to the true conversion count. When a bucket exceeds
    `cap` samples, every stored sample is kept with probability 1/2 and its
    weight doubled, and keep_frac halves — the subsample stays uniform over
    the WHOLE run instead of freezing after the first `cap` samples (the
    previous behaviour, which biased percentiles towards the first images).

    Mergeable (merge / state_dict / from_state), so resumable chunked runs give
    the same scalar statistics as one long run.

    Per spec 12 this only OBSERVES; it never feeds back into the gain.
    """

    def __init__(self, keep_frac: float = 0.002, seed: int = 0, cap: int = 400_000):
        self.keep_frac, self.cap = keep_frac, cap
        self._rng = np.random.default_rng(seed)
        self.b = {}

    def _bucket(self, label):
        if label not in self.b:
            self.b[label] = dict(n=0, clip_hi=0, clip_lo=0, rail_hi=0,
                                 rail_lo=0, amp_sat=0, cm_would_clip=0, cm_n=0,
                                 voa_min=np.inf, voa_max=-np.inf,
                                 voa_sum=0.0, voa_sumsq=0.0,
                                 vadc_min=np.inf, vadc_max=-np.inf,
                                 vadc_sum=0.0, vadc_sumsq=0.0,
                                 cm_absmax=0.0, cm_gain_absmax=0.0,
                                 kf=float(self.keep_frac), n_kept=0,
                                 voa=[], vadc=[], code=[], w=[])
        return self.b[label]

    def _thin(self, s) -> None:
        """Halve the stored subsample (uniformly), doubling surviving weights."""
        while s["n_kept"] > self.cap:
            kept = 0
            new = {k: [] for k in _SAMPLES}
            for i in range(len(s["w"])):
                keep = self._rng.random(s["w"][i].size) < 0.5
                for k in _SAMPLES:
                    arr = s[k][i][keep]
                    if k == "w":
                        arr = arr * 2.0
                    new[k].append(arr)
                kept += int(keep.sum())
            for k in _SAMPLES:
                s[k] = new[k]
            s["n_kept"] = kept
            s["kf"] = s["kf"] / 2.0

    def record(self, label, v_oa, v_adc, d, clip_hi=0, clip_lo=0,
               rail_hi=0, rail_lo=0, amp_sat=0,
               cm_term=None, cm_gain=1.0, cm_would_clip=0, cm_n=0):
        s = self._bucket(label)
        v_oa = np.asarray(v_oa).ravel()
        v_adc = np.asarray(v_adc).ravel()
        n = v_oa.size
        s["n"] += n
        s["clip_hi"] += int(clip_hi)
        s["clip_lo"] += int(clip_lo)
        s["rail_hi"] += int(rail_hi)
        s["rail_lo"] += int(rail_lo)
        s["amp_sat"] += int(amp_sat)
        s["cm_would_clip"] += int(cm_would_clip)
        s["cm_n"] += int(cm_n)
        if cm_term is not None and np.size(cm_term):
            a = float(np.max(np.abs(cm_term)))
            s["cm_absmax"] = max(s["cm_absmax"], a)
            s["cm_gain_absmax"] = max(s["cm_gain_absmax"], a * abs(cm_gain))
        if not n:
            return
        s["voa_min"] = min(s["voa_min"], float(v_oa.min()))
        s["voa_max"] = max(s["voa_max"], float(v_oa.max()))
        s["voa_sum"] += float(v_oa.sum())
        s["voa_sumsq"] += float(np.dot(v_oa, v_oa))
        s["vadc_min"] = min(s["vadc_min"], float(v_adc.min()))
        s["vadc_max"] = max(s["vadc_max"], float(v_adc.max()))
        s["vadc_sum"] += float(v_adc.sum())
        s["vadc_sumsq"] += float(np.dot(v_adc, v_adc))
        k = max(1, int(n * s["kf"]))
        k = min(k, n)
        idx = self._rng.integers(0, n, size=k)
        s["voa"].append(v_oa[idx].astype(np.float32))
        s["vadc"].append(v_adc[idx].astype(np.float32))
        s["code"].append(np.asarray(d).ravel()[idx].astype(np.int32))
        s["w"].append(np.full(k, n / k, dtype=np.float64))
        s["n_kept"] += k
        if s["n_kept"] > self.cap:
            self._thin(s)

    # -- merge / serialisation (for resumable chunked runs) ---------------- #

    def merge(self, other: "AnalogStats") -> None:
        for label, o in other.b.items():
            s = self._bucket(label)
            for k in ("n", "clip_hi", "clip_lo", "rail_hi", "rail_lo",
                      "amp_sat", "cm_would_clip", "cm_n",
                      "voa_sum", "voa_sumsq", "vadc_sum", "vadc_sumsq",
                      "n_kept"):
                s[k] += o[k]
            for k in ("voa_min", "vadc_min"):
                s[k] = min(s[k], o[k])
            for k in ("voa_max", "vadc_max", "cm_absmax", "cm_gain_absmax"):
                s[k] = max(s[k], o[k])
            s["kf"] = min(s["kf"], o["kf"])
            for k in _SAMPLES:
                s[k].extend(o[k])
            if s["n_kept"] > self.cap:
                self._thin(s)

    def state_dict(self) -> tuple[dict, dict]:
        """(json-able scalars, {name: ndarray}) — e.g. for np.savez."""
        scal, arrs = {}, {}
        for label, s in self.b.items():
            key = str(label)
            scal[key] = {k: (float(s[k]) if isinstance(s[k], float) else int(s[k]))
                         for k in _SCALARS}
            for k in _SAMPLES:
                arrs[f"{key}__{k}"] = (np.concatenate(s[k]) if s[k]
                                       else np.zeros(0))
        return scal, arrs

    @classmethod
    def from_state(cls, scal: dict, arrs: dict, keep_frac: float = 0.002,
                   seed: int = 0, cap: int = 400_000) -> "AnalogStats":
        st = cls(keep_frac=keep_frac, seed=seed, cap=cap)
        for key, sv in scal.items():
            label = None if key == "None" else key
            s = st._bucket(label)
            for k in _SCALARS:
                s[k] = sv[k]
            for k in _SAMPLES:
                a = np.asarray(arrs[f"{key}__{k}"])
                s[k] = [a] if a.size else []
        return st

    # -- reporting ---------------------------------------------------------- #

    def summary(self, adc_vref=0.6, adc_bits=8) -> dict:
        out = {}
        half = adc_vref / 2.0
        for label, s in self.b.items():
            if not s["n"]:
                continue
            voa = (np.concatenate(s["voa"]) if s["voa"] else np.zeros(0, np.float32))
            vad = (np.concatenate(s["vadc"]) if s["vadc"] else np.zeros(0, np.float32))
            w = (np.concatenate(s["w"]) if s["w"] else np.zeros(0))
            n = s["n"]
            mean = s["voa_sum"] / n
            var = max(0.0, s["voa_sumsq"] / n - mean * mean)
            vmean = s["vadc_sum"] / n
            vvar = max(0.0, s["vadc_sumsq"] / n - vmean * vmean)
            r = dict(n=n, voa_min=s["voa_min"], voa_max=s["voa_max"],
                     voa_mean=mean, voa_std=var ** 0.5,
                     vadc_min=s["vadc_min"], vadc_max=s["vadc_max"],
                     vadc_mean=vmean, vadc_std=vvar ** 0.5,
                     clip_hi=s["clip_hi"], clip_lo=s["clip_lo"],
                     clip_pct=100.0 * (s["clip_hi"] + s["clip_lo"]) / n,
                     clip_hi_pct=100.0 * s["clip_hi"] / n,
                     clip_lo_pct=100.0 * s["clip_lo"] / n,
                     rail_hi=s["rail_hi"], rail_lo=s["rail_lo"],
                     amp_sat=s["amp_sat"], amp_sat_pct=100.0 * s["amp_sat"] / n,
                     cm_residue_absmax_v=s["cm_absmax"],
                     cm_residue_after_gain_absmax_v=s["cm_gain_absmax"],
                     cm_counterfactual_clip=s["cm_would_clip"],
                     cm_counterfactual_n=s["cm_n"],
                     cm_counterfactual_clip_pct=(100.0 * s["cm_would_clip"] / s["cm_n"]
                                                 if s["cm_n"] else 0.0),
                     n_samples=int(voa.size),
                     # exact |V_ADC_IN| max over the whole run / usable swing
                     util_max_pct=100.0 * max(abs(s["vadc_min"]),
                                              abs(s["vadc_max"])) / half)
            if voa.size:
                a = np.abs(voa).astype(np.float64)
                r.update(voa_median=_weighted_percentile(voa.astype(np.float64), w, 50),
                         voa_p01=_weighted_percentile(voa.astype(np.float64), w, 1),
                         voa_p99=_weighted_percentile(voa.astype(np.float64), w, 99),
                         voa_absp50=_weighted_percentile(a, w, 50),
                         voa_absp95=_weighted_percentile(a, w, 95),
                         voa_absp99=_weighted_percentile(a, w, 99),
                         voa_absp999=_weighted_percentile(a, w, 99.9),
                         voa_absmax_sampled=float(a.max()))
            if vad.size:
                vd = vad.astype(np.float64)
                av = np.abs(vd)
                r.update(vadc_median=_weighted_percentile(vd, w, 50),
                         vadc_p01=_weighted_percentile(vd, w, 1),
                         vadc_p99=_weighted_percentile(vd, w, 99),
                         vadc_absp50=_weighted_percentile(av, w, 50),
                         vadc_absp95=_weighted_percentile(av, w, 95),
                         vadc_absp99=_weighted_percentile(av, w, 99),
                         vadc_absp999=_weighted_percentile(av, w, 99.9),
                         # how much of the usable +/-Vref/2 swing is occupied
                         util_p999_pct=100.0 * _weighted_percentile(av, w, 99.9) / half,
                         util_median_pct=100.0 * _weighted_percentile(av, w, 50) / half)
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
        if dac_bits < act_bits:
            raise ValueError(
                f"dac_bits={dac_bits} < act_bits={act_bits}: an activation code "
                f"cannot be represented on a narrower DAC without an explicit "
                f"requantization, which this model does not perform.")
        align = str(cfg.dac_input_alignment).lower()
        if align not in DAC_ALIGNMENTS:
            raise ValueError(f"dac_input_alignment must be one of "
                             f"{DAC_ALIGNMENTS}, got {cfg.dac_input_alignment!r}")
        self.dac_alignment = align
        # integer factor applied to the NN activation code before the DAC
        self.dac_shift = (dac_bits - act_bits) if align == "msb" else 0
        self.dac_code_scale = 2 ** self.dac_shift

        self.dac = CapacitiveDAC(bits=dac_bits, vref=cfg.dac_vref,
                                 nonidealities=cfg.dac_non, **cfg.dac_kwargs)
        self.macro = C2CMacro(rows=rows, weight_bits=weight_bits, m=cfg.macro_m,
                              adapter=SignedWeightAdapter(weight_bits),
                              nonidealities=cfg.macro_non)
        self.amp = PreADCGain(cfg.gain_default, cfg.gain_non)
        self.adc = SARADC(cfg.adc_bits, cfg.adc_vref, cfg.adc_non,
                          SARDAC(cfg.adc_bits, cfg.adc_vref, cfg.sar_dac_non))

        self.stats = stats if stats is not None else AnalogStats()
        self.label = None                 # current operation, for stats+gain
        self._prog = None                 # programmed (sign, cap)
        self._colsum_eff = None           # per-column sum of sign*cap (CM diag.)

    # -- scale ------------------------------------------------------------- #

    @property
    def k_mac(self) -> float:
        """Volts of V_OA per unit of sum(X_q W_q). Architecture-derived.

        dac_code_scale converts one unit of the NN activation code into DAC
        codes (1 unless an activation is MSB-aligned onto a wider DAC).
        """
        return self.dac.v_per_unit * self.dac_code_scale * self.macro.gain_per_unit_v()

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
                self.dac.signed_input, self.dac_shift, self.macro.m,
                self.weight_bits, round(self.gain, 12), self.adc.bits,
                self.adc.vref)

    def requantizer(self) -> Requantizer:
        return Requantizer(self.lsb_p)

    def accumulator(self, shape) -> DigitalAccumulator:
        return DigitalAccumulator(shape, self.domain_key)

    # -- weight programming ------------------------------------------------ #

    def program(self, w_q: np.ndarray) -> None:
        self._prog = self.macro.program(w_q)
        sign, cap = self._prog
        # weight-only quantity for the common-mode diagnostic (no data involved)
        self._colsum_eff = (sign.astype(np.float64) * cap).sum(axis=0)

    # -- one analog conversion --------------------------------------------- #

    def dac_codes(self, x_q: np.ndarray) -> np.ndarray:
        """NN activation codes -> codes on the DAC's own bit grid (alignment)."""
        x_q = np.asarray(x_q)
        if self.dac_code_scale == 1:
            return x_q
        return x_q.astype(np.int64) * self.dac_code_scale

    def execute(self, x_q: np.ndarray) -> np.ndarray:
        """Signed activation codes [T, rows] -> signed ADC codes [T, cols].

        Every stage is a real object boundary carrying a physical quantity.
        """
        if self._prog is None:
            raise RuntimeError("no weights programmed; call program() first")
        sign, cap = self._prog

        v_ia = self.dac.transfer(self.dac_codes(x_q))       # 1. input DAC
        v_oa = self.macro.mac(v_ia, sign, cap)              # 2. C2C macro
        sat0 = self.amp.amp_saturated
        v_adc_in = self.amp.apply(v_oa)                     # 3. pre-ADC gain

        hi0, lo0 = self.adc.n_clipped_hi, self.adc.n_clipped_lo
        rh0, rl0 = self.adc.n_rail_hi, self.adc.n_rail_lo
        d = self.adc.convert(v_adc_in)                      # 4. SAR ADC

        # ---- diagnostics only (nothing below changes d) ------------------ #
        cm_term, cm_clip, cm_n = None, 0, 0
        v_cm = self.dac.v_common_mode
        if v_cm and self._colsum_eff is not None:
            # term an offset-binary DAC adds to each column's V_OA if the
            # common mode is NOT cancelled; the model cancels it ideally.
            cm_term = v_cm * self._colsum_eff / float(self.macro.m)   # [cols]
            g = self.amp.gain * (1.0 + self.amp.non.gain_error)
            cf = g * (v_oa + cm_term[None, :])
            cm_clip = int(np.count_nonzero(np.abs(cf) > self.adc.vref / 2.0))
            cm_n = int(cf.size)
        self.stats.record(self.label, v_oa, v_adc_in, d,
                          self.adc.n_clipped_hi - hi0,
                          self.adc.n_clipped_lo - lo0,
                          rail_hi=self.adc.n_rail_hi - rh0,
                          rail_lo=self.adc.n_rail_lo - rl0,
                          amp_sat=self.amp.amp_saturated - sat0,
                          cm_term=cm_term, cm_gain=self.amp.gain,
                          cm_would_clip=cm_clip, cm_n=cm_n)
        return d

    # -- introspection ----------------------------------------------------- #

    def describe(self) -> dict:
        return dict(dac=self.dac.describe(), macro=self.macro.describe(),
                    gain=self.amp.describe(), adc=self.adc.describe(),
                    dac_alignment=self.dac_alignment,
                    dac_code_scale=self.dac_code_scale,
                    amp_saturation_modeled=self.amp.saturation_modeled,
                    k_mac=self.k_mac, lsb_p=self.lsb_p,
                    domain_key=self.domain_key)

    def scale_report(self) -> str:
        d = self.dac
        lines = [
            "analog scale chain",
            f"  DAC {d.name!r}: {d.bits}b, Vref={d.vref} V, "
            f"v_per_unit = {d.v_per_unit:.6e} V/code, "
            f"signed encoding = {d.signed_encoding}",
            f"  act->DAC alignment: {self.dac_alignment} "
            f"(x{self.dac_code_scale} per activation unit)",
            f"  macro: 1/(m*2^B) = {self.macro.gain_per_unit_v():.6e}, m={self.macro.m}",
            f"  K_MAC = {self.k_mac:.6e} V per unit sum(Xq*Wq)",
            f"  gain G = {self.gain:g}  (operation {self.label!r})",
            f"  ADC: {self.adc.bits}b, Vref={self.adc.vref} V, "
            f"LSB = {self.adc.lsb_volts*1e3:.4f} mV, "
            f"signed swing = +/-{self.adc.signed_swing_volts*1e3:.1f} mV",
            f"  LSB_P = {self.lsb_p:.4f} column-sum units per code",
        ]
        return "\n".join(lines)
