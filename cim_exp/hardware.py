"""
hardware.py — the primary hardware matrix, job identities and cache keys.

PRIMARY MATRIX (24 configurations per model)
    precision  : W4/A4, W4/A8, W8/A4, W8/A8
    ADC        : 8, 10, 12 bits, V_REF_ADC = 0.6 V
    gain set   : "gain1"       qkv=1,     qk=1,    av=1
                 "cal_w8a8"    qkv=23.77, qk=8.41, av=2.92
    fixed      : 32x32 macro, analog chain ON, CAPACITIVE (unsigned CDAC)
                 8-bit, V_REF_DAC = 0.6 V, A4 MSB-aligned on the 8-bit DAC,
                 ideal nonidealities, amplifier swing NOT limited (saturation
                 not modelled), sites qkv/qk/av on CIM, out-proj/MLP in FP32.

The calibrated gains were derived for W8/A8 (P99.99 full scales on synthetic
calibration inputs, CLAUDE.md §6). They are held FIXED across precisions here
and are not claimed optimal for any other precision.

Mixed precision (W4/A8, W8/A4) is an INFERENCE configuration; no checkpoint
was trained at mixed precision.

QAT checkpoint selection ("matched", default): weight bits 4 -> qat_int4,
weight bits 8 -> qat_int8. "both" additionally runs each QAT checkpoint at the
other weight precision (flagged precision_matches_training = False).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

PRECISIONS = ((4, 4), (4, 8), (8, 4), (8, 8))
ADC_BITS = (8, 10, 12)
GAIN_SETS = {
    "gain1": {"qkv": 1.0, "qk": 1.0, "av": 1.0},
    "cal_w8a8": {"qkv": 23.77, "qk": 8.41, "av": 2.92},
}
GAIN_SET_NOTES = {
    "gain1": "no amplification (G=1 on every operation)",
    "cal_w8a8": "calibrated for W8/A8 (capacitive DAC, P99.99 full scale); "
                "held fixed across precisions; not claimed optimal for other precisions",
}
OPS = ("qkv", "qk", "av")


@dataclass(frozen=True)
class HardwareConfig:
    weight_bits: int
    act_bits: int
    adc_bits: int
    gain_set: str
    adc_vref: float = 0.6
    dac_bits: int = 8
    dac_vref: float = 0.6
    dac_alignment: str = "msb"
    rows: int = 32
    cols: int = 32
    num_arrays: int = 1
    token_block: int | None = None
    cim_ops: tuple = OPS
    cim_out_proj: bool = False
    # None => ideal linear amplifier, saturation NOT modelled (primary default:
    # no amplifier swing specification exists). A value in volts clips |V_ADC_IN|
    # at the amplifier output and counts saturations separately from the ADC.
    amp_swing_limit_v: float | None = None
    nonidealities: str = "ideal"
    # Experiment B only: explicit (qkv, qk, av) gains for a calibrated gain set
    # whose name is `gain_set` and whose provenance is `gain_note`. Never used by
    # the primary (Experiment A) matrix.
    custom_gains: tuple | None = None
    gain_note: str = ""

    @property
    def dac_type(self) -> str:
        """Fixed label: the simulator models only the hardware-specified
        unsigned 8-bit capacitive DAC (no longer a configuration field)."""
        return "CAPACITIVE"

    @property
    def gains(self) -> dict:
        if self.custom_gains is not None:
            return dict(zip(OPS, (float(g) for g in self.custom_gains)))
        return dict(GAIN_SETS[self.gain_set])

    @property
    def gain_description(self) -> str:
        return self.gain_note or GAIN_SET_NOTES.get(self.gain_set, "")

    @property
    def label(self) -> str:
        amp = "" if self.amp_swing_limit_v is None else f"_amp{self.amp_swing_limit_v:g}V"
        return f"W{self.weight_bits}A{self.act_bits}_ADC{self.adc_bits}_{self.gain_set}{amp}"

    @property
    def precision_label(self) -> str:
        return f"W{self.weight_bits}/A{self.act_bits}"

    @property
    def mixed_precision(self) -> bool:
        return self.weight_bits != self.act_bits

    def as_dict(self) -> dict:
        d = asdict(self)
        d["cim_ops"] = list(self.cim_ops)
        d["custom_gains"] = None if self.custom_gains is None else list(self.custom_gains)
        d["gains"] = self.gains
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "HardwareConfig":
        d = dict(d)
        d.pop("gains", None)
        # cached job.json files written before Oct 2026 carry a dac_type field
        dt = d.pop("dac_type", "CAPACITIVE")
        if str(dt).upper() != "CAPACITIVE":
            raise ValueError(f"dac_type {dt!r}: only the capacitive DAC is modelled")
        d["cim_ops"] = tuple(d["cim_ops"])
        if d.get("custom_gains") is not None:
            d["custom_gains"] = tuple(d["custom_gains"])
        return cls(**d)

    def cim_config(self, analog_stats=None):
        """-> (CIMConfig, AnalogConfig). Analog chain always ON."""
        from cim_sim import CIMConfig
        from cim_sim.analog import AnalogConfig, GainNonidealities
        if self.nonidealities != "ideal":
            raise NotImplementedError("only the ideal analog chain is configured here")
        an = AnalogConfig(enabled=True, dac_bits=self.dac_bits,
                          dac_vref=self.dac_vref, dac_input_alignment=self.dac_alignment,
                          gain_default=1.0, gain_per_op=self.gains,
                          adc_bits=self.adc_bits, adc_vref=self.adc_vref,
                          gain_non=GainNonidealities(v_swing_limit=self.amp_swing_limit_v))
        cfg = CIMConfig(rows=self.rows, cols=self.cols, weight_bits=self.weight_bits,
                        act_bits=self.act_bits, token_block=self.token_block,
                        adc_enabled=False, analog=an)
        return cfg


def primary_matrix(amp_swing_limit_v: float | None = None) -> list[HardwareConfig]:
    """Experiment A: fixed gain sets (G=1 and the W8/A8-calibrated set), never
    recalibrated per checkpoint or precision."""
    return [HardwareConfig(weight_bits=w, act_bits=a, adc_bits=b, gain_set=g,
                           amp_swing_limit_v=amp_swing_limit_v)
            for (w, a) in PRECISIONS for b in ADC_BITS for g in GAIN_SETS]


def qat_model_for(weight_bits: int) -> str:
    return {4: "qat_int4", 8: "qat_int8"}[weight_bits]


def planned_cim_jobs(qat_selection: str = "matched",
                     amp_swing_limit_v: float | None = None) -> list[tuple[str, HardwareConfig]]:
    jobs = []
    for hw in primary_matrix(amp_swing_limit_v):
        jobs.append(("pretrained", hw))
        if qat_selection == "matched":
            jobs.append((qat_model_for(hw.weight_bits), hw))
        elif qat_selection == "both":
            jobs += [("qat_int4", hw), ("qat_int8", hw)]
        else:
            raise ValueError(qat_selection)
    return jobs


# --------------------------------------------------------------------------- #
# cache keys
# --------------------------------------------------------------------------- #


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def short_hash(obj, n: int = 16) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()[:n]


def code_fingerprint(repo_root: Path) -> str:
    """SHA-256 over every source file that can change an output number."""
    files = sorted((repo_root / "cim_sim").rglob("*.py"))
    files += [repo_root / "tiny_vit.py", repo_root / "run_tinyvit.py",
              repo_root / "cim_exp" / "engine.py", repo_root / "cim_exp" / "imagenette.py"]
    h = hashlib.sha256()
    for f in files:
        h.update(f.relative_to(repo_root).as_posix().encode())
        # normalise line endings so a CRLF checkout gives the same fingerprint
        h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


@dataclass
class RunContext:
    """Everything shared by all jobs of one run (stage)."""
    run_name: str
    manifest_sha256: str
    preprocess_id: str
    code_sha256: str
    torch_version: str
    numpy_version: str
    threads: int = 1
    batch_size: int = 1
    chunk_size: int = 25
    stats_keep_frac: float = 2e-4
    stats_cap: int = 400_000
    extra: dict = field(default_factory=dict)

    def base(self) -> dict:
        return dict(manifest=self.manifest_sha256, preprocess=self.preprocess_id,
                    code=self.code_sha256, torch=self.torch_version,
                    numpy=self.numpy_version, threads=self.threads,
                    batch_size=self.batch_size)


def reference_key(ctx: RunContext, model_id: str, ckpt_sha256: str) -> str:
    return "ref_" + model_id + "_" + short_hash(dict(kind="fp32_reference", model=model_id,
                                                     ckpt=ckpt_sha256, **ctx.base()))


def cim_key(ctx: RunContext, model_id: str, ckpt_sha256: str, hw: HardwareConfig) -> str:
    return ("cim_" + model_id + "_" + hw.label + "_" +
            short_hash(dict(kind="cim_analog", model=model_id, ckpt=ckpt_sha256,
                            hw=hw.as_dict(), chunk_size=ctx.chunk_size,
                            stats_keep_frac=ctx.stats_keep_frac,
                            stats_cap=ctx.stats_cap, **ctx.base())))
