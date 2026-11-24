"""
calibration.py — EXPERIMENT B: per-operation gain calibration and robustness.

Experiment B is a SEPARATE follow-up. It never changes the gains of Experiment A
(G = 1 and the fixed W8/A8 set 23.77 / 8.41 / 2.92) and writes only to
results/<stage>/expB/ (its CIM jobs share the run's cache directory, keyed by
their own gains, so nothing can mix).

QUESTION
--------
Do gains calibrated on one data distribution / checkpoint / precision clip on
another, and what do calibrated gains buy relative to the fixed set?

CALIBRATION DATA (strictly separate from evaluation)
----------------------------------------------------
Imagenette **train** split, `per_class` images per class (default 10 -> 100
images), seeded selection (seed 4242, the historical CAL_SEED). The evaluation
images are the run's **val** manifest. Disjointness is checked by file SHA-256
and the command refuses to continue on any overlap. (Imagenette train images
are ImageNet-1k training images; that matters for accuracy, not for measuring
activation ranges.)

WHAT IS MEASURED
----------------
For every (calibration checkpoint, precision) the network is run with an EXACT
probe in place of the analog chain: the probe returns the exact integer column
sum P for every 32-row depth block (so the network sees operand quantization
only - no ADC, no gain, hence no circularity between the gain being calibrated
and the activations used to calibrate it) and records an exact histogram of |P|
per operation (qkv, qk, av). In the analog chain V_ADC_IN = G * K_MAC * P, and
the ADC clips when |V_ADC_IN| > V_REF_ADC / 2, so for a full scale FS (in
column-sum units):

    G(FS)               = (V_REF_ADC / 2) / (K_MAC * FS)
    clip fraction (G)   = P(|P| > (V_REF_ADC/2) / (K_MAC * G))      [predicted]

K_MAC is the architecture constant of that precision (8-bit CDAC, MSB-aligned
activations, m = 32, B = weight bits). This is the same formula that produced
the fixed gains (CLAUDE.md §6), now on real images and per precision/checkpoint.

CALIBRATION OBJECTIVE (pre-declared, not tuned on evaluation data)
------------------------------------------------------------------
PRIMARY level "p99.99": FS = the 99.99th percentile of |P| on the calibration
set, i.e. a clipping TARGET of <= 0.01 % of conversions per operation on the
calibration data — the same criterion as the original W8/A8 gains.
SENSITIVITY levels (reported, never used to pick a "winner" on evaluation
data): "p99.9" (higher gain, target <= 0.1 %), "p99.999" (lower gain, <= 0.001 %),
"p100" (largest observed |P|: no clipping on the calibration data).
Gains are rounded to 0.01 (register-like resolution; < 0.05 % change of FS).

The trade-off is explicit: a higher gain uses more of the ADC range (finer
effective LSB) but clips more; a lower gain clips less but quantizes coarser.
The robustness evaluation reports both sides (clipping per operation and the
resulting logit error / accuracy) on held-out images.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .hardware import OPS, PRECISIONS, HardwareConfig, qat_model_for

CAL_SEED = 4242
LEVELS = {"p99.9": 99.9, "p99.99": 99.99, "p99.999": 99.999, "p100": 100.0}
PRIMARY_LEVEL = "p99.99"
# held-out clipping above this multiple of the calibration target is reported as a
# material generalisation failure (sampling noise alone gives ratios near 1)
MATERIAL_EXCESS = 2.0
LEVEL_TARGET_PCT = {k: round(100.0 - q, 6) for k, q in LEVELS.items()}


# --------------------------------------------------------------------------- #
# exact probe (drop-in for AnalogChain inside CIMArray; observation only)
# --------------------------------------------------------------------------- #


class _NoADC:
    n_clipped_hi = 0
    n_clipped_lo = 0


class ProbeChain:
    """Exact digital MAC with a per-operation |P| histogram.

    Implements exactly the AnalogChain interface CIMArray/cim_matmul use
    (set_operation, program, execute, domain_key, lsb_p, adc counters), with
    lsb_p = 1 so the scheduler's single rescale yields the exact integer
    result - bit-identical to the legacy ADC-off path (tested).
    """

    def __init__(self, hmax: int, flush_every: int = 4_000_000):
        self.hmax = int(hmax)
        self.hist = {}
        self.maxabs = {}
        self._buf = {}
        self._buf_n = 0
        self.flush_every = flush_every
        self.label = None
        self.adc = _NoADC()
        self.lsb_p = 1.0
        self.domain_key = ("exact_probe",)
        self._W = None

    def set_operation(self, op, act_signed=True):
        self.label = op

    def program(self, w_q):
        self._W = np.asarray(w_q, dtype=np.int64)

    def execute(self, x_q):
        P = np.asarray(x_q, dtype=np.int64) @ self._W
        a = np.abs(P).ravel()
        self._buf.setdefault(self.label, []).append(a)
        self._buf_n += a.size
        if self._buf_n >= self.flush_every:
            self.flush()
        return P

    def flush(self):
        for op, parts in self._buf.items():
            a = np.concatenate(parts)
            m = int(a.max()) if a.size else 0
            if m > self.hmax:
                raise ValueError(f"|P|={m} exceeds histogram range {self.hmax}")
            h = self.hist.setdefault(op, np.zeros(self.hmax + 1, dtype=np.int64))
            h += np.bincount(a, minlength=self.hmax + 1)
            self.maxabs[op] = max(self.maxabs.get(op, 0), m)
        self._buf, self._buf_n = {}, 0


def hist_range(weight_bits: int, act_bits: int, rows: int = 32) -> int:
    """Largest possible |P| for one depth block (unsigned activations worst case)."""
    return rows * 2 ** (weight_bits - 1) * 2 ** act_bits


def observe(repo_root: Path, ckpt_path: Path, weight_bits: int, act_bits: int,
            inputs_path: Path) -> dict:
    """Run the calibration images through the exact-probe network; return the
    per-operation |P| histograms."""
    from . import engine as EN
    EN._setup_paths(repo_root)
    EN.configure_threads(1)
    import torch
    import run_tinyvit
    from cim_sim import CIMArray
    from cim_sim.attention import convert_attention
    from .imagenette import to_model_input

    hw = HardwareConfig(weight_bits=weight_bits, act_bits=act_bits, adc_bits=8,
                        gain_set="gain1")
    model = run_tinyvit.build_model(ckpt_path)
    arr = CIMArray(hw.cim_config())
    probe = ProbeChain(hist_range(weight_bits, act_bits))
    arr.chain = probe                                   # exact, observation only
    convert_attention(model, arr)
    u8 = np.load(inputs_path, mmap_mode="r")
    t0 = time.time()
    with torch.no_grad():
        for i in range(u8.shape[0]):
            model(to_model_input(u8[i:i + 1]))
            probe.flush()
    return dict(hist=probe.hist, maxabs=probe.maxabs, n_images=int(u8.shape[0]),
                wall_s=time.time() - t0)


# --------------------------------------------------------------------------- #
# full scale / gain arithmetic
# --------------------------------------------------------------------------- #


def tail_counts(hist: np.ndarray) -> np.ndarray:
    """tail[v] = count(|P| > v)."""
    return hist.sum() - np.cumsum(hist)


def full_scale(hist: np.ndarray, q: float) -> int:
    """Smallest integer FS with count(|P| > FS) <= floor(n * (1 - q/100))."""
    n = int(hist.sum())
    if q >= 100.0:
        nz = np.nonzero(hist)[0]
        return int(nz[-1]) if nz.size else 1
    allowed = int(np.floor(n * (1.0 - q / 100.0)))
    tail = tail_counts(hist)
    return max(1, int(np.argmax(tail <= allowed)))


def k_mac(weight_bits: int, act_bits: int, op: str) -> float:
    from cim_sim.analog import AnalogChain
    hw = HardwareConfig(weight_bits=weight_bits, act_bits=act_bits, adc_bits=8,
                        gain_set="gain1")
    ch = AnalogChain(hw.cim_config().analog, hw.rows, weight_bits, act_bits)
    ch.set_operation(op, act_signed=(op != "av"))
    return ch.k_mac


def gain_for_fs(fs: float, kmac: float, adc_vref: float = 0.6) -> float:
    return (adc_vref / 2.0) / (kmac * fs)


def predicted_clip_pct(hist: np.ndarray, gain: float, kmac: float,
                       adc_vref: float = 0.6) -> float:
    """% of calibration conversions with |G*K_MAC*P| > V_REF_ADC/2 (exact-MAC network)."""
    n = int(hist.sum())
    thr = (adc_vref / 2.0) / (kmac * gain)            # in column-sum units
    v = int(np.floor(thr))
    if v >= hist.size - 1:
        return 0.0
    return 100.0 * float(tail_counts(hist)[v]) / n


def calibrate(obs: dict, weight_bits: int, act_bits: int, adc_vref: float = 0.6) -> dict:
    """-> {level: {"gains": {op: g}, "fs": {op: FS}, ...}} plus fixed-gain predictions."""
    from .hardware import GAIN_SETS
    out = {"levels": {}, "kmac": {}, "n_colsums": {}, "maxabs": {}, "predicted_clip_pct": {}}
    for op in OPS:
        out["kmac"][op] = k_mac(weight_bits, act_bits, op)
        out["n_colsums"][op] = int(obs["hist"][op].sum())
        out["maxabs"][op] = int(obs["maxabs"][op])
    for lvl, q in LEVELS.items():
        fs = {op: full_scale(obs["hist"][op], q) for op in OPS}
        g = {op: round(gain_for_fs(fs[op], out["kmac"][op], adc_vref), 2) for op in OPS}
        out["levels"][lvl] = dict(quantile_pct=q, clip_target_pct=LEVEL_TARGET_PCT[lvl],
                                  fs=fs, gains=g)
    # predicted calibration-set clipping for every level and for the fixed sets
    sets = {f"calB_{lvl}": out["levels"][lvl]["gains"] for lvl in LEVELS}
    sets.update(GAIN_SETS)
    for name, g in sets.items():
        out["predicted_clip_pct"][name] = {
            op: predicted_clip_pct(obs["hist"][op], g[op], out["kmac"][op], adc_vref)
            for op in OPS}
    return out


# --------------------------------------------------------------------------- #
# robustness job planning
# --------------------------------------------------------------------------- #


def model_for(source: str, weight_bits: int) -> str:
    return "pretrained" if source == "pretrained" else qat_model_for(weight_bits)


def plan_jobs(cal: dict, adc_bits_list, levels, cal_manifest_sha: str,
              cross_apply: bool = False, precision_transfer: bool = True
              ) -> list[tuple[str, HardwareConfig, dict]]:
    """cal[(source, 'W{w}A{a}')] -> calibration dict. Returns
    (tested_model, HardwareConfig, provenance) for every Experiment B run,
    including the Experiment A baselines (G=1, fixed W8/A8 set) for comparison;
    those baselines have identical HardwareConfigs (and cache keys) to Exp. A."""
    jobs = []
    seen = set()

    def add(tested, hw, prov):
        k = (tested, hw.label, hw.custom_gains)
        if k not in seen:
            seen.add(k)
            jobs.append((tested, hw, prov))

    for src in ("pretrained", "qat"):
        for (w, a) in PRECISIONS:
            ptag = f"W{w}A{a}"
            if (src, ptag) not in cal:
                continue
            cal_model = model_for(src, w)
            for adc in adc_bits_list:
                testeds = [cal_model]
                if cross_apply:
                    other = model_for("qat" if src == "pretrained" else "pretrained", w)
                    testeds.append(other)
                for tested in testeds:
                    for gs in ("gain1", "cal_w8a8"):
                        add(tested, HardwareConfig(weight_bits=w, act_bits=a, adc_bits=adc,
                                                   gain_set=gs),
                            dict(kind="experiment_A_baseline", source=gs))
                    for lvl in levels:
                        if tested != cal_model and lvl != PRIMARY_LEVEL:
                            continue          # cross-checkpoint transfer: primary level only
                        g = cal[(src, ptag)]["levels"][lvl]["gains"]
                        name = f"calB_{src}_{ptag}_{lvl}"
                        note = (f"Experiment B: calibrated on {cal_model} at {ptag}, objective "
                                f"{lvl} (clip target <= {LEVEL_TARGET_PCT[lvl]}% per op on the "
                                f"calibration set), cal manifest {cal_manifest_sha[:12]}")
                        add(tested, HardwareConfig(weight_bits=w, act_bits=a, adc_bits=adc,
                                                   gain_set=name,
                                                   custom_gains=tuple(g[op] for op in OPS),
                                                   gain_note=note),
                            dict(kind=("calibrated_matched" if tested == cal_model
                                       else "calibrated_cross_checkpoint"),
                                 cal_model=cal_model, cal_precision=ptag, level=lvl,
                                 clip_target_pct=LEVEL_TARGET_PCT[lvl]))
                    # precision transfer: W8A8-calibrated (primary level) applied here
                    if precision_transfer and ptag != "W8A8" and (src, "W8A8") in cal \
                            and tested == cal_model:
                        g = cal[(src, "W8A8")]["levels"][PRIMARY_LEVEL]["gains"]
                        src_model = model_for(src, 8)
                        name = f"calB_{src}_W8A8_{PRIMARY_LEVEL}_xfer"
                        note = (f"Experiment B precision transfer: calibrated on {src_model} at "
                                f"W8A8 ({PRIMARY_LEVEL}), applied at {ptag}")
                        add(tested, HardwareConfig(weight_bits=w, act_bits=a, adc_bits=adc,
                                                   gain_set=name,
                                                   custom_gains=tuple(g[op] for op in OPS),
                                                   gain_note=note),
                            dict(kind="calibrated_precision_transfer", cal_model=src_model,
                                 cal_precision="W8A8", level=PRIMARY_LEVEL,
                                 clip_target_pct=LEVEL_TARGET_PCT[PRIMARY_LEVEL]))
    return jobs


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def build_rows(jobs, finals: dict, refs: dict, cal: dict, y1000, y10, idx10,
               hashes: dict, cal_meta: dict) -> list[dict]:
    from . import metrics as M
    rows = []
    for tested, hw, prov in jobs:
        key = finals.get((tested, hw.label, hw.custom_gains))
        if key is None:
            continue
        meta, lg = key
        ref = refs[tested]
        mm = M.model_metrics(lg, y1000, y10, idx10)
        summ = meta["analog_summary"]
        ptag = f"W{hw.weight_bits}A{hw.act_bits}"
        src_for_pred = "pretrained" if tested == "pretrained" else "qat"
        pred = None
        c = cal.get((src_for_pred, ptag))
        if c is not None and "_hist" in c:      # predicted clipping of THESE gains on the
            pred = {op: predicted_clip_pct(c["_hist"][op], hw.gains[op],   # calibration set
                                           c["kmac"][op], hw.adc_vref) for op in OPS}
        r = dict(tested_model=tested, tested_checkpoint_sha256=hashes[tested],
                 run_id=meta.get("key", ""), precision=hw.precision_label,
                 weight_bits=hw.weight_bits, act_bits=hw.act_bits, adc_bits=hw.adc_bits,
                 adc_vref_v=hw.adc_vref, dac_bits=hw.dac_bits, dac_vref_v=hw.dac_vref,
                 gain_set=hw.gain_set, gain_kind=prov["kind"],
                 gain_qkv=hw.gains["qkv"], gain_qk=hw.gains["qk"], gain_av=hw.gains["av"],
                 gain_provenance=hw.gain_description,
                 calibration_checkpoint=prov.get("cal_model", ""),
                 calibration_precision=prov.get("cal_precision", ""),
                 calibration_objective=prov.get("level", ""),
                 clip_target_pct_per_op=prov.get("clip_target_pct", ""),
                 calibration_manifest_sha256=cal_meta.get("manifest_sha256", ""),
                 calibration_split=cal_meta.get("split", ""),
                 calibration_n_images=cal_meta.get("n_images", ""),
                 reference=f"{tested} FP32 software",
                 agreement_vs_own_fp32_pct=M.agreement_pct(lg, ref),
                 rel_logit_rms_err_pct_vs_own_fp32=M.rel_rms_pct(lg, ref),
                 **mm,
                 adc_clip_pct_all=100.0 * meta["cim_stats"]["adc_saturations"]
                 / meta["cim_stats"]["adc_conversions"],
                 images_with_any_clipping=meta["images_with_any_clipping"],
                 status=meta["status"])
        for op in OPS:
            v = summ[op]
            r[f"heldout_clip_pct_{op}"] = 100.0 * (v["clip_hi"] + v["clip_lo"]) / v["n"]
            r[f"calibration_predicted_clip_pct_{op}"] = ("" if pred is None or pred.get(op) is None
                                                         else pred[op])
            r[f"adc_util_p999_pct_{op}"] = v.get("util_p999_pct", float("nan"))
            tgt = prov.get("clip_target_pct")
            if tgt in (None, ""):
                r[f"heldout_to_target_ratio_{op}"] = ""
                r[f"heldout_exceeds_target_{op}"] = ""
            elif tgt == 0:          # p100: any held-out clipping exceeds a zero target
                r[f"heldout_to_target_ratio_{op}"] = (float("inf") if r[f"heldout_clip_pct_{op}"] > 0
                                                      else 0.0)
                r[f"heldout_exceeds_target_{op}"] = r[f"heldout_clip_pct_{op}"] > 0
            else:
                ratio = r[f"heldout_clip_pct_{op}"] / tgt
                r[f"heldout_to_target_ratio_{op}"] = ratio
                r[f"heldout_exceeds_target_{op}"] = ratio > MATERIAL_EXCESS
        rows.append(r)
    return rows


def render_md(rows: list[dict], cal: dict, cal_meta: dict, levels) -> str:
    L = ["# Experiment B — gain calibration and robustness\n",
         "Separate from the primary (Experiment A) matrix: Experiment A keeps G = 1 and the "
         "fixed W8/A8 gains (23.77 / 8.41 / 2.92) for every precision and checkpoint; nothing "
         "here changes them.\n",
         f"- Calibration data: Imagenette `{cal_meta.get('split')}` split, "
         f"{cal_meta.get('n_images')} images ({cal_meta.get('per_class')} per class, seed "
         f"{cal_meta.get('seed')}), manifest SHA-256 `{str(cal_meta.get('manifest_sha256'))[:16]}…`, "
         f"verified disjoint from the evaluation manifest (by file SHA-256).",
         "- Measurement: exact column sums per 32-row depth block (operand quantization only; "
         "no ADC/gain in the calibration network), per operation.",
         f"- Objective (pre-declared): **{PRIMARY_LEVEL}** — FS = P99.99 of |P| on the calibration "
         "set, i.e. clip target <= 0.01% of conversions per operation; G = (V_REF_ADC/2)/(K_MAC*FS), "
         "rounded to 0.01. Sensitivity levels " + ", ".join(f"`{l}`" for l in levels if l != PRIMARY_LEVEL)
         + " are reported, never used to select a gain on evaluation data.",
         "- Evaluation: held-out Imagenette val images (the run's manifest); reference = the "
         "tested checkpoint's own FP32 software output.\n",
         "## Calibrated gains\n",
         "| cal. checkpoint | precision | level | G_qkv | G_qk | G_av | FS_qkv | FS_qk | FS_av | "
         "pred. clip % cal (qkv/qk/av) | fixed-set pred. clip % cal (qkv/qk/av) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (src, ptag), c in sorted(cal.items()):
        fx = c["predicted_clip_pct"]["cal_w8a8"]
        for lvl in levels:
            lv = c["levels"][lvl]
            pc = c["predicted_clip_pct"][f"calB_{lvl}"]
            L.append(f"| {c['cal_model']} | {ptag} | {lvl} | " +
                     " | ".join(f"{lv['gains'][o]:.2f}" for o in OPS) + " | " +
                     " | ".join(str(lv["fs"][o]) for o in OPS) + " | " +
                     "/".join(f"{pc[o]:.4f}" for o in OPS) + " | " +
                     "/".join(f"{fx[o]:.4f}" for o in OPS) + " |")
    L.append("\n## Held-out evaluation\n")
    L.append("Clip % = clipped conversions / conversions of that operation on the held-out "
             "images. `held-out/target` = worst operation's held-out clip % divided by the "
             "calibration clip target of that gain's objective (calibrated rows only); "
             f"EXCEEDS marks ratios > {MATERIAL_EXCESS:g}x (a p100 gain has a zero target, so any "
             "held-out clipping exceeds it). Per-operation ratios are in gain_robustness.csv.\n")
    hdr = ("| tested | W/A | ADC | gain set | G qkv/qk/av | clip % qkv | clip % qk | clip % av | "
           "held-out/target | util p99.9 % qkv/qk/av | rel RMS % vs own FP32 | agree % vs own FP32 | "
           "top-1 % (1000) | top-1 % (10) | CE |")
    L += [hdr, "|" + "---|" * 15]
    for r in sorted(rows, key=lambda r: (r["tested_model"], r["weight_bits"], r["act_bits"],
                                         r["adc_bits"], r["gain_kind"], r["gain_set"])):
        rat = [r.get(f"heldout_to_target_ratio_{o}") for o in OPS]
        rat = [x for x in rat if x not in ("", None)]
        if not rat:
            exc = "n/a"
        else:
            worst = max(rat)
            flag = [o for o in OPS if r.get(f"heldout_exceeds_target_{o}") is True]
            exc = (f"{worst:.2f}x" if worst != float("inf") else "inf") + (
                f" EXCEEDS ({','.join(flag)})" if flag else "")
        L.append(f"| {r['tested_model']} | {r['precision']} | {r['adc_bits']} | {r['gain_set']} | "
                 f"{r['gain_qkv']:.2f}/{r['gain_qk']:.2f}/{r['gain_av']:.2f} | "
                 f"{r['heldout_clip_pct_qkv']:.3g} | {r['heldout_clip_pct_qk']:.3g} | "
                 f"{r['heldout_clip_pct_av']:.3g} | {exc} | "
                 + "/".join(f"{r[f'adc_util_p999_pct_{o}']:.0f}" for o in OPS) +
                 f" | {r['rel_logit_rms_err_pct_vs_own_fp32']:.2f} | "
                 f"{r['agreement_vs_own_fp32_pct']:.1f} | {r['top1_1000way_pct']:.2f} | "
                 f"{r['top1_10way_pct']:.2f} | {r['ce_1000way_nats']:.4f} |")
    L.append("\nInterpretation guide: compare rows of the same tested checkpoint, precision "
             "and ADC. A calibrated gain that lowers logit error without held-out clipping "
             "above its target generalised; one whose held-out clipping exceeds the target "
             "did not. `_xfer` rows test precision transfer (W8A8-calibrated gains used at "
             "another precision); `calibrated_cross_checkpoint` rows (with --cross-apply) "
             "test checkpoint transfer. Same training-data-overlap and provisional-DAC "
             "caveats as the primary REPORT.md.\n")
    return "\n".join(L)
