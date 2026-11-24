"""
trainpoint.py — EXPERIMENT T (optional): evaluate every checkpoint at the QAT
TRAINING operating point.

Why: the supplied QAT context states that both QAT checkpoints were fine-tuned
through a surrogate of the simulator's LEGACY functional ADC
(`CIMConfig(adc_enabled=True, adc_bits=8)`, worst-case full scale
rows*2^(w-1)*2^(a-1), per depth block, 32x32, sites qkv/qk/av) at
W4/A4 (INT4) and W8/A8 (INT8) — NOT through the analog chain of the primary
matrix. This experiment runs exactly that operating point through the
reference NumPy simulator (the same evaluation the QAT context calls
"row 3 vs row 4": pretrained vs QAT at the same ADC setting), so one can check
that each checkpoint behaves as trained before interpreting its analog-chain
results. Cross cells (INT4 checkpoint at W8/A8 and vice versa) help detect a
mislabelled checkpoint (the QAT context records one earlier "INT8" run that
silently trained at 4/4).

Separate outputs: results/<stage>/expT/. Not part of the five-case tables.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

POINTS = ((4, 4), (8, 8))
MODELS = ("pretrained", "qat_int4", "qat_int8")
LEGACY_ADC_BITS = 8


def job_key(base: dict, model: str, sha: str, w: int, a: int) -> str:
    from .hardware import short_hash
    return f"tp_{model}_W{w}A{a}_" + short_hash(dict(kind="trainpoint_legacy_adc", model=model,
                                                     ckpt=sha, w=w, a=a,
                                                     adc_bits=LEGACY_ADC_BITS, **base))


def run_job(unit: dict) -> dict:
    """One (checkpoint, precision) at the legacy functional ADC; per-image counters."""
    repo = Path(unit["repo_root"])
    from . import engine as EN
    EN._setup_paths(repo)
    EN.configure_threads(1)
    import torch
    import run_tinyvit
    from cim_sim import CIMArray, CIMConfig
    from cim_sim.attention import convert_attention
    from .imagenette import to_model_input

    out = Path(unit["out"])
    if out.exists():
        return dict(key=unit["key"], skipped=True)
    cfg = CIMConfig(rows=32, cols=32, weight_bits=unit["w"], act_bits=unit["a"],
                    adc_enabled=True, adc_bits=LEGACY_ADC_BITS, adc_vref=0.6)
    model = run_tinyvit.build_model(Path(unit["ckpt"]))
    arr = CIMArray(cfg)
    convert_attention(model, arr)
    if arr.chain is not None or not cfg.adc_enabled:
        raise RuntimeError("training-point check must use the LEGACY functional ADC")
    u8 = np.load(unit["inputs"], mmap_mode="r")
    n = u8.shape[0]
    logits = np.empty((n, 1000), np.float32)
    conv = np.zeros(n, np.int64)
    sat = np.zeros(n, np.int64)
    t0 = time.time()
    with torch.no_grad():
        for i in range(n):
            c0, s0 = arr.stats.adc_conversions, arr.stats.adc_saturations
            logits[i] = model(to_model_input(u8[i:i + 1])).numpy()[0]
            conv[i] = arr.stats.adc_conversions - c0
            sat[i] = arr.stats.adc_saturations - s0
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, logits=logits, conversions=conv, saturations=sat)
    tmp.replace(out)
    out.with_suffix(".json").write_text(json.dumps(dict(
        key=unit["key"], model=unit["model"], w=unit["w"], a=unit["a"],
        path="legacy_functional_adc", adc_bits=LEGACY_ADC_BITS,
        full_scale=cfg.rows * 2 ** (unit["w"] - 1) * 2 ** (unit["a"] - 1),
        wall_s=time.time() - t0), indent=1))
    return dict(key=unit["key"], skipped=False)


def report(rows_in: list[dict], refs: dict, y1000, y10, idx10) -> tuple[list[dict], str]:
    from . import metrics as M
    rows = []
    for r in rows_in:
        lg = r["logits"]
        ref = refs[r["model"]]
        mm = M.model_metrics(lg, y1000, y10, idx10)
        rows.append(dict(model=r["model"], checkpoint_sha256=r["sha"],
                         precision=f"W{r['w']}/A{r['a']}", weight_bits=r["w"], act_bits=r["a"],
                         execution_path="cim_legacy_functional_adc",
                         adc_bits=LEGACY_ADC_BITS, legacy_full_scale=r["fs"],
                         is_qat_training_point=(
                             (r["model"] == "qat_int4" and r["w"] == 4)
                             or (r["model"] == "qat_int8" and r["w"] == 8)),
                         agreement_vs_own_fp32_pct=M.agreement_pct(lg, ref),
                         rel_logit_rms_err_pct_vs_own_fp32=M.rel_rms_pct(lg, ref), **mm,
                         legacy_adc_saturation_pct=100.0 * r["sat"].sum() / r["conv"].sum(),
                         images_with_saturation=int((r["sat"] > 0).sum()),
                         run_id=r["key"]))
    base = {(x["model"], x["weight_bits"]): x for x in rows}
    L = ["# Experiment T — checkpoints at the QAT training operating point\n",
         "Operating point (from the supplied QAT context): 32x32, sites qkv/qk/av, LEGACY "
         "functional ADC 8-bit (worst-case full scale, per depth block), W4/A4 for QAT-INT4 "
         "and W8/A8 for QAT-INT8, reference NumPy simulator. Not the analog chain.\n",
         "| model | W/A | training point | top-1 % (1000) | top-1 % (10) | CE | Δtop-1 vs pretrained "
         "(same point) | ΔCE vs pretrained | agree % vs own FP32 | rel RMS % vs own FP32 | "
         "legacy ADC sat. % |",
         "|" + "---|" * 11]
    for x in sorted(rows, key=lambda x: (x["weight_bits"], x["model"])):
        p = base.get(("pretrained", x["weight_bits"]))
        dt = x["top1_1000way_pct"] - p["top1_1000way_pct"] if p else float("nan")
        dc = x["ce_1000way_nats"] - p["ce_1000way_nats"] if p else float("nan")
        L.append(f"| {x['model']} | {x['precision']} | {x['is_qat_training_point']} | "
                 f"{x['top1_1000way_pct']:.2f} | {x['top1_10way_pct']:.2f} | "
                 f"{x['ce_1000way_nats']:.4f} | {dt:+.2f} | {dc:+.4f} | "
                 f"{x['agreement_vs_own_fp32_pct']:.1f} | "
                 f"{x['rel_logit_rms_err_pct_vs_own_fp32']:.2f} | "
                 f"{x['legacy_adc_saturation_pct']:.4f} |")
    L.append("\nExpected if the checkpoints behave as trained: at W4/A4 QAT-INT4 beats the "
             "pretrained model (top-1 up, CE down), and at W8/A8 QAT-INT8 does. A QAT "
             "checkpoint that does not improve on the pretrained model at its OWN training "
             "point, or that does better at the other precision, should be investigated "
             "(e.g. mislabelled run) before its analog-chain results are interpreted. "
             "Training-data-overlap caveat as in REPORT.md.\n")
    return rows, "\n".join(L)
