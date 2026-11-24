"""
reporting.py — builds every result table from cached raw outputs. Nothing in
any table is typed by hand: all numbers come from the cached logits / analog
statistics through cim_exp.metrics.

THE FIVE COMPARISONS (names are used verbatim in CSVs and REPORT.md)
  CASE1  pretrained CIM  vs  QAT CIM            (same hardware config; pairwise)
  CASE2  QAT FP32 software vs pretrained FP32 software (hardware-independent)
  CASE3  QAT FP32 software vs the SAME QAT checkpoint on CIM
  CASE4  pretrained FP32 software vs pretrained CIM
  CASE5  QAT CIM vs pretrained FP32 software    (combined model + hardware)

Agreement / error columns always name their reference; there is no
reference-free "agreement" column anywhere.
"""

from __future__ import annotations

import csv
import io
import json
import time
from pathlib import Path

import numpy as np

from . import metrics as M
from .hardware import (GAIN_SET_NOTES, GAIN_SETS, OPS, PRECISIONS, ADC_BITS,
                       HardwareConfig, planned_cim_jobs, primary_matrix)
from .imagenette import IMAGENET_INDICES, IMAGENETTE_CLASSES

CASES = {
    "CASE1": "pretrained CIM vs QAT CIM (same hardware config); model difference "
             "under identical simulated hardware - not a pure QAT-error measurement",
    "CASE2": "QAT FP32 software vs pretrained FP32 software; observed software-model "
             "difference associated with the QAT checkpoints (independent of ADC/gain)",
    "CASE3": "QAT FP32 software vs the same QAT checkpoint on CIM; effect of CIM "
             "execution (digital operand quantization + modelled analog chain)",
    "CASE4": "pretrained FP32 software vs pretrained CIM; effect of CIM execution "
             "on the original model",
    "CASE5": "QAT CIM vs pretrained FP32 software; COMBINED difference (checkpoint "
             "+ operand precision + modelled hardware) - not pure hardware error",
}

AGREE_COLS = ["agreement_pretrained_cim_vs_qat_cim_pct",
              "agreement_vs_pretrained_fp32_pct",
              "agreement_qat_int4_fp32_vs_qat_int8_fp32_pct",
              "agreement_vs_own_qat_fp32_pct"]
ERR_COLS = ["rel_logit_rms_diff_pct_qat_cim_vs_pretrained_cim",
            "rel_logit_rms_diff_pct_vs_pretrained_fp32",
            "rel_logit_rms_diff_pct_qat_int4_fp32_vs_qat_int8_fp32",
            "rel_logit_rms_err_pct_vs_own_fp32",
            "combined_rel_logit_rms_diff_pct_vs_pretrained_fp32"]
CASE_COLUMNS = {
    "CASE1": ("agreement_pretrained_cim_vs_qat_cim_pct",
              "rel_logit_rms_diff_pct_qat_cim_vs_pretrained_cim"),
    "CASE2": ("agreement_vs_pretrained_fp32_pct",
              "rel_logit_rms_diff_pct_vs_pretrained_fp32"),
    "CASE2b": ("agreement_qat_int4_fp32_vs_qat_int8_fp32_pct",
               "rel_logit_rms_diff_pct_qat_int4_fp32_vs_qat_int8_fp32"),
    "CASE3": ("agreement_vs_own_qat_fp32_pct", "rel_logit_rms_err_pct_vs_own_fp32"),
    "CASE4": ("agreement_vs_pretrained_fp32_pct", "rel_logit_rms_err_pct_vs_own_fp32"),
    "CASE5": ("agreement_vs_pretrained_fp32_pct",
              "combined_rel_logit_rms_diff_pct_vs_pretrained_fp32"),
}

HW_COLS = ["hw_label", "weight_bits", "act_bits", "precision", "mixed_precision_inference",
           "dac_type", "dac_bits", "dac_vref_v", "dac_alignment", "adc_bits",
           "adc_vref_v", "gain_set", "gain_qkv", "gain_qk", "gain_av", "gain_note",
           "amp_saturation_modeled", "amp_swing_limit_v", "cim_sites", "num_arrays",
           "rows", "cols"]
PER_OP_COLS = [f"{k}_{op}" for op in ("qkv", "qk", "av")
               for k in ("adc_clip_pct", "adc_clip_hi_pct", "adc_clip_lo_pct",
                         "adc_util_p999_pct", "adc_util_max_pct", "vadc_absp999_mv",
                         "voa_absp999_mv", "amp_saturation_pct")]


def per_op_columns(summ: dict, amp_modeled: bool, prefix: str = "") -> dict:
    """Per-operation clipping / ADC-utilisation columns for one CIM run.
    Percentages use that operation's own conversion count as denominator."""
    d = {}
    for op in ("qkv", "qk", "av"):
        v = summ.get(op, {})
        n = v.get("n", 0) or 0
        pct = (lambda x: 100.0 * x / n) if n else (lambda x: float("nan"))
        d[f"{prefix}adc_clip_pct_{op}"] = pct(v.get("clip_hi", 0) + v.get("clip_lo", 0))
        d[f"{prefix}adc_clip_hi_pct_{op}"] = pct(v.get("clip_hi", 0))
        d[f"{prefix}adc_clip_lo_pct_{op}"] = pct(v.get("clip_lo", 0))
        d[f"{prefix}adc_util_p999_pct_{op}"] = v.get("util_p999_pct", float("nan"))
        d[f"{prefix}adc_util_max_pct_{op}"] = v.get("util_max_pct", float("nan"))
        d[f"{prefix}vadc_absp999_mv_{op}"] = 1e3 * v.get("vadc_absp999", float("nan"))
        d[f"{prefix}voa_absp999_mv_{op}"] = 1e3 * v.get("voa_absp999", float("nan"))
        d[f"{prefix}amp_saturation_pct_{op}"] = (pct(v.get("amp_sat", 0)) if amp_modeled
                                                 else "not_modeled")
    return d


# --------------------------------------------------------------------------- #
# small CSV / markdown helpers
# --------------------------------------------------------------------------- #


def _fmt(v):
    if isinstance(v, float):
        if np.isnan(v):
            return "nan"
        return repr(v)
    return v


def write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    if columns is None:
        columns = []
        for r in rows:
            for k in r:
                if k not in columns:
                    columns.append(k)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore",
                       lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: _fmt(r.get(k, "")) for k in columns})
    path.write_text(buf.getvalue(), encoding="utf-8")


def md_table(rows: list[dict], cols: list[tuple[str, str, str]]) -> str:
    """cols: (key, header, format) ; format like '.2f' or '' """
    out = ["| " + " | ".join(h for _, h, _ in cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        cells = []
        for k, _, f in cols:
            v = r.get(k, "")
            if isinstance(v, (float, np.floating)) and f:
                v = "nan" if np.isnan(v) else format(v, f)
            cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def hw_columns(hw: HardwareConfig | None) -> dict:
    if hw is None:
        return {k: "" for k in HW_COLS} | {"hw_label": "none (FP32 software)"}
    g = hw.gains
    return dict(hw_label=hw.label, weight_bits=hw.weight_bits, act_bits=hw.act_bits,
                precision=hw.precision_label, mixed_precision_inference=hw.mixed_precision,
                dac_type=hw.dac_type, dac_bits=hw.dac_bits, dac_vref_v=hw.dac_vref,
                dac_alignment=hw.dac_alignment, adc_bits=hw.adc_bits,
                adc_vref_v=hw.adc_vref, gain_set=hw.gain_set, gain_qkv=g["qkv"],
                gain_qk=g["qk"], gain_av=g["av"], gain_note=hw.gain_description,
                amp_saturation_modeled=hw.amp_swing_limit_v is not None,
                amp_swing_limit_v=("none (ideal linear amplifier)"
                                   if hw.amp_swing_limit_v is None else hw.amp_swing_limit_v),
                cim_sites="+".join(hw.cim_ops) + ("+proj" if hw.cim_out_proj else ""),
                num_arrays=hw.num_arrays, rows=hw.rows, cols=hw.cols)


# --------------------------------------------------------------------------- #
# assemble
# --------------------------------------------------------------------------- #


class Results:
    """Loads the manifest labels, references and finalized CIM jobs."""

    def __init__(self, run_dir: Path, run_cfg: dict, registry_hashes: dict,
                 trained_bits: dict, ref_keys: dict, cim_keys: dict):
        from .imagenette import read_manifest
        self.run_dir = run_dir
        self.run_cfg = run_cfg
        rows, msha = read_manifest(run_dir / "input_manifest.csv")
        self.manifest_rows = rows
        self.manifest_sha256 = msha
        self.y1000 = np.array([r["label_1000way"] for r in rows])
        self.y10 = np.array([r["label_10way"] for r in rows])
        self.hashes = registry_hashes
        self.trained_bits = trained_bits
        self.refs, self.cims, self.missing = {}, {}, []
        for mid, key in ref_keys.items():
            p = run_dir / "cache" / "reference" / f"{key}.npz"
            if p.exists():
                meta = json.loads(p.with_suffix(".json").read_text())
                self.refs[mid] = dict(key=key, logits=np.load(p)["logits"], meta=meta)
            else:
                self.missing.append(("reference", mid, key))
        for (mid, label), (key, hw) in cim_keys.items():
            p = run_dir / "cache" / "cim" / key / "final.npz"
            if p.exists():
                meta = json.loads(p.with_suffix(".json").read_text())
                self.cims[(mid, label)] = dict(key=key, hw=hw,
                                               logits=np.load(p)["logits"], meta=meta)
            else:
                self.missing.append(("cim", mid, key))

    # -- common columns ---------------------------------------------------- #

    def ctx_cols(self) -> dict:
        return dict(input_manifest=str(self.run_dir / "input_manifest.csv"),
                    manifest_sha256=self.manifest_sha256,
                    dataset=self.run_cfg["variant"], dataset_split="val",
                    stage=self.run_cfg["stage"], n_images=len(self.manifest_rows),
                    preprocessing=self.run_cfg["preprocess_id"])

    def model_metrics(self, logits) -> dict:
        return M.model_metrics(logits, self.y1000, self.y10, IMAGENET_INDICES)


def _side(prefix: str, model_id: str, path: str, run_id: str, sha: str,
          mm: dict) -> dict:
    d = {f"{prefix}_model": model_id, f"{prefix}_execution_path": path,
         f"{prefix}_run_id": run_id, f"{prefix}_checkpoint_sha256": sha}
    for k, v in mm.items():
        if k != "n_images":
            d[f"{prefix}_{k}"] = v
    return d


def build_tables(R: Results, qat_selection: str, amp=None) -> dict:
    amp_limit = amp            # amplifier swing limit of the planned jobs (None = ideal)
    """-> dict of table name -> list of row dicts."""
    ctx = R.ctx_cols()
    raw, pair, clip = [], [], []
    mm_cache = {}

    def mm_ref(mid):
        if ("ref", mid) not in mm_cache:
            mm_cache[("ref", mid)] = R.model_metrics(R.refs[mid]["logits"])
        return mm_cache[("ref", mid)]

    def mm_cim(mid, label):
        if ("cim", mid, label) not in mm_cache:
            mm_cache[("cim", mid, label)] = R.model_metrics(R.cims[(mid, label)]["logits"])
        return mm_cache[("cim", mid, label)]

    # ---------------- raw metrics ---------------- #
    for mid, ref in R.refs.items():
        raw.append(dict(run_id=ref["key"], model=mid, checkpoint_sha256=R.hashes[mid],
                        qat_trained_weight_bits=R.trained_bits.get(mid) or "",
                        execution_path="fp32_software", **hw_columns(None),
                        **{**ctx, **mm_ref(mid)},
                        status="OK" if mm_ref(mid)["n_nonfinite_images"] == 0
                        else "INVALID_NONFINITE"))
    for (mid, label), c in sorted(R.cims.items()):
        hw, meta = c["hw"], c["meta"]
        tb = R.trained_bits.get(mid)
        tot = meta["cim_stats"]
        conv = tot["adc_conversions"]
        summ = meta["analog_summary"]
        ch = sum(v["clip_hi"] for v in summ.values())
        cl = sum(v["clip_lo"] for v in summ.values())
        amp_mod = meta["chain"]["amp_saturation_modeled"]
        amp_cnt = sum(v["amp_sat"] for v in summ.values())
        raw.append(dict(
            run_id=c["key"], model=mid, checkpoint_sha256=R.hashes[mid],
            qat_trained_weight_bits=tb or "",
            precision_matches_training=("" if tb is None else tb == hw.weight_bits),
            execution_path="cim_analog", **hw_columns(hw), **{**ctx, **mm_cim(mid, label)},
            adc_conversions=conv, adc_clip_hi=ch, adc_clip_lo=cl, adc_clip_total=ch + cl,
            adc_clip_hi_pct=100.0 * ch / conv, adc_clip_lo_pct=100.0 * cl / conv,
            adc_clip_total_pct=100.0 * (ch + cl) / conv,
            images_with_any_clipping=meta["images_with_any_clipping"],
            images_with_any_clipping_pct=100.0 * meta["images_with_any_clipping"] / meta["n_images"],
            amp_saturation=(amp_cnt if amp_mod else "not_modeled"),
            **per_op_columns(summ, amp_mod),
            conversions_per_image=meta["conversions_per_image"],
            consistency_ok=meta["consistency_ok"], wall_s=meta["wall_s_total"],
            status=meta["status"]))
        # ---------------- clipping / analog diagnostics ---------------- #
        for op in list(OPS) + ["ALL"]:
            if op == "ALL":
                v = dict(n=conv, clip_hi=ch, clip_lo=cl,
                         rail_hi=sum(x["rail_hi"] for x in summ.values()),
                         rail_lo=sum(x["rail_lo"] for x in summ.values()),
                         amp_sat=amp_cnt, images=meta["images_with_any_clipping"])
                g = ""
            else:
                v = dict(summ[op])
                v["images"] = meta["images_with_clipping_per_op"][op]
                g = hw.gains[op]
            row = dict(run_id=c["key"], model=mid, **hw_columns(hw), operation=op,
                       applied_gain=g, adc_conversions=v["n"],
                       clip_hi=v["clip_hi"], clip_hi_pct=100.0 * v["clip_hi"] / v["n"],
                       clip_lo=v["clip_lo"], clip_lo_pct=100.0 * v["clip_lo"] / v["n"],
                       clip_total=v["clip_hi"] + v["clip_lo"],
                       clip_total_pct=100.0 * (v["clip_hi"] + v["clip_lo"]) / v["n"],
                       images_with_clipping=v["images"],
                       images_with_clipping_pct=100.0 * v["images"] / meta["n_images"],
                       rail_code_hi=v["rail_hi"], rail_code_lo=v["rail_lo"],
                       amp_saturation=(v["amp_sat"] if amp_mod else "not_modeled"),
                       amp_saturation_pct=(100.0 * v["amp_sat"] / v["n"] if amp_mod
                                           else "not_modeled"),
                       n_images=meta["n_images"])
            if op != "ALL":
                for k in ("voa_min", "voa_max", "voa_mean", "voa_std", "voa_p01",
                          "voa_median", "voa_p99", "voa_absp50", "voa_absp99",
                          "voa_absp999", "vadc_min", "vadc_max", "vadc_mean",
                          "vadc_std", "vadc_p01", "vadc_median", "vadc_p99",
                          "vadc_absp50", "vadc_absp99", "vadc_absp999",
                          "util_median_pct", "util_p999_pct", "util_max_pct",
                          "cm_residue_absmax_v", "cm_residue_after_gain_absmax_v",
                          "cm_counterfactual_clip_pct", "n_samples"):
                    row[k] = v.get(k, "")
            row["adc_vref_v"] = hw.adc_vref
            row["dac_vref_v"] = hw.dac_vref
            clip.append(row)

    # ---------------- pairwise comparisons ---------------- #
    def add_pair(case, ref_side, test_side, hw, agree_col, err_col, a, e,
                 extra=None):
        row = dict(case=case, case_description=CASES[case.rstrip("b")],
                   **ref_side, **test_side, **hw_columns(hw), **ctx)
        for k in AGREE_COLS:
            row[k] = ""
        for k in ERR_COLS:
            row[k] = ""
        row[agree_col] = a
        row[err_col] = e
        row.update(extra or {})
        st = []
        for side in ("reference", "tested"):
            if row.get(f"{side}_n_nonfinite_images", 0):
                st.append(f"{side}_nonfinite")
        row["status"] = "OK" if not st else "INVALID:" + ",".join(st)
        pair.append(row)

    def ref_side(prefix, mid):
        return _side(prefix, mid, "fp32_software", R.refs[mid]["key"], R.hashes[mid],
                     mm_ref(mid))

    def cim_side(prefix, mid, label):
        c = R.cims[(mid, label)]
        d = _side(prefix, mid, "cim_analog", c["key"], R.hashes[mid], mm_cim(mid, label))
        d[f"{prefix}_adc_clip_total_pct"] = (
            100.0 * c["meta"]["cim_stats"]["adc_saturations"]
            / c["meta"]["cim_stats"]["adc_conversions"])
        d[f"{prefix}_images_with_any_clipping"] = c["meta"]["images_with_any_clipping"]
        d.update(per_op_columns(c["meta"]["analog_summary"],
                                c["meta"]["chain"]["amp_saturation_modeled"], prefix + "_"))
        tb = R.trained_bits.get(mid)
        d[f"{prefix}_precision_matches_training"] = "" if tb is None else tb == c["hw"].weight_bits
        return d

    # CASE2 (once, hardware independent)
    if "pretrained" in R.refs:
        for q in ("qat_int4", "qat_int8"):
            if q in R.refs:
                a = M.agreement_pct(R.refs[q]["logits"], R.refs["pretrained"]["logits"])
                e = M.rel_rms_pct(R.refs[q]["logits"], R.refs["pretrained"]["logits"])
                add_pair("CASE2", ref_side("reference", "pretrained"),
                         ref_side("tested", q), None, *CASE_COLUMNS["CASE2"], a, e)
    if "qat_int4" in R.refs and "qat_int8" in R.refs:
        a = M.agreement_pct(R.refs["qat_int4"]["logits"], R.refs["qat_int8"]["logits"])
        e = M.rel_rms_pct(R.refs["qat_int4"]["logits"], R.refs["qat_int8"]["logits"])
        add_pair("CASE2b", ref_side("reference", "qat_int8"), ref_side("tested", "qat_int4"),
                 None, *CASE_COLUMNS["CASE2b"], a, e)

    for mid_t, hw in planned_cim_jobs(qat_selection, amp_limit):
        label = hw.label
        if (mid_t, label) not in R.cims:
            continue
        cl = R.cims[(mid_t, label)]["logits"]
        if mid_t == "pretrained":
            # CASE4
            if "pretrained" in R.refs:
                pl = R.refs["pretrained"]["logits"]
                add_pair("CASE4", ref_side("reference", "pretrained"),
                         cim_side("tested", "pretrained", label), hw,
                         *CASE_COLUMNS["CASE4"], M.agreement_pct(cl, pl), M.rel_rms_pct(cl, pl))
            continue
        # QAT CIM job
        if mid_t in R.refs:                                      # CASE3
            ql = R.refs[mid_t]["logits"]
            add_pair("CASE3", ref_side("reference", mid_t), cim_side("tested", mid_t, label),
                     hw, *CASE_COLUMNS["CASE3"], M.agreement_pct(cl, ql), M.rel_rms_pct(cl, ql))
        if "pretrained" in R.refs:                               # CASE5
            pl = R.refs["pretrained"]["logits"]
            add_pair("CASE5", ref_side("reference", "pretrained"),
                     cim_side("tested", mid_t, label), hw, *CASE_COLUMNS["CASE5"],
                     M.agreement_pct(cl, pl), M.rel_rms_pct(cl, pl))
        if ("pretrained", label) in R.cims:                      # CASE1
            pcl = R.cims[("pretrained", label)]["logits"]
            add_pair("CASE1", cim_side("reference", "pretrained", label),
                     cim_side("tested", mid_t, label), hw, *CASE_COLUMNS["CASE1"],
                     M.agreement_pct(cl, pcl), M.rel_rms_pct(cl, pcl))

    return dict(raw=raw, pair=pair, clip=clip)


# --------------------------------------------------------------------------- #
# coverage
# --------------------------------------------------------------------------- #


def coverage(R: Results, tables: dict, qat_selection: str, amp=None) -> dict:
    exp_jobs = planned_cim_jobs(qat_selection, amp)
    got = {(r["model"], r["hw_label"]) for r in tables["raw"]
           if r["execution_path"] == "cim_analog"}
    missing_jobs = [f"{m}:{hw.label}" for m, hw in exp_jobs if (m, hw.label) not in got]
    n_pre = sum(1 for m, _ in exp_jobs if m == "pretrained")
    n_qat = len(exp_jobs) - n_pre
    expected = {"CASE1": n_qat, "CASE2": 2, "CASE2b": 1, "CASE3": n_qat,
                "CASE4": n_pre, "CASE5": n_qat}
    counts = {k: sum(1 for r in tables["pair"] if r["case"] == k) for k in expected}
    untraceable = [i for i, r in enumerate(tables["pair"])
                   if not (r.get("reference_run_id") and r.get("tested_run_id")
                           and r.get("reference_checkpoint_sha256")
                           and r.get("tested_checkpoint_sha256")
                           and r.get("manifest_sha256"))]
    invalid = [r["run_id"] for r in tables["raw"] if r["status"] != "OK"]
    ok = (not missing_jobs and counts == expected and not untraceable
          and not R.missing and not invalid)
    return dict(complete=ok, expected_counts=expected, counts=counts,
                missing_jobs=missing_jobs, missing_cache=R.missing,
                untraceable_rows=untraceable, invalid_runs=invalid,
                n_raw_rows=len(tables["raw"]), n_pair_rows=len(tables["pair"]))


# --------------------------------------------------------------------------- #
# class-mapping check (empirical)
# --------------------------------------------------------------------------- #


def class_mapping_check(R: Results) -> list[dict]:
    """Empirical mapping check on the pretrained FP32 reference.

    PASS criterion (per class): the 10-WAY modal prediction (argmax restricted to
    the 10 mapped ImageNet indices) is the class itself. This detects a wrong or
    permuted index (e.g. parachute mapped to 702) but is not fooled by legitimate
    ImageNet near-synonyms ("tape player" vs "cassette player"), which can win
    the 1000-way argmax on correctly mapped images. The 1000-way modal prediction
    is reported for information only.
    """
    if "pretrained" not in R.refs:
        return []
    lg = R.refs["pretrained"]["logits"]
    pred = lg.argmax(1)
    pred10 = lg[:, IMAGENET_INDICES].argmax(1)
    rows = []
    for i, c in enumerate(IMAGENETTE_CLASSES):
        m = R.y10 == i
        if not m.any():
            continue
        vals, cnt = np.unique(pred[m], return_counts=True)
        modal = int(vals[np.argmax(cnt)])
        v10, c10 = np.unique(pred10[m], return_counts=True)
        modal10 = int(v10[np.argmax(c10)])
        rows.append(dict(label_10way=i, class_name=c.name, folder_wnid=c.folder_wnid,
                         mapped_imagenet_index=c.imagenet_index,
                         mapped_imagenet_wnid=c.imagenet_wnid,
                         n_images=int(m.sum()),
                         modal_prediction_10way=modal10,
                         modal_count_10way=int(c10.max()),
                         pretrained_fp32_top1_10way_pct=100.0 * float((pred10[m] == i).mean()),
                         modal_prediction_1000way_info=modal,
                         modal_count_1000way_info=int(cnt.max()),
                         pretrained_fp32_top1_1000way_pct=100.0 * float((pred[m] == c.imagenet_index).mean()),
                         mapping_confirmed=modal10 == i))
    return rows


# --------------------------------------------------------------------------- #
# write everything
# --------------------------------------------------------------------------- #


def write_all(R: Results, out_dir: Path, qat_selection: str, env: dict,
              ckpt_reports: dict, run_manifest_rows: list[dict], amp=None,
              extra_context: dict | None = None) -> dict:
    t = build_tables(R, qat_selection, amp)
    cov = coverage(R, t, qat_selection, amp)
    cmap = class_mapping_check(R)
    cov["class_mapping_confirmed"] = bool(cmap) and all(r["mapping_confirmed"] for r in cmap)
    cov["complete"] = cov["complete"] and cov["class_mapping_confirmed"]
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(out_dir / "run_manifest.csv", run_manifest_rows)
    write_csv(out_dir / "raw_metrics.csv", t["raw"])
    write_csv(out_dir / "pairwise_comparisons.csv", t["pair"])
    base = ["run_id", "model", "execution_path", "checkpoint_sha256",
            "qat_trained_weight_bits", "precision_matches_training"] + HW_COLS
    write_csv(out_dir / "accuracy_summary.csv", t["raw"],
              base + ["n_images", "top1_1000way_pct", "top1_10way_pct",
                      "n_nonfinite_images", "status", "manifest_sha256"])
    write_csv(out_dir / "cross_entropy_summary.csv", t["raw"],
              base + ["n_images", "ce_1000way_nats", "ce_10way_nats",
                      "n_nonfinite_images", "status", "manifest_sha256"])
    pid = ["case", "reference_model", "reference_execution_path", "reference_run_id",
           "tested_model", "tested_execution_path", "tested_run_id",
           "tested_precision_matches_training"] + HW_COLS
    write_csv(out_dir / "agreement_summary.csv", t["pair"],
              pid + AGREE_COLS + ["n_images", "status", "manifest_sha256"])
    write_csv(out_dir / "logit_error_summary.csv", t["pair"],
              pid + ERR_COLS + ["n_images", "status", "manifest_sha256"])
    write_csv(out_dir / "clipping_summary.csv", t["clip"])
    for case in ("CASE1", "CASE2", "CASE3", "CASE4", "CASE5"):
        rows = [r for r in t["pair"] if r["case"] in (case, case + "b")]
        write_csv(out_dir / f"{case.lower()}_table.csv", rows)
    write_csv(out_dir / "class_mapping_check.csv", cmap)
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2))
    (out_dir / "coverage.json").write_text(json.dumps(cov, indent=2, default=str))
    (out_dir / "checkpoints.json").write_text(json.dumps(ckpt_reports, indent=2, default=str))
    report = render_report(R, t, cov, cmap, env, ckpt_reports, qat_selection,
                           amp=amp, extra=extra_context or {})
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")
    return cov


# --------------------------------------------------------------------------- #
# REPORT.md
# --------------------------------------------------------------------------- #

CAVEATS = """\
- **Real images, 10 classes.** Inputs are the official fast.ai Imagenette
  validation split (`{variant}`), {n} images. Imagenette covers 10 ImageNet
  classes; it is not the 1000-class ImageNet validation set. Top-1 is reported
  1000-way (primary: argmax over all 1000 TinyViT outputs) and 10-way (argmax
  restricted to the 10 Imagenette classes).
- **Training-data overlap (important).** Imagenette2 re-splits roughly all ImageNet-1k
  train+val images of its 10 classes 70/30, so most of its *val* images are ImageNet-1k
  *training* images. File-name provenance of this manifest: {prov}. The pretrained model
  was trained on ImageNet (22k->1k), and per the supplied QAT context the QAT checkpoints
  were fine-tuned on ImageNet-1k train and model-selected (early stopping) on ImageNet-1k
  val. Absolute accuracies are therefore optimistic, and QAT-vs-pretrained differences
  (CASE1/2/5) are confounded by this exposure; CIM-vs-own-FP32 comparisons (CASE3/4) on
  identical images are much less affected.
- **Image resolution.** imagenette2-160 images have a 160 px short side and are
  upsampled to 256 before the 224 centre crop, so absolute accuracies are lower
  than full-resolution ImageNet numbers. All models and paths see identical inputs.
- **Agreement is not accuracy.** Every agreement column names its reference.
  Accuracy and cross-entropy are always against the true labels.
- **Different questions.** FP32-reference comparisons (CASE3/4) measure what CIM
  execution does to one checkpoint; pairwise CIM comparisons (CASE1) measure how
  two checkpoints differ under identical simulated hardware; CASE5 mixes
  checkpoint, precision and hardware effects and is not pure hardware error.
- **Fixed gains (Experiment A).** QKV 23.77 / QK 8.41 / AV 2.92 were derived for W8/A8
  with a P99.99 full-scale criterion on *synthetic* calibration inputs (CLAUDE.md §6).
  They are held fixed across all precisions and checkpoints, alongside G = 1, and are
  not claimed optimal for W4/A4, W4/A8 or W8/A4. Real-image calibration and its
  robustness are Experiment B (separate outputs, never mixed into these tables).
- **Mixed precision is inference-only.** W4/A8 and W8/A4 use the QAT checkpoint
  selected by weight precision (W4 -> QAT-INT4, W8 -> QAT-INT8); neither was trained
  at that mixed precision.
- **How the QAT checkpoints were trained (from the supplied QAT context; the training
  code itself was not supplied).** Fine-tuned from this pretrained TinyViT-5M through a
  differentiable surrogate of the simulator's quantizer (same per-token / per-column
  symmetric scheme as the CIM path here) **and of the LEGACY functional ADC**
  (8-bit, worst-case full scale, per depth block) at W4/A4 (INT4) and W8/A8 (INT8). They
  were **not** trained against the analog chain (CDAC -> C2C -> gain -> SAR) used in this
  matrix, so QAT results here measure transfer to a different ADC model. The optional
  `trainpoint` command evaluates each checkpoint at its own training operating point.
  The files store FP32 latent weights and no quantizer state.
- **What "CIM" includes.** The CIM path = digital operand quantization of the qkv,
  QK^T and AV matmuls + the modelled analog chain (unsigned 8-bit CDAC @ 0.6 V ->
  C2C macro -> per-op gain -> SAR ADC @ 0.6 V -> digital accumulation). LayerNorm,
  softmax, attention biases, output projection, MLP, convolutions and the head run
  in FP32. Error attribution between DAC, macro, amplifier and ADC requires
  dedicated controlled comparisons, which this matrix does not contain.
- **Provisional assumptions (not hardware-confirmed).** Confirmed: the unsigned 8-bit
  CDAC, V_REF 0.6 V, V_out = V_REF*code/256. Provisional: signed activations are
  offset-binary coded onto it (code = x + 128) with the V_REF/2 common mode removed
  ideally; INT4 activation codes are MSB-aligned on the 8-bit DAC (code x 16); signed
  weights use a behavioural sign-magnitude abstraction; all nonidealities are off.
- **Amplifier.** {amp}. No amplifier swing/saturation specification exists, so the
  primary experiment uses an ideal linear amplifier and reports amplifier saturation as
  `not_modeled` (never as zero); a swing limit can be configured for diagnostics. ADC
  clipping is always measured separately.
- **No ADC-disabled runs** are in the matrix: disabling the ADC is not equivalent to
  FP32 execution or to the absence of all hardware effects.
"""


def _pivot(rows, row_key, col_key, val_key, row_order, col_order, fmt=".2f"):
    idx = {(r[row_key], r[col_key]): r.get(val_key, "") for r in rows}
    out = ["| | " + " | ".join(str(c) for c in col_order) + " |",
           "|---|" + "|".join("---" for _ in col_order) + "|"]
    for rk in row_order:
        cells = []
        for ck in col_order:
            v = idx.get((rk, ck), "")
            cells.append(format(v, fmt) if isinstance(v, float) else str(v or "-"))
        out.append(f"| {rk} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def render_report(R, t, cov, cmap, env, ckpts, qat_selection, amp=None,
                  extra=None) -> str:
    extra = extra or {}
    ctx = R.ctx_cols()
    L = []
    A = L.append
    A(f"# CIM TinyViT-5M experiment report — stage `{ctx['stage']}`\n")
    A(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} from cached raw outputs by "
      f"`run_experiment.py report`. Every number below is computed from "
      f"`raw_metrics.csv` / `pairwise_comparisons.csv` inputs; nothing is typed by hand.\n")
    A(f"- Dataset: `{ctx['dataset']}` split `val`, **{ctx['n_images']} real images** "
      f"(manifest `{ctx['input_manifest']}`, SHA-256 `{ctx['manifest_sha256'][:16]}…`)")
    A(f"- Preprocessing: `{ctx['preprocessing']}`")
    A(f"- QAT checkpoint selection: `{qat_selection}`")
    A(f"- Code fingerprint: `{env.get('code_sha256', '')[:16]}…`, torch "
      f"{env.get('torch')}, numpy {env.get('numpy')}, threads/forward = 1, batch = 1")
    A("")
    A("## Coverage and validity\n")
    if not cov.get("class_mapping_confirmed", False):
        A("> **WARNING — class mapping NOT confirmed on these images** (see the "
          "class-mapping section). Do not interpret any accuracy, cross-entropy or "
          "class-dependent number in this report until this is resolved.\n")
    A(f"- Complete (all jobs, all rows traceable, no invalid runs, mapping confirmed): "
      f"**{cov['complete']}**")
    A(f"- Comparison rows per case (got / expected): " + ", ".join(
        f"{k} {cov['counts'][k]}/{v}" for k, v in cov["expected_counts"].items()))
    if cov["missing_jobs"]:
        A(f"- Missing CIM jobs ({len(cov['missing_jobs'])}): " +
          ", ".join(cov["missing_jobs"][:12]) + (" …" if len(cov["missing_jobs"]) > 12 else ""))
    if cov["invalid_runs"]:
        A(f"- **Invalid runs (excluded from interpretation):** {cov['invalid_runs']}")
    if cov["untraceable_rows"]:
        A(f"- **Untraceable comparison rows:** {cov['untraceable_rows']}")
    A("")
    A("## Caveats and assumptions\n")
    prov = extra.get("provenance") or {}
    prov_s = (", ".join(f"{k}: {v}" for k, v in prov.items()) if prov
              else "not recorded (re-run `prepare`)")
    amp_s = ("Ideal linear amplifier, no swing limit (saturation not modelled)" if amp is None
             else f"DIAGNOSTIC: linear amplifier with output swing limit {amp:g} V "
                  f"(saturation counted per operation, separately from ADC clipping)")
    A(CAVEATS.format(variant=ctx["dataset"], n=ctx["n_images"], prov=prov_s, amp=amp_s))
    A("## Checkpoints\n")
    rows = []
    for mid, rep in ckpts.items():
        rows.append(dict(model=mid, sha=rep.get("sha256", "")[:16] + "…",
                         n=rep.get("n_tensors"), head=str(rep.get("head_shape")),
                         q=rep.get("n_quantizer_keys"),
                         g4=rep.get("qkv_grid_test", {}).get("int4_mean_offgrid"),
                         d=rep.get("median_rel_param_diff_vs_pretrained", float("nan"))))
    A(md_table(rows, [("model", "model", ""), ("sha", "SHA-256", ""),
                      ("n", "tensors", ""), ("head", "head", ""),
                      ("q", "quantizer keys", ""), ("g4", "qkv int4 off-grid", ".3f"),
                      ("d", "median rel. param diff vs pretrained", ".4f")]))
    A("\n(off-grid ≈ 0.25 means the stored weights are not on an int4 grid: latent FP32 QAT weights.)\n")

    A("## Class-mapping verification (pretrained FP32, empirical)\n")
    if cmap:
        A(md_table(cmap, [("class_name", "class", ""), ("folder_wnid", "folder", ""),
                          ("mapped_imagenet_index", "mapped idx", ""), ("n_images", "n", ""),
                          ("modal_prediction_10way", "10-way modal (label)", ""),
                          ("modal_count_10way", "count", ""),
                          ("pretrained_fp32_top1_10way_pct", "top-1 % 10-way", ".1f"),
                          ("modal_prediction_1000way_info", "1000-way modal (info)", ""),
                          ("pretrained_fp32_top1_1000way_pct", "top-1 % 1000-way", ".1f"),
                          ("mapping_confirmed", "confirmed", "")]))
        A("\nConfirmed = the 10-way modal prediction (argmax over the 10 mapped indices) "
          "is the class itself; this detects wrong/permuted indices without being fooled "
          "by ImageNet near-synonyms (e.g. 'tape player' 848 for cassette players), which "
          "can win the 1000-way argmax. The parachute folder `n03888605` is ImageNet "
          "'parallel bars' (702); it is mapped to parachute (701) by class.\n")
    else:
        A("_pretrained FP32 reference not available yet._\n")

    pair = [r for r in t["pair"] if r["status"].startswith("OK")]

    A("## CASE 2 — QAT FP32 software vs pretrained FP32 software (hardware-independent)\n")
    A(CASES["CASE2"] + "\n")
    c2 = [r for r in pair if r["case"] in ("CASE2", "CASE2b")]
    for r in c2:
        r["_agree"] = r.get("agreement_vs_pretrained_fp32_pct") or r.get(
            "agreement_qat_int4_fp32_vs_qat_int8_fp32_pct")
        r["_err"] = r.get("rel_logit_rms_diff_pct_vs_pretrained_fp32") or r.get(
            "rel_logit_rms_diff_pct_qat_int4_fp32_vs_qat_int8_fp32")
    A(md_table(c2, [("reference_model", "reference (FP32)", ""),
                    ("tested_model", "tested (FP32)", ""),
                    ("_agree", "top-1 agreement %", ".2f"),
                    ("_err", "rel. logit RMS diff %", ".2f"),
                    ("reference_top1_1000way_pct", "ref top-1 % (1000-way)", ".2f"),
                    ("tested_top1_1000way_pct", "tested top-1 % (1000-way)", ".2f"),
                    ("tested_top1_10way_pct", "tested top-1 % (10-way)", ".2f"),
                    ("reference_ce_1000way_nats", "ref CE", ".4f"),
                    ("tested_ce_1000way_nats", "tested CE", ".4f")]))
    A("")

    hw_cols = [("precision", "W/A", ""), ("adc_bits", "ADC", ""), ("gain_set", "gains", "")]

    def case_section(case, title, agree_hdr, err_hdr, extra_cols):
        rows = [r for r in pair if r["case"] == case]
        ac, ec = CASE_COLUMNS[case]
        A(f"## {case} — {title}\n")
        A(CASES[case] + "\n")
        if not rows:
            A("_no completed rows yet._\n")
            return
        A(md_table(rows, [("tested_model", "tested", "")] + hw_cols +
                   [(ac, agree_hdr, ".2f"), (ec, err_hdr, ".2f")] + extra_cols))
        A("")

    tcols = [("tested_top1_1000way_pct", "CIM top-1 % (1000)", ".2f"),
             ("tested_top1_10way_pct", "CIM top-1 % (10)", ".2f"),
             ("tested_ce_1000way_nats", "CIM CE", ".4f"),
             ("tested_adc_clip_total_pct", "ADC clip % all", ".4f"),
             ("tested_adc_clip_pct_qkv", "clip % QKV", ".4f"),
             ("tested_adc_clip_pct_qk", "clip % QK", ".4f"),
             ("tested_adc_clip_pct_av", "clip % AV", ".4f"),
             ("tested_images_with_any_clipping", "imgs clipped", "")]
    case_section("CASE4", "pretrained FP32 software vs pretrained CIM",
                 "agreement vs pretrained FP32 %", "rel. logit RMS err vs own FP32 %",
                 [("reference_top1_1000way_pct", "FP32 top-1 %", ".2f")] + tcols)
    case_section("CASE3", "QAT FP32 software vs same QAT checkpoint on CIM",
                 "agreement vs own QAT FP32 %", "rel. logit RMS err vs own FP32 %",
                 [("reference_top1_1000way_pct", "QAT FP32 top-1 %", ".2f"),
                  ("tested_precision_matches_training", "W bits = trained", "")] + tcols)
    case_section("CASE5", "QAT CIM vs pretrained FP32 software (combined)",
                 "agreement vs pretrained FP32 %",
                 "combined rel. logit RMS diff vs pretrained FP32 %", tcols)
    case_section("CASE1", "pretrained CIM vs QAT CIM (same hardware)",
                 "agreement pretrained-CIM vs QAT-CIM %",
                 "rel. logit RMS diff QAT-CIM vs pretrained-CIM %",
                 [("reference_top1_1000way_pct", "pretrained CIM top-1 %", ".2f"),
                  ("tested_top1_1000way_pct", "QAT CIM top-1 %", ".2f"),
                  ("reference_ce_1000way_nats", "pretrained CIM CE", ".4f"),
                  ("tested_ce_1000way_nats", "QAT CIM CE", ".4f")])

    # precision comparison + ADC/gain comparison
    precs = [f"W{w}/A{a}" for w, a in PRECISIONS]
    combos = [f"ADC{b}/{g}" for b in ADC_BITS for g in GAIN_SETS]
    raw_cim = [dict(r, combo=f"ADC{r['adc_bits']}/{r['gain_set']}",
                    model_class=("pretrained" if r["model"] == "pretrained" else "QAT"))
               for r in t["raw"] if r["execution_path"] == "cim_analog" and r["status"] == "OK"]
    A("## Precision comparison (CIM top-1 %, 1000-way)\n")
    A("Rows: weight/activation precision; columns: ADC bits / gain set. QAT rows use "
      "the checkpoint matched to the weight precision.\n")
    for mc in ("pretrained", "QAT"):
        sub = [r for r in raw_cim if r["model_class"] == mc and
               (mc == "pretrained" or r.get("precision_matches_training") in (True, "True"))]
        A(f"**{mc}**\n")
        A(_pivot(sub, "precision", "combo", "top1_1000way_pct", precs, combos))
        A("")
    A("## ADC / gain comparison (relative logit RMS error vs own FP32, %)\n")
    for case, mc in (("CASE4", "pretrained"), ("CASE3", "QAT (matched)")):
        sub = [dict(r, combo=f"ADC{r['adc_bits']}/{r['gain_set']}") for r in pair
               if r["case"] == case and (case == "CASE4" or
                                         r.get("tested_precision_matches_training") in (True, "True"))]
        A(f"**{mc}** ({case})\n")
        A(_pivot(sub, "precision", "combo", CASE_COLUMNS[case][1], precs, combos))
        A("")
    A("## ADC clipping (% of conversions, all operations)\n")
    for mc in ("pretrained", "QAT"):
        sub = [r for r in raw_cim if r["model_class"] == mc and
               (mc == "pretrained" or r.get("precision_matches_training") in (True, "True"))]
        A(f"**{mc}**\n")
        A(_pivot(sub, "precision", "combo", "adc_clip_total_pct", precs, combos, ".4f"))
        A("")
    A("## Per-operation analog diagnostics\n")
    A("From `clipping_summary.csv` (operation rows). V in mV. `swing p99.9` = "
      "|V_ADC_IN| P99.9 / (V_REF_ADC/2). Amplifier saturation is `not_modeled` "
      "(no swing limit configured). `CM cf. clip %` = share of conversions that WOULD "
      "clip if the CDAC's V_REF/2 common-mode term were not cancelled before the "
      "amplifier (counterfactual diagnostic; never applied).\n")
    crow = []
    for r in t["clip"]:
        if r["operation"] == "ALL":
            continue
        crow.append(dict(r, voa999=(r.get("voa_absp999") or 0) * 1e3,
                         vadc999=(r.get("vadc_absp999") or 0) * 1e3))
    crow.sort(key=lambda r: (r["model"], r["weight_bits"], r["act_bits"], r["adc_bits"],
                             r["gain_set"], OPS.index(r["operation"])))
    A(md_table(crow, [("model", "model", ""), ("precision", "W/A", ""),
                      ("adc_bits", "ADC", ""), ("gain_set", "gains", ""),
                      ("operation", "op", ""), ("applied_gain", "G", ""),
                      ("clip_hi_pct", "clip hi %", ".4f"), ("clip_lo_pct", "clip lo %", ".4f"),
                      ("images_with_clipping", "imgs", ""),
                      ("voa999", "abs V_OA p99.9", ".3f"), ("vadc999", "abs V_ADC p99.9", ".2f"),
                      ("util_p999_pct", "swing p99.9 %", ".1f"),
                      ("cm_counterfactual_clip_pct", "CM cf. clip %", ".2f"),
                      ("amp_saturation", "amp sat", "")]))
    A("")
    return "\n".join(L)
