#!/usr/bin/env python3
"""
qat_evaluate.py — evaluation. Every hardware-aware number comes from the ACTUAL
simulator (cim_sim) through cim_qat.sim_eval (batch 1, CPU processes); the
training surrogate is never used here.

  # 1) FP32 preprocessing sanity check on VAL (GPU, no CIM; never TEST)
  python qat_evaluate.py fp32-check --paths configs/paths.json --device cuda

  # 2) ImageNetV2 label-integrity check (pretrained FP32; prints mapping checks only)
  python qat_evaluate.py v2-integrity --paths configs/paths.json --device cuda

  # 3) one simulator evaluation (resumable). TEST needs --final.
  python qat_evaluate.py sim --paths configs/paths.json --split val --model pretrained \
         --kind analog --precision W8A8 --gain-set cal --workers 24
  python qat_evaluate.py sim --paths configs/paths.json --split test --final \
         --model run:w8a8_kd_pilot5ep:best --kind analog --precision W8A8 --gain-set cal --workers 24

  # (W4A4 only, before ANY TEST read) decision D2: which W4A4 run is reported
  python qat_evaluate.py select-d2 --paths configs/paths.json

  # 4) report for one split and precision (identical-input assertions + paired stats)
  python qat_evaluate.py report --paths configs/paths.json --split test --final --precision W8A8 \
         --new w8a8_kd_pilot5ep

Models: registry names (pretrained, old_int8, old_int4; SHA-256-checked, never
modified) or run:<run_name>:best|final (a FINISHED training run; best = the
VAL-selected primary checkpoint, final = the pre-declared secondary).
Kinds: fp32 | adc_off (operand quantization only) | analog (capacitive DAC ->
C2C -> per-op gain -> SAR ADC 8 b) | legacy_adc (old QAT training point).
Gain sets (analog): cal (frozen calibrated gains of that precision) | unity
(G = 1) | hist (W8A8 only: 23.77 / 8.41 / 2.92, derived on synthetic inputs).
Outputs: <work_root>/eval/<split>/<tag>/ (tag = <model>__<kind>[_<P>][__<gainset>]).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cim_qat.common import single_thread_env_before_numpy, tag_job  # noqa: E402
single_thread_env_before_numpy()     # BLAS sizes its thread pool when numpy loads: set 1 thread first
tag_job("eval")                      # qat_stop.py eval stops this process and all its workers
from cim_qat.common import (now_iso, read_json, select_device, sha256_file,  # noqa: E402
                            write_json, write_provenance)

import numpy as np  # noqa: E402

OPS = ("qkv", "qk", "av")
HIST_GAINS_W8A8 = (23.77, 8.41, 2.92)
PURPOSE = dict(val="eval_val", calcheck="eval_calcheck", test="final_eval",
               test_th07="final_eval", test_top="final_eval")
TEST_SPLITS = ("test", "test_th07", "test_top")         # matched-frequency + the v2.8 extension variants
POOL_ORDER_R = TEST_SPLITS
SPLIT_CHOICES = ["val", "calcheck", *TEST_SPLITS]


def parse_prec(p):
    p = p.upper()
    if p not in ("W8A8", "W4A4"):
        sys.exit("--precision must be W8A8 or W4A4 in this phase")
    return (8, 8) if p == "W8A8" else (4, 4)


def open_splits(dp, split, final):
    from cim_qat.data import Splits
    if split in TEST_SPLITS and not final:
        sys.exit("--split test reads ImageNetV2, the final TEST set; pass --final to confirm this is the "
                 "final evaluation (QAT_DESIGN.md §8). Nothing was read.")
    return Splits(dp.splits_dir, purpose=PURPOSE.get(split, "eval"), allow_test=(split in TEST_SPLITS))


def resolve_eval_model(spec: str, dp, need_finished: bool) -> dict:
    """-> dict(label, path, sha256, source, run=...)"""
    from cim_qat.model import REGISTRY, resolve_model
    if spec.startswith("run:"):
        _, name, which = spec.split(":")
        if which not in ("best", "final"):
            sys.exit("run models are run:<run_name>:best or run:<run_name>:final")
        rd = Path(dp.work_root) / "runs" / name
        summ = rd / "summary.json"
        if need_finished and not summ.exists():
            sys.exit(f"{rd}: no summary.json; the run is not finished. A checkpoint is evaluated on "
                     "TEST only after its run has finished and been selected on VAL.")
        fname = dict(best="best_model.pth", final="final_model.pth")[which]
        p = rd / "checkpoints" / fname
        reg = read_json(rd / "checkpoints" / "checkpoints.json")
        h = sha256_file(p)
        if reg.get(fname, {}).get("sha256") != h:
            sys.exit(f"{p}: sha256 {h[:16]} does not match checkpoints.json; refusing")
        cfg = read_json(rd / "config_resolved.json")
        return dict(label=f"{name}_{which}", path=p, sha256=h, source="new_qat_run", run=name,
                    which=which, run_gains_sha256=sha256_file(cfg["config"]["frozen_gains"]),
                    run_precision=cfg["config"]["precision"])
    label, p, h = resolve_model(spec)
    return dict(label=label, path=p, sha256=h,
                source="registry" if spec in REGISTRY else "file")


def gains_for(a, dp, wb, ab):
    from cim_qat.hw_config import HWConfig, load_frozen
    if a.gain_set == "cal":
        f = Path(dp.work_root) / "calibration" / f"W{wb}A{ab}" / "frozen_gains.json"
        if not f.exists():
            sys.exit(f"{f} not found: calibrate and freeze first (runbook stage 8)")
        hw, rec = load_frozen(f, wb, ab)
        return hw, dict(gain_set="cal", file=str(f), sha256=sha256_file(f), level=rec["level"],
                        cal_manifest_sha256=rec["cal_manifest_sha256"])
    if a.gain_set == "unity":
        return HWConfig(wb, ab, (1.0, 1.0, 1.0)), dict(gain_set="unity")
    if (wb, ab) != (8, 8):
        sys.exit("--gain-set hist exists only for W8A8 (the historical gains are W8A8 gains)")
    return HWConfig(wb, ab, HIST_GAINS_W8A8), dict(gain_set="hist", source="CLAUDE.md §6 (synthetic-input P99.99)")


def eval_dir(dp, split, limit):
    return Path(dp.work_root) / "eval" / (split + (f"_limit{limit}" if limit else ""))


# --------------------------------------------------------------------------- #


def batched_logits(model, ds, dev, num_workers):
    import torch
    dl = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=num_workers)
    lg, lab = [], []
    with torch.no_grad():
        for x, y, _ in dl:
            lg.append(model(x.to(dev)).float().cpu().numpy())
            lab.append(y.numpy())
    return np.concatenate(lg), np.concatenate(lab)


def cmd_fp32_check(a, dp):
    """VAL only. FP32 accuracy under bicubic (the protocol) and bilinear (old QAT's filter)."""
    import torch

    from cim_qat import metrics as M
    from cim_qat.data import EvalImages, Splits
    from cim_qat.model import build_fp32, resolve_model
    sp = Splits(dp.splits_dir, purpose="eval_val")
    dev = select_device(a.device)
    out = {}
    for interp in a.interp:
        for name in a.models:
            label, path, sha = resolve_model(name)
            m = build_fp32(path).to(dev).eval()
            ds = EvalImages(sp, "val", interp)
            if a.limit:
                ds = torch.utils.data.Subset(ds, list(np.linspace(0, len(ds) - 1, a.limit).astype(int)))
            lg, lab = batched_logits(m, ds, dev, a.num_workers)
            s = M.summarize(M.per_image(lg, lab))
            out[f"{label}|{interp}"] = dict(sha256=sha, **s)
            print(f"  {label:12s} {interp:8s} n={s['n']:6d} top1 {s['top1_pct']:.2f}% (+-{s['top1_se_pct']:.2f}) "
                  f"top5 {s['top5_pct']:.2f}% NLL {s['nll']:.4f} ECE {s['ece_pct']:.2f}%")
    rep = dict(time_utc=now_iso(), split="val", n_val=len(sp.rows("val")), limit=a.limit, results=out,
               note=("FP32 only (no CIM), batched on the GPU. The TinyViT authors report 80.7% top-1 for this "
                     "checkpoint on ImageNet-1k val; their exact eval protocol was not verified here."))
    p = Path(dp.work_root) / "eval" / f"fp32_check_val{f'_limit{a.limit}' if a.limit else ''}.json"
    write_json(p, rep)
    print(f"-> {p}")


def cmd_v2_integrity(a, dp):
    """Label/class-mapping integrity of ImageNetV2 with the pretrained FP32 model.

    Prints ONLY mapping statistics: for how many of the 1000 classes the modal
    pretrained prediction over that class's images equals the folder label,
    under (a) label = int(folder) [used] and (b) torchvision ImageFolder's
    lexicographic folder order [wrong, for contrast]. Accuracy is not printed;
    the pretrained model is not trained, so this cannot steer any design choice.
    """
    from cim_qat.data import EvalImages, Splits, imagefolder_lexicographic_labels
    from cim_qat.model import build_fp32
    from cim_qat.data import assert_not_frozen
    split = a.split
    p = Path(dp.work_root) / "eval" / ("v2_integrity.json" if split == "test" else f"v2_integrity_{split}.json")
    assert_not_frozen(dp.work_root, p)
    sp = Splits(dp.splits_dir, purpose="v2_integrity", allow_test=True)
    dev = select_device(a.device)
    m = build_fp32().to(dev).eval()
    ds = EvalImages(sp, split)
    lg, lab = batched_logits(m, ds, dev, a.num_workers)
    pred = lg.argmax(1)
    folder = np.array([int(Path(r["rel_path"]).parts[0]) for r in ds.rows])
    assert np.array_equal(folder, lab), "manifest label != int(folder)"
    lex = imagefolder_lexicographic_labels()
    lab_lex = np.array([lex[f] for f in folder])

    def modal_match(labels):
        n = 0
        for c in range(1000):
            sel = labels == c
            if sel.any():
                vals, cnt = np.unique(pred[sel], return_counts=True)
                n += int(vals[cnt.argmax()] == c)
        return n
    ma, mb = modal_match(lab), modal_match(lab_lex)
    ok = ma >= a.min_classes and mb <= 50
    rep = dict(time_utc=now_iso(), split=split, n_images=int(len(lab)), test_manifest_sha256=sp.sha(split),
               classes_modal_match_int_folder=ma, classes_modal_match_imagefolder_lexicographic=mb,
               pass_rule=f"int-folder >= {a.min_classes} of 1000 and lexicographic <= 50", ok=bool(ok),
               note="data-integrity check only; accuracy deliberately not reported here")
    write_json(p, rep)
    print(f"[{'PASS' if ok else 'FAIL'}] ImageNetV2 ({split}) label mapping: modal pretrained prediction == label for "
          f"{ma}/1000 classes with label=int(folder); {mb}/1000 with the ImageFolder lexicographic order "
          f"(-> {p})")
    sys.exit(0 if ok else 1)


def cmd_sim(a, dp):
    from cim_qat import sim_eval
    from cim_qat.data import assert_not_frozen, frozen_files
    if a.split in TEST_SPLITS and a.limit:
        sys.exit("--limit is not allowed on TEST (no partial looks at the final set)")
    sp = open_splits(dp, a.split, a.final)
    mod = resolve_eval_model(a.model, dp, need_finished=(a.split in TEST_SPLITS))
    job = dict(kind=a.kind, ckpt=str(mod["path"]), with_own_fp32=a.kind != "fp32", interpolation=a.interp)
    tag = f"{mod['label']}__{a.kind.replace('_', '')}"
    rec = dict(model=mod, kind=a.kind, split=a.split, final=a.final, limit=a.limit,
               split_manifest_sha256=sp.sha(a.split), purpose=PURPOSE.get(a.split),
               time_utc=now_iso())
    if a.kind != "fp32":
        wb, ab = parse_prec(a.precision)
        if mod.get("run_precision") and mod["run_precision"].upper() != a.precision.upper():
            print(f"NOTE: {mod['label']} was trained at {mod['run_precision']}, evaluated at {a.precision}")
        if a.kind == "analog":
            hw, grec = gains_for(a, dp, wb, ab)
            if grec["gain_set"] == "cal" and mod.get("run_gains_sha256") and \
                    mod["run_gains_sha256"] != grec["sha256"] and mod["run_precision"].upper() == a.precision.upper():
                sys.exit("the frozen gains file differs from the one this run was trained with; refusing")
            rec["gains"] = dict(grec, values=dict(zip(OPS, hw.gains)))
            tag += f"_W{wb}A{ab}__{grec['gain_set']}"
        else:
            from cim_qat.hw_config import HWConfig
            hw = HWConfig(wb, ab, (1.0, 1.0, 1.0))          # gains unused by adc_off / legacy_adc
            tag += f"_W{wb}A{ab}"
        job["hw"] = hw.job_dict()
    if a.interp != "bicubic":
        tag += f"_{a.interp}"
    out = eval_dir(dp, a.split, a.limit) / tag
    # v2.8: a finished row is left untouched (re-running it used to rewrite its records and
    # log a new read). Checked BEFORE the manifest is read, so a skip logs no access.
    if (out / "summary.json").exists() and (out / "result.npz").exists():
        jold = read_json(out / "job.json")["job"]
        drop = ("preprocess_id", "ckpt")        # the checkpoint path depends on the code folder: compare its hash
        same = ({k: v for k, v in jold.items() if k not in drop}
                == {k: v for k, v in json.loads(json.dumps(job)).items() if k not in drop}
                and read_json(out / "eval_record.json")["model"]["sha256"] == mod["sha256"])
        if not same:
            sys.exit(f"{out} holds a different finished job; refusing (use a new folder)")
        print(f"already finished: {tag} (left untouched) -> {out}")
        return
    fz = frozen_files(dp.work_root)
    if any(os.path.relpath(f, dp.work_root) in fz for f in out.glob("*")):
        sys.exit(f"{out} contains frozen files but is not finished; refusing to write into it")
    rows = sp.rows(a.split)
    if a.limit:
        rows = [rows[i] for i in np.linspace(0, len(rows) - 1, a.limit).astype(int)]
    items = [(r["idx"], str(sp.path(r))) for r in rows]
    rec["n"] = len(rows)
    out.mkdir(parents=True, exist_ok=True)
    assert_not_frozen(dp.work_root, out / "provenance.json", out / "eval_record.json", out / "labels.npy")
    write_provenance(out, dict(stage="evaluation", tag=tag))       # details in eval_record.json
    write_json(out / "eval_record.json", rec)
    np.save(out / "labels.npy", np.array([r["label"] for r in rows]))
    res = sim_eval.run(job, items, out, workers=a.workers, chunk=a.chunk)
    s = res["summary"]
    print(f"done: {tag} n={s['n']} cpu {s['cpu_seconds_per_image']:.2f} s/img -> {out}")


# --------------------------------------------------------------------------- #
# decision D2: which W4A4 run is reported (declared 27 Oct 2026)
# --------------------------------------------------------------------------- #

D2_RULE = ("report the W4A4 run with the higher best-VAL hardware top-1 (surrogate); ties -> lower "
           "VAL NLL; still tied -> the pilot. Declared 27 Oct 2026, before the pilot's epoch-5 result.")


def cmd_select_d2(a, dp):
    """Apply D2 to the FINISHED pilot (and its fork, if one was made); record the
    choice in eval/D2_selection.json. Refused once any TEST read has happened."""
    from cim_qat.data import is_test_split, log_access, read_access_logs
    W = Path(dp.work_root)
    out = W / "eval" / "D2_selection.json"
    if out.exists():
        sys.exit(f"{out} exists: the D2 selection is made once")
    if any(e.get("purpose") == "final_eval" and is_test_split(str(e.get("split"))) for e in read_access_logs(W)):
        sys.exit("TEST has already been read: D2 must be applied BEFORE any TEST evaluation; report this")
    cands = []
    for name in (a.pilot, a.fork):
        rd = W / "runs" / name
        if not rd.exists():
            if name == a.pilot:
                sys.exit(f"{rd} not found")
            continue                                     # no fork made (the pilot did not early-stop)
        if not (rd / "summary.json").exists():
            sys.exit(f"{rd} is not finished (no summary.json); D2 is applied to finished runs only")
        s = read_json(rd / "summary.json")
        b = s["best_eval"]
        cands.append(dict(run=name, best_tag=b["tag"], best_epoch=b["epoch"], top1_pct=b["hw_val"]["top1_pct"],
                          nll=b["hw_val"]["nll"], best_model_sha256=s["selection"]["primary"].get("sha256"),
                          stopped_reason=s["stopped_reason"], forked_from=(s.get("forked_from") or {}).get("source_run")))
    order = sorted(cands, key=lambda c: (-c["top1_pct"], c["nll"], c["run"] != a.pilot))
    sel = order[0]
    rec = dict(decision="D2", rule=D2_RULE, time_utc=now_iso(), candidates=cands, selected=sel["run"],
               selected_best_model_sha256=sel["best_model_sha256"],
               same_checkpoint_as_pilot=(len(cands) == 2 and cands[0]["best_model_sha256"] == cands[1]["best_model_sha256"]))
    write_json(out, rec)
    log_access(W, "none", "d2_selection", None, dict(selected=sel["run"]))
    for c in cands:
        print(f"  {c['run']:28s} best {c['best_tag']} (epoch {c['best_epoch']:.0f}) VAL top-1 {c['top1_pct']:.3f}% "
              f"NLL {c['nll']:.4f}  [{c['stopped_reason']}]")
    print(f"D2 -> report {sel['run']}" + (" (its best checkpoint is the pilot's: identical file)"
                                          if rec["same_checkpoint_as_pilot"] else "") + f"  -> {out}")


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def comparisons(P, old, new, new_ce):
    """Pre-declared paired comparisons (QAT_DESIGN.md §9): (id, what it isolates, a, b) = b vs a."""
    pt = "pretrained"
    C = [("C1", "operand quantization (exact integer MAC, no ADC)", f"{pt}__fp32", f"{pt}__adcoff_{P}"),
         ("C2", "ADC + analog chain at unity gain", f"{pt}__adcoff_{P}", f"{pt}__analog_{P}__unity"),
         ("C3", "ADC + analog chain at calibrated gains", f"{pt}__adcoff_{P}", f"{pt}__analog_{P}__cal"),
         ("C4", "gain calibration (unity -> calibrated)", f"{pt}__analog_{P}__unity", f"{pt}__analog_{P}__cal")]
    if P == "W8A8":
        C.append(("C4b", "calibrated vs historical synthetic-input gains", f"{pt}__analog_{P}__hist",
                  f"{pt}__analog_{P}__cal"))
    C.append(("C5", "old QAT vs PTQ (same hardware, same gains)", f"{pt}__analog_{P}__cal", f"{old}__analog_{P}__cal"))
    C.append(("C8", "old QAT checkpoint: its training point (legacy ADC) -> new hardware",
              f"{old}__legacyadc_{P}", f"{old}__analog_{P}__cal"))
    for n in new:
        C.append(("C6", "new QAT (VAL-selected) vs PTQ", f"{pt}__analog_{P}__cal", f"{n}_best__analog_{P}__cal"))
        C.append(("C7", "new QAT vs old QAT (same hardware, same gains)", f"{old}__analog_{P}__cal",
                  f"{n}_best__analog_{P}__cal"))
        C.append(("C7b", "new QAT: last vs VAL-selected checkpoint", f"{n}_best__analog_{P}__cal",
                  f"{n}_final__analog_{P}__cal"))
    for n in new_ce:
        for k in new:
            C.append(("C9", "CE pilot vs KD (same budget)", f"{k}_best__analog_{P}__cal", f"{n}_best__analog_{P}__cal"))
    return C


def load_run(d: Path):
    with np.load(d / "result.npz") as z:
        r = {k: z[k] for k in z.files}
    r["labels"] = np.load(d / "labels.npy")
    r["summary"] = read_json(d / "summary.json")
    r["record"] = read_json(d / "eval_record.json")
    return r


D2_PILOT, D2_FORK = "w4a4_kd_pilot10ep", "w4a4_kd_pilot10ep_full"
UNIFORM_FILE = "analog_summary_uniform.json"


def analog_summary_for(run_dir: Path, summary: dict):
    """-> (per-op analog summary, method): the re-merged one if available."""
    p = run_dir / UNIFORM_FILE
    if p.exists():
        return read_json(p)["analog_summary"], "uniform"
    if summary.get("analog_summary_method"):
        return summary.get("analog_summary"), "uniform"
    return summary.get("analog_summary"), ("legacy" if summary.get("analog_summary") else None)


def cmd_restat(a, dp):
    """Recompute the analog statistics summary of finished analog runs from their
    chunk files with the v2.7 merge (cim_qat.analog_merge). Reads only the
    simulator's own chunk statistics (no image, no label); writes
    <run>/analog_summary_uniform.json and leaves every result and summary unchanged."""
    from cim_qat.analog_merge import METHOD, summary_from_chunks
    from cim_qat.data import assert_not_frozen
    if a.split in TEST_SPLITS and not a.final:
        sys.exit("--split test needs --final")
    base = eval_dir(dp, a.split, a.limit)
    done = 0
    for d in sorted(base.iterdir()):
        sp = d / "summary.json"
        if not sp.exists():
            continue
        summ = read_json(sp)
        if summ["job"]["kind"] != "analog" or (a.only and a.only not in d.name):
            continue
        if (d / UNIFORM_FILE).exists() and not a.force:
            print(f"  {d.name}: already done")
            continue
        hw = summ["job"]["hw"]
        new = summary_from_chunks(d, adc_vref=hw.get("adc_vref", 0.6), adc_bits=hw.get("adc_bits", 8))
        old = summ.get("analog_summary") or {}
        bad = [op for op in OPS if old.get(op, {}).get("n") != new.get(op, {}).get("n")
               or old.get(op, {}).get("clip_hi") != new.get(op, {}).get("clip_hi")]
        if bad:
            sys.exit(f"{d.name}: exact counts differ between summary.json and the chunk files for {bad}; refusing")
        assert_not_frozen(dp.work_root, d / UNIFORM_FILE)
        write_json(d / UNIFORM_FILE, dict(time_utc=now_iso(), method=METHOD, analog_summary=new))
        done += 1
        print(f"  {d.name}: util median % " + "/".join(f"{old.get(o, {}).get('util_median_pct', float('nan')):.1f}->"
                                                       f"{new[o]['util_median_pct']:.1f}" for o in OPS)
              + "  p99.9 % " + "/".join(f"{old.get(o, {}).get('util_p999_pct', float('nan')):.0f}->"
                                       f"{new[o]['util_p999_pct']:.0f}" for o in OPS))
    print(f"restat: {done} run(s) written in {base}")


def cmd_report(a, dp):
    from cim_qat import metrics as M
    P = a.precision.upper()
    parse_prec(P)
    if a.split in TEST_SPLITS and not a.final:
        sys.exit("--split test needs --final")
    base = eval_dir(dp, a.split, a.limit)
    old = a.old or ("old_int8" if P == "W8A8" else "old_int4")
    tags = sorted(p.name for p in base.iterdir() if (p / "result.npz").exists()
                  and (p.name.endswith("__fp32") or f"_{P}" in p.name))
    if "pretrained__fp32" not in tags:
        sys.exit(f"{base}/pretrained__fp32 missing: run `sim --kind fp32 --model pretrained` first")
    runs = {t: load_run(base / t) for t in tags}
    # ---- identical images / preprocessing / labels (requirement E) ----
    ref = runs["pretrained__fp32"]
    problems = []
    for t, r in runs.items():
        for k in ("idx", "input_sha", "labels"):
            if not np.array_equal(r[k], ref[k]):
                problems.append(f"{t}: {k} differs from pretrained__fp32")
        if r["summary"]["job"]["preprocess_id"] != ref["summary"]["job"]["preprocess_id"]:
            problems.append(f"{t}: preprocessing differs")
        if r["record"]["split_manifest_sha256"] != ref["record"]["split_manifest_sha256"]:
            problems.append(f"{t}: split manifest differs")
    if problems:
        print("FAIL: runs are not on identical inputs:\n  " + "\n  ".join(problems))
        sys.exit(1)
    labels = ref["labels"]
    fp_logits = ref["logits"]
    pis, rows = {}, []
    for t, r in runs.items():
        pis[t] = M.per_image(r["logits"], labels, fp_logits)
        row = dict(run=t, kind=r["record"]["kind"], model_sha256=r["record"]["model"]["sha256"],
                   gains=r["record"].get("gains", {}).get("values"), **M.summarize(pis[t]))
        if "logits_own_fp32" in r:
            s = M.summarize(M.per_image(r["logits_own_fp32"], labels, fp_logits))
            row["own_fp32_top1_pct"] = s["top1_pct"]
            row["kl_own_fp32"] = M.summarize(M.per_image(r["logits"], labels, r["logits_own_fp32"]))["kl_ref"]
        an, row["util_stats_method"] = analog_summary_for(base / t, r["summary"])
        if an:
            for op in OPS:
                o = an.get(op, {})
                row[f"{op}"] = dict(conversions=o.get("n"), clip_pct=o.get("clip_pct"),
                                    clip_hi_pct=o.get("clip_hi_pct"), clip_lo_pct=o.get("clip_lo_pct"),
                                    rail_pct=(100.0 * (o.get("rail_hi", 0) + o.get("rail_lo", 0)) / o["n"]
                                              if o.get("n") else None),
                                    util_median_pct=o.get("util_median_pct"),
                                    util_p999_pct=o.get("util_p999_pct"), util_max_pct=o.get("util_max_pct"))
        if "adc_saturations" in r:
            row["images_with_any_saturation_pct"] = 100.0 * float((r["adc_saturations"] > 0).mean())
            row["saturation_pct_all_ops"] = (100.0 * r["adc_saturations"].sum() / max(1, r["adc_conversions"].sum()))
        rows.append(row)
    comps = []
    cl = comparisons(P, old, a.new or [], a.new_ce or [])
    pb, fb = f"{D2_PILOT}_best__analog_{P}__cal", f"{D2_FORK}_best__analog_{P}__cal"
    if pb in pis and fb in pis:
        cl.append(("C10", "decision D2: fork (10 epochs, no early stop) vs early-stopped pilot, both VAL-selected"
                   + (" (on VAL this favours the fork: VAL chose both checkpoints and the D2 winner)"
                      if a.split == "val" else ""), pb, fb))
    for cid, what, ta, tb in cl:
        if ta in pis and tb in pis:
            comps.append(dict(id=cid, isolates=what, a=ta, b=tb, **M.compare(pis[ta], pis[tb])))
        else:
            comps.append(dict(id=cid, isolates=what, a=ta, b=tb, missing=[t for t in (ta, tb) if t not in pis]))
    rep = dict(time_utc=now_iso(), split=a.split, precision=P, n=int(len(labels)),
               split_manifest_sha256=ref["record"]["split_manifest_sha256"],
               preprocess_id=ref["summary"]["job"]["preprocess_id"],
               input_sha_digest=ref["summary"]["input_sha_digest"], identical_inputs=True,
               rows=rows, comparisons=comps,
               caveat=CAVEAT.get(a.split, ""))
    out = base / f"REPORT_{a.split}_{P}"
    from cim_qat.data import assert_not_frozen
    assert_not_frozen(dp.work_root, out.with_suffix(".json"), out.with_suffix(".md"))
    write_json(out.with_suffix(".json"), rep)
    out.with_suffix(".md").write_text(render_report(rep))
    print(out.with_suffix(".md").read_text())
    print(f"\n-> {out}.md / .json")



def _fmt_cmp(c):
    t, n = c["top1_diff_pct"], c["nll_diff"]
    m = c["mcnemar"]
    return (f"{t['mean']:+.2f} ({t['ci95'][0]:+.2f}, {t['ci95'][1]:+.2f}) | {m['p']:.3g} ({m['a_only']}/{m['b_only']}) | "
            f"{n['mean']:+.4f} ({n['ci95'][0]:+.4f}, {n['ci95'][1]:+.4f})")


def cmd_report_pooled(a, dp):
    """v2.8 extension: one report over the three ImageNetV2 variants, combined ONLY here.
    Every distinct image (file content, SHA-256) is counted once: first occurrence in the
    order matched-frequency, threshold-0.7, top-images. Reads finished results only (no image,
    no manifest read; nothing re-run); writes eval/test_pooled/REPORT_test_pooled_<P>.{md,json}."""
    import hashlib
    from cim_qat import metrics as M
    from cim_qat.data import POOL_ORDER, assert_not_frozen
    P = a.precision.upper()
    parse_prec(P)
    if not a.final:
        sys.exit("the pooled report includes TEST: pass --final")
    old = a.old or ("old_int8" if P == "W8A8" else "old_int4")
    W, sd = Path(dp.work_root), dp.splits_dir
    out_dir = W / "eval" / "test_pooled"
    outp = out_dir / f"REPORT_test_pooled_{P}"
    assert_not_frozen(dp.work_root, outp.with_suffix(".json"), outp.with_suffix(".md"))
    tags = {}
    for s in POOL_ORDER:
        b = W / "eval" / s
        if not b.exists():
            sys.exit(f"{b} missing: evaluate every variant first")
        tags[s] = {d.name for d in b.iterdir() if (d / "result.npz").exists()
                   and (d.name.endswith("__fp32") or f"_{P}" in d.name)}
    common = sorted(set.intersection(*tags.values()))
    left_out = {s: sorted(tags[s] - set(common)) for s in POOL_ORDER}
    if "pretrained__fp32" not in common:
        sys.exit("pretrained__fp32 must be finished on every variant")
    # per split: reference images, labels, content hashes; identical inputs within the split
    ref = {}
    for s in POOL_ORDER:
        b = W / "eval" / s
        with np.load(b / "pretrained__fp32" / "result.npz") as z:
            idx, ish = z["idx"], z["input_sha"]
        lab = np.load(b / "pretrained__fp32" / "labels.npy")
        hrows = open(sd / f"hashes_{s}.csv").read().splitlines()[1:]
        h_of = {int(l.split(",", 1)[0]): l.split(",", 1)[1] for l in hrows}
        content = [h_of[int(i)] for i in idx]
        for t in common:
            with np.load(b / t / "result.npz") as z:
                if not (np.array_equal(z["idx"], idx) and np.array_equal(z["input_sha"], ish)
                        and np.array_equal(np.load(b / t / "labels.npy"), lab)):
                    sys.exit(f"{s}/{t}: not on the same images as {s}/pretrained__fp32; refusing")
        ref[s] = dict(idx=idx, lab=lab, content=content,
                      manifest=read_json(b / "pretrained__fp32" / "eval_record.json")["split_manifest_sha256"])
    # pooled selection
    first, sel, dups = {}, {s: [] for s in POOL_ORDER}, []
    for s in POOL_ORDER:
        for k, h in enumerate(ref[s]["content"]):
            if h in first:
                dups.append((s, k, *first[h]))
            else:
                first[h] = (s, k)
                sel[s].append(k)
    label_conflicts = sum(int(ref[s]["lab"][k] != ref[s0]["lab"][k0]) for s, k, s0, k0 in dups)
    labels = np.concatenate([ref[s]["lab"][sel[s]] for s in POOL_ORDER])
    n = int(len(labels))

    def pooled_logits(t):
        L = {}
        for s in POOL_ORDER:
            with np.load(W / "eval" / s / t / "result.npz") as z:
                L[s] = z["logits"]
        mism = sum(int(not np.array_equal(L[s][k], L[s0][k0])) for s, k, s0, k0 in dups)
        return np.concatenate([L[s][sel[s]] for s in POOL_ORDER]), mism, L

    fp, _, fpL = pooled_logits("pretrained__fp32")
    pis, rows, dup_mism, per_split = {}, [], {}, {s: {} for s in POOL_ORDER}
    for t in common:
        lg, mism, L = pooled_logits(t)
        dup_mism[t] = mism
        pis[t] = M.per_image(lg, labels, fp)
        rows.append(dict(run=t, **M.summarize(pis[t])))
        for s in POOL_ORDER:
            per_split[s][t] = M.per_image(L[s], ref[s]["lab"], fpL[s])
    cl = comparisons(P, old, a.new or [], [])
    pb, fb = f"{D2_PILOT}_best__analog_{P}__cal", f"{D2_FORK}_best__analog_{P}__cal"
    if pb in pis and fb in pis:
        cl.append(("C10", "decision D2: fork vs early-stopped pilot, both VAL-selected", pb, fb))
    comps = [dict(id=cid, isolates=what, a=ta, b=tb, **M.compare(pis[ta], pis[tb])) if ta in pis and tb in pis
             else dict(id=cid, isolates=what, a=ta, b=tb, missing=[x for x in (ta, tb) if x not in pis])
             for cid, what, ta, tb in cl]
    post = []
    for nm in a.new or []:
        tb = f"{nm}_best__analog_{P}__cal"
        if tb in pis:
            post.append(dict(id="P1", isolates="POST HOC: new QAT (VAL-selected) on the CIM simulator vs the FP32 "
                             "pretrained model", a="pretrained__fp32", b=tb,
                             pooled=M.compare(pis["pretrained__fp32"], pis[tb]),
                             per_variant={s: M.compare(per_split[s]["pretrained__fp32"], per_split[s][tb])
                                          for s in POOL_ORDER}))
    comp = [dict(split=s, n_images=int(len(ref[s]["idx"])), contributes=len(sel[s]),
                 duplicates_dropped=len(ref[s]["idx"]) - len(sel[s]), manifest_sha256=ref[s]["manifest"])
            for s in POOL_ORDER]
    rep = dict(time_utc=now_iso(), split="test_pooled", precision=P, n=n, composition=comp,
               duplicate_rule="each distinct file content (SHA-256) once; first occurrence in the order "
                              + ", ".join(POOL_ORDER),
               duplicates=len(dups), duplicate_label_conflicts=label_conflicts,
               duplicate_logit_mismatches=dup_mism, rows_left_out=left_out,
               pooled_content_digest=hashlib.sha256("".join(sorted(first)).encode()).hexdigest(),
               rows=rows, comparisons=comps, post_hoc=post, caveat=CAVEAT["test_pooled"],
               per_variant_top1={s: {t: 100 * float(per_split[s][t]["correct1"].mean()) for t in common}
                                 for s in POOL_ORDER})
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(outp.with_suffix(".json"), rep)
    outp.with_suffix(".md").write_text(render_pooled(rep))
    print(outp.with_suffix(".md").read_text())
    print(f"\n-> {outp}.md / .json")


def render_pooled(rep):
    V = {"test": "matched-frequency", "test_th07": "threshold-0.7", "test_top": "top-images"}
    L = [f"# {rep['precision']} — {SPLIT_TITLE['test_pooled']} (n = {rep['n']:,} distinct images)\n",
         f"Status: EXECUTED ({rep['time_utc']}). Combined at report time from finished per-variant results; "
         "nothing was re-run. KL and agreement are vs the pretrained FP32 model on the same images.\n",
         f"> {rep['caveat']}\n",
         "## Composition\n", "| variant | images evaluated | counted in the pool | duplicates of an earlier variant |",
         "|---|---|---|---|"]
    L += [f"| {V[c['split']]} | {c['n_images']:,} | {c['contributes']:,} | {c['duplicates_dropped']:,} |"
          for c in rep["composition"]]
    mm = max(rep["duplicate_logit_mismatches"].values()) if rep["duplicate_logit_mismatches"] else 0
    L.append(f"\nDuplicate rule: {rep['duplicate_rule']}. {rep['duplicates']:,} duplicate images; label conflicts "
             f"{rep['duplicate_label_conflicts']}; duplicates with different outputs (should be 0: same image, same "
             f"model): {mm}.")
    if any(rep["rows_left_out"].values()):
        L.append(f"Rows not finished on all three variants (left out): {rep['rows_left_out']}")
    L += ["\n## Results (pooled)\n",
          "| run | top-1 % (±SE) | top-5 % | NLL | KL(FP32‖run) | agree FP32 % | " +
          " | ".join(f"top-1 % {V[s]}" for s in POOL_ORDER_R) + " |",
          "|---|---|---|---|---|---|" + "---|" * len(POOL_ORDER_R)]
    for r in rep["rows"]:
        L.append(f"| {r['run']} | {r['top1_pct']:.2f} (±{r['top1_se_pct']:.2f}) | {r['top5_pct']:.2f} | {r['nll']:.4f} | "
                 f"{r['kl_ref']:.4f} | {r['agree_ref_pct']:.2f} | "
                 + " | ".join(f"{rep['per_variant_top1'][s][r['run']]:.2f}" for s in POOL_ORDER_R) + " |")
    c10 = any(c["id"] == "C10" for c in rep["comparisons"])
    L += ["\n## Paired comparisons on the pooled images (b vs a; C1–C9 pre-declared"
          + (", C10 declared 29 Oct before TEST" if c10 else "") + ")\n",
          "| id | isolates | a | b | Δtop-1 pp (95% CI) | McNemar p (a-only/b-only) | ΔNLL (95% CI) |",
          "|---|---|---|---|---|---|---|"]
    for c in rep["comparisons"]:
        if "missing" in c:
            L.append(f"| {c['id']} | {c['isolates']} | {c['a']} | {c['b']} | PENDING (missing {', '.join(c['missing'])}) | | |")
        else:
            L.append(f"| {c['id']} | {c['isolates']} | {c['a']} | {c['b']} | {_fmt_cmp(c)} |")
    for c in rep["post_hoc"]:
        L += [f"\n## {c['id']} — {c['isolates']}\n",
              "Chosen after the TEST results were seen: descriptive, not a pre-declared test.\n",
              "| images | Δtop-1 pp, b − a (95% CI) | McNemar p (a-only/b-only) | ΔNLL (95% CI) |", "|---|---|---|---|",
              f"| pooled ({rep['n']:,}) | {_fmt_cmp(c['pooled'])} |"]
        L += [f"| {V[s]} ({c['per_variant'][s]['top1_diff_pct']['n']:,}) | {_fmt_cmp(c['per_variant'][s])} |"
              for s in POOL_ORDER_R]
    return "\n".join(L) + "\n"

EXT_NOTE = ("EXTENSION requested after the results were frozen (29 Oct 2026): descriptive only. The "
            "pre-declared primary TEST result is ImageNetV2 matched-frequency (10,000 images).")
CAVEAT = dict(
    val="VAL = full ImageNet-1k val, also used to SELECT the old and new QAT checkpoints (optimistic for those rows).",
    test="TEST = ImageNetV2 matched-frequency: a different distribution from ImageNet val; compare models WITHIN "
         "this table, never its absolute numbers with the VAL table.",
    test_th07="ImageNetV2 threshold-0.7 (" + EXT_NOTE + ") Compare models WITHIN this table.",
    test_top="ImageNetV2 top-images (" + EXT_NOTE + ") Compare models WITHIN this table.",
    test_pooled="All three ImageNetV2 variants pooled, each distinct image counted once (" + EXT_NOTE + ") "
                "The variants differ in difficulty; compare models WITHIN this table.")
SPLIT_TITLE = dict(val="ImageNet-1k VAL (full 50,000)", test="TEST = ImageNetV2 matched-frequency",
                   test_th07="ImageNetV2 threshold-0.7 (extension)", test_top="ImageNetV2 top-images (extension)",
                   test_pooled="ImageNetV2, all three variants pooled (extension)")


def _f(v, fmt):
    return "—" if v is None else format(v, fmt)


def render_report(rep):
    nm = SPLIT_TITLE.get(rep["split"], rep["split"])
    L = [f"# {rep['precision']} — {nm} (n = {rep['n']})\n",
         f"Status: EXECUTED ({rep['time_utc']}). All CIM rows from the actual cim_sim simulator (batch 1, CPU). "
         "KL and agreement are vs the pretrained FP32 model on the same images.",
         f"Identical inputs verified: manifest `{rep['split_manifest_sha256'][:16]}`, per-image input-tensor "
         f"SHA-256 digest `{rep['input_sha_digest'][:16]}`, preprocessing `{rep['preprocess_id']}`.\n",
         f"> {rep['caveat']}\n",
         "| run | top-1 % (±SE) | top-5 % | NLL | ECE % | KL(FP32‖run) | agree FP32 % | own-FP32 top-1 % | KL(own FP32‖run) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rep["rows"]:
        L.append(f"| {r['run']} | {r['top1_pct']:.2f} (±{r['top1_se_pct']:.2f}) | {r['top5_pct']:.2f} | "
                 f"{r['nll']:.4f} | {r['ece_pct']:.2f} | {r['kl_ref']:.4f} | {r['agree_ref_pct']:.2f} | "
                 f"{_f(r.get('own_fp32_top1_pct'), '.2f')} | {_f(r.get('kl_own_fp32'), '.4f')} |")
    L += ["\n## Hardware diagnostics per operation (analog runs)\n",
          "| run | gains qkv/qk/av | clip % qkv/qk/av | rail % qkv/qk/av | ADC util median % qkv/qk/av | "
          "util p99.9 % qkv/qk/av | ADC saturation % (all ops) | images with any saturation % |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rep["rows"]:
        if r["kind"] not in ("analog", "legacy_adc"):
            continue
        g = r.get("gains") or {}
        per = "qkv" in r

        def col(key, fmt):
            return "/".join(_f(r[o][key], fmt) for o in OPS) if per else "— (legacy ADC: per-op stats not recorded)"
        dag = " †" if r.get("util_stats_method") == "legacy" else ""
        L.append(f"| {r['run']} | " + ("/".join(_f(g.get(o), '.2f') for o in OPS) if g else "—") + " | "
                 + col("clip_pct", ".4f") + " | " + col("rail_pct", ".3f") + " | "
                 + col("util_median_pct", ".1f") + dag + " | " + col("util_p999_pct", ".0f") + dag + " | "
                 + _f(r.get("saturation_pct_all_ops"), ".4f") + " | "
                 + _f(r.get("images_with_any_saturation_pct"), ".1f") + " |")
    if any(r.get("util_stats_method") == "legacy" for r in rep["rows"]):
        L.append("\n† ADC-utilisation percentiles from the pre-v2.7 merge of chunk statistics, which concentrates "
                 "the weight on a few samples: unreliable. Run `qat_evaluate.py restat` and re-run this report. "
                 "Clip, rail and saturation columns are exact counts and are not affected.")
    else:
        L.append("\nADC-utilisation percentiles: weighted uniform subsample (~2 M conversions per operation, "
                 "cim_qat.analog_merge); clip, rail and saturation columns are exact counts.")
    c10 = any(c["id"] == "C10" for c in rep["comparisons"])
    L += ["\n## Paired comparisons (b vs a, same images; C1–C9 pre-declared"
          + (", C10 declared 29 Oct before TEST" if c10 else "") + ")\n",
          "| id | isolates | a | b | Δtop-1 pp (95% CI) | McNemar p (a-only/b-only) | ΔNLL (95% CI) | ΔKL (95% CI) |",
          "|---|---|---|---|---|---|---|---|"]
    for c in rep["comparisons"]:
        if "missing" in c:
            L.append(f"| {c['id']} | {c['isolates']} | {c['a']} | {c['b']} | PENDING (missing {', '.join(c['missing'])}) | | | |")
            continue
        t, n, k = c["top1_diff_pct"], c["nll_diff"], c.get("kl_ref_diff")
        L.append(f"| {c['id']} | {c['isolates']} | {c['a']} | {c['b']} | {t['mean']:+.2f} ({t['ci95'][0]:+.2f}, "
                 f"{t['ci95'][1]:+.2f}) | {c['mcnemar']['p']:.3g} ({c['mcnemar']['a_only']}/{c['mcnemar']['b_only']}) | "
                 f"{n['mean']:+.4f} ({n['ci95'][0]:+.4f}, {n['ci95'][1]:+.4f}) | "
                 + (f"{k['mean']:+.4f} ({k['ci95'][0]:+.4f}, {k['ci95'][1]:+.4f})" if k else "—") + " |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fp32-check")
    f.add_argument("--models", nargs="+", default=["pretrained", "old_int8", "old_int4"])
    f.add_argument("--interp", nargs="+", default=["bicubic", "bilinear"])
    f.add_argument("--device", default="cuda")
    f.add_argument("--num-workers", type=int, default=12)
    f.add_argument("--limit", type=int, default=None)
    v = sub.add_parser("v2-integrity")
    v.add_argument("--device", default="cuda")
    v.add_argument("--num-workers", type=int, default=12)
    v.add_argument("--min-classes", type=int, default=700)
    v.add_argument("--split", default="test", choices=list(TEST_SPLITS),
                   help="which ImageNetV2 variant (default: matched-frequency, the original check)")
    s = sub.add_parser("sim")
    s.add_argument("--split", required=True, choices=SPLIT_CHOICES)
    s.add_argument("--model", required=True)
    s.add_argument("--kind", choices=["analog", "adc_off", "fp32", "legacy_adc"], required=True)
    s.add_argument("--precision", default=None)
    s.add_argument("--gain-set", choices=["cal", "unity", "hist"], default="cal")
    s.add_argument("--interp", default="bicubic", choices=["bicubic", "bilinear"])
    s.add_argument("--workers", type=int, default=16)
    s.add_argument("--chunk", type=int, default=50)
    s.add_argument("--final", action="store_true")
    s.add_argument("--limit", type=int, default=None)
    d2 = sub.add_parser("select-d2")
    d2.add_argument("--pilot", default="w4a4_kd_pilot10ep")
    d2.add_argument("--fork", default="w4a4_kd_pilot10ep_full")
    rp = sub.add_parser("report-pooled", help="v2.8 extension: the three ImageNetV2 variants pooled")
    rp.add_argument("--precision", required=True)
    rp.add_argument("--new", nargs="*", default=[])
    rp.add_argument("--old", default=None)
    rp.add_argument("--final", action="store_true")
    rs = sub.add_parser("restat")
    rs.add_argument("--split", required=True, choices=SPLIT_CHOICES)
    rs.add_argument("--final", action="store_true")
    rs.add_argument("--limit", type=int, default=None)
    rs.add_argument("--only", default="", help="only run folders whose name contains this text")
    rs.add_argument("--force", action="store_true", help="recompute even if already done")
    r = sub.add_parser("report")
    r.add_argument("--split", required=True, choices=SPLIT_CHOICES)
    r.add_argument("--precision", required=True)
    r.add_argument("--new", nargs="*", default=[], help="new-QAT run names (KD) for C6/C7")
    r.add_argument("--new-ce", nargs="*", default=[], help="CE pilot run names for C9")
    r.add_argument("--old", default=None)
    r.add_argument("--final", action="store_true")
    r.add_argument("--limit", type=int, default=None)
    for p in (f, v, s, r, d2, rs, rp):
        p.add_argument("--paths", required=True)
    a = ap.parse_args()
    if a.cmd == "sim" and a.kind != "fp32" and not a.precision:
        sys.exit("--precision is required for CIM kinds")
    from cim_qat.data import DataPaths
    dp = DataPaths.load(a.paths)
    dict(fp32_check=cmd_fp32_check, v2_integrity=cmd_v2_integrity, sim=cmd_sim,
         report=cmd_report, select_d2=cmd_select_d2, restat=cmd_restat,
         report_pooled=cmd_report_pooled)[a.cmd.replace("-", "_")](a, dp)


if __name__ == "__main__":
    main()
