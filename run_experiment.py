#!/usr/bin/env python3
"""
run_experiment.py — real-image (Imagenette) evaluation of pretrained and QAT
TinyViT-5M on the modelled analog CIM datapath. Deterministic and resumable.

Typical sequence (PowerShell, from the repository root):

    python run_experiment.py download-data                 # official fast.ai archive
    python run_experiment.py inspect-checkpoints --download-pretrained
    python run_experiment.py prepare --stage smoke         # 100-image manifest
    python run_experiment.py verify  --stage smoke         # validation suite
    python run_experiment.py reference --stage smoke       # FP32 logits, 3 models
    python run_experiment.py cim --stage smoke --workers 4 # 48 CIM jobs, resumable
    python run_experiment.py report --stage smoke          # CSVs + REPORT.md
    python run_experiment.py status --stage smoke

    (repeat with --stage full after the smoke results have been checked)

Optional, separate follow-ups (never change the primary tables):
    python run_experiment.py trainpoint --stage smoke --workers 4          # Experiment T
    python run_experiment.py prepare-calibration --stage smoke             # Experiment B
    python run_experiment.py calibrate --stage smoke
    python run_experiment.py gain-robustness --stage smoke --workers 4

Outputs: results/<stage>/  (see EXPERIMENTS.md for every file).
No synthetic fallback exists anywhere in this script: if the dataset or a
checkpoint is missing it stops with an explanation.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

# 1 thread per forward pass, set before numpy/torch import (determinism).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "_shim"))
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from cim_exp import checkpoints as CK  # noqa: E402
from cim_exp import engine as EN  # noqa: E402
from cim_exp import hardware as HWM  # noqa: E402
from cim_exp import imagenette as IM  # noqa: E402
from cim_exp import reporting as RP  # noqa: E402

DEFAULT_CKPT_CONFIG = REPO / "configs" / "checkpoints.json"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def run_dir(a) -> Path:
    return REPO / "results" / (a.run_name or a.stage)


def load_run_cfg(a) -> dict:
    p = run_dir(a) / "run_config.json"
    if not p.exists():
        sys.exit(f"{p} not found — run:  python run_experiment.py prepare --stage {a.stage}")
    return json.loads(p.read_text())


def registry(a, allow_download=False) -> tuple[dict, dict]:
    reg = CK.load_registry(Path(a.checkpoints), REPO)
    hashes = {}
    for mid, spec in reg.items():
        CK.ensure_present(spec, allow_download=allow_download)
        hashes[mid] = CK.verify_hash(spec)
    return reg, hashes


def context(a, run_cfg: dict) -> HWM.RunContext:
    import torch
    return HWM.RunContext(run_name=a.run_name or a.stage,
                          manifest_sha256=run_cfg["manifest_sha256"],
                          preprocess_id=run_cfg["preprocess_id"],
                          code_sha256=HWM.code_fingerprint(REPO),
                          torch_version=torch.__version__,
                          numpy_version=np.__version__,
                          # fixed at `prepare` time: part of every CIM cache key
                          chunk_size=int(run_cfg.get("chunk_size", a.chunk_size)))


def environment(code_sha: str) -> dict:
    import torch
    try:
        import PIL
        pil = PIL.__version__
    except Exception:                                          # pragma: no cover
        pil = None
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                             text=True, timeout=10).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                                    capture_output=True, text=True,
                                    timeout=10).stdout.strip())
    except Exception:
        git, dirty = None, None
    return dict(python=sys.version, platform=platform.platform(),
                machine=platform.machine(), processor=platform.processor(),
                cpu_count=os.cpu_count(), torch=torch.__version__, numpy=np.__version__,
                pillow=pil, threads_per_forward=1, batch_size=1,
                code_sha256=code_sha, git_commit=git, git_dirty=dirty,
                recorded=time.strftime("%Y-%m-%d %H:%M:%S %z"))


def cim_jobs(a, ctx, hashes, reg) -> list[dict]:
    jobs = []
    for mid, hw in HWM.planned_cim_jobs(a.qat_selection, a.amp_swing_limit):
        if a.models and mid not in a.models:
            continue
        if a.only and hw.label not in a.only:
            continue
        key = HWM.cim_key(ctx, mid, hashes[mid], hw)
        jobs.append(dict(model=mid, hw=hw, key=key, ckpt=reg[mid].path))
    return jobs


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


def cmd_download_data(a):
    root = REPO / a.data_root
    arc = IM.download(a.variant, root)
    IM.extract(arc, root, a.variant)
    p = IM.write_provenance(root, a.variant, arc)
    n = len(IM.list_split(root / a.variant))
    print(f"OK: {root / a.variant}  ({n} validation images; provenance {p.name})")


def cmd_inspect(a):
    reg, hashes = registry(a, allow_download=a.download_pretrained)
    import torch
    pre = torch.load(reg["pretrained"].path, map_location="cpu", weights_only=True)
    pre = pre.get("model", pre)
    reps = {}
    for mid, spec in reg.items():
        rep = CK.inspect(spec, reference_sd=None if mid == "pretrained" else pre)
        rep["sha256"] = hashes[mid]
        rep["expected_sha256"] = spec.expected_sha256
        rep["trained_weight_bits"] = spec.trained_weight_bits
        if mid in CK.SUPPLIED_QAT_SHA256:
            rep["matches_supplied_file"] = hashes[mid] == CK.SUPPLIED_QAT_SHA256[mid]
        if mid == "pretrained":
            rep["matches_official_pretrained"] = hashes[mid] == CK.PRETRAINED_SHA256
        model = CK.build(spec)                    # strict key/shape load
        rep["builds_strict"] = True
        rep["n_attention_modules"] = sum(type(m).__name__ == "Attention"
                                         for m in model.modules())
        reps[mid] = rep
    out = REPO / "results" / "checkpoint_inspection.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(reps, indent=2, default=str))
    for mid, r in reps.items():
        print(f"{mid:11s} sha256={r['sha256'][:16]}…  tensors={r['n_tensors']} "
              f"head={r['head_shape']} dtypes={r['dtypes']} quantizer_keys={r['n_quantizer_keys']} "
              f"int4-offgrid={r['qkv_grid_test']['int4_mean_offgrid']:.3f} "
              f"strict-load={r['builds_strict']}")
    print(f"written {out}")
    return reps


def cmd_prepare(a):
    notes = IM.verify_mapping_tables()
    for n in notes:
        print("mapping note:", n)
    ds = REPO / a.data_root / a.variant
    rows = IM.build_manifest(ds, a.stage, per_class=a.per_class, seed=a.seed)
    rd = run_dir(a)
    msha = IM.write_manifest(rows, rd / "input_manifest.csv")
    inputs = rd / "cache" / f"inputs_{msha[:16]}.npy"
    if not inputs.exists():
        print(f"preprocessing {len(rows)} images -> {inputs}")
        IM.build_input_cache(ds, rows, inputs)
    old = rd / "run_config.json"
    chunk = a.chunk_size
    if old.exists():                      # keep the chunking of an existing run
        chunk = json.loads(old.read_text()).get("chunk_size", chunk)
    cfg = dict(stage=a.stage, variant=a.variant, dataset_dir=str(ds), split=IM.EVAL_SPLIT,
               chunk_size=chunk,
               per_class=(a.per_class if a.stage == "smoke" else None), seed=a.seed,
               n_images=len(rows), manifest_sha256=msha, inputs_cache=str(inputs),
               preprocess_id=IM.PREPROCESS_ID, mapping_notes=notes,
               provenance=IM.filename_provenance(rows))
    (rd / "run_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"OK: {len(rows)} images, manifest sha256 {msha[:16]}…  -> {rd}")
    print(f"file-name provenance: {cfg['provenance']}  (ImageNet-1k train-style names = "
          f"images the pretrained and QAT models were trained on; see REPORT caveats)")


def _ref_paths(a, ctx, hashes):
    d = run_dir(a) / "cache" / "reference"
    return {mid: d / f"{HWM.reference_key(ctx, mid, hashes[mid])}.npz" for mid in CK.MODEL_IDS}


def cmd_reference(a):
    run_cfg = load_run_cfg(a)
    reg, hashes = registry(a, allow_download=a.download_pretrained)
    ctx = context(a, run_cfg)
    for mid, p in _ref_paths(a, ctx, hashes).items():
        if p.exists():
            print(f"[cached] FP32 reference {mid}: {p.name}")
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        print(f"FP32 reference {mid} ({run_cfg['n_images']} images) ...")
        EN.run_reference(REPO, reg[mid].path, Path(run_cfg["inputs_cache"]), p,
                         dict(model=mid, checkpoint=str(reg[mid].path),
                              checkpoint_sha256=hashes[mid], **ctx.base(),
                              run_name=ctx.run_name))
        lg = np.load(p)["logits"]
        nf = int((~np.isfinite(lg).all(1)).sum())
        print(f"   done: logits {lg.shape}, non-finite images: {nf}")
        if nf:
            print(f"   WARNING: {mid} FP32 produced non-finite logits")


def _units(a, ctx, jobs, run_cfg):
    n = run_cfg["n_images"]
    cs = ctx.chunk_size
    n_chunks = (n + cs - 1) // cs
    units = []
    for j in jobs:
        jd = run_dir(a) / "cache" / "cim" / j["key"]
        if (jd / "final.json").exists():
            continue
        jd.mkdir(parents=True, exist_ok=True)
        (jd / "job.json").write_text(json.dumps(dict(model=j["model"], hw=j["hw"].as_dict(),
                                                     key=j["key"], n_chunks=n_chunks,
                                                     **ctx.base()), indent=1))
        for c in range(n_chunks):
            if (jd / f"chunk_{c:05d}.json").exists():
                continue
            units.append(dict(repo_root=str(REPO), job_key=j["key"], job_dir=str(jd),
                              hw=j["hw"].as_dict(), ckpt_path=str(j["ckpt"]), chunk=c,
                              i0=c * cs, i1=min(n, (c + 1) * cs),
                              inputs_path=run_cfg["inputs_cache"],
                              stats_keep_frac=ctx.stats_keep_frac,
                              stats_cap=ctx.stats_cap))
    return units, n_chunks


def _run_jobs(a, ctx, jobs, run_cfg, hashes) -> dict:
    """Run (resumably, in parallel) and finalize CIM jobs; returns {key: final meta}."""
    units, n_chunks = _units(a, ctx, jobs, run_cfg)
    print(f"{len(jobs)} CIM jobs, {n_chunks} chunks each; {len(units)} chunk units to run "
          f"with {a.workers} worker process(es)")
    t0, done = time.time(), 0
    if units:
        if a.workers <= 1:
            it = map(EN.run_cim_chunk, units)
            pool = None
        else:
            import multiprocessing as mp
            pool = mp.get_context("spawn").Pool(a.workers)
            it = pool.imap_unordered(EN.run_cim_chunk, units)
        try:
            for r in it:
                done += 1
                el = time.time() - t0
                eta = el / done * (len(units) - done)
                print(f"  [{done}/{len(units)}] {r['job_key']} chunk {r['chunk']} "
                      f"{r['wall_s']:.0f}s   elapsed {el / 60:.1f} min, ETA {eta / 60:.1f} min",
                      flush=True)
        finally:
            if pool is not None:
                pool.close()
                pool.join()
    finals = {}
    for j in jobs:
        jd = run_dir(a) / "cache" / "cim" / j["key"]
        if (jd / "final.json").exists():
            finals[j["key"]] = json.loads((jd / "final.json").read_text())
            continue
        if all((jd / f"chunk_{c:05d}.json").exists() for c in range(n_chunks)):
            out = EN.finalize_job(jd, n_chunks, dict(model=j["model"], hw=j["hw"].as_dict(),
                                                     key=j["key"], checkpoint=str(j["ckpt"]),
                                                     checkpoint_sha256=hashes[j["model"]],
                                                     **ctx.base()),
                                  ctx.stats_cap, keep_chunks=a.keep_chunks)
            finals[j["key"]] = out
            summ = out["analog_summary"]
            per_op = " ".join(f"{op} {summ[op]['clip_pct']:.4f}%" for op in HWM.OPS)
            flag = "" if out["status"] == "OK" else f"  <-- {out['status']}"
            print(f"finalized {j['key']}: ADC clip per op: {per_op}{flag}")
    return finals


def cmd_cim(a):
    run_cfg = load_run_cfg(a)
    reg, hashes = registry(a, allow_download=a.download_pretrained)
    ctx = context(a, run_cfg)
    if a.amp_swing_limit is not None:
        print(f"NOTE: amplifier swing limit {a.amp_swing_limit} V -> a DIAGNOSTIC variant of "
              f"the matrix (labels end in _amp{a.amp_swing_limit:g}V); the primary matrix "
              f"uses an ideal amplifier. Use a separate --run-name.")
    _run_jobs(a, ctx, cim_jobs(a, ctx, hashes, reg), run_cfg, hashes)


def cmd_status(a):
    run_cfg = load_run_cfg(a)
    reg, hashes = registry(a)
    ctx = context(a, run_cfg)
    n_chunks = (run_cfg["n_images"] + ctx.chunk_size - 1) // ctx.chunk_size
    for mid, p in _ref_paths(a, ctx, hashes).items():
        print(f"reference {mid:11s}: {'done' if p.exists() else 'MISSING'}")
    jobs = cim_jobs(a, ctx, hashes, reg)
    nd = 0
    for j in jobs:
        jd = run_dir(a) / "cache" / "cim" / j["key"]
        if (jd / "final.json").exists():
            st = json.loads((jd / "final.json").read_text())["status"]
            nd += 1
            s = f"done ({st})"
        else:
            k = sum((jd / f"chunk_{c:05d}.json").exists() for c in range(n_chunks))
            s = f"{k}/{n_chunks} chunks"
        print(f"  {j['model']:11s} {j['hw'].label:24s} {s}")
    print(f"{nd}/{len(jobs)} CIM jobs finalized")


def cmd_report(a):
    run_cfg = load_run_cfg(a)
    reg, hashes = registry(a)
    ctx = context(a, run_cfg)
    ref_keys = {mid: HWM.reference_key(ctx, mid, hashes[mid]) for mid in CK.MODEL_IDS}
    cim_keys = {}
    run_manifest = []
    import torch
    pre = torch.load(reg["pretrained"].path, map_location="cpu", weights_only=True)
    pre = pre.get("model", pre)
    ck_reports = {}
    for mid, spec in reg.items():
        rep = CK.inspect(spec, reference_sd=None if mid == "pretrained" else pre)
        rep["sha256"] = hashes[mid]
        ck_reports[mid] = rep
    for mid in CK.MODEL_IDS:
        p = run_dir(a) / "cache" / "reference" / f"{ref_keys[mid]}.npz"
        run_manifest.append(dict(run_id=ref_keys[mid], kind="fp32_reference", model=mid,
                                 checkpoint=str(reg[mid].path),
                                 checkpoint_sha256=hashes[mid], hw_label="",
                                 status="complete" if p.exists() else "missing",
                                 manifest_sha256=run_cfg["manifest_sha256"],
                                 n_images=run_cfg["n_images"],
                                 cache_path=str(p.relative_to(REPO))))
    for mid, hw in HWM.planned_cim_jobs(a.qat_selection, a.amp_swing_limit):
        key = HWM.cim_key(ctx, mid, hashes[mid], hw)
        cim_keys[(mid, hw.label)] = (key, hw)
        fj = run_dir(a) / "cache" / "cim" / key / "final.json"
        st = json.loads(fj.read_text())["status"] if fj.exists() else "missing"
        run_manifest.append(dict(run_id=key, kind="cim_analog", model=mid,
                                 checkpoint=str(reg[mid].path), checkpoint_sha256=hashes[mid],
                                 hw_label=hw.label, status=st,
                                 manifest_sha256=run_cfg["manifest_sha256"],
                                 n_images=run_cfg["n_images"],
                                 cache_path=str(fj.parent.relative_to(REPO))))
    R = RP.Results(run_dir(a), run_cfg, hashes,
                   {mid: reg[mid].trained_weight_bits for mid in CK.MODEL_IDS},
                   ref_keys, cim_keys)
    env = environment(ctx.code_sha256)
    prov = run_cfg.get("provenance") or IM.filename_provenance(R.manifest_rows)
    cov = RP.write_all(R, run_dir(a), a.qat_selection, env, ck_reports, run_manifest,
                       amp=a.amp_swing_limit, extra_context=dict(provenance=prov))
    print(f"report written to {run_dir(a) / 'REPORT.md'}")
    print(f"coverage complete: {cov['complete']}   rows per case: {cov['counts']} "
          f"(expected {cov['expected_counts']})")
    if cov["invalid_runs"]:
        print(f"INVALID runs: {cov['invalid_runs']}")
    if not cov.get("class_mapping_confirmed"):
        print("WARNING: class mapping NOT confirmed on these images "
              "(see class_mapping_check.csv) - results must not be interpreted")
    if not cov["complete"] and not a.allow_incomplete:
        sys.exit("report is INCOMPLETE (see coverage.json); rerun the missing steps or "
                 "pass --allow-incomplete to accept a partial report")


def cmd_verify(a):
    """The validation suite of the experiment plan; writes verify.json."""
    from cim_exp import verify as VF
    run_cfg = load_run_cfg(a)
    reg, hashes = registry(a, allow_download=a.download_pretrained)
    res = VF.run_all(REPO, a, run_cfg, reg, hashes, run_dir(a))
    out = run_dir(a) / "verify.json"
    out.write_text(json.dumps(res, indent=2, default=str))
    bad = [k for k, v in res["checks"].items() if v["status"] == "FAIL"]
    print(f"\nverify: {sum(v['status'] == 'PASS' for v in res['checks'].values())} PASS, "
          f"{len(bad)} FAIL, {sum(v['status'] == 'SKIP' for v in res['checks'].values())} SKIP "
          f"-> {out}")
    if bad:
        sys.exit(f"FAILED checks: {bad}")


# --------------------------------------------------------------------------- #
# Experiment B — gain calibration and robustness (separate, optional)
# --------------------------------------------------------------------------- #


def _expb_dir(a) -> Path:
    return run_dir(a) / "expB"


def cmd_prepare_calibration(a):
    """Calibration manifest from the Imagenette TRAIN split, disjoint from evaluation."""
    from cim_exp import calibration as CB
    run_cfg = load_run_cfg(a)
    ds = REPO / a.data_root / a.variant
    rows = IM.build_manifest(ds, "smoke", per_class=a.cal_per_class, seed=CB.CAL_SEED,
                             split="train")
    eval_rows, _ = IM.read_manifest(run_dir(a) / "input_manifest.csv")
    overlap = {r["file_sha256"] for r in rows} & {r["file_sha256"] for r in eval_rows}
    if overlap:
        raise ValueError(f"calibration and evaluation images overlap ({len(overlap)} files "
                         f"by SHA-256); refusing to calibrate")
    d = _expb_dir(a)
    msha = IM.write_manifest(rows, d / "calibration_manifest.csv")
    inputs = d / "cache" / f"cal_inputs_{msha[:16]}.npy"
    if not inputs.exists():
        print(f"preprocessing {len(rows)} calibration images -> {inputs}")
        IM.build_input_cache(ds, rows, inputs)
    cfg = dict(split="train", per_class=a.cal_per_class, seed=CB.CAL_SEED,
               n_images=len(rows), manifest_sha256=msha, inputs_cache=str(inputs),
               eval_manifest_sha256=run_cfg["manifest_sha256"], overlap_with_eval=0,
               preprocess_id=IM.PREPROCESS_ID, provenance=IM.filename_provenance(rows))
    (d / "calibration_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"OK: {len(rows)} calibration images (train split, seed {CB.CAL_SEED}), "
          f"disjoint from the {len(eval_rows)} evaluation images -> {d}")


def _load_cal_cfg(a) -> dict:
    p = _expb_dir(a) / "calibration_config.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - run:  python run_experiment.py "
                                f"prepare-calibration --stage {a.stage}")
    return json.loads(p.read_text())


def _hist_path(a, model, sha, w, ab, cal_cfg) -> Path:
    key = HWM.short_hash(dict(model=model, ckpt=sha, w=w, a=ab, cal=cal_cfg["manifest_sha256"],
                              code=HWM.code_fingerprint(REPO)))
    return _expb_dir(a) / "cache" / f"hist_{model}_W{w}A{ab}_{key}.npz"


def _load_calibration(a, reg, hashes, cal_cfg, compute: bool) -> dict:
    from cim_exp import calibration as CB
    cal = {}
    for src in a.cal_sources:
        for (w, ab) in HWM.PRECISIONS:
            model = CB.model_for(src, w)
            hp = _hist_path(a, model, hashes[model], w, ab, cal_cfg)
            if not hp.exists():
                if not compute:
                    continue
                print(f"calibrating {model} at W{w}A{ab} on {cal_cfg['n_images']} images ...",
                      flush=True)
                obs = CB.observe(REPO, reg[model].path, w, ab, Path(cal_cfg["inputs_cache"]))
                hp.parent.mkdir(parents=True, exist_ok=True)
                np.savez(hp, **{f"hist_{op}": obs["hist"][op] for op in HWM.OPS},
                         maxabs=np.array([obs["maxabs"][op] for op in HWM.OPS]),
                         n_images=obs["n_images"])
            z = np.load(hp)
            obs = dict(hist={op: z[f"hist_{op}"] for op in HWM.OPS},
                       maxabs=dict(zip(HWM.OPS, z["maxabs"].tolist())),
                       n_images=int(z["n_images"]))
            c = CB.calibrate(obs, w, ab)
            c["cal_model"] = model
            c["cal_checkpoint_sha256"] = hashes[model]
            c["_hist"] = obs["hist"]
            cal[(src, f"W{w}A{ab}")] = c
    return cal


def cmd_calibrate(a):
    from cim_exp import calibration as CB
    from cim_exp.reporting import write_csv
    reg, hashes = registry(a, allow_download=a.download_pretrained)
    cal_cfg = _load_cal_cfg(a)
    cal = _load_calibration(a, reg, hashes, cal_cfg, compute=True)
    rows = []
    for (src, ptag), c in sorted(cal.items()):
        for name, pred in c["predicted_clip_pct"].items():
            lvl = name[len("calB_"):] if name.startswith("calB_") else None
            g = c["levels"][lvl]["gains"] if lvl else HWM.GAIN_SETS[name]
            for op in HWM.OPS:
                rows.append(dict(calibration_checkpoint=c["cal_model"],
                                 calibration_checkpoint_sha256=c["cal_checkpoint_sha256"],
                                 precision=ptag, gain_set=name,
                                 objective=(lvl or "fixed (Experiment A)"),
                                 clip_target_pct=(CB.LEVEL_TARGET_PCT[lvl] if lvl else ""),
                                 operation=op, gain=g[op],
                                 full_scale_colsum=(c["levels"][lvl]["fs"][op] if lvl else ""),
                                 k_mac_v_per_unit=c["kmac"][op],
                                 predicted_calibration_clip_pct=pred[op],
                                 max_abs_colsum=c["maxabs"][op], n_colsums=c["n_colsums"][op],
                                 calibration_manifest_sha256=cal_cfg["manifest_sha256"],
                                 calibration_split=cal_cfg["split"],
                                 calibration_n_images=cal_cfg["n_images"]))
    d = _expb_dir(a)
    write_csv(d / "calibration_gains.csv", rows)
    js = {f"{c['cal_model']}|{ptag}": {k: v for k, v in c.items() if k != "_hist"}
          for (src, ptag), c in cal.items()}
    (d / "calibration.json").write_text(json.dumps(dict(calibration=cal_cfg, results=js,
                                                        primary_level=CB.PRIMARY_LEVEL),
                                                   indent=1, default=str))
    print(f"\n{'checkpoint':11s} {'W/A':5s}  primary ({CB.PRIMARY_LEVEL}) gains qkv/qk/av   "
          f"fixed-set predicted clip % (qkv/qk/av)")
    for (src, ptag), c in sorted(cal.items()):
        g = c["levels"][CB.PRIMARY_LEVEL]["gains"]
        fx = c["predicted_clip_pct"]["cal_w8a8"]
        print(f"{c['cal_model']:11s} {ptag:5s}  {g['qkv']:7.2f} {g['qk']:6.2f} {g['av']:6.2f}"
              f"           {fx['qkv']:.4f} / {fx['qk']:.4f} / {fx['av']:.4f}")
    print(f"written {d / 'calibration_gains.csv'} and calibration.json")


def cmd_gain_robustness(a):
    from cim_exp import calibration as CB
    from cim_exp.imagenette import IMAGENET_INDICES
    from cim_exp.reporting import write_csv
    run_cfg = load_run_cfg(a)
    reg, hashes = registry(a, allow_download=a.download_pretrained)
    ctx = context(a, run_cfg)
    cal_cfg = _load_cal_cfg(a)
    cal = _load_calibration(a, reg, hashes, cal_cfg, compute=False)
    if not cal:
        raise FileNotFoundError("no calibration found - run `calibrate` first")
    plan = CB.plan_jobs(cal, a.b_adc_bits, a.levels, cal_cfg["manifest_sha256"],
                        cross_apply=a.cross_apply,
                        precision_transfer=not a.no_precision_transfer)
    jobs = [dict(model=m, hw=hw, key=HWM.cim_key(ctx, m, hashes[m], hw), ckpt=reg[m].path)
            for m, hw, _ in plan]
    print(f"Experiment B: {len(jobs)} CIM evaluations on the {run_cfg['n_images']} held-out "
          f"evaluation images (Experiment A baselines share Experiment A's cache)")
    finals_by_key = _run_jobs(a, ctx, jobs, run_cfg, hashes)
    refs = {}
    for mid, p in _ref_paths(a, ctx, hashes).items():
        if not p.exists():
            raise FileNotFoundError(f"FP32 reference for {mid} missing - run `reference` first")
        refs[mid] = np.load(p)["logits"]
    finals = {}
    for (m, hw, _), j in zip(plan, jobs):
        if j["key"] in finals_by_key:
            lg = np.load(run_dir(a) / "cache" / "cim" / j["key"] / "final.npz")["logits"]
            finals[(m, hw.label, hw.custom_gains)] = (finals_by_key[j["key"]], lg)
    rows_m, _ = IM.read_manifest(run_dir(a) / "input_manifest.csv")
    y1000 = np.array([r["label_1000way"] for r in rows_m])
    y10 = np.array([r["label_10way"] for r in rows_m])
    rows = CB.build_rows(plan, finals, refs, cal, y1000, y10, IMAGENET_INDICES, hashes, cal_cfg)
    d = _expb_dir(a)
    write_csv(d / "gain_robustness.csv", rows)
    (d / "EXPERIMENT_B.md").write_text(CB.render_md(rows, cal, cal_cfg, a.levels),
                                       encoding="utf-8")
    missing = len(plan) - len(rows)
    print(f"written {d / 'gain_robustness.csv'} and EXPERIMENT_B.md "
          f"({len(rows)} rows{'; ' + str(missing) + ' jobs incomplete' if missing else ''})")


# --------------------------------------------------------------------------- #
# Experiment T — QAT training operating point (legacy functional ADC), optional
# --------------------------------------------------------------------------- #


def cmd_trainpoint(a):
    from cim_exp import trainpoint as TP
    from cim_exp.imagenette import IMAGENET_INDICES
    from cim_exp.reporting import write_csv
    run_cfg = load_run_cfg(a)
    reg, hashes = registry(a, allow_download=a.download_pretrained)
    ctx = context(a, run_cfg)
    d = run_dir(a) / "expT"
    (d / "cache").mkdir(parents=True, exist_ok=True)
    units = []
    for m in TP.MODELS:
        for (w, ab) in TP.POINTS:
            key = TP.job_key(ctx.base(), m, hashes[m], w, ab)
            units.append(dict(repo_root=str(REPO), key=key, model=m, w=w, a=ab,
                              ckpt=str(reg[m].path), inputs=run_cfg["inputs_cache"],
                              out=str(d / "cache" / f"{key}.npz")))
    todo = [u for u in units if not Path(u["out"]).exists()]
    print(f"Experiment T: {len(units)} jobs ({len(todo)} to run), legacy functional 8-bit ADC")
    if todo:
        if a.workers <= 1:
            for u in todo:
                TP.run_job(u)
                print(f"  done {u['key']}", flush=True)
        else:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(min(a.workers, len(todo))) as pool:
                for r in pool.imap_unordered(TP.run_job, todo):
                    print(f"  done {r['key']}", flush=True)
    refs = {mid: np.load(p)["logits"] for mid, p in _ref_paths(a, ctx, hashes).items()
            if p.exists()}
    if len(refs) < 3:
        raise FileNotFoundError("FP32 references missing - run `reference` first")
    rows_in = []
    for u in units:
        z = np.load(u["out"])
        meta = json.loads(Path(u["out"]).with_suffix(".json").read_text())
        rows_in.append(dict(model=u["model"], w=u["w"], a=u["a"], key=u["key"],
                            sha=hashes[u["model"]], logits=z["logits"],
                            conv=z["conversions"], sat=z["saturations"],
                            fs=meta["full_scale"]))
    rows_m, _ = IM.read_manifest(run_dir(a) / "input_manifest.csv")
    y1000 = np.array([r["label_1000way"] for r in rows_m])
    y10 = np.array([r["label_10way"] for r in rows_m])
    rows, md = TP.report(rows_in, refs, y1000, y10, IMAGENET_INDICES)
    write_csv(d / "trainpoint.csv", rows)
    (d / "TRAINPOINT.md").write_text(md, encoding="utf-8")
    print(md)


def cmd_all(a):
    cmd_prepare(a)
    cmd_verify(a)
    cmd_reference(a)
    cmd_cim(a)
    cmd_report(a)


# --------------------------------------------------------------------------- #


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, stage=True):
        sp.add_argument("--checkpoints", default=str(DEFAULT_CKPT_CONFIG),
                        help="checkpoint registry JSON (default configs/checkpoints.json)")
        sp.add_argument("--data-root", default="data")
        sp.add_argument("--variant", default=IM.DEFAULT_VARIANT, choices=sorted(IM.VARIANTS))
        sp.add_argument("--download-pretrained", action="store_true",
                        help="download the official pretrained checkpoint if missing")
        if stage:
            sp.add_argument("--stage", choices=["smoke", "full"], required=True)
            sp.add_argument("--run-name", default=None,
                            help="results/<run-name> instead of results/<stage>")
            sp.add_argument("--per-class", type=int, default=10)
            sp.add_argument("--seed", type=int, default=0)
            sp.add_argument("--chunk-size", type=int, default=25,
                            help="images per resumable unit; fixed by `prepare` for the run")
            sp.add_argument("--qat-selection", choices=["matched", "both"], default="matched")
            sp.add_argument("--workers", type=int, default=1)
            sp.add_argument("--models", nargs="+", choices=list(CK.MODEL_IDS))
            sp.add_argument("--only", nargs="+", help="hardware labels, e.g. W8A8_ADC10_cal_w8a8")
            sp.add_argument("--keep-chunks", action="store_true")
            sp.add_argument("--allow-incomplete", action="store_true")
            sp.add_argument("--amp-swing-limit", type=float, default=None,
                            help="amplifier output swing limit in volts (DIAGNOSTIC; default "
                                 "none = ideal linear amplifier, saturation not modelled)")
            # Experiment B
            sp.add_argument("--cal-per-class", type=int, default=10,
                            help="Experiment B calibration images per class (train split)")
            sp.add_argument("--cal-sources", nargs="+", default=["pretrained", "qat"],
                            choices=["pretrained", "qat"])
            sp.add_argument("--levels", nargs="+", default=["p99.9", "p99.99", "p99.999", "p100"],
                            choices=["p99.9", "p99.99", "p99.999", "p100"])
            sp.add_argument("--b-adc-bits", nargs="+", type=int, default=[8],
                            help="ADC resolutions for the Experiment B robustness runs")
            sp.add_argument("--cross-apply", action="store_true",
                            help="Experiment B: also apply each checkpoint's gains to the other")
            sp.add_argument("--no-precision-transfer", action="store_true")

    for name in ("download-data",):
        common(sub.add_parser(name), stage=False)
    common(sub.add_parser("inspect-checkpoints"), stage=False)
    for name in ("prepare", "verify", "reference", "cim", "report", "status", "all",
                 "prepare-calibration", "calibrate", "gain-robustness", "trainpoint"):
        common(sub.add_parser(name))
    a = p.parse_args()
    try:
        _dispatch(a)
    except (FileNotFoundError, KeyError, ValueError) as e:
        # missing dataset / checkpoint / config, or a hash mismatch: explain and stop
        sys.exit(f"ERROR: {e}")
    except OSError as e:
        if a.cmd == "download-data":
            url = IM.VARIANTS[a.variant]
            sys.exit(f"ERROR: download failed ({e}).\nDownload {url} in a browser, save it "
                     f"as {REPO / a.data_root / (a.variant + '.tgz')} and run this command "
                     f"again (it will then only extract).")
        raise


def _dispatch(a):
    {"download-data": cmd_download_data, "inspect-checkpoints": cmd_inspect,
     "prepare": cmd_prepare, "verify": cmd_verify, "reference": cmd_reference,
     "cim": cmd_cim, "report": cmd_report, "status": cmd_status,
     "all": cmd_all, "prepare-calibration": cmd_prepare_calibration,
     "calibrate": cmd_calibrate, "gain-robustness": cmd_gain_robustness,
     "trainpoint": cmd_trainpoint}[a.cmd](a)


if __name__ == "__main__":
    main()
