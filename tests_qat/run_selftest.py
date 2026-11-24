#!/usr/bin/env python3
"""
tests_qat/run_selftest.py — END-TO-END plumbing test on FAKE data (CPU).

Builds a fake ImageNet-1k (1000 classes; train 4/class, flat val 2/class +
LOC_val_solution.csv) and a fake ImageNetV2 (folders 0..999, 2/class) in a
temporary folder WITH A SPACE in its name, then runs every pipeline command
exactly as the runbook does (as subprocesses), on mini splits where a command
would otherwise take hours on CPU:

  prepare inspect/make --hash-train -> mini splits -> calibrate run/freeze ->
  surrogate-vs-simulator (incl. T4 code path) -> training mechanics ->
  selection/early-stopping rules -> thread limit / orphan / stop-tool tests ->
  smoke training -> training A (uninterrupted)
  and B (killed + resumed; final weights must be identical) -> TEST guards
  (refusals) -> simulator evaluation of the whole experiment matrix on VAL and
  TEST -> reports (identical-input assertion) -> v2-integrity -> audit (pass)
  -> audit negative tests (planted leaks must FAIL).

Fake images carry no class information: accuracies are meaningless (~0.1 %).
This test establishes that the code runs and that the guards work — nothing
about models or hardware.

  python tests_qat/run_selftest.py [--keep] [--workers 2]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
PY = sys.executable
RES = []


def check(name, ok, detail=""):
    RES.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    return ok


def run(args, expect=0, timeout=3600, env=None, tail=12):
    t0 = time.time()
    p = subprocess.run([PY] + [str(a) for a in args], cwd=REPO, capture_output=True, text=True,
                       timeout=timeout, env=dict(os.environ, **(env or {})))
    out = p.stdout + p.stderr
    ok = (p.returncode == expect) if expect is not None else True
    name = " ".join(str(a) for a in args[:3])
    check(f"{name} (exit {p.returncode}, expected {expect}, {time.time() - t0:.0f}s)", ok)
    if not ok:
        print("\n".join(out.splitlines()[-40:]))
    elif tail:
        print("    " + "\n    ".join(out.strip().splitlines()[-tail:]))
    return p.returncode, out


def fake_data(root: Path):
    import numpy as np
    from PIL import Image

    from cim_qat import data as D
    rng = np.random.default_rng(0)
    syn = D.synsets()
    (root / "raw" / "val").mkdir(parents=True)
    rows = []
    for ci, w in enumerate(syn):
        d = root / "raw" / "train" / w
        d.mkdir(parents=True)
        for k in range(4):
            Image.fromarray(rng.integers(0, 255, (int(rng.integers(60, 120)), int(rng.integers(60, 120)), 3),
                                         dtype=np.uint8)).save(d / f"{w}_{k}.JPEG")
        for k in range(2):
            name = f"ILSVRC2012_val_{ci * 2 + k + 1:08d}"
            Image.fromarray(rng.integers(0, 255, (80, 100, 3), dtype=np.uint8)).save(root / "raw" / "val" / f"{name}.JPEG")
            rows.append((name, f"{w} 10 10 50 50"))
        d2 = root / "v2" / D.V2_DIRNAME / str(ci)
        d2.mkdir(parents=True)
        for k in range(2):
            Image.fromarray(rng.integers(0, 255, (90, 90, 3), dtype=np.uint8)).save(d2 / f"img{k}_{ci}.jpeg")
    with open(root / "raw" / "LOC_val_solution.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["ImageId", "PredictionString"])
        wr.writerows(rows)


def _hash(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _subset_variant(sd: Path, split: str, must: list, n: int):
    """Plumbing only: shrink a registered variant to n images (keeping `must` rel_paths)."""
    from cim_qat.common import sha256_file
    from cim_qat.data import read_hashes, read_manifest, write_manifest
    reg = json.loads((sd / "extra_tests.json").read_text())
    rows, hs = read_manifest(sd / f"manifest_{split}.csv"), read_hashes(sd / f"hashes_{split}.csv")
    pick = [i for i, r in enumerate(rows) if r["rel_path"] in must]
    pick += [i for i in range(len(rows)) if i not in pick][: n - len(pick)]
    pick.sort()
    sub = [dict(rows[i], idx=k) for k, i in enumerate(pick)]
    msha = write_manifest(sub, sd / f"manifest_{split}.csv")
    with open(sd / f"hashes_{split}.csv", "w") as f:
        f.write("idx,sha256\n")
        for k, i in enumerate(pick):
            f.write(f"{k},{hs[i]}\n")
    reg[split].update(n=len(sub), manifest_sha256=msha, hashes_sha256=sha256_file(sd / f"hashes_{split}.csv"),
                      mini=True)
    (sd / "extra_tests.json").write_text(json.dumps(reg, indent=1))


def extension(tmp: Path, mini: Path, mpaths: Path, W: str, sim: list):
    """v2.8: checksum freeze guards + two extra ImageNetV2 variants evaluated in their own
    folders and pooled at report time (fake data)."""
    import numpy as np
    from PIL import Image

    from cim_qat import data as D
    print("\n== v2.8: freeze, then the ImageNetV2 extension variants")
    # 1. freeze everything the real freeze covers (no chunks)
    lines = []
    for top in ("eval", "runs", "calibration", "splits", "audit"):
        for f in sorted((mini / top).rglob("*")):
            if f.is_file() and "chunks" not in f.parts and f.name != "last.pth":
                lines.append(f"{_hash(f)}  {f.relative_to(mini).as_posix()}")
    lines.append(f"{_hash(mini / 'access_log.jsonl')}  access_log.jsonl")
    (mini / "FINAL_SHA256SUMS_selftest.txt").write_text("\n".join(lines) + "\n")
    nlog = len((mini / "access_log.jsonl").read_text().splitlines())
    # 2. guards: no overwrite of a frozen report; a finished TEST row is skipped untouched, nothing logged
    run(["qat_evaluate.py", "report", "--paths", mpaths, "--split", "test", "--final", "--precision", "W8A8",
         "--new", "e2e_A"], expect=1, tail=1)
    rc, out = run(sim + ["--split", "test", "--final", "--model", "pretrained", "--kind", "fp32"], tail=1)
    check("finished TEST row skipped untouched; frozen access log not appended",
          "already finished" in out and len((mini / "access_log.jsonl").read_text().splitlines()) == nlog
          and not (mini / "access_log_ext.jsonl").exists())
    # 3. two fake variants: 1 image per class + planted duplicates
    sinfo = json.loads((mini / "splits" / "splits.json").read_text())       # manifests read directly: no
    val0 = D.read_manifest(mini / "splits" / "manifest_val.csv")[0]          # access-log entry from the test
    t0 = D.read_manifest(mini / "splits" / "manifest_test.csv")[3]
    rng = np.random.default_rng(7)
    roots = {}
    for split, dirname in D.EXT_TESTS.items():
        r = tmp / "v2x" / split / dirname
        for c in range(1000):
            (r / str(c)).mkdir(parents=True)
            Image.fromarray(rng.integers(0, 255, (40, 48, 3), dtype=np.uint8)).save(r / str(c) / f"{split}_{c}.png")
        roots[split] = r
    shutil.copy(D.resolve(sinfo, val0), roots["test_th07"] / "5" / "dup_of_val.JPEG")             # -> excluded
    shutil.copy(D.resolve(sinfo, t0), roots["test_th07"] / str(t0["label"]) / "dup_of_test.jpeg")  # kept, pooled once
    shutil.copy(roots["test_th07"] / "0" / "test_th07_0.png", roots["test_top"] / "0" / "dup_of_th07.png")
    pm = json.loads(Path(mpaths).read_text())
    pm["imagenetv2_extra_roots"] = {k: str(v.parent) for k, v in roots.items()}
    Path(mpaths).write_text(json.dumps(pm, indent=1))
    for split in D.EXT_TESTS:
        run(["qat_prepare_data.py", "add-test-variant", "--paths", mpaths, "--variant", split,
             "--archive", f"fake-{split}.tar", "--archive-sha256", "0" * 64, "--workers", W], tail=2)
    reg = json.loads((mini / "splits" / "extra_tests.json").read_text())
    th, tp = reg.get("test_th07", {}), reg.get("test_top", {})
    check("variant registration: VAL duplicate excluded, TEST and cross-variant duplicates counted, mapping ok",
          th.get("excluded") == 1 and th.get("n") == 1001 and th["content_overlap_before_exclusion"]["test"] == 1
          and tp.get("excluded") == 0 and tp["content_overlap_before_exclusion"].get("test_th07") == 1
          and th["mapping"]["ok"] and tp["mapping"]["ok"], json.dumps({k: (v.get("n"), v.get("excluded"))
                                                                        for k, v in reg.items()}))
    run(["qat_prepare_data.py", "add-test-variant", "--paths", mpaths, "--variant", "test_th07"], expect=1, tail=1)
    check("splits.json untouched by the variants", _hash(mini / "splits" / "splits.json")
          == next(l.split()[0] for l in lines if l.endswith("splits/splits.json")))
    _subset_variant(mini / "splits", "test_th07", [f"{t0['label']}/dup_of_test.jpeg", "0/test_th07_0.png"], 8)
    _subset_variant(mini / "splits", "test_top", ["0/dup_of_th07.png"], 8)
    # 4. guard, evaluation in new folders, reports
    run(sim + ["--split", "test_th07", "--model", "pretrained", "--kind", "fp32"], expect=1, tail=1)
    P = ["--precision", "W8A8"]
    for split in D.EXT_TESTS:
        for m in (["--model", "pretrained", "--kind", "fp32"],
                  ["--model", "pretrained", "--kind", "analog", "--gain-set", "cal"] + P,
                  ["--model", "run:e2e_A:best", "--kind", "analog", "--gain-set", "cal"] + P):
            run(sim + ["--split", split, "--final"] + m, tail=1)
        run(["qat_evaluate.py", "report", "--paths", mpaths, "--split", split, "--final", "--precision", "W8A8",
             "--new", "e2e_A"], tail=1)
    run(["qat_evaluate.py", "v2-integrity", "--paths", mpaths, "--split", "test_th07", "--device", "cpu",
         "--num-workers", "0", "--min-classes", "700"], expect=1, tail=1)       # fake labels: FAIL, own file
    check("v2-integrity of a variant writes its own file", (mini / "eval" / "v2_integrity_test_th07.json").exists())
    run(["qat_evaluate.py", "report-pooled", "--paths", mpaths, "--precision", "W8A8", "--new", "e2e_A", "--final"],
        tail=4)
    rp = json.loads((mini / "eval" / "test_pooled" / "REPORT_test_pooled_W8A8.json").read_text())
    check("pooled report: 16 + 7 + 7 distinct images, 2 duplicates with identical outputs, P1 (post hoc) present",
          rp["n"] == 30 and rp["duplicates"] == 2 and rp["duplicate_label_conflicts"] == 0
          and not any(rp["duplicate_logit_mismatches"].values()) and len(rp["post_hoc"]) == 1
          and len(rp["rows"]) == 3, f"n {rp['n']} dup {rp['duplicates']}")
    ext_log = D.read_access_logs(mini)
    check("variant reads logged (final_eval / prepare / v2_integrity) in the new log only",
          any(e.get("split") == "test_th07" and e.get("purpose") == "final_eval" for e in ext_log)
          and all(e["_log"] != "access_log.jsonl" for e in ext_log if e.get("split") in D.EXT_TESTS)
          and len((mini / "access_log.jsonl").read_text().splitlines()) == nlog)
    # 5. audit: 0 FAIL, X checks present; a changed frozen file must FAIL
    rc, out = run(["qat_audit.py", "--paths", mpaths], expect=0, tail=3)
    check("audit: frozen files unchanged, no leakage, variant rows on the registered manifests",
          "[PASS] X0.FINAL_SHA256SUMS_selftest.txt" in out and "[PASS] X3.test_th07.leakage" in out
          and "[WARN] X3.test_th07.excluded" in out and "[WARN] X4.test_top.overlap" in out
          and "[PASS] X6.test_top.e2e_A_best__analog_W8A8__cal.model" in out)
    # 6. v2.8.1 regression: freezing the extension log itself (runbook 13.6) must not turn its own,
    #    earlier reads into X5 failures; appending a read after that freeze must FAIL (X0 and X5)
    ext_files = [mini / "access_log_ext.jsonl"] + sorted(f for f in (mini / "eval" / "test_th07").rglob("*")
                                                          if f.is_file() and "chunks" not in f.parts)
    (mini / "FINAL_SHA256SUMS_selftest_ext.txt").write_text(
        "".join(f"{_hash(f)}  {f.relative_to(mini).as_posix()}\n" for f in ext_files))
    time.sleep(1.1)
    rc, out = run(["qat_audit.py", "--paths", mpaths], expect=0, tail=2)
    check("audit after freezing the extension log: X5 passes for reads made before that freeze",
          "[PASS] X0.FINAL_SHA256SUMS_selftest_ext.txt" in out and "[PASS] X5.test_th07.log" in out
          and (mini / "access_log_ext2.jsonl").exists())
    elog = mini / "access_log_ext.jsonl"
    keep = elog.read_bytes()
    with open(elog, "a") as f:
        f.write(json.dumps(dict(time_utc="2099-01-01T00:00:00+00:00", split="test_th07", purpose="final_eval")) + "\n")
    rc, out = run(["qat_audit.py", "--paths", mpaths], expect=1, tail=0)
    check("audit FAILs a read appended to a frozen extension log (X0 and X5)",
          "[FAIL] X0.FINAL_SHA256SUMS_selftest_ext.txt" in out and "[FAIL] X5.test_th07.log" in out)
    elog.write_bytes(keep)
    victim = mini / "eval" / "test" / "REPORT_test_W8A8.md"
    keep = victim.read_bytes()
    victim.write_bytes(keep + b"\n")
    rc, out = run(["qat_audit.py", "--paths", mpaths], expect=1, tail=0)
    check("audit FAILs when a frozen file changed", "[FAIL] X0.FINAL_SHA256SUMS_selftest.txt" in out)
    victim.write_bytes(keep)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    W = str(a.workers)
    t_start = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="cim qat selftest "))
    print(f"fake data + work dirs in {tmp!s}")
    fake_data(tmp)
    paths = tmp / "paths.json"
    paths.write_text(json.dumps(dict(
        imagenet_train_roots=[str(tmp / "raw" / "train")], imagenet_val_root=str(tmp / "raw" / "val"),
        imagenet_val_labels_csv=str(tmp / "raw" / "LOC_val_solution.csv"),
        imagenetv2_root=str(tmp / "v2" / "imagenetv2-matched-frequency-format-val"),
        work_root=str(tmp / "work full")), indent=1))

    print("\n== data")
    rc, out = run(["qat_prepare_data.py", "inspect", "--paths", paths], expect=1)
    check("inspect reports the fake val count as a failure (50 000 expected)", "[FAIL] ImageNet val" in out)
    run(["qat_prepare_data.py", "make", "--paths", paths, "--workers", W, "--hash-train"], expect=0)
    run(["qat_prepare_data.py", "make", "--paths", paths], expect=1, tail=2)          # refuses to overwrite
    info = json.loads((tmp / "work full" / "splits" / "splits.json").read_text())
    check("exact accounting from disk", info["accounting"]["identity_holds"]
          and info["accounting"]["train"] == 4000 - 2000 - 200, json.dumps(info["accounting"])[:160])
    mini = tmp / "work mini"
    mpaths = tmp / "paths_mini.json"
    subprocess.run([PY, "tests_qat/mini_splits.py", str(tmp / "work full" / "splits"), str(mini),
                    json.dumps(dict(cal=24, calcheck=8, train=32, val=16, test=16)), str(paths), str(mpaths)],
                   cwd=REPO, check=True)

    print("\n== calibration (mini CAL / CALCHECK)")
    run(["qat_calibrate.py", "scale-check", "--precision", "W8A8"], tail=2)
    run(["qat_calibrate.py", "scale-check", "--precision", "W4A4"], tail=2)
    run(["qat_calibrate.py", "run", "--paths", mpaths, "--precision", "W8A8", "--workers", W], tail=4)
    run(["qat_calibrate.py", "freeze", "--paths", mpaths, "--precision", "W8A8", "--level", "p99.99"], tail=2)
    fg = mini / "calibration" / "W8A8" / "frozen_gains.json"

    print("\n== surrogate vs simulator, mechanics, selection rules")
    run(["test_surrogate_vs_simulator.py", "--device", "cpu", "--frozen-gains", fg, "--paths", mpaths,
         "--n-real", "4", "--n-real-t3", "2", "--workers", W, "--skip-grid", "--force-t4",
         "--out", tmp / "surrogate_validation"], tail=3)
    run(["test_training_mechanics.py", "--device", "cpu", "--paths", mpaths, "--frozen-gains", fg,
         "--batch-size", "4", "--num-workers", "0", "--out", tmp / "mechanics"], tail=3)
    run(["tests_qat/test_selection.py", "--paths", mpaths, "--frozen-gains", fg], tail=7)
    run(["tests_qat/test_device_placement.py"], tail=5)
    run(["tests_qat/test_threads.py"], tail=3, timeout=1800)
    run(["tests_qat/test_stats_merge.py"], tail=3, timeout=1800)

    print("\n== training (smoke, A uninterrupted, B killed + resumed)")
    base = json.loads((REPO / "configs" / "qat_w8a8_kd.json").read_text())
    base.update(paths=str(mpaths), epochs=3, batch_size=8, num_workers=1, val_batch_size=8,
                spot_check_n=4, spot_check_workers=int(W), log_every=1, early_stopping_patience=2)
    cfgs = {}
    for n in ("e2e_A", "e2e_B"):
        cfgs[n] = tmp / f"cfg {n}.json"
        cfgs[n].write_text(json.dumps(dict(base, run_name=n), indent=1))
    run(["qat_train.py", "--config", cfgs["e2e_A"], "--device", "cpu", "--smoke-steps", "3"], tail=4)
    run(["qat_train.py", "--config", cfgs["e2e_A"], "--device", "cpu"], tail=4)
    rd = mini / "runs"
    log_b = open(tmp / "trainB.log", "w")
    pb = subprocess.Popen([PY, "qat_train.py", "--config", str(cfgs["e2e_B"]), "--device", "cpu"], cwd=REPO,
                          stdout=log_b, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 1800:                     # kill after the first epoch-end evaluation
        ev = rd / "e2e_B" / "evals.jsonl"
        if ev.exists() and len([l for l in ev.read_text().splitlines() if '"tag"' in l]) >= 2:
            break
        if pb.poll() is not None:
            break
        time.sleep(2)
    killed = pb.poll() is None
    if killed:
        pb.send_signal(signal.SIGKILL)
        pb.wait()
    check("run B killed mid-training", killed)
    run(["qat_train.py", "--config", cfgs["e2e_B"], "--device", "cpu"], expect=1, tail=1)   # no --resume
    run(["qat_train.py", "--config", cfgs["e2e_B"], "--device", "cpu", "--resume"], tail=3)
    from cim_qat.common import sha256_file
    ha = sha256_file(rd / "e2e_A" / "checkpoints" / "final_model.pth")
    hb = sha256_file(rd / "e2e_B" / "checkpoints" / "final_model.pth")
    check("resumed run == uninterrupted run (final weights bit-identical)", ha == hb, f"{ha[:12]} {hb[:12]}")
    sa = json.loads((rd / "e2e_A" / "summary.json").read_text())
    check("summary: selection on VAL, best = primary, TEST unused",
          sa["selection"]["split"] == "val" and sa["test_split_used"] is False and
          sa["selection"]["primary"]["file"] == "best_model.pth", sa["stopped_reason"])
    run(["qat_train.py", "--config", cfgs["e2e_A"], "--device", "cpu", "--resume"], expect=1, tail=1)  # finished

    print("\n== decision D2: fork of an early-stopped run, then the D2 selection")
    run(["tests_qat/test_fork.py", "--paths", mpaths, "--frozen-gains", fg, "--out-root", rd, "--workers", W],
        tail=9, timeout=3600)
    rc, out = run(["qat_evaluate.py", "select-d2", "--paths", mpaths], tail=3)
    check("D2 selection recorded (tie -> the pilot; same checkpoint file)",
          "D2 -> report w4a4_kd_pilot10ep" in out and "identical file" in out)
    run(["qat_evaluate.py", "select-d2", "--paths", mpaths], expect=1, tail=1)     # made once

    print("\n== TEST guards (all must refuse, nothing read)")
    nlog = len((mini / "access_log.jsonl").read_text().splitlines())
    sim = ["qat_evaluate.py", "sim", "--paths", mpaths, "--workers", W, "--chunk", "4"]
    run(sim + ["--split", "test", "--model", "pretrained", "--kind", "fp32"], expect=1, tail=1)
    run(sim + ["--split", "test", "--final", "--limit", "4", "--model", "pretrained", "--kind", "fp32"], expect=1, tail=1)
    run(sim + ["--split", "test", "--final", "--model", "run:e2e_A_smoke:best", "--kind", "fp32"], expect=1, tail=1)
    check("refused TEST commands left no TEST read in the access log",
          not any('"split": "test"' in l for l in (mini / "access_log.jsonl").read_text().splitlines()[nlog:]))

    print("\n== simulator evaluation matrix (W8A8) on VAL and TEST")
    P = ["--precision", "W8A8"]
    matrix = [["--model", "pretrained", "--kind", "fp32"],
              ["--model", "pretrained", "--kind", "adc_off"] + P,
              ["--model", "pretrained", "--kind", "analog", "--gain-set", "unity"] + P,
              ["--model", "pretrained", "--kind", "analog", "--gain-set", "cal"] + P,
              ["--model", "pretrained", "--kind", "analog", "--gain-set", "hist"] + P,
              ["--model", "old_int8", "--kind", "analog", "--gain-set", "cal"] + P,
              ["--model", "old_int8", "--kind", "legacy_adc"] + P,
              ["--model", "run:e2e_A:best", "--kind", "analog", "--gain-set", "cal"] + P,
              ["--model", "run:e2e_A:final", "--kind", "analog", "--gain-set", "cal"] + P]
    for split in ("val", "test"):
        for m in matrix:
            run(sim + ["--split", split] + (["--final"] if split == "test" else []) + m, tail=1)
    # decision D2 diagnostic rows (VAL): pilot best and fork best -> comparison C10
    for m in ("run:w4a4_kd_pilot10ep:best", "run:w4a4_kd_pilot10ep_full:best"):
        run(sim + ["--split", "val", "--model", m, "--kind", "analog", "--gain-set", "cal"] + P, tail=1)
    # a run summarised by the pre-v2.7 merge is flagged in the report until `restat` re-merges it
    legacy = mini / "eval" / "val" / "pretrained__analog_W8A8__cal"
    sj = json.loads((legacy / "summary.json").read_text())
    sj.pop("analog_summary_method", None)
    (legacy / "summary.json").write_text(json.dumps(sj))
    run(["qat_evaluate.py", "report", "--paths", mpaths, "--split", "val", "--precision", "W8A8",
         "--new", "e2e_A"], tail=0)
    md = (mini / "eval" / "val" / "REPORT_val_W8A8.md").read_text()
    check("report flags (†) util percentiles of a run merged the pre-v2.7 way", "† ADC-utilisation" in md)
    run(["qat_evaluate.py", "restat", "--paths", mpaths, "--split", "val"], tail=3)
    check("restat wrote the re-merged statistics of every VAL analog run",
          (legacy / "analog_summary_uniform.json").exists())
    bt = {d.name: json.loads((d / "summary.json").read_text()).get("blas_threads_per_worker")
          for d in sorted((mini / "eval" / "val").iterdir()) if (d / "summary.json").exists()}
    check("every VAL simulator run recorded 1 BLAS thread per worker", bt and all(v == [1] for v in bt.values()),
          f"{len(bt)} runs")
    run(["qat_evaluate.py", "report", "--paths", mpaths, "--split", "val", "--precision", "W8A8",
         "--new", "e2e_A"], tail=3)
    rv = json.loads((mini / "eval" / "val" / "REPORT_val_W8A8.json").read_text())
    c10 = [c for c in rv["comparisons"] if c["id"] == "C10"]
    check("VAL report: C10 (D2 pilot vs fork) present; identical checkpoints -> 0 discordant images",
          len(c10) == 1 and "missing" not in c10[0] and c10[0]["mcnemar"]["a_only"] == 0
          and c10[0]["mcnemar"]["b_only"] == 0)
    check("VAL report: every analog row uses the re-merged statistics (no †)",
          all(r.get("util_stats_method") == "uniform" for r in rv["rows"] if r["kind"] == "analog")
          and "† ADC-utilisation" not in (mini / "eval" / "val" / "REPORT_val_W8A8.md").read_text())
    run(["qat_evaluate.py", "report", "--paths", mpaths, "--split", "test", "--final", "--precision", "W8A8",
         "--new", "e2e_A"], tail=3)
    rep = json.loads((mini / "eval" / "test" / "REPORT_test_W8A8.json").read_text())
    check("TEST report: identical inputs across all 9 runs, all comparisons present",
          rep["identical_inputs"] and len(rep["rows"]) == 9 and not any("missing" in c for c in rep["comparisons"]),
          f"rows {len(rep['rows'])}")
    # identical-input assertion must fire if an image differs
    victim = mini / "eval" / "test" / "pretrained__adcoff_W8A8"
    import numpy as np
    shutil.copytree(victim, tmp / "victim_backup")
    with np.load(victim / "result.npz") as z:
        d = {k: z[k] for k in z.files}
    d["input_sha"] = d["input_sha"].copy()
    d["input_sha"][0] = b"0" * 64
    np.savez(victim / "result.npz", **d)
    run(["qat_evaluate.py", "report", "--paths", mpaths, "--split", "test", "--final", "--precision", "W8A8"],
        expect=1, tail=2)
    shutil.rmtree(victim)
    shutil.copytree(tmp / "victim_backup", victim)
    run(["qat_evaluate.py", "v2-integrity", "--paths", mpaths, "--device", "cpu", "--num-workers", "0",
         "--min-classes", "700"], expect=1, tail=1)        # fake labels: must FAIL the integrity rule

    print("\n== audit")
    run(["qat_audit.py", "--paths", mpaths], expect=0, tail=4)
    # negative 1: a training config that references ImageNetV2
    bad_cfg = REPO / "configs" / "qat_zz_selftest_leak.json"
    bad_cfg.write_text(json.dumps(dict(base, run_name="leak", notes="select on imagenetv2 please")))
    try:
        rc, out = run(["qat_audit.py", "--paths", mpaths], expect=1, tail=0)
        check("audit FAILs a config referring to ImageNetV2", "[FAIL] L1.qat_zz_selftest_leak.json" in out)
    finally:
        bad_cfg.unlink()
    # negative 2: a training process that read TEST
    al = mini / "access_log.jsonl"
    keep = al.read_text()
    with open(al, "a") as f:
        f.write(json.dumps(dict(time_utc="2026-01-01T00:00:00+00:00", split="test", purpose="train")) + "\n")
    rc, out = run(["qat_audit.py", "--paths", mpaths], expect=1, tail=0)
    check("audit FAILs a training-time TEST read in the access log", "[FAIL] L4.test_purposes" in out)
    al.write_text(keep)
    # negative 3: an eval record of a TEST run made without --final
    er = mini / "eval" / "test" / "pretrained__fp32" / "eval_record.json"
    keep = er.read_text()
    er.write_text(keep.replace('"final": true', '"final": false'))
    rc, out = run(["qat_audit.py", "--paths", mpaths], expect=1, tail=0)
    check("audit FAILs a TEST evaluation not marked --final", "[FAIL] L6.pretrained__fp32" in out)
    er.write_text(keep)
    # negative 4: a VAL eval row of a run replaced by a different split
    ev = rd / "e2e_A" / "evals.jsonl"
    keep = ev.read_text()
    ev.write_text(keep.replace('"split": "val"', '"split": "test"', 1))
    rc, out = run(["qat_audit.py", "--paths", mpaths], expect=1, tail=0)
    check("audit FAILs a run whose selection evaluation was not on VAL", "[FAIL] L2.e2e_A.evals" in out)
    ev.write_text(keep)
    # negative 5: a fork that changed a setting D2 does not allow
    crp = rd / "w4a4_kd_pilot10ep_full" / "config_resolved.json"
    keep = crp.read_text()
    d = json.loads(keep)
    d["fork"]["changed"]["lr"] = [1e-05, 2e-05]
    crp.write_text(json.dumps(d))
    rc, out = run(["qat_audit.py", "--paths", mpaths], expect=1, tail=0)
    check("audit FAILs a fork that changed a non-allowed setting", "[FAIL] L8.w4a4_kd_pilot10ep_full.changes" in out)
    crp.write_text(keep)
    run(["qat_audit.py", "--paths", mpaths], expect=0, tail=1)

    extension(tmp, mini, mpaths, W, sim)

    n_ok = sum(ok for _, ok in RES)
    print(f"\nSELFTEST: {n_ok}/{len(RES)} passed in {(time.time() - t_start) / 60:.1f} min "
          f"({'kept ' + str(tmp) if a.keep else 'temporary files removed'})")
    if not a.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(0 if n_ok == len(RES) else 1)


if __name__ == "__main__":
    main()
