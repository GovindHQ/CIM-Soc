#!/usr/bin/env python3
"""
qat_prepare_data.py — dataset verification and the fixed QAT splits.

  # 1. read-only inspection: paths, counts, labels, ImageNetV2 structure, work root
  python qat_prepare_data.py inspect --paths configs/paths.json

  # 2. fix the splits ONCE (before any calibration/training)
  python qat_prepare_data.py make --paths configs/paths.json --workers 16 --hash-train

  # (only if `make` FAILED and nothing has used its output: archive it and make again)
  python qat_prepare_data.py make --paths configs/paths.json --workers 16 --hash-train --replace-failed

  # (only if ImageNetV2 was not available at `make` time)
  python qat_prepare_data.py add-test --paths configs/paths.json --workers 16

  # (v2.8 extension) register one more ImageNetV2 variant as its own guarded TEST split,
  # without touching splits.json or any existing file (QAT_DESIGN.md §0.4 D3)
  python qat_prepare_data.py add-test-variant --paths configs/paths.json --variant test_th07 \
         --archive imagenetv2-threshold0.7.tar.gz --archive-sha256 <sha> --workers 16

  # print the stored split summary again
  python qat_prepare_data.py show --paths configs/paths.json

`make` writes <work_root>/splits/: manifest_{cal,calcheck,train,val,test}.csv,
hashes_*.csv (SHA-256 per file), train_excluded.csv (if any), splits.json
(seed, exact counts and their accounting, manifest hashes, path- and
content-level disjointness, ImageNetV2 class-mapping checks).
It never prints or computes any accuracy. Exit code 1 if a check fails.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cim_qat.common import read_json  # noqa: E402
from cim_qat import data as D  # noqa: E402


def _gb(n):
    return f"{n / 1e9:.1f} GB"


def cmd_inspect(a) -> int:
    raw = read_json(a.paths)
    print(f"paths file: {Path(a.paths).resolve()}")
    print(json.dumps(raw, indent=1))
    bad = 0
    # work root: exists or creatable, writable, free space
    wr = Path(raw["work_root"])
    probe = wr if wr.exists() else next((p for p in wr.parents if p.exists()), Path("/"))
    try:
        if not wr.exists():
            print(f"[INFO] work_root does not exist yet; parent {probe} "
                  f"{'is writable' if os.access(probe, os.W_OK) else 'is NOT writable'}")
        free = shutil.disk_usage(probe).free
        ok_w = os.access(probe, os.W_OK)
        print(f"[{'PASS' if ok_w and free > 50e9 else 'FAIL'}] work_root {wr}: writable={ok_w}, free {_gb(free)} "
              "(need >= 50 GB: hash caches, eval chunks, checkpoints)")
        bad += not (ok_w and free > 50e9)
    except OSError as e:
        print(f"[FAIL] work_root {wr}: {e}")
        bad += 1
    roots = raw.get("imagenet_train_roots") or [raw.get("imagenet_train_root")]
    dp = D.DataPaths(train_roots=[str(r) for r in roots], val_root=str(raw.get("imagenet_val_root")),
                     val_labels_csv=raw.get("imagenet_val_labels_csv"),
                     imagenetv2_root=raw.get("imagenetv2_root"), work_root=str(raw["work_root"]))
    missing = [p for p in dp.train_roots + [dp.val_root] if not Path(p).is_dir()]
    if dp.val_labels_csv and not Path(dp.val_labels_csv).is_file():
        missing.append(dp.val_labels_csv)
    for p in missing:
        print(f"[FAIL] not found: {p!r}")
    if missing:
        return 1
    if dp.imagenetv2_root and not Path(dp.imagenetv2_root).is_dir():
        print(f"[FAIL] imagenetv2_root not found: {dp.imagenetv2_root!r} (download it: runbook stage 5.1)")
        bad += 1
        dp.imagenetv2_root = None
    tr, tdiag = D.list_train(dp.train_roots)
    ok = len(tr) == 1000 and tdiag["per_class_min"] > 3
    print(f"[{'PASS' if ok else 'FAIL'}] ImageNet train: {tdiag['files']:,} image files in 1000 class folders "
          f"(per class {tdiag['per_class_min']}..{tdiag['per_class_max']}), symlinks {tdiag['symlinks']:,}, "
          f"non-image files {tdiag['non_image_files']}, non-synset folders {len(tdiag['extra_folders'])}")
    if tdiag["files"] != 1281167:
        print(f"  NOTE: {tdiag['files']:,} != 1,281,167 (the official count); the splits use the count on disk.")
    bad += not ok
    va, vdiag = D.list_val(dp.val_root, dp.val_labels_csv)
    ok = vdiag["files"] == 50000 and vdiag["per_class_min"] == vdiag["per_class_max"] == 50
    print(f"[{'PASS' if ok else 'FAIL'}] ImageNet val ({vdiag['layout']}): {vdiag['files']:,} images"
          + (f", CSV rows {vdiag['csv_rows']:,}" if 'csv_rows' in vdiag else "")
          + f", every image labelled exactly once, per class {vdiag['per_class_min']}..{vdiag['per_class_max']}, "
          f"symlinks {vdiag['symlinks']:,}")
    bad += not ok
    if dp.imagenetv2_root:
        root = D.find_v2_root(dp.imagenetv2_root)
        if root is None or Path(root) != Path(dp.imagenetv2_root):
            print(f"[FAIL] imagenetv2_root {dp.imagenetv2_root} is not the class-index folder"
                  + (f"; use {root}" if root else ""))
            bad += 1
        else:
            try:
                v2, d2 = D.list_v2(root)
                ok = d2["files"] == 10000 and d2["per_class_min"] == d2["per_class_max"] == 10
                print(f"[{'PASS' if ok else 'FAIL'}] ImageNetV2 matched-frequency: {d2['files']:,} images in folders "
                      f"0..999 (per class {d2['per_class_min']}..{d2['per_class_max']}), symlinks {d2['symlinks']}, "
                      f"non-image files {d2['non_image_files']}; label = int(folder name)")
                bad += not ok
            except RuntimeError as e:
                print(f"[FAIL] {e}")
                bad += 1
    elif not raw.get("imagenetv2_root"):
        print("[INFO] imagenetv2_root not set: TEST cannot be prepared yet (QAT_RUNBOOK.md stage 5.1)")
    print("\nNo files were written." if not bad else f"\n{bad} check(s) failed.")
    return 1 if bad else 0


def summarize(info) -> int:
    rep = info["disjointness"]
    acc = info["accounting"]
    print(json.dumps(dict(counts=info["counts"], accounting=acc, size_check=info["size_check"],
                          content_between=rep["between"], content_within=rep["within"],
                          content_not_checked=rep["not_checked"], read_errors=rep["errors"],
                          path_overlap=rep["path_overlap"], v2_mapping=info.get("v2_mapping"),
                          train_hashed=rep["train_hashed"]), indent=1))
    bad_size = {k: v for k, v in info["size_check"].items() if not v["ok"]}
    if bad_size:
        print(f"WARNING: split sizes differ from the plan: {bad_size}")
    if not rep["train_hashed"]:
        print("WARNING: TRAIN was not content-hashed: TRAIN-vs-CAL/CALCHECK/VAL/TEST content duplicates "
              "are NOT checked (path-level disjointness is). Use --hash-train (QAT_DESIGN.md §8).")
    for k in ("cal|val", "calcheck|val", "train|val", "train|test", "val|test"):
        if rep["between"].get(k):
            print(f"NOTE: {rep['between'][k]} identical files between {k} (examples {rep['examples'].get(k)})")
    sk = info.get("heldout_skipped", [])
    print(f"held-out eligibility rule: {info.get('heldout_eligibility_rule', 'none (made before v2.1)')}; "
          f"{len(sk)} candidate(s) skipped" + "".join(
              f"\n  class {x['class_index']} {x['root']}/{x['rel_path']} (would have been "
              f"{x['would_have_been'].upper()}): {x['reason']}" for x in sk))
    if not info.get("test_available"):
        print("NOTE: TEST (ImageNetV2) not prepared yet; run `add-test` after downloading it.")
    if not rep["ok"]:
        print("FAIL: a critical check failed (see splits.json['disjointness'], ['accounting'], "
              "['v2_mapping']). Do not proceed.")
        return 1
    print(f"OK: TRAIN = {acc['train_files_on_disk']:,} - {acc['cal']:,} - {acc['calcheck']:,} - "
          f"{acc['train_excluded_content_duplicates']:,} = {acc['train']:,}; VAL = {acc['val']:,}; "
          f"TEST = {acc['test']:,}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("inspect", "make", "add-test", "add-test-variant", "show"):
        p = sub.add_parser(name)
        p.add_argument("--paths", required=True)
        if name in ("make", "add-test", "add-test-variant"):
            p.add_argument("--workers", type=int, default=8)
        if name == "add-test-variant":
            p.add_argument("--variant", required=True, choices=sorted(D.EXT_TESTS))
            p.add_argument("--archive", default="", help="file name of the downloaded archive (recorded)")
            p.add_argument("--archive-sha256", default="", help="its SHA-256 (recorded; none is published)")
        if name == "make":
            p.add_argument("--seed", type=int, default=D.DEFAULT_SPLIT_SEED)
            p.add_argument("--hash-train", action="store_true",
                           help="SHA-256 every TRAIN file (~1.27M, I/O heavy, resumable) so that content "
                                "duplicates of CAL/CALCHECK are excluded and TRAIN-vs-VAL/TEST duplicates counted")
            p.add_argument("--force", action="store_true")
            p.add_argument("--replace-failed", action="store_true",
                           help="move the outputs of a FAILED make (never used: checked against the access "
                                "log) to splits/failed_<time>/ and make again, reusing the hash caches")
    a = ap.parse_args()
    if a.cmd == "inspect":
        sys.exit(cmd_inspect(a))
    dp = D.DataPaths.load(a.paths)
    if a.cmd == "make":
        Path(dp.work_root).mkdir(parents=True, exist_ok=True)
        if a.replace_failed:
            dst = D.archive_failed(dp)
            print(f"previous FAILED splits moved to {dst} (hash caches kept and reused)")
        info = D.prepare(dp, a.seed, a.workers, a.hash_train, a.force)
    elif a.cmd == "add-test":
        info = D.add_test(dp, a.workers)
    elif a.cmd == "add-test-variant":
        e = D.add_test_variant(dp, a.variant, a.workers, a.archive_sha256, a.archive)
        print(json.dumps({k: v for k, v in e.items() if k != "rule"}, indent=1))
        print(f"rule: {e['rule']}")
        ov = e["content_overlap_before_exclusion"]
        print(f"{'OK' if e['ok'] else 'FAIL'}: {a.variant} = {e['n']:,} images "
              f"({e['files_on_disk']:,} on disk - {e['excluded']} excluded: content equal to "
              f"CAL {ov['cal']}, CALCHECK {ov['calcheck']}, VAL {ov['val']}, TRAIN {ov['train']}); "
              f"shared content with matched-frequency TEST {ov['test']}"
              + "".join(f", with {k} {v}" for k, v in ov.items() if k in D.EXT_TESTS)
              + f"; label = int(folder) {e['mapping']['label_is_int_folder']}, "
              f"{e['mapping']['classes']} classes, {e['mapping']['per_class_min']}-{e['mapping']['per_class_max']} per class")
        sys.exit(0 if e["ok"] else 1)
    else:
        info = read_json(dp.splits_dir / "splits.json")
    rc = summarize(info)
    if rc == 0:
        print(f"splits: {dp.splits_dir}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
