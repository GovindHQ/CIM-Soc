#!/usr/bin/env python3
"""
qat_eval_matrix.py — run the pre-declared evaluation matrix (QAT_DESIGN.md §10)
for one precision and one split, by calling `qat_evaluate.py sim` once per row
(each row resumable; finished rows are skipped by the simulator's chunk cache).

  python qat_eval_matrix.py --paths configs/paths.json --precision W8A8 --split val \
         --new w8a8_kd_pilot5ep --workers 24 [--dry-run]
  python qat_eval_matrix.py --paths configs/paths.json --precision W8A8 --split test --final \
         --new w8a8_kd_pilot5ep --workers 24
  (v2.8) --split test_th07 / test_top: the extension variants, same rows, own folders.
  Finished rows are skipped without touching them (qat_evaluate.py sim, v2.8).

Rows (P = precision, OLD = old_int8 for W8A8 / old_int4 for W4A4):
  E1  pretrained  fp32
  E2  pretrained  adc_off  P              operand quantization only
  E3  pretrained  analog   P  unity       PTQ, G = 1
  E4  pretrained  analog   P  cal         PTQ, calibrated gains   (baseline)
  E4b pretrained  analog   W8A8 hist      PTQ, historical gains (W8A8 only)
  E5  OLD         analog   P  cal         old QAT, as delivered
  E5b OLD         legacy_adc P            old QAT at its training point (diagnostic)
  E6  run:<new>:best / run:<new>:final  analog P cal   new QAT primary / secondary
  E7  run:<ce>:best / final (optional, --new-ce)
--skip E3,E5b etc. drops rows (state why in the experiment log).
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# every `qat_evaluate.py sim` started below inherits these: one BLAS thread per
# simulator process (parallelism is by processes), and the job tag used by qat_stop.py
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("CIM_QAT_JOB", f"eval:{os.getpid()}")


def rows(P, new, new_ce):
    old = "old_int8" if P == "W8A8" else "old_int4"
    p = ["--precision", P]
    R = [("E1", ["--model", "pretrained", "--kind", "fp32"]),
         ("E2", ["--model", "pretrained", "--kind", "adc_off"] + p),
         ("E3", ["--model", "pretrained", "--kind", "analog", "--gain-set", "unity"] + p),
         ("E4", ["--model", "pretrained", "--kind", "analog", "--gain-set", "cal"] + p)]
    if P == "W8A8":
        R.append(("E4b", ["--model", "pretrained", "--kind", "analog", "--gain-set", "hist"] + p))
    R += [("E5", ["--model", old, "--kind", "analog", "--gain-set", "cal"] + p),
          ("E5b", ["--model", old, "--kind", "legacy_adc"] + p)]
    for n in new:
        R += [("E6", ["--model", f"run:{n}:best", "--kind", "analog", "--gain-set", "cal"] + p),
              ("E6s", ["--model", f"run:{n}:final", "--kind", "analog", "--gain-set", "cal"] + p)]
    for n in new_ce:
        R += [("E7", ["--model", f"run:{n}:best", "--kind", "analog", "--gain-set", "cal"] + p),
              ("E7s", ["--model", f"run:{n}:final", "--kind", "analog", "--gain-set", "cal"] + p)]
    return R


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", required=True)
    ap.add_argument("--precision", required=True, choices=["W8A8", "W4A4"])
    ap.add_argument("--split", required=True, choices=["val", "test", "test_th07", "test_top"],
                    help="test = ImageNetV2 matched-frequency; test_th07 / test_top = the v2.8 extension variants")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--new", nargs="*", default=[])
    ap.add_argument("--new-ce", nargs="*", default=[])
    ap.add_argument("--skip", default="", help="comma-separated row ids to skip, e.g. E3,E5b")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.split.startswith("test") and not a.final:
        sys.exit("--split test* is a final ImageNetV2 evaluation: pass --final")
    skip = {s.strip() for s in a.skip.split(",") if s.strip()}
    here = Path(__file__).resolve().parent
    base = [sys.executable, str(here / "qat_evaluate.py"), "sim", "--paths", a.paths, "--split", a.split,
            "--workers", str(a.workers)] + (["--final"] if a.final else [])
    failed = []
    for rid, args in rows(a.precision, a.new, a.new_ce):
        if rid in skip:
            print(f"--- {rid} SKIPPED (--skip)")
            continue
        cmd = base + args
        print(f"--- {rid}: " + " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
        if a.dry_run:
            continue
        rc = subprocess.run(cmd).returncode
        if rc:
            failed.append(rid)
            print(f"!!! {rid} failed (exit {rc}); continuing with the next row")
    if failed:
        sys.exit(f"rows failed: {failed}")


if __name__ == "__main__":
    main()
