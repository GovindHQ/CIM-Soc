#!/usr/bin/env python3
"""
tools_qat/imagenette_percentile_check.py RESULTS_DIR [--keep-frac 2e-4] [--cap 400000]

READ-ONLY check of the older Imagenette experiments (run_experiment.py / cim_exp): which jobs'
percentile columns (ADC util p99.9 / median, V_OA / V_ADC_IN percentiles) went through the
defective AnalogStats.merge() thinning (QAT_DESIGN.md §13, v2.7). Exact columns (accuracy,
clip counts, util max) are unaffected in every job.

Per operation, with n conversions, keep fraction kf and cap C:
  * every recorded call keeps k = max(1, int(n_call*kf)) >= n_call*kf/2 samples, so an
    un-thinned run keeps at least n*kf/2 samples: if n*kf/2 > C, thinning certainly happened
    -> AFFECTED;
  * once thinning happens the merged subsample never drops below ~C/2: a final n_samples < C/2
    means it was never thinned -> unaffected;
  * otherwise -> UNCERTAIN (treat as affected).
"""
import argparse
import json
import sys
from pathlib import Path


def verdict(n, n_samples, kf, cap):
    if n_samples < cap / 2:
        return "unaffected"
    if n * kf / 2 > cap:
        return "AFFECTED"
    return "UNCERTAIN"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--keep-frac", type=float, default=2e-4, help="stats_keep_frac the jobs ran with")
    ap.add_argument("--cap", type=int, default=400_000, help="stats_cap the jobs ran with")
    a = ap.parse_args()
    files = sorted(Path(a.results_dir).rglob("final.json"))
    if not files:
        sys.exit(f"no final.json under {a.results_dir}")
    count = {}
    for f in files:
        d = json.loads(f.read_text())
        kf, cap = d.get("stats_keep_frac", a.keep_frac), d.get("stats_cap", a.cap)
        summ = d.get("analog_summary") or {}
        if not summ:
            continue
        v = {op: verdict(s["n"], s.get("n_samples", 0), kf, cap) for op, s in summ.items()}
        worst = "AFFECTED" if "AFFECTED" in v.values() else "UNCERTAIN" if "UNCERTAIN" in v.values() else "unaffected"
        count[worst] = count.get(worst, 0) + 1
        print(f"{worst:10s}  " + "  ".join(f"{op}: n {s['n']:.3g} samples {s.get('n_samples', 0)}"
                                        for op, s in summ.items()) + f"  {f.parent.name}")
    print("\n" + ", ".join(f"{k}: {v}" for k, v in sorted(count.items())) + f"  (of {len(files)} final.json)")


if __name__ == "__main__":
    main()
