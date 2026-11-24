#!/usr/bin/env python3
"""
tools_qat/pair_compare.py — one paired comparison between two finished simulator runs, with the
reports' own statistics (paired bootstrap 95% CI, exact McNemar, dNLL, dKL). READ-ONLY.

  python tools_qat/pair_compare.py --paths configs/paths.json --split test --a pretrained__fp32 \
         --b w4a4_kd_pilot10ep_full_best__analog_W4A4__cal

Any comparison chosen after the TEST reports were seen is POST HOC: label it so in the thesis.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", required=True)
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--a", required=True, help="run folder name (reference)")
    ap.add_argument("--b", required=True, help="run folder name (compared)")
    a = ap.parse_args()
    from cim_qat import metrics as M
    from cim_qat.data import DataPaths
    base = Path(DataPaths.load(a.paths).work_root) / "eval" / a.split
    ref = np.load(base / "pretrained__fp32" / "result.npz")
    lab = np.load(base / "pretrained__fp32" / "labels.npy")
    pis = {}
    for t in (a.a, a.b):
        z = np.load(base / t / "result.npz")
        if not (np.array_equal(z["idx"], ref["idx"]) and np.array_equal(z["input_sha"], ref["input_sha"])):
            sys.exit(f"{t}: not on the same images as pretrained__fp32; refusing")
        pis[t] = M.per_image(z["logits"], lab, ref["logits"])
    c = M.compare(pis[a.a], pis[a.b])
    t, n, k, m = c["top1_diff_pct"], c["nll_diff"], c["kl_ref_diff"], c["mcnemar"]
    print(f"{a.split}: b = {a.b}\n      a = {a.a}\n      n = {t['n']} images (b - a)")
    print(f"top-1 a {100 * pis[a.a]['correct1'].mean():.2f} %  b {100 * pis[a.b]['correct1'].mean():.2f} %")
    print(f"dtop-1 {t['mean']:+.3f} pp  95% CI ({t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f})  SE {t['se']:.3f}")
    print(f"McNemar exact p {m['p']:.3g}  (a right only {m['a_only']}, b right only {m['b_only']})")
    print(f"dNLL {n['mean']:+.4f} ({n['ci95'][0]:+.4f}, {n['ci95'][1]:+.4f});  "
          f"dKL(FP32||run) {k['mean']:+.4f} ({k['ci95'][0]:+.4f}, {k['ci95'][1]:+.4f})")


if __name__ == "__main__":
    main()
