#!/usr/bin/env python3
"""
tools_qat/val_overlap_sensitivity.py — VAL results with and without the VAL images whose
content also occurs in TRAIN (audit S4.info.train|val). READ-ONLY: reads the split hash files
and the finished VAL simulator results; reads no image and no TEST file; writes
<work_root>/eval/val/SENSITIVITY_train_overlap.{md,json}.

  python tools_qat/val_overlap_sensitivity.py --paths configs/paths.json [--new-w4 RUN] [--new-w8 RUN]
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
    ap.add_argument("--new-w8", default="w8a8_kd_pilot5ep")
    ap.add_argument("--new-w4", default="w4a4_kd_pilot10ep_full")
    a = ap.parse_args()
    from cim_qat import metrics as M
    from cim_qat.common import now_iso, write_json
    from cim_qat.data import DataPaths, read_hashes
    dp = DataPaths.load(a.paths)
    sd = Path(dp.splits_dir)
    hv_idx = [int(l.split(",", 1)[0]) for l in open(sd / "hashes_val.csv").read().splitlines()[1:]]
    hv = read_hashes(sd / "hashes_val.csv")
    htrain = set(read_hashes(sd / "hashes_train.csv"))
    sha_of = dict(zip(hv_idx, hv))
    base = Path(dp.work_root) / "eval" / "val"
    ref = np.load(base / "pretrained__fp32" / "result.npz")
    idx, ref_logits = ref["idx"], ref["logits"]
    labels = np.load(base / "pretrained__fp32" / "labels.npy")
    overlap = np.array([sha_of[int(i)] in htrain for i in idx])
    n_ov = int(overlap.sum())
    print(f"VAL images {len(idx)}; content also in TRAIN: {n_ov} ({100 * n_ov / len(idx):.2f} %)")
    keep = ~overlap
    runs = {}
    for d in sorted(base.iterdir()):
        if (d / "result.npz").exists() and (d.name.endswith("__fp32") or "_W8A8" in d.name or "_W4A4" in d.name):
            z = np.load(d / "result.npz")
            if not np.array_equal(z["idx"], idx):
                sys.exit(f"{d.name}: image order differs from pretrained__fp32; refusing")
            runs[d.name] = M.per_image(z["logits"], labels, ref_logits)
    sub = lambda pi, m: {k: v[m] for k, v in pi.items()}   # noqa: E731
    rows = []
    for t, pi in runs.items():
        a_all, a_cl, a_ov = (100 * pi["correct1"].mean(), 100 * pi["correct1"][keep].mean(),
                             100 * pi["correct1"][overlap].mean())
        rows.append(dict(run=t, top1_all=a_all, top1_without_overlap=a_cl, top1_overlap_only=a_ov))
    P = {"W8A8": ("old_int8", a.new_w8), "W4A4": ("old_int4", a.new_w4)}
    comps = []
    for prec, (old, new) in P.items():
        pairs = [("C3", f"pretrained__adcoff_{prec}", f"pretrained__analog_{prec}__cal"),
                 ("C5", f"pretrained__analog_{prec}__cal", f"{old}__analog_{prec}__cal"),
                 ("C6", f"pretrained__analog_{prec}__cal", f"{new}_best__analog_{prec}__cal"),
                 ("C7", f"{old}__analog_{prec}__cal", f"{new}_best__analog_{prec}__cal")]
        if prec == "W4A4":
            pairs.append(("C10", f"w4a4_kd_pilot10ep_best__analog_{prec}__cal", f"{new}_best__analog_{prec}__cal"))
        for cid, ta, tb in pairs:
            if ta in runs and tb in runs:
                full = M.compare(runs[ta], runs[tb])["top1_diff_pct"]
                cl = M.compare(sub(runs[ta], keep), sub(runs[tb], keep))["top1_diff_pct"]
                comps.append(dict(precision=prec, id=cid, a=ta, b=tb, all=full, without_overlap=cl))
    rep = dict(time_utc=now_iso(), n_val=int(len(idx)), n_overlap=n_ov, rows=rows, comparisons=comps,
               note="overlap = VAL images whose SHA-256 content hash equals a TRAIN image's (hashes_val/train.csv)")
    write_json(base / "SENSITIVITY_train_overlap.json", rep)
    L = [f"# VAL sensitivity: without the {n_ov} VAL images whose content is also in TRAIN\n",
         f"VAL n = {len(idx)}; overlap n = {n_ov}; remaining n = {int(keep.sum())}. Read-only; {rep['time_utc']}.\n",
         "| run | top-1 % all | top-1 % without overlap | top-1 % on the overlap images |", "|---|---|---|---|"]
    L += [f"| {r['run']} | {r['top1_all']:.2f} | {r['top1_without_overlap']:.2f} | {r['top1_overlap_only']:.2f} |" for r in rows]
    L += ["\n| precision | id | a | b | Δtop-1 pp all (95% CI) | Δtop-1 pp without overlap (95% CI) |", "|---|---|---|---|---|---|"]
    for c in comps:
        f = lambda t: f"{t['mean']:+.2f} ({t['ci95'][0]:+.2f}, {t['ci95'][1]:+.2f})"   # noqa: E731
        L.append(f"| {c['precision']} | {c['id']} | {c['a']} | {c['b']} | {f(c['all'])} | {f(c['without_overlap'])} |")
    (base / "SENSITIVITY_train_overlap.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n-> {base / 'SENSITIVITY_train_overlap.md'}")


if __name__ == "__main__":
    main()
