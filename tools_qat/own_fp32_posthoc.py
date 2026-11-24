#!/usr/bin/env python3
"""
tools_qat/own_fp32_posthoc.py  --  POST HOC, descriptive, READ-ONLY on every existing result.

Uses only finished result.npz files (stored logits). It reads no image, no split manifest, runs no
simulator and changes no existing file. The only thing it writes is a NEW folder
<work_root>/eval/posthoc_own_fp32/ (refuses to overwrite a frozen file).

Two commands:

  check     verify that result.npz holds `logits_own_fp32` for the rows needed, that every row is on the
            same images as pretrained__fp32 (idx, input hash, labels), and two determinism sanity checks.
  compute   per test set (matched-frequency, threshold-0.7, top-images) and POOLED (each distinct image
            once, same rule as report-pooled), for W8A8 and W4A4, new QAT (primary) and old QAT:

              A = the QAT checkpoint in plain FP32          ("QAT FP32", = own-FP32)
              C = the same checkpoint on the simulator       ("QAT-CIM", calibrated gains)
              P = pretrained FP32                            ("Pretrained FP32")
              Q = pretrained on the simulator, calibrated    ("Pretrained-CIM")

            All differences are  second minus first,  in top-1 points, paired on the same images:
              Case 4 : P -> Q      cost of the hardware, no QAT
              Case 1 : Q -> C      what QAT recovers on the hardware
              Case 2 : P -> A      what QAT training did to the model (no hardware)
              Case 3 : A -> C      what the hardware does to the QAT model
              Case 5 : P -> C      final gap to the original model        (= Case 2 + Case 3, exactly)

Also compares the pooled top-1 of P, Q and C with the numbers in the existing REPORT_test_pooled_*.json
(self-check that the same images and the same rule were used).

  python tools_qat/own_fp32_posthoc.py check   --paths configs/paths.json
  python tools_qat/own_fp32_posthoc.py compute --paths configs/paths.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SPLITS = ("test", "test_th07", "test_top")          # = cim_qat.data.POOL_ORDER (dedupe order)
LABEL = {"test": "matched-frequency 10k", "test_th07": "threshold-0.7 10k", "test_top": "top-images 10k"}
PREC = {"W8A8": dict(old="old_int8", new="w8a8_kd_pilot5ep"),
        "W4A4": dict(old="old_int4", new="w4a4_kd_pilot10ep_full")}
OUT_NAME = "posthoc_own_fp32"


def runs_for(P):
    c = PREC[P]
    return dict(fp32="pretrained__fp32",
                pre_cim=f"pretrained__analog_{P}__cal",
                new=f"{c['new']}_best__analog_{P}__cal",
                old=f"{c['old']}__analog_{P}__cal",
                old_legacy=f"{c['old']}__legacyadc_{P}")


def work_root(paths):
    return Path(json.load(open(paths))["work_root"])


def load(W, s, run):
    f = W / "eval" / s / run / "result.npz"
    if not f.exists():
        raise FileNotFoundError(f)
    return np.load(f)


# ------------------------------------------------------------------------------------------ check
def cmd_check(a):
    W = work_root(a.paths)
    ok_all = True
    print(f"work_root {W}\n")
    for s in SPLITS:
        base = W / "eval" / s
        ref = load(W, s, "pretrained__fp32")
        lab = np.load(base / "pretrained__fp32" / "labels.npy")
        print(f"== {s}: n = {len(ref['idx'])} images, pretrained__fp32 keys {sorted(ref.files)}")
        for P in PREC:
            R = runs_for(P)
            for key in ("pre_cim", "new", "old", "old_legacy"):
                run = R[key]
                try:
                    z = load(W, s, run)
                except FileNotFoundError:
                    print(f"  [FAIL] {s}/{run}: result.npz missing"); ok_all = False; continue
                has = "logits_own_fp32" in z.files
                same = (np.array_equal(z["idx"], ref["idx"]) and np.array_equal(z["input_sha"], ref["input_sha"])
                        and np.array_equal(np.load(base / run / "labels.npy"), lab))
                shp = z["logits_own_fp32"].shape if has else None
                good = has and same and shp == (len(ref["idx"]), 1000)
                ok_all &= good
                print(f"  [{'PASS' if good else 'FAIL'}] {run}: logits_own_fp32 {'present ' + str(shp) if has else 'MISSING'}; "
                      f"same images/inputs/labels as pretrained__fp32: {same}")
            # determinism sanity checks
            z = load(W, s, R["pre_cim"])
            if "logits_own_fp32" in z.files:
                d = float(np.abs(z["logits_own_fp32"] - ref["logits"]).max())
                print(f"  [{'PASS' if d == 0 else 'NOTE'}] {P}: own-FP32 of the pretrained CIM row vs pretrained__fp32 logits, "
                      f"max |diff| = {d:.3g}")
            zo, zl = load(W, s, R["old"]), load(W, s, R["old_legacy"])
            if "logits_own_fp32" in zo.files and "logits_own_fp32" in zl.files:
                d = float(np.abs(zo["logits_own_fp32"] - zl["logits_own_fp32"]).max())
                print(f"  [{'PASS' if d == 0 else 'NOTE'}] {P}: old QAT own-FP32 in its analog row vs its legacy-ADC row, "
                      f"max |diff| = {d:.3g}")
        print()
    print("OVERALL:", "PASS - the stored logits are enough for the post hoc analysis" if ok_all
          else "FAIL - see the lines above; the analysis would need those rows re-run")
    return 0 if ok_all else 1


# ------------------------------------------------------------------------------------------ compute
def dedupe(W, ref_idx):
    """Same rule as qat_evaluate.cmd_report_pooled: each distinct file content (SHA-256) once, first
    occurrence in the order test, test_th07, test_top."""
    sd = W / "splits"
    content = {}
    for s in SPLITS:
        rows = open(sd / f"hashes_{s}.csv").read().splitlines()[1:]
        h_of = {int(l.split(",", 1)[0]): l.split(",", 1)[1] for l in rows}
        content[s] = [h_of[int(i)] for i in ref_idx[s]]
    first, sel = {}, {s: [] for s in SPLITS}
    for s in SPLITS:
        for k, h in enumerate(content[s]):
            if h not in first:
                first[h] = (s, k)
                sel[s].append(k)
    return sel


def fmt_cmp(c):
    t = c["top1_diff_pct"]
    return f"{t['mean']:+.2f} ({t['ci95'][0]:+.2f}, {t['ci95'][1]:+.2f})"


def cmd_compute(a):
    from cim_qat import metrics as M
    from cim_qat.data import assert_not_frozen
    W = work_root(a.paths)
    out_dir = W / "eval" / OUT_NAME
    outs = [out_dir / "own_fp32_posthoc.json", out_dir / "own_fp32_posthoc.md"]
    assert_not_frozen(str(W), *map(str, outs))

    # inputs per split
    data, ref_idx = {}, {}
    for s in SPLITS:
        ref = load(W, s, "pretrained__fp32")
        ref_idx[s] = ref["idx"]
        data[s] = dict(lab=np.load(W / "eval" / s / "pretrained__fp32" / "labels.npy"), fp32=ref["logits"], z={})
    sel = dedupe(W, ref_idx)

    result, md = {}, []
    md.append("# POST HOC: own-FP32 and the case decomposition (stored logits only; descriptive)\n")
    md.append("All differences are second minus first, top-1 points, paired on the same images, 95% paired-bootstrap "
              "CI (cim_qat.metrics, seed 0, 2000 resamples). Pooled = each distinct image once (first occurrence in the "
              "order matched-frequency, threshold-0.7, top-images). Chosen after the TEST results were seen.\n")
    for P in PREC:
        R = runs_for(P)
        result[P] = {}
        # logits of every needed row per split
        L = {}
        for s in SPLITS:
            L[s] = dict(P=data[s]["fp32"], Q=load(W, s, R["pre_cim"])["logits"])
            for fam in ("new", "old"):
                z = load(W, s, R[fam])
                if "logits_own_fp32" not in z.files:
                    sys.exit(f"{s}/{R[fam]}: logits_own_fp32 missing; run `check` first")
                L[s][fam + "_C"] = z["logits"]
                L[s][fam + "_A"] = z["logits_own_fp32"]
        # sets: each split and pooled
        sets = {s: (L[s], data[s]["lab"], data[s]["fp32"]) for s in SPLITS}
        pooled = {k: np.concatenate([L[s][k][sel[s]] for s in SPLITS]) for k in L["test"]}
        plab = np.concatenate([data[s]["lab"][sel[s]] for s in SPLITS])
        sets["pooled"] = (pooled, plab, pooled["P"])
        n_pool = len(plab)

        md.append(f"\n## {P}  (pooled n = {n_pool:,} distinct images)\n")
        for fam, famname in (("new", "NEW QAT (primary)"), ("old", "OLD QAT (supporting)")):
            rows_top1, rows_cmp = [], []
            for name, (lg, lab, fp) in sets.items():
                pi = {k: M.per_image(v, lab, fp) for k, v in lg.items()}
                top1 = {k: 100 * pi[k]["correct1"].mean() for k in pi}
                cmps = {
                    "case4_P_to_Q": M.compare(pi["P"], pi["Q"]),
                    "case1_Q_to_C": M.compare(pi["Q"], pi[fam + "_C"]),
                    "case2_P_to_A": M.compare(pi["P"], pi[fam + "_A"]),
                    "case3_A_to_C": M.compare(pi[fam + "_A"], pi[fam + "_C"]),
                    "case5_P_to_C": M.compare(pi["P"], pi[fam + "_C"]),
                }
                agree = {k: 100 * pi[k]["agree"].mean() for k in (fam + "_A", fam + "_C", "Q")}
                # exact additivity of the point estimates: Case 2 + Case 3 = Case 5
                add = cmps["case2_P_to_A"]["top1_diff_pct"]["mean"] + cmps["case3_A_to_C"]["top1_diff_pct"]["mean"]
                assert abs(add - cmps["case5_P_to_C"]["top1_diff_pct"]["mean"]) < 1e-9, "Case 2 + Case 3 != Case 5"
                result[P].setdefault(fam, {})[name] = dict(
                    n=int(len(lab)), top1_pct={k: float(v) for k, v in top1.items()},
                    agree_vs_pretrained_fp32_pct={k: float(v) for k, v in agree.items()},
                    comparisons={k: dict(top1_diff_pct=v["top1_diff_pct"], mcnemar=v["mcnemar"])
                                 for k, v in cmps.items()})
                setname = "POOLED" if name == "pooled" else LABEL[name]
                rows_top1.append(f"| {setname} ({len(lab):,}) | {top1['P']:.2f} | {top1['Q']:.2f} | "
                                 f"{top1[fam + '_A']:.2f} | {top1[fam + '_C']:.2f} | "
                                 f"{agree[fam + '_A']:.1f} | {agree[fam + '_C']:.1f} |")
                rows_cmp.append(f"| {setname} | {fmt_cmp(cmps['case4_P_to_Q'])} | {fmt_cmp(cmps['case1_Q_to_C'])} | "
                                f"{fmt_cmp(cmps['case2_P_to_A'])} | {fmt_cmp(cmps['case3_A_to_C'])} | "
                                f"{fmt_cmp(cmps['case5_P_to_C'])} |")
            md.append(f"\n### {P}, {famname}: top-1 % and agreement with pretrained FP32\n")
            md.append("| test set (images) | P: pretrained FP32 | Q: pretrained-CIM | A: QAT FP32 (own) | C: QAT-CIM | "
                      "agree A vs P % | agree C vs P % |\n|---|---|---|---|---|---|---|")
            md += rows_top1
            md.append(f"\n### {P}, {famname}: paired differences, top-1 points (95% CI)\n")
            md.append("| test set | Case 4: P→Q | Case 1: Q→C | Case 2: P→A | Case 3: A→C | Case 5: P→C |\n|---|---|---|---|---|---|")
            md += rows_cmp

        # self-check against the existing pooled report
        rp = W / "eval" / "test_pooled" / f"REPORT_test_pooled_{P}.json"
        if rp.exists():
            rep = json.load(open(rp))
            rows = {r["run"]: r["top1_pct"] for r in rep["rows"]}
            chk = []
            for key, run in (("P", R["fp32"]), ("Q", R["pre_cim"]), ("new_C", R["new"]), ("old_C", R["old"])):
                mine = 100 * M.per_image(pooled[key], plab).get("correct1").mean()
                chk.append((run, mine, rows.get(run)))
            md.append(f"\n### {P} self-check: my pooled top-1 vs REPORT_test_pooled_{P}.json\n")
            md.append("| row | this script | existing report | match |\n|---|---|---|---|")
            for run, mine, theirs in chk:
                good = theirs is not None and abs(mine - theirs) < 1e-9
                md.append(f"| {run} | {mine:.4f} | {theirs if theirs is None else format(theirs, '.4f')} | {'yes' if good else 'NO'} |")
                result[P].setdefault("selfcheck", {})[run] = dict(mine=float(mine), report=theirs, match=bool(good))
        else:
            md.append(f"\n(no {rp.name} found: self-check skipped)\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    outs[0].write_text(json.dumps(result, indent=1))
    outs[1].write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwritten (new files only): {outs[0]}\n                          {outs[1]}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("check", "compute"):
        p = sub.add_parser(n)
        p.add_argument("--paths", required=True)
    a = ap.parse_args()
    sys.exit(dict(check=cmd_check, compute=cmd_compute)[a.cmd](a))


if __name__ == "__main__":
    main()
