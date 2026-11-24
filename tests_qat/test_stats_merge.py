#!/usr/bin/env python3
"""
tests_qat/test_stats_merge.py — the v2.7 merge of per-chunk analog statistics.

  python tests_qat/test_stats_merge.py

  M1  a synthetic chunked run (300 chunks, chunk-to-chunk scale variation, worker
      RNG streams persisting across chunks as in sim_eval): the uniform re-merge
      recovers the exact |V_ADC_IN| median and p99.9 for three different worker
      layouts, its weights sum to the conversion count, and no sample carries
      more than 0.1 % of the weight
  M2  control: the cim_sim AnalogStats.merge() path on the same chunks is far off
      (this is the defect the re-merge fixes; if it ever passes, cim_sim changed)
  M3  exact scalars (n, clip, rail, min, max, sums) of the re-merge == AnalogStats.merge()
  M4  a real simulator run (pretrained, analog W8A8, 6 fake images, chunk 1) with
      2 and with 3 workers: identical logits and exact counts, summary carries the
      v2.7 method, util percentiles from the two layouts agree within 10 %
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
RES = []


def check(name, ok, detail=""):
    RES.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


def synthetic():
    from cim_sim.analog.chain import AnalogStats, _weighted_percentile
    from cim_qat.analog_merge import uniform_merge_states
    N_CH, CALLS, NPC, KF, CAP = 300, 50, 2000, 0.03, 4000
    gen = np.random.default_rng(123)
    scales = gen.lognormal(0, 0.6, N_CH)
    data = [[gen.standard_normal(NPC) * scales[c] * 0.01 for _ in range(CALLS)] for c in range(N_CH)]
    allv = np.abs(np.concatenate([np.concatenate(d) for d in data]))
    true = {q: float(np.percentile(allv, q)) for q in (50, 99.9)}

    def chunks(seed0, n_workers):
        workers = [AnalogStats(keep_frac=KF, cap=CAP, seed=seed0 + i) for i in range(n_workers)]
        out = []
        for c in range(N_CH):
            st = workers[c % n_workers]
            st.reset()
            for v in data[c]:
                st.record("qkv", v, v, np.zeros(v.size), clip_hi=c % 2, rail_lo=c % 3)
            out.append(st.state_dict())
        return out

    def pct(st):
        s = st.b["qkv"]
        w = np.concatenate(s["w"])
        v = np.abs(np.concatenate(s["vadc"]).astype(float))
        return {q: _weighted_percentile(v, w, q) for q in (50, 99.9)}, w.max() / w.sum(), w.sum() / s["n"]

    med_u, worst_legacy = [], 0.0
    for seed0, nw in ((100, 12), (200, 16), (300, 16)):
        ch = chunks(seed0, nw)
        leg = AnalogStats(keep_frac=KF, cap=CAP, seed=0)
        for scal, arrs in ch:
            leg.merge(AnalogStats.from_state(scal, arrs, keep_frac=KF, cap=CAP))
        uni = uniform_merge_states(lambda arrays=True: iter(ch), budget=20_000)
        pu, share, wn = pct(uni)
        pl, _, _ = pct(leg)
        e50, e999 = pu[50] / true[50] - 1, pu[99.9] / true[99.9] - 1
        med_u.append(pu[50])
        check(f"M1 uniform re-merge, {nw} workers: median {100 * e50:+.1f} %, p99.9 {100 * e999:+.1f} % vs exact; "
              f"sum(w)/n {wn:.3f}; max weight share {100 * share:.3f} %",
              abs(e50) < 0.05 and abs(e999) < 0.10 and abs(wn - 1) < 0.02 and share < 1e-3)
        worst_legacy = max(worst_legacy, abs(pl[50] / true[50] - 1), abs(pl[99.9] / true[99.9] - 1))
        su, sl = uni.b["qkv"], leg.b["qkv"]
        same = all(su[k] == sl[k] for k in ("n", "clip_hi", "clip_lo", "rail_hi", "rail_lo", "voa_min", "voa_max",
                                            "vadc_min", "vadc_max")) and abs(su["vadc_sumsq"] / sl["vadc_sumsq"] - 1) < 1e-12
        check(f"M3 exact scalars of the re-merge == AnalogStats.merge() ({nw} workers)", same)
    check("M1 uniform re-merge: median spread across the three layouts < 5 %",
          (max(med_u) - min(med_u)) / true[50] < 0.05)
    check(f"M2 control: AnalogStats.merge() percentile error {100 * worst_legacy:.0f} % (the defect; expect > 10 %)",
          worst_legacy > 0.10)


def real_run(tmp: Path):
    import torch
    from PIL import Image
    from cim_qat import sim_eval
    from cim_qat.analog_merge import METHOD
    from cim_qat.hw_config import HWConfig
    from cim_qat.model import PRETRAINED
    rng = np.random.default_rng(0)
    items = []
    for k in range(6):
        f = tmp / f"img{k}.png"
        Image.fromarray(rng.integers(0, 255, (80, 90, 3), dtype=np.uint8)).save(f)
        items.append((k, str(f)))
    job = dict(kind="analog", ckpt=str(PRETRAINED), with_own_fp32=False, interpolation="bicubic",
               hw=HWConfig(8, 8, (22.92, 8.75, 2.72)).job_dict())
    torch.set_num_threads(1)
    r2 = sim_eval.run(job, items, tmp / "w2", workers=2, chunk=1, progress=False)
    r3 = sim_eval.run(job, items, tmp / "w3", workers=3, chunk=1, progress=False)
    s2, s3 = r2["summary"], r3["summary"]
    a2, a3 = s2["analog_summary"], s3["analog_summary"]
    check("M4 real run, 2 vs 3 workers: logits bit-identical", np.array_equal(r2["logits"], r3["logits"]))
    check("M4 exact counts identical (n, clip, rail per op)",
          all(a2[o][k] == a3[o][k] for o in ("qkv", "qk", "av") for k in ("n", "clip_hi", "clip_lo", "rail_hi", "rail_lo")))
    check("M4 summary records the v2.7 method", s2.get("analog_summary_method") == METHOD == s3.get("analog_summary_method"))
    rel = max(abs(a2[o][k] / a3[o][k] - 1) for o in ("qkv", "qk", "av") for k in ("util_median_pct", "util_p999_pct")
              if a3[o][k])
    check(f"M4 util median / p99.9 agree between the layouts within 10 % (max rel. diff {100 * rel:.1f} %)", rel < 0.10)
    json.dumps(a2)                                          # json-serialisable


def main():
    synthetic()
    with tempfile.TemporaryDirectory(prefix="cim qat stats ") as t:
        real_run(Path(t))
    n = sum(ok for _, ok in RES)
    print(f"\nSTATS MERGE: {n}/{len(RES)} passed")
    sys.exit(0 if n == len(RES) else 1)


if __name__ == "__main__":
    main()
