#!/usr/bin/env python3
"""
adc_analysis.py — turn ONE recorded pre-ADC distribution into the whole ADC
design sweep, offline.

The point of this script is that the sweep costs nothing. The distribution of
column sums entering the ADC does not depend on the ADC: the converter sits
downstream of the analog sum, and its full scale is a design constant, not an
input. So a single instrumented forward pass fixes the distribution, and every
candidate (full scale, resolution) pair can then be scored against that
distribution exactly — no reruns, no model, no PyTorch.

Usage:
    python run_tinyvit.py --images 20 --image-dir images --adc-stats \\
        --probe-out probe_4b.npz
    python adc_analysis.py probe_4b.npz

What the table means
--------------------
  clip%        fraction of column sums whose magnitude exceeds the chosen full
               scale and is therefore saturated
  RMSE         root-mean-square error of the digitised value against the exact
               column sum, in column-sum units, INCLUDING clipping error
  codes        how many distinct ADC codes the workload ever produces, out of
               2^bits. A small number here is the direct measure of wasted
               converter
  ENOB         effective number of bits, (SNR_dB - 1.76) / 6.02, the standard
               way of saying "this nominally N-bit converter is really doing
               the job of an M-bit one on this signal"

The comparison that answers the QAT question is the ENOB column read down a
fixed row of `bits`: how many effective bits does re-sizing the full scale buy
you, for free, before anyone retrains anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from cim_sim.adc_probe import (abs_percentiles_from_hist, adc_error_from_hist,
                               moments_from_hist, percentiles_from_hist)

BITS = (4, 5, 6, 7, 8, 9, 10, 12)


def load(path: str):
    z = np.load(path)
    fs = int(z["full_scale"])
    hists = {k[len("hist::"):]: z[k] for k in z.files if k.startswith("hist::")}
    meta = {k[len("meta::"):]: z[k] for k in z.files if k.startswith("meta::")}
    bounds = {k[len("bound::"):]: int(z[k]) for k in z.files
              if k.startswith("bound::")}
    return fs, hists, meta, bounds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="file written by run_tinyvit.py --probe-out")
    ap.add_argument("--target-enob", type=float, default=None,
                    help="if given, also report the cheapest (FS, bits) pair "
                         "reaching this ENOB for each operation")
    args = ap.parse_args()

    fs_span, hists, meta, bounds = load(args.npz)
    values = np.arange(-fs_span, fs_span + 1, dtype=np.int64)

    print()
    print("=" * 92)
    print(f"ADC DESIGN SWEEP from {args.npz}")
    print("=" * 92)
    if meta:
        print("  run: " + "  ".join(f"{k}={int(v)}" for k, v in sorted(meta.items())))
    print(f"  histogram span : +/-{fs_span}")
    print("  NOTE: each operation is scored against its OWN theoretical bound. "
          "The AV\n        path quantizes activations unsigned, so its bound is "
          "~2x the signed one.")
    print()

    for tag in sorted(hists):
        counts = hists[tag].astype(np.int64)
        total = int(counts.sum())
        if total == 0:
            continue

        mean, std, vmin, vmax = moments_from_hist(counts, values)
        p99, p999, p9999 = abs_percentiles_from_hist(
            counts, values, [99, 99.9, 99.99])
        amax = float(max(abs(vmin), abs(vmax)))

        fs_th = bounds.get(tag, fs_span)
        print("-" * 92)
        print(f"{tag}    ({total:,} column sums, theoretical bound +/-{fs_th})")
        print("-" * 92)
        print(f"  signed range [{vmin}, {vmax}]   mean {mean:.2f}   std {std:.2f}")
        print(f"  |y| percentiles:  P99 {p99:.0f}   P99.9 {p999:.0f}   "
              f"P99.99 {p9999:.0f}   max {amax:.0f}")
        print(f"  max |y| is {100.0 * amax / fs_th:.2f}% of theoretical full scale")
        print()

        candidates = [
            ("theoretical", fs_th),
            ("max |y|", int(max(amax, 1))),
            ("P99.99", int(max(p9999, 1))),
            ("P99.9", int(max(p999, 1))),
            ("P99", int(max(p99, 1))),
        ]
        # Drop duplicates while keeping order.
        seen, uniq = set(), []
        for name, v in candidates:
            if v not in seen:
                seen.add(v)
                uniq.append((name, v))

        header = f"  {'full scale':<14}{'value':>9}   " + "".join(
            f"{b:>2}b ENOB  " for b in BITS)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name, fsv in uniq:
            row = f"  {name:<14}{fsv:>9}   "
            for b in BITS:
                r = adc_error_from_hist(counts, values, fsv, b)
                row += f"{r['enob']:>8.2f}  "
            print(row)
        print()

        # Detail for the two ends of the trade so the numbers behind ENOB are
        # visible rather than implied.
        for name, fsv in (uniq[0], uniq[min(3, len(uniq) - 1)]):
            print(f"  detail — full scale = {name} (+/-{fsv})")
            print(f"    {'bits':>5}{'clip%':>10}{'RMSE':>10}{'max err':>10}"
                  f"{'codes used':>13}{'ENOB':>8}")
            for b in BITS:
                r = adc_error_from_hist(counts, values, fsv, b)
                codes = f"{r['codes_used']}/{r['codes_total']}"
                print(f"    {b:>5}{r['clip_pct']:>10.4f}{r['rmse']:>10.2f}"
                      f"{r['max_err']:>10.0f}{codes:>13}{r['enob']:>8.2f}")
            print()

        if args.target_enob is not None:
            best = None
            for name, fsv in uniq:
                for b in BITS:
                    r = adc_error_from_hist(counts, values, fsv, b)
                    if r["enob"] >= args.target_enob:
                        if best is None or b < best[2]:
                            best = (name, fsv, b, r)
                        break
            if best:
                name, fsv, b, r = best
                print(f"  cheapest configuration reaching ENOB >= "
                      f"{args.target_enob}: {b}-bit ADC at full scale {name} "
                      f"(+/-{fsv}), clipping {r['clip_pct']:.4f}%")
            else:
                print(f"  no configuration in the sweep reaches ENOB >= "
                      f"{args.target_enob}")
            print()

    print("=" * 92)


if __name__ == "__main__":
    main()
