#!/usr/bin/env python3
"""
experiments.py — the four thesis sweeps, as standalone experiments that reuse
the existing simulator unchanged.

Nothing in cim_sim/ or run_tinyvit.py is modified by this file. It imports the
same building blocks the CLI modes already use (build_model, synthetic_images,
convert_attention, CIMConfig, TraceLog) and computes the same two metrics those
modes already compute:

    output agreement (%) = fraction of images whose CIM top-1 class matches the
                           float32 top-1 class
                           == (out.argmax(-1) == ref.argmax(-1)).mean()
                           — identical to the 'top-1 agree' column in
                           run_tinyvit.py's --sweep-bits / --compare-adc modes.

    latency (cycles)     = TraceLog.total_parallel_cycles()
                           — the parallel wall-clock array-timestep count,
                           == run_tinyvit.py --compare-arrays' 'wall-clock
                           parallel cycles'. NOT the P=1 figure; this is the
                           real multi-array latency.

Y-axis is deliberately "Output agreement (%)", NOT accuracy: the inputs are
synthetic (see run_tinyvit.synthetic_images), so this measures how faithfully
the CIM path tracks the float path on identical inputs, not model accuracy on
real data.

The four experiments (each fixes everything except one swept axis):

    1. weight-bit sweep     act=4,  ADC off,        P=1 ; sweep w in {2,4,6,8}
    2. activation-bit sweep w=8,    ADC off,        P=1 ; sweep a in {2,4,6,8}
    3. adc-bit sweep        w=8,a=8, ADC ON,        P=1 ; sweep adc in {6,8,10,12}
    4. array scaling        w=8,a=8, ADC ON 10-bit      ; sweep P in {1,2,4,8}
                            -> plots latency (cycles) vs P, using parallel cycles

Usage
-----
    python experiments.py --all                 # run all four, write CSV + PNG
    python experiments.py --exp weight          # just experiment 1
    python experiments.py --exp activation
    python experiments.py --exp adc
    python experiments.py --exp arrays
    python experiments.py --plot-only           # re-draw PNGs from existing CSVs
    python experiments.py --all --images 32     # more images -> finer agreement

Outputs land in ./results/ :
    exp1_weight_bits.csv     exp1_weight_bits.png
    exp2_activation_bits.csv exp2_activation_bits.png
    exp3_adc_bits.csv        exp3_adc_bits.png
    exp4_array_scaling.csv   exp4_array_scaling.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "_shim"))
sys.path.insert(0, str(HERE))

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")           # no display needed; write straight to PNG
import matplotlib.pyplot as plt

from cim_sim import CIMArray, CIMConfig, TraceLog
from cim_sim.attention import convert_attention
from run_tinyvit import build_model, synthetic_images

RESULTS = HERE / "results"
DEFAULT_IMAGES = 16

# ---- consistent thesis-quality plot styling across all four figures ------- #
PLOT_STYLE = dict(color="#1f4e79", marker="o", markersize=7,
                  linewidth=2, markerfacecolor="#1f4e79",
                  markeredgecolor="white", markeredgewidth=1.2)
LATENCY_STYLE = dict(color="#8c1d18", marker="s", markersize=7,
                     linewidth=2, markerfacecolor="#8c1d18",
                     markeredgecolor="white", markeredgewidth=1.2)


def _style_axes(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=12, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=8)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.7)
    ax.tick_params(labelsize=11)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


# --------------------------------------------------------------------------- #
# shared measurement helpers — one place, reused by all agreement experiments
# --------------------------------------------------------------------------- #


def _reference_logits(x: torch.Tensor) -> torch.Tensor:
    """Float32 TinyViT logits — the thing CIM output is compared against."""
    model = build_model()
    with torch.no_grad():
        return model(x)


def _agreement(cfg: CIMConfig, x: torch.Tensor, ref: torch.Tensor,
               num_arrays: int = 1) -> float:
    """Output agreement (%) for one config: build a fresh model + array(s),
    convert attention, run, compare top-1 against ref. This is exactly the
    computation in run_tinyvit.py's sweep modes, factored out."""
    model = build_model()
    if num_arrays == 1:
        convert_attention(model, CIMArray(cfg))
    else:
        convert_attention(model, arrays=[CIMArray(cfg) for _ in range(num_arrays)])
    with torch.no_grad():
        out = model(x)
    return 100.0 * (out.argmax(-1) == ref.argmax(-1)).float().mean().item()


def _latency_cycles(cfg: CIMConfig, x: torch.Tensor, num_arrays: int) -> int:
    """Parallel wall-clock latency (cycles) for one config, via the existing
    TraceLog.total_parallel_cycles(). A log is needed, so attention is
    converted with log= and the pass is run for its trace, not its logits."""
    model = build_model()
    log = TraceLog()
    arrays = [CIMArray(cfg) for _ in range(num_arrays)]
    convert_attention(model, arrays=arrays, log=log)
    with torch.no_grad():
        model(x)
    return log.total_parallel_cycles()


# --------------------------------------------------------------------------- #
# CSV / plot io
# --------------------------------------------------------------------------- #


def _write_csv(path: Path, header: list[str], rows: list[tuple]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote {path}")


def _read_csv(path: Path) -> tuple[list[str], list[list[float]]]:
    with path.open() as f:
        r = csv.reader(f)
        header = next(r)
        rows = [[float(v) for v in row] for row in r]
    return header, rows


def _line_plot(csv_path: Path, png_path: Path, xlabel: str, ylabel: str,
               title: str, style: dict, integer_x: bool = True,
               ylim: tuple | None = None, latency: bool = False) -> None:
    header, rows = _read_csv(csv_path)
    xs = [r[0] for r in rows]
    ys = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=200)
    ax.plot(xs, ys, **style)
    _style_axes(ax, xlabel, ylabel, title)
    if integer_x:
        ax.set_xticks(xs)
    if ylim:
        ax.set_ylim(*ylim)
    else:
        # give headroom so top annotation isn't clipped
        lo, hi = min(ys), max(ys)
        pad = (hi - lo) * 0.12 if hi > lo else hi * 0.1
        ax.set_ylim(lo - pad, hi + pad)

    # annotate each point with its value; latency uses thousands separators and
    # sits to the upper-right so it never overlaps the descending line
    for xv, yv in zip(xs, ys):
        if latency:
            label = f"{int(yv):,}"
            ax.annotate(label, (xv, yv), textcoords="offset points",
                        xytext=(6, 10), ha="left", fontsize=8.5,
                        color=style["color"])
        else:
            ax.annotate(f"{yv:g}", (xv, yv), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=9,
                        color=style["color"])
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {png_path}")


# --------------------------------------------------------------------------- #
# the four experiments
# --------------------------------------------------------------------------- #


def exp_weight_bits(x, ref, sweep=(2, 4, 6, 8)) -> None:
    """1. weight-bit sweep. act=4, ADC off, P=1."""
    print("\n[1] weight-bit sweep  (act=4, ADC disabled, arrays=1)")
    rows = []
    for wb in sweep:
        cfg = CIMConfig(weight_bits=wb, act_bits=4, adc_enabled=False)
        agr = _agreement(cfg, x, ref)
        print(f"    weight_bits={wb}: agreement={agr:.2f}%")
        rows.append((wb, agr))
    _write_csv(RESULTS / "exp1_weight_bits.csv",
               ["weight_bits", "output_agreement_pct"], rows)


def exp_activation_bits(x, ref, sweep=(2, 4, 6, 8)) -> None:
    """2. activation-bit sweep. w=8, ADC off, P=1."""
    print("\n[2] activation-bit sweep  (weight=8, ADC disabled, arrays=1)")
    rows = []
    for ab in sweep:
        cfg = CIMConfig(weight_bits=8, act_bits=ab, adc_enabled=False)
        agr = _agreement(cfg, x, ref)
        print(f"    act_bits={ab}: agreement={agr:.2f}%")
        rows.append((ab, agr))
    _write_csv(RESULTS / "exp2_activation_bits.csv",
               ["activation_bits", "output_agreement_pct"], rows)


def exp_adc_bits(x, ref, sweep=(6, 8, 10, 12)) -> None:
    """3. ADC-bit sweep. w=8, a=8, ADC enabled, P=1."""
    print("\n[3] ADC-bit sweep  (weight=8, act=8, ADC enabled, arrays=1)")
    rows = []
    for adc in sweep:
        cfg = CIMConfig(weight_bits=8, act_bits=8,
                        adc_enabled=True, adc_bits=adc)
        agr = _agreement(cfg, x, ref)
        print(f"    adc_bits={adc}: agreement={agr:.2f}%")
        rows.append((adc, agr))
    _write_csv(RESULTS / "exp3_adc_bits.csv",
               ["adc_bits", "output_agreement_pct"], rows)


def exp_array_scaling(x, sweep=(1, 2, 4, 8)) -> None:
    """4. array scaling / latency. w=8, a=8, ADC 10-bit enabled.
    Plots parallel wall-clock latency (cycles) vs number of arrays."""
    print("\n[4] array scaling  (weight=8, act=8, ADC 10-bit enabled)")
    rows = []
    for P in sweep:
        cfg = CIMConfig(weight_bits=8, act_bits=8,
                        adc_enabled=True, adc_bits=10)
        cyc = _latency_cycles(cfg, x, num_arrays=P)
        print(f"    arrays={P}: parallel latency={cyc:,} cycles")
        rows.append((P, cyc))
    _write_csv(RESULTS / "exp4_array_scaling.csv",
               ["num_arrays", "parallel_latency_cycles"], rows)


# --------------------------------------------------------------------------- #
# plotting (reads CSVs, so it works standalone via --plot-only)
# --------------------------------------------------------------------------- #


def plot_all() -> None:
    print("\nplotting from CSVs in results/ ...")
    specs = [
        ("exp1_weight_bits", "Weight bits", "Output agreement (%)",
         "Output Agreement vs Weight Bit-width\n(activation=4b, ADC off, 1 array)",
         PLOT_STYLE, True, (0, 105)),
        ("exp2_activation_bits", "Activation bits", "Output agreement (%)",
         "Output Agreement vs Activation Bit-width\n(weight=8b, ADC off, 1 array)",
         PLOT_STYLE, True, (0, 105)),
        ("exp3_adc_bits", "ADC bits", "Output agreement (%)",
         "Output Agreement vs ADC Bit-width\n(weight=8b, activation=8b, ADC on, 1 array)",
         PLOT_STYLE, True, (0, 105)),
        ("exp4_array_scaling", "Number of CIM arrays",
         "CIM attention latency (cycles)",
         "CIM Attention Latency vs Array Count\n(weight=8b, activation=8b, ADC 10b)",
         LATENCY_STYLE, True, None),
    ]
    for stem, xl, yl, title, style, int_x, ylim in specs:
        csv_path = RESULTS / f"{stem}.csv"
        if not csv_path.exists():
            print(f"  skip {stem}: {csv_path} not found (run the experiment first)")
            continue
        _line_plot(csv_path, RESULTS / f"{stem}.png", xl, yl, title,
                   style, integer_x=int_x, ylim=ylim,
                   latency=(stem == "exp4_array_scaling"))


# --------------------------------------------------------------------------- #


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true", help="run all four experiments")
    p.add_argument("--exp", choices=["weight", "activation", "adc", "arrays"],
                   help="run a single experiment")
    p.add_argument("--plot-only", action="store_true",
                   help="redraw PNGs from existing CSVs, run nothing")
    p.add_argument("--images", type=int, default=DEFAULT_IMAGES,
                   help=f"images for the agreement metric (default {DEFAULT_IMAGES}; "
                        "more -> finer agreement resolution, slower)")
    a = p.parse_args()

    RESULTS.mkdir(exist_ok=True)

    if a.plot_only:
        plot_all()
        return

    if not a.all and not a.exp:
        p.error("nothing to do: pass --all, --exp <name>, or --plot-only")

    x = synthetic_images(a.images)
    print(f"synthetic inputs: {a.images} images "
          f"(agreement resolution = {100/a.images:.1f}% per image)")

    # reference logits are shared by experiments 1-3; compute once if needed
    need_ref = a.all or a.exp in ("weight", "activation", "adc")
    ref = _reference_logits(x) if need_ref else None

    if a.all or a.exp == "weight":
        exp_weight_bits(x, ref)
    if a.all or a.exp == "activation":
        exp_activation_bits(x, ref)
    if a.all or a.exp == "adc":
        exp_adc_bits(x, ref)
    if a.all or a.exp == "arrays":
        exp_array_scaling(x)

    plot_all()
    print("\ndone.")


if __name__ == "__main__":
    main()