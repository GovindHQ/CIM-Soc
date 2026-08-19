#!/usr/bin/env python3
"""
run_tinyvit.py — run pretrained TinyViT-5M with every attention matmul executed
on the simulated 32x32 CIM array, and compare against the float reference.

Usage:
    python run_tinyvit.py                    # full model, 2 synthetic images
    python run_tinyvit.py --stage3-only      # just the stage-3 attention block
    python run_tinyvit.py --token-block 8    # sweep the psum/write tradeoff
    python run_tinyvit.py --act-bits 8 --weight-bits 8

CAVEAT ON THE ACCURACY NUMBERS
------------------------------
No ImageNet data is available in this environment, so the inputs are synthetic
(smooth gradients and blobs, normalized like ImageNet). Those are closer to
natural image statistics than white noise, but they are NOT a validation set.
Treat the logit agreement as an indication of whether the pipeline is coherent,
not as an accuracy measurement. Run this on real images before quoting a number.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "_shim"))
sys.path.insert(0, str(HERE))

import numpy as np
import torch

from cim_sim import CIMArray, CIMConfig, TraceLog, cosine_sim, report, tile_map
from cim_sim.attention import (CIMAttention, convert_attention,
                               set_cim_enabled, set_cim_ops)
from tiny_vit import TinyViT

CKPT = HERE / "tinyvit5m.pth"
CKPT_URL = ("https://github.com/wkcn/TinyViT-model-zoo/releases/download/"
            "checkpoints/tiny_vit_5m_22kto1k_distill.pth")


def ensure_checkpoint() -> None:
    """Fetch the pretrained TinyViT-5M weights on first run (~22 MB)."""
    if CKPT.exists():
        return
    import urllib.request
    print(f"downloading TinyViT-5M weights -> {CKPT}")
    urllib.request.urlretrieve(CKPT_URL, CKPT)

# TinyViT-5M-224 configuration, verbatim from the reference source.
TINYVIT_5M = dict(
    img_size=224,
    embed_dims=[64, 128, 160, 320],
    depths=[2, 2, 6, 2],
    num_heads=[2, 4, 5, 10],
    window_sizes=[7, 7, 14, 7],
    mlp_ratio=4.0,
    drop_path_rate=0.0,
    num_classes=1000,
)


# --------------------------------------------------------------------------- #
# model + inputs
# --------------------------------------------------------------------------- #


def build_model() -> TinyViT:
    ensure_checkpoint()
    model = TinyViT(**TINYVIT_5M)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = sd["model"] if "model" in sd else sd
    sd = {k: v for k, v in sd.items() if not k.endswith("attention_bias_idxs")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    model.eval()
    return model


def synthetic_images(n: int = 2, size: int = 224, seed: int = 0) -> torch.Tensor:
    """Smooth synthetic images, ImageNet-normalized.

    Not real data. Natural images have low-frequency-dominated spectra, so
    smooth gradients and blobs exercise the activation ranges far more
    realistically than white noise, which would make quantization look worse
    than it is.
    """
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:size, 0:size] / size
    imgs = []
    for _ in range(n):
        chans = []
        for _c in range(3):
            f = rng.uniform(1, 4, size=4)
            p = rng.uniform(0, 2 * np.pi, size=4)
            img = (np.sin(2 * np.pi * f[0] * xs + p[0])
                   + np.sin(2 * np.pi * f[1] * ys + p[1])
                   + np.exp(-((xs - rng.uniform(.2, .8)) ** 2
                              + (ys - rng.uniform(.2, .8)) ** 2) / 0.05) * 2)
            img = (img - img.min()) / (img.max() - img.min() + 1e-9)
            chans.append(img)
        imgs.append(np.stack(chans))
    x = torch.tensor(np.stack(imgs), dtype=torch.float32)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #


def run_stage3_only(cfg: CIMConfig) -> None:
    """Isolate one stage-3 attention module: dim 160, 5 heads, key_dim 32, N=196."""
    model = build_model()
    attn = None
    for name, m in model.named_modules():
        if type(m).__name__ == "Attention" and m.nh_kd == 160:
            attn, attn_name = m, name
            break
    assert attn is not None, "no stage-3 attention found"

    B, N, C = 1, 196, 160
    torch.manual_seed(0)
    x = torch.randn(B, N, C) * 0.5

    with torch.no_grad():
        ref = attn(x)

    array = CIMArray(cfg)
    log = TraceLog()
    cim = CIMAttention(attn, array, log=log, name="stage3")
    cim.eval()
    with torch.no_grad():
        out = cim(x)

    print(f"\nstage-3 attention module: {attn_name}")
    print(f"  dim={C}  heads={attn.num_heads}  key_dim={attn.key_dim}  "
          f"d={attn.d}  qkv=[{C} -> {attn.qkv.out_features}]  N={N}")
    print()
    for label, (n, k, m) in [
        ("qkv proj", (B * N, C, attn.qkv.out_features)),
        ("QK^T    ", (N, attn.key_dim, N)),
        ("AV      ", (N, N, attn.d)),
    ]:
        print(f"  {label}  {tile_map(n, k, m)}")
    print()
    print(report(log))
    err = (out - ref).norm() / ref.norm()
    print(f"\n  output relative error vs float32: {err:.2%}"
          f"   cosine: {cosine_sim(out.detach().numpy(), ref.detach().numpy()):.4f}")


def run_full(cfg: CIMConfig, n_images: int, cim_out_proj: bool) -> None:
    model = build_model()
    x = synthetic_images(n_images)

    with torch.no_grad():
        ref_logits = model(x)

    array = CIMArray(cfg)
    log = TraceLog()
    replaced = convert_attention(model, array, log=log, cim_out_proj=cim_out_proj)
    print(f"replaced {len(replaced)} attention modules:")
    for r in replaced:
        print(f"    {r}")

    with torch.no_grad():
        cim_logits = model(x)

    print()
    print(report(log))

    print()
    print("=" * 78)
    print("ACCURACY  (synthetic inputs — see the caveat in the docstring)")
    print("=" * 78)
    ref_top = ref_logits.argmax(-1)
    cim_top = cim_logits.argmax(-1)
    for i in range(x.shape[0]):
        r, c = ref_logits[i], cim_logits[i]
        print(f"  image {i}: top-1 float={ref_top[i].item():>4}  "
              f"cim={cim_top[i].item():>4}  "
              f"{'MATCH' if ref_top[i]==cim_top[i] else 'DIFFER'}   "
              f"logit cos={cosine_sim(c.numpy(), r.numpy()):.4f}   "
              f"rel err={(c-r).norm()/r.norm():.2%}")

    k = 5
    agree = [(len(set(ref_logits[i].topk(k).indices.tolist())
                  & set(cim_logits[i].topk(k).indices.tolist())), k)
             for i in range(x.shape[0])]
    print(f"  top-{k} overlap: " + ", ".join(f"{a}/{b}" for a, b in agree))


def run_full_multi_array(cfg: CIMConfig, n_images: int, num_arrays: int,
                         cim_out_proj: bool) -> None:
    """Same as run_full, but scheduled across a pool of `num_arrays` arrays."""
    model = build_model()
    x = synthetic_images(n_images)

    with torch.no_grad():
        ref_logits = model(x)

    arrays = [CIMArray(cfg) for _ in range(num_arrays)]
    log = TraceLog()
    replaced = convert_attention(model, arrays=arrays, log=log,
                                 cim_out_proj=cim_out_proj)
    print(f"replaced {len(replaced)} attention modules, sharing a pool of "
          f"{num_arrays} arrays")

    with torch.no_grad():
        cim_logits = model(x)

    print()
    print(report(log))

    tot = log.total()
    cycles = log.total_parallel_cycles()
    speedup = tot.array_ops / cycles if cycles else float("nan")
    print()
    print("=" * 78)
    print("PARALLELISM")
    print("=" * 78)
    print(f"  array pool size (P)         : {num_arrays}")
    print(f"  total physical array_ops    : {tot.array_ops:,}  "
          f"(= what P=1 wall-clock cycles would have been)")
    print(f"  wall-clock parallel cycles  : {cycles:,}")
    print(f"  ideal speedup vs P=1        : {speedup:.2f}x")
    print(f"  parallel utilization        : {log.parallel_utilization():.1%}  "
          f"(active array slots / total array slots across all groups)")

    print()
    print("=" * 78)
    print("ACCURACY  (synthetic inputs — see the caveat in the docstring)")
    print("=" * 78)
    ref_top = ref_logits.argmax(-1)
    cim_top = cim_logits.argmax(-1)
    for i in range(x.shape[0]):
        r, c = ref_logits[i], cim_logits[i]
        print(f"  image {i}: top-1 float={ref_top[i].item():>4}  "
              f"cim={cim_top[i].item():>4}  "
              f"{'MATCH' if ref_top[i]==cim_top[i] else 'DIFFER'}   "
              f"logit cos={cosine_sim(c.numpy(), r.numpy()):.4f}   "
              f"rel err={(c-r).norm()/r.norm():.2%}")


def run_ablation(cfg: CIMConfig, n_images: int) -> None:
    """Which of the three matmul sites actually costs the accuracy?

    Each row puts exactly one site (or all three) on the 4-bit array and leaves
    the rest in float32. This is the number that answers 'is this viable', and
    it is the number that tells you which site needs a precision escape hatch.
    """
    x = synthetic_images(n_images)

    model = build_model()
    with torch.no_grad():
        ref = model(x)

    combos = [("float baseline", set()),
              ("qkv proj only", {"qkv"}),
              ("QK^T only", {"qk"}),
              ("AV only", {"av"}),
              ("QK^T + AV", {"qk", "av"}),
              ("all three", {"qkv", "qk", "av"})]

    print(f"\n{'sites on array':<18}{'logit rel err':>15}{'logit cos':>12}"
          f"{'top-1 agree':>13}{'top-5 overlap':>15}")
    print("-" * 73)
    for label, ops in combos:
        model = build_model()
        array = CIMArray(cfg)
        convert_attention(model, array)
        set_cim_ops(model, ops)
        with torch.no_grad():
            out = model(x)
        err = ((out - ref).norm() / ref.norm()).item()
        cos = cosine_sim(out.numpy(), ref.numpy())
        agree = (out.argmax(-1) == ref.argmax(-1)).float().mean().item()
        ov = np.mean([len(set(ref[i].topk(5).indices.tolist())
                          & set(out[i].topk(5).indices.tolist()))
                      for i in range(x.shape[0])])
        print(f"{label:<18}{err:>14.2%}{cos:>12.4f}"
              f"{agree:>12.0%}{ov:>13.1f}/5")


def run_bit_sweep(cfg: CIMConfig, n_images: int) -> None:
    """Same workload at several bit widths. The array geometry is unchanged;
    only the number of levels per cell and per broadcast input changes."""
    x = synthetic_images(n_images)
    model = build_model()
    with torch.no_grad():
        ref = model(x)

    print(f"\n{'w/a bits':<12}{'logit rel err':>15}{'logit cos':>12}"
          f"{'top-1 agree':>13}{'top-5 overlap':>15}")
    print("-" * 67)
    for bits in (2, 3, 4, 6, 8):
        model = build_model()
        c = CIMConfig(rows=cfg.rows, cols=cfg.cols, act_bits=bits,
                      weight_bits=bits, token_block=cfg.token_block)
        array = CIMArray(c)
        convert_attention(model, array)
        with torch.no_grad():
            out = model(x)
        err = ((out - ref).norm() / ref.norm()).item()
        cos = cosine_sim(out.numpy(), ref.numpy())
        agree = (out.argmax(-1) == ref.argmax(-1)).float().mean().item()
        ov = np.mean([len(set(ref[i].topk(5).indices.tolist())
                          & set(out[i].topk(5).indices.tolist()))
                      for i in range(x.shape[0])])
        print(f"{f'{bits}b/{bits}b':<12}{err:>14.2%}{cos:>12.4f}"
              f"{agree:>12.0%}{ov:>13.1f}/5")


def run_adc_compare(cfg: CIMConfig, n_images: int, adc_bits_list: list[int]) -> None:
    """Error contributed by the ADC alone, isolated from operand quantization.

    Row 1 is the float baseline. Row 2 is the existing ADC-disabled CIM path
    (exact integer column sums) at cfg's weight_bits/act_bits — this is what
    every earlier --ablate / --sweep-bits run in this file already measured.
    Every row after that turns the ADC on at a given resolution, holding
    weight_bits/act_bits fixed, so the DELTA between row 2 and any ADC row is
    specifically the ADC's contribution, not operand quantization's.
    """
    x = synthetic_images(n_images)
    model = build_model()
    with torch.no_grad():
        ref = model(x)

    print(f"\noperand precision fixed at {cfg.weight_bits}b weights / "
          f"{cfg.act_bits}b activations, Vref={cfg.adc_vref}V")
    print(f"\n{'config':<20}{'logit rel err':>15}{'logit cos':>12}"
          f"{'top-1 agree':>13}{'adc conv.':>12}{'adc sat.':>10}")
    print("-" * 82)

    rows = [("ADC disabled (exact)", False, None)]
    for b in adc_bits_list:
        rows.append((f"ADC {b}-bit", True, b))

    for label, adc_on, adc_bits in rows:
        model = build_model()
        c = CIMConfig(rows=cfg.rows, cols=cfg.cols, weight_bits=cfg.weight_bits,
                      act_bits=cfg.act_bits, token_block=cfg.token_block,
                      adc_enabled=adc_on, adc_bits=adc_bits or 8,
                      adc_vref=cfg.adc_vref)
        array = CIMArray(c)
        log = TraceLog()
        convert_attention(model, array, log=log)
        with torch.no_grad():
            out = model(x)
        err = ((out - ref).norm() / ref.norm()).item()
        cos = cosine_sim(out.numpy(), ref.numpy())
        agree = (out.argmax(-1) == ref.argmax(-1)).float().mean().item()
        tot = log.total()
        print(f"{label:<20}{err:>14.2%}{cos:>12.4f}{agree:>12.0%}"
              f"{tot.adc_conversions:>12,}{tot.adc_saturations:>10,}")

    print("\n  'adc conv.' = 0 on the disabled row by construction (no ADC "
          "calls made). Saturation count > 0 would mean the configured "
          "full_scale is too small for this workload at this precision — "
          "not expected here since full_scale is sized from weight_bits/"
          "act_bits directly (see array.py's ADC docstring).")


def run_parallel_compare(base_cfg: CIMConfig, n_images: int,
                         array_counts: list[int]) -> None:
    """Area(=array count placeholder — see note)/efficiency/parallelism vs P.

    For each P: build a fresh pool of P identical arrays, run the full model,
    and report:
        accuracy       — must not move with P (parallelism is a scheduling
                          choice, not a numerical one; this is the regression
                          check, same idea as TEST 4 in test_multi_array.py)
        MAC utilization — CIMStats.utilization: fraction of issued MACs inside
                          one 32x32 tile that are real data, not padding.
                          Unaffected by P; printed as a sanity check.
        parallel util.  — MatmulTrace.array_parallel_utilization, weighted
                          across the whole log: how fully the P macros are
                          populated with distinct output tiles.
        ideal speedup   — total_array_ops / total_parallel_cycles. array_ops
                          summed across arrays is the total physical work,
                          which does NOT depend on P (see BONUS test in
                          test_multi_array.py); parallel_cycles is wall-clock.
                          Their ratio is exactly what P=1's wall-clock would
                          have been, divided by this run's wall-clock — i.e.
                          the ideal speedup, no separate P=1 run required.
        relative area   — NOT a real area number. See the note below.
    """
    x = synthetic_images(n_images)
    model = build_model()
    with torch.no_grad():
        ref = model(x)

    print(f"\n{'P (arrays)':<12}{'logit rel err':>15}{'MAC util':>10}"
          f"{'parallel util':>15}{'ideal speedup':>15}{'rel. area*':>12}")
    print("-" * 79)
    for P in array_counts:
        model = build_model()
        arrays = [CIMArray(base_cfg) for _ in range(P)]
        log = TraceLog()
        convert_attention(model, arrays=arrays, log=log)
        with torch.no_grad():
            out = model(x)
        err = ((out - ref).norm() / ref.norm()).item()
        tot = log.total()
        cycles = log.total_parallel_cycles()
        speedup = tot.array_ops / cycles if cycles else float("nan")
        print(f"{P:<12}{err:>14.2%}{tot.utilization:>10.1%}"
              f"{log.parallel_utilization():>15.1%}{speedup:>14.2f}x"
              f"{P:>11}x")

    print(f"\n  * 'rel. area' = P (one placeholder unit per macro). This is NOT"
          f"\n    a real area estimate — no per-macro or per-ADC area/mm^2 "
          f"number exists"
          f"\n    anywhere in this codebase yet. See the note in the chat "
          f"reply for what's")
    print(f"    needed to replace it with a real figure.")


def run_token_block_sweep(cfg: CIMConfig) -> None:
    """Same stage-3 workload at several token block sizes T."""
    model = build_model()
    attn = next(m for m in model.modules()
                if type(m).__name__ == "Attention" and m.nh_kd == 160)
    torch.manual_seed(0)
    x = torch.randn(1, 196, 160) * 0.5

    print(f"\n{'T':>6}{'tile writes':>14}{'cycles':>12}{'psum words':>13}"
          f"{'psum kB@20b':>14}{'rel err':>10}")
    print("-" * 69)
    ref = None
    for T in (None, 98, 32, 8, 4, 1):
        c = CIMConfig(rows=cfg.rows, cols=cfg.cols, act_bits=cfg.act_bits,
                      weight_bits=cfg.weight_bits, token_block=T)
        array = CIMArray(c)
        with torch.no_grad():
            out = CIMAttention(attn, array, name="s3").eval()(x)
        if ref is None:
            ref = out
        s = array.stats
        print(f"{('all' if T is None else T):>6}{s.weight_tile_writes:>14,}"
              f"{s.array_ops:>12,}{s.psum_words_live:>13,}"
              f"{s.psum_words_live*20/8/1024:>13.1f}"
              f"{(out-ref).norm()/ref.norm():>10.1e}")
    print("  (rel err is against T=all: identical, as it must be)")


# --------------------------------------------------------------------------- #


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage3-only", action="store_true")
    p.add_argument("--sweep-token-block", action="store_true")
    p.add_argument("--ablate", action="store_true")
    p.add_argument("--sweep-bits", action="store_true")
    p.add_argument("--compare-adc", action="store_true",
                   help="float baseline vs ADC-disabled vs ADC-enabled at "
                        "--sweep-adc-bits resolutions, 1 array")
    p.add_argument("--sweep-adc-bits", type=int, nargs="+", default=[4, 6, 8, 10])
    p.add_argument("--compare-arrays", action="store_true",
                   help="accuracy/utilization/speedup vs --sweep-arrays array counts")
    p.add_argument("--sweep-arrays", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--images", type=int, default=2)
    p.add_argument("--act-bits", type=int, default=4)
    p.add_argument("--weight-bits", type=int, default=4)
    p.add_argument("--token-block", type=int, default=None)
    p.add_argument("--cim-out-proj", action="store_true")
    p.add_argument("--num-arrays", type=int, default=1,
                   help="parallel array pool size for the default full-model run")
    p.add_argument("--adc-enabled", action="store_true")
    p.add_argument("--adc-bits", type=int, default=8)
    p.add_argument("--adc-vref", type=float, default=0.6)
    a = p.parse_args()

    cfg = CIMConfig(rows=32, cols=32, act_bits=a.act_bits,
                    weight_bits=a.weight_bits, token_block=a.token_block,
                    adc_enabled=a.adc_enabled, adc_bits=a.adc_bits,
                    adc_vref=a.adc_vref)
    adc_str = (f"ADC {cfg.adc_bits}b @ Vref={cfg.adc_vref}V"
              if cfg.adc_enabled else "ADC disabled (exact column sums)")
    print(f"CIM fabric: {cfg.rows}x{cfg.cols}, {cfg.weight_bits}b weights, "
          f"{cfg.act_bits}b activations, {cfg.planes} plane, "
          f"{a.num_arrays} array(s), T={cfg.token_block or 'all tokens'}, {adc_str}")

    if a.compare_adc:
        run_adc_compare(cfg, a.images, a.sweep_adc_bits)
    elif a.compare_arrays:
        run_parallel_compare(cfg, a.images, a.sweep_arrays)
    elif a.ablate:
        run_ablation(cfg, a.images)
    elif a.sweep_bits:
        run_bit_sweep(cfg, a.images)
    elif a.sweep_token_block:
        run_token_block_sweep(cfg)
    elif a.stage3_only:
        run_stage3_only(cfg)
    elif a.num_arrays > 1:
        run_full_multi_array(cfg, a.images, a.num_arrays, a.cim_out_proj)
    else:
        run_full(cfg, a.images, a.cim_out_proj)


if __name__ == "__main__":
    main()