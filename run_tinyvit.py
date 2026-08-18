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
        ref_logits = model(x) #reference normal floating point output in ref_logits

    array = CIMArray(cfg)
    log = TraceLog()
    replaced = convert_attention(model, array, log=log, cim_out_proj=cim_out_proj) #replace attention with cimattention
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
    ref_top = ref_logits.argmax(-1) #to check with has the highest score
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
    p.add_argument("--images", type=int, default=2)
    p.add_argument("--act-bits", type=int, default=4)
    p.add_argument("--weight-bits", type=int, default=4)
    p.add_argument("--token-block", type=int, default=None)
    p.add_argument("--cim-out-proj", action="store_true")
    a = p.parse_args()

    cfg = CIMConfig(rows=32, cols=32, act_bits=a.act_bits,
                    weight_bits=a.weight_bits, token_block=a.token_block)
    print(f"CIM fabric: {cfg.rows}x{cfg.cols}, {cfg.weight_bits}b weights, "
          f"{cfg.act_bits}b activations, {cfg.planes} plane, "
          f"{cfg.n_arrays} array, T={cfg.token_block or 'all tokens'}")

    if a.ablate:
        run_ablation(cfg, a.images)
    elif a.sweep_bits:
        run_bit_sweep(cfg, a.images)
    elif a.sweep_token_block:
        run_token_block_sweep(cfg)
    elif a.stage3_only:
        run_stage3_only(cfg)
    else:
        run_full(cfg, a.images, a.cim_out_proj)


if __name__ == "__main__":
    main()
