#!/usr/bin/env python3
"""
test_surrogate_vs_simulator.py — links the differentiable surrogate (cim_qat)
to the reference simulator (cim_sim) on identical inputs and configurations.

TIERS
  T1  per-matmul, randomised grid: W8A8/W4A4 x 6 gain sets (incl. tie-heavy
      G = 1, 2 and heavy clipping G = 300) x 6 shapes (ragged K/M) x qkv/qk/av.
  T1b per-matmul boundary cases built on purpose: exact ADC ties, operand
      rounding ties (x/s = k + 1/2), the saturation edge (|P| = 128*LSB_P),
      zero rows/columns/matrices, tiny values, ragged K in {1,17,31,33,49,196},
      N = 1, unsigned AV with exact zeros and one-hot rows, batched leading dims
      vs cim_matmul_batched.
  T2  each of the 10 TinyViT attention modules, identical input.
  T3  the full TinyViT-5M, batch 1, synthetic images (+ real images if given).
  T4  (CUDA + real images) GPU surrogate vs CPU simulator, with the FP32
      GPU-vs-CPU noise floor and GPU batch-size invariance.

EXPECTATIONS (pre-declared)
  T1, T1b, T2, T3 on CPU: 0 differing elements, identical clip/conversion
      counters (T1/T1b). Exact equality is expected because every operation is
      either an exact integer matmul or the same float64 operation in the same
      order as NumPy (QAT_DESIGN.md §4).
  T1, T1b on CUDA (TF32 off and on): 0 differing elements (integer-valued
      float32 matmul is exact in any order; quantization/ADC run in float64).
  T2 on CUDA: NOT expected to be exact (LayerNorm, softmax and the projection
      run in CUDA float32); reported only.
  T4: (i) EXACT: the CPU surrogate at batch 1 equals the simulator on every
      real image (0 differing logits).
      (ii) DISTRIBUTIONAL: the hardware model is discontinuous (rounding), so a
      1-ulp change in the input already changes the SIMULATOR's own output
      ("discretization avalanche"). The yardstick is therefore the simulator's
      self-sensitivity floor F = mean KL(sim(x) || sim(x + 1 ulp)) on the same
      images. PASS if mean KL(sim || GPU surrogate) <= 2 F; otherwise REVIEW
      (reported, never silently accepted). Also reported: the CPU batch-order
      floor (CPU surrogate batched vs simulator) and the hardware effect
      KL(FP32 || sim) for scale.

  python test_surrogate_vs_simulator.py --device cpu
  python test_surrogate_vs_simulator.py --device cuda --frozen-gains <...>/frozen_gains.json \
         --paths configs/paths.json --n-real 500 --workers 24 --out <dir>
Exit code 1 if any exact tier fails.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cim_qat.common import configure_determinism, now_iso, select_device, write_json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

RESULTS = []


def record(tier, name, ok, **kw):
    RESULTS.append(dict(tier=tier, name=name, ok=bool(ok), **kw))
    extra = "  ".join(f"{k}={v}" for k, v in kw.items())
    print(f"  [{'PASS' if ok else 'FAIL'}] {tier} {name}  {extra}")


# --------------------------------------------------------------------------- #
# per-matmul helpers
# --------------------------------------------------------------------------- #


def sim_matmul(hw, op, X, W):
    from cim_sim import CIMArray, cim_matmul, cim_matmul_batched
    from cim_qat.hw_config import OP_SIGNED
    arr = CIMArray(hw.cim_config())
    Xd, Wd = np.asarray(X, np.float64), np.asarray(W, np.float64)
    if Xd.ndim == 2:
        y = cim_matmul(Xd, Wd, arr, act_signed=OP_SIGNED[op], op=op)
    else:
        y = cim_matmul_batched(Xd, Wd, arr, act_signed=OP_SIGNED[op], op=op)
    a = arr.chain.adc
    return y.astype(np.float32), (a.n_clipped_hi, a.n_clipped_lo, arr.stats.adc_conversions)


def sur_matmul(hw, op, X, W, dev):
    from cim_qat.hw_config import op_specs
    from cim_qat.surrogate import OpStats, cim_matmul_surrogate
    st = OpStats()
    y = cim_matmul_surrogate(torch.from_numpy(np.asarray(X, np.float32)).to(dev),
                             torch.from_numpy(np.asarray(W, np.float32)).to(dev),
                             op_specs(hw)[op], st)
    return y.cpu().numpy(), (st.clip_hi, st.clip_lo, st.conversions_sim)


def exact_fs_hits(hw, op, X, W):
    """Conversions with |P_b| EXACTLY equal to the full scale 128*LSB_P.

    Known, explained counter difference (QAT_DESIGN.md §4.3): there the
    simulator's volt-domain value is 1 ulp above V_REF/2 (e.g. 0.30000000000000004 V),
    so its strict `> V_REF` clip counter fires, while the exact value is ON the
    boundary (not clipped by the definition). The output code is the same
    saturated code either way, so outputs are unaffected.
    """
    from cim_sim.quant import quantize
    from cim_qat.hw_config import OP_SIGNED, op_specs
    fs = 128 * op_specs(hw)[op].lsb_p
    if abs(fs - round(fs)) > 1e-9:
        return 0, 0
    fs = int(round(fs))
    Xd, Wd = np.asarray(X, np.float64), np.asarray(W, np.float64)
    if Xd.ndim == 2:
        Xd, Wd = Xd[None], Wd[None]
    Xd = Xd.reshape(-1, *Xd.shape[-2:])
    Wd = Wd.reshape(-1, *Wd.shape[-2:])
    hi = lo = 0
    for b in range(Xd.shape[0]):
        xq, _ = quantize(Xd[b], hw.act_bits, axis=1, signed=OP_SIGNED[op])
        wq, _ = quantize(Wd[b], hw.weight_bits, axis=0, signed=True)
        K = xq.shape[1]
        for r0 in range(0, K, 32):
            P = xq[:, r0:r0 + 32].astype(np.int64) @ wq[r0:r0 + 32].astype(np.int64)
            hi += int((P == fs).sum())
            lo += int((P == -fs).sum())
    return hi, lo


def compare_case(tier, name, hw, op, X, W, dev, extra=None):
    # Both sides must receive IDENTICAL values: in the model, CIM operands are
    # float32 torch tensors, which the simulator widens to float64 exactly.
    X = np.asarray(X, np.float32)
    W = np.asarray(W, np.float32)
    ys, cs = sim_matmul(hw, op, X, W)
    yu, cu = sur_matmul(hw, op, X, W, dev)
    nbad = int((ys != yu).sum())
    counters_ok = cs == cu
    if not counters_ok:
        fh, fl = exact_fs_hits(hw, op, X, W)
        explained = (cs[2] == cu[2] and cs[0] - cu[0] == fh and cs[1] - cu[1] == fl)
        if explained:
            extra = dict(extra or {}, exact_full_scale_hits=(fh, fl),
                         note="clip-counter difference fully explained by exact |P|=FS hits")
            counters_ok = True
    ok = nbad == 0 and counters_ok
    if extra is None and ok:
        return ok, nbad
    record(tier, name, ok, mismatches=nbad, counters_sim=cs, counters_sur=cu, **(extra or {}))
    return ok, nbad


def pinned_ints(rng, shape, q, axis, signed=True):
    """Integer-valued float matrix whose max |.| along `axis` is exactly q,
    so the quantizer's scale is exactly 1 and the codes are the values."""
    a = rng.integers(-q if signed else 0, q + 1, size=shape).astype(np.float64)
    a = np.moveaxis(a, axis, -1)
    a[..., 0] = q if (not signed or rng.random() < 0.5) else -q
    return np.moveaxis(a, -1, axis)


def t1_grid(dev, label):
    from cim_qat.hw_config import HWConfig
    rng = np.random.default_rng(7)
    n = bad = 0
    for (wb, ab), gains, shape in itertools.product(
            [(8, 8), (4, 4)],
            [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (23.77, 8.41, 2.92), (16.0, 4.0, 1.0),
             (300.0, 300.0, 300.0), (7.13, 3.31, 1.37)],
            [(37, 75, 45), (196, 160, 480), (49, 32, 49), (49, 49, 32), (196, 196, 32), (5, 320, 960)]):
        hw = HWConfig(wb, ab, gains)
        for op in ("qkv", "qk", "av"):
            N, K, M = shape
            X = rng.normal(size=(N, K)).astype(np.float32)
            if op == "av":
                X = np.abs(X) ** 3
                X = (X / X.sum(1, keepdims=True)).astype(np.float32)
            W = rng.normal(size=(K, M)).astype(np.float32)
            ok, nb = compare_case("T1", f"{label} W{wb}A{ab} G={gains} {shape} {op}", hw, op, X, W, dev)
            n += 1
            bad += (not ok)
    record("T1", f"{label}: randomised grid", bad == 0, cases=n, failing_cases=bad)


def t1_boundary(dev, label):
    from cim_qat.hw_config import HWConfig
    rng = np.random.default_rng(11)

    # (a) exact ADC ties at power-of-two LSB_P
    for (wb, ab), G in [((4, 4), 1.0), ((4, 4), 2.0), ((8, 8), 1.0), ((8, 8), 64.0)]:
        hw = HWConfig(wb, ab, (G, G, G))
        qa, qw = 2 ** (ab - 1) - 1, 2 ** (wb - 1) - 1
        X = pinned_ints(rng, (256, 64), qa, axis=1)
        W = pinned_ints(rng, (64, 64), qw, axis=0)
        from cim_qat.hw_config import op_specs
        L = op_specs(hw)["qkv"].lsb_p
        P = np.concatenate([X[:, :32] @ W[:32], X[:, 32:] @ W[32:]])
        ties = int((np.abs(np.mod(P / L, 1.0) - 0.5) < 1e-12).sum())
        compare_case("T1b", f"{label} ADC exact ties W{wb}A{ab} G={G}", hw, "qkv", X, W, dev,
                     extra=dict(exact_ties_in_P=ties, lsb_p=L))
        assert ties > 0, "tie construction failed"

    # (b) operand rounding ties x/s = k + 1/2 (half-to-even in both)
    for (wb, ab) in [(4, 4), (8, 8)]:
        hw = HWConfig(wb, ab, (5.0, 5.0, 5.0))
        qa = 2 ** (ab - 1) - 1
        X = pinned_ints(rng, (64, 64), qa, axis=1)
        X[:, 1:] += np.where(rng.random((64, 63)) < 0.5, 0.5, 0.0)
        X = np.clip(X, -qa, qa)
        W = rng.normal(size=(64, 40))
        halves = int((np.mod(X, 1.0) == 0.5).sum())
        compare_case("T1b", f"{label} operand rounding ties W{wb}A{ab}", hw, "qkv", X, W, dev,
                     extra=dict(half_values=halves))

    # (c) saturation edge: W4A4 G=8 -> LSB_P = 4, FS = 128*4 = 512.
    # Column 0 of W is [7]*31 + [1]; row r of X is built so that P[r, 0] hits a
    # chosen target exactly: 7*S + a*1 = T  (S = sum of 31 entries in 0..7).
    hw = HWConfig(4, 4, (8.0, 8.0, 8.0))
    targets = [t * sg for t in range(500, 525) for sg in (1, -1)]
    X = np.zeros((len(targets), 32))
    for r, T in enumerate(targets):
        S = int(round(abs(T) / 7))
        a_ = abs(T) - 7 * S
        ent = np.zeros(31)
        ent[0], rest = 7, S - 7
        for j in range(1, 31):
            ent[j] = min(7, rest)
            rest -= ent[j]
        X[r, :31], X[r, 31] = ent, a_
        X[r] *= np.sign(T)
    W = pinned_ints(rng, (32, 16), 7, axis=0)
    W[:, 0] = 7
    W[31, 0] = 1
    P = X @ W
    assert np.array_equal(P[:, 0], np.array(targets, float)), "edge construction failed"
    edge = {v: int((P[:, 0] == v).sum()) for v in (-514, -513, -512, -511, -510, 510, 511, 512, 513, 514)}
    compare_case("T1b", f"{label} saturation edge W4A4 G=8 (P = +-500..+-524, FS = 512)", hw, "qkv",
                 X, W, dev, extra=dict(P_edge_counts=edge))

    # (d) zeros, tiny values, eps scale path
    hw = HWConfig(8, 8, (23.77, 8.41, 2.92))
    X = rng.normal(size=(20, 70))
    X[3] = 0.0
    X[5] = 1e-30
    W = rng.normal(size=(70, 33))
    W[:, 4] = 0.0
    W[:, 7] = -1e-30
    compare_case("T1b", f"{label} zero/tiny rows and columns", hw, "qkv", X, W, dev, extra={})
    compare_case("T1b", f"{label} all-zero matrices", hw, "qkv", np.zeros((5, 64)), np.zeros((64, 32)),
                 dev, extra={})

    # (e) ragged K, N = 1, ragged M
    for K in (1, 17, 31, 33, 49, 196):
        for (wb, ab), G in [((8, 8), 23.77), ((4, 4), 6.5)]:
            hw = HWConfig(wb, ab, (G, G, G))
            compare_case("T1b", f"{label} ragged K={K} W{wb}A{ab}", hw, "qk",
                         rng.normal(size=(1 if K == 17 else 9, K)), rng.normal(size=(K, 33)), dev, extra={})

    # (f) unsigned AV: exact zeros, one-hot rows, uniform rows
    for (wb, ab), G in [((8, 8), 2.92), ((4, 4), 1.0)]:
        hw = HWConfig(wb, ab, (G, G, G))
        A = np.abs(rng.normal(size=(49, 49))) ** 6
        A[A < 0.05] = 0.0
        A[0] = 0.0
        A[0, 3] = 1.0
        A[1] = 1.0 / 49
        A = A / np.maximum(A.sum(1, keepdims=True), 1e-30)
        compare_case("T1b", f"{label} unsigned AV zeros/one-hot W{wb}A{ab}", hw, "av", A,
                     rng.normal(size=(49, 32)), dev, extra=dict(zero_frac=float((A == 0).mean())))

    # (g) batched leading dims vs cim_matmul_batched
    hw = HWConfig(8, 8, (23.77, 8.41, 2.92))
    q = rng.normal(size=(2, 5, 196, 32))
    kt = rng.normal(size=(2, 5, 32, 196))
    compare_case("T1b", f"{label} batched QK^T [2,5,196,32]@[2,5,32,196]", hw, "qk", q, kt, dev, extra={})
    a = np.abs(rng.normal(size=(2, 5, 49, 49))) ** 3
    a /= a.sum(-1, keepdims=True)
    compare_case("T1b", f"{label} batched AV [2,5,49,49]@[2,5,49,32]", hw, "av", a,
                 rng.normal(size=(2, 5, 49, 32)), dev, extra={})

    # (h) heavy clipping
    for (wb, ab) in [(8, 8), (4, 4)]:
        hw = HWConfig(wb, ab, (300.0, 300.0, 300.0))
        compare_case("T1b", f"{label} heavy clipping G=300 W{wb}A{ab}", hw, "qkv",
                     rng.normal(size=(64, 160)), rng.normal(size=(160, 96)), dev, extra={})


# --------------------------------------------------------------------------- #
# module / model tiers
# --------------------------------------------------------------------------- #


def t2_attention(hw, dev, exact: bool, label):
    from cim_sim import CIMArray
    from cim_sim.attention import CIMAttention
    from cim_qat.attention import CIMAttentionSurrogate
    from cim_qat.hw_config import op_specs
    from cim_qat.model import build_fp32
    m = build_fp32()
    torch.manual_seed(3)
    mods = [(n, x) for n, x in m.named_modules() if type(x).__name__ == "Attention"]
    worst = 0.0
    nbad_total = 0
    for name, ref in mods:
        C = ref.norm.normalized_shape[0]
        N = ref.attention_bias_idxs.shape[0]
        x = torch.randn(3, N, C)
        sim = CIMAttention(ref, array=CIMArray(hw.cim_config()), name=name).eval()
        sur = CIMAttentionSurrogate(copy.deepcopy(ref), op_specs(hw), name=name).to(dev).eval()
        with torch.no_grad():
            ys = sim(x)
            yu = sur(x.to(dev)).cpu()
        nb = int((ys != yu).sum())
        nbad_total += nb
        worst = max(worst, float((ys - yu).abs().max() / ys.abs().max()))
        if exact:
            record("T2", f"{label} {name} N={N} C={C}", nb == 0, mismatches=nb)
    if not exact:
        record("T2", f"{label} 10 attention modules (GPU float ops; informational)", True,
               differing_elements=nbad_total, worst_rel_maxabs=f"{worst:.2e}")


def t3_full(hw, images: torch.Tensor, label):
    from cim_sim import CIMArray
    from cim_sim.attention import convert_attention
    from cim_qat.attention import convert_attention_surrogate
    from cim_qat.hw_config import op_specs
    from cim_qat.model import build_fp32
    base = build_fp32()
    sim, sur = copy.deepcopy(base), copy.deepcopy(base)
    convert_attention(sim, CIMArray(hw.cim_config()))
    convert_attention_surrogate(sur, op_specs(hw))
    nb, t_sim, t_sur = 0, 0.0, 0.0
    with torch.no_grad():
        for i in range(images.shape[0]):
            x = images[i:i + 1]
            t = time.time()
            ys = sim(x)
            t_sim += time.time() - t
            t = time.time()
            yu = sur(x)
            t_sur += time.time() - t
            nb += int((ys != yu).sum())
    record("T3", f"{label} full model, {images.shape[0]} images, batch 1", nb == 0,
           mismatching_logits=nb, sim_s_per_img=round(t_sim / images.shape[0], 2),
           sur_cpu_s_per_img=round(t_sur / images.shape[0], 2))


def t4_gpu_vs_sim(hw, dev, splits, n_real, workers, out):
    from cim_qat import metrics as M
    from cim_qat import sim_eval
    from cim_qat.attention import convert_attention_surrogate, merged_stats, set_collect_stats
    from cim_qat.data import EvalImages
    from cim_qat.hw_config import op_specs
    from cim_qat.model import PRETRAINED, build_fp32
    rows = splits.rows("val")
    idx = np.linspace(0, len(rows) - 1, n_real).astype(int)
    items = [(rows[i]["idx"], str(splits.path(rows[i]))) for i in idx]
    labels = np.array([rows[i]["label"] for i in idx])
    job = dict(kind="analog", ckpt=str(PRETRAINED), hw=hw.job_dict(), with_own_fp32=True)
    sim = sim_eval.run(job, items, Path(out) / "t4_sim", workers=workers)
    sim_u = sim_eval.run(dict(job, input_perturb="ulp_up", with_own_fp32=False), items,
                         Path(out) / "t4_sim_ulp", workers=workers)
    ds = EvalImages(splits, "val", indices=list(idx))
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
    fp_gpu = build_fp32().to(dev).eval()
    sur_cpu = build_fp32()
    convert_attention_surrogate(sur_cpu, op_specs(hw))
    sur_cpu.eval()
    sur = copy.deepcopy(sur_cpu).to(dev).eval()
    set_collect_stats(sur, True)
    lg_sur, lg_fp, lg_cpu_b, first = [], [], [], None
    with torch.no_grad():
        for x, _, _ in dl:
            lg_cpu_b.append(sur_cpu(x).numpy())          # CPU, batched: batch-order floor
            xd = x.to(dev)
            lg_sur.append(sur(xd).cpu().numpy())
            lg_fp.append(fp_gpu(xd).cpu().numpy())
            if first is None:
                first = xd
        ms = merged_stats(sur)
        set_collect_stats(sur, False)
        b1 = torch.cat([sur(first[i:i + 1]) for i in range(first.shape[0])]).cpu().numpy()
    # EXACT check: inputs built exactly as the simulator workers build them
    # (cim_exp to_model_input -> channels-last strides, batch 1, ONE thread).
    # A DataLoader batch is contiguous NCHW: same values, but a different conv
    # kernel/rounding, which the CIM rounding then amplifies.
    from cim_qat.common import single_thread
    from cim_qat.data import load_uint8_eval, to_model_input
    with torch.no_grad(), single_thread():
        lg_cpu_1 = [sur_cpu(to_model_input(load_uint8_eval(p)[None])).numpy() for _, p in items]
    lg_sur, lg_fp = np.concatenate(lg_sur), np.concatenate(lg_fp)
    lg_cpu_b, lg_cpu_1 = np.concatenate(lg_cpu_b), np.concatenate(lg_cpu_1)
    assert lg_cpu_1.shape == lg_sur.shape
    S = lambda lg, ref: M.summarize(M.per_image(lg, labels, ref))           # noqa: E731
    s_hw = S(sim["logits"], sim["logits_own_fp32"])        # hardware effect, FP32-CPU ref
    s_dev = S(lg_sur, sim["logits"])                       # GPU surrogate vs simulator
    s_floor = S(sim_u["logits"], sim["logits"])            # simulator self-sensitivity
    s_reord = S(lg_cpu_b, sim["logits"])                   # CPU batch-order floor
    s_fp = S(lg_fp, sim["logits_own_fp32"])                # FP32 GPU vs CPU
    n_exact = int((lg_cpu_1 == sim["logits"]).all(1).sum())
    record("T4", f"CPU surrogate batch 1 == simulator on {n_real} real VAL images", n_exact == n_real,
           bit_exact_images=f"{n_exact}/{n_real}")
    ratio = s_dev["kl_ref"] / max(s_floor["kl_ref"], 1e-15)
    ok = s_dev["kl_ref"] <= 2 * s_floor["kl_ref"]
    record("T4", f"GPU surrogate vs CPU simulator, judged against the 1-ulp floor", ok,
           verdict="PASS" if ok else "REVIEW", kl_sim_to_gpu=f"{s_dev['kl_ref']:.2e}",
           kl_floor_1ulp=f"{s_floor['kl_ref']:.2e}", ratio_to_floor=f"{ratio:.2f}",
           kl_cpu_batch_order=f"{s_reord['kl_ref']:.2e}", kl_hw_effect_fp32_to_sim=f"{s_hw['kl_ref']:.2e}",
           top1_agree_gpu_vs_sim=f"{s_dev['agree_ref_pct']:.2f}%", top1_agree_floor=f"{s_floor['agree_ref_pct']:.2f}%")
    record("T4", "FP32 noise floor: GPU vs CPU (informational)", True,
           kl=f"{s_fp['kl_ref']:.2e}", agree=f"{s_fp['agree_ref_pct']:.2f}%",
           rel_logit_rms=f"{s_fp['rel_logit_rms_pct']:.4f}%")
    a = sim["summary"]["analog_summary"]
    clip_cmp = {op: dict(sim=a[op]["clip_hi"] + a[op]["clip_lo"],
                         sur=ms[op].clip_hi + ms[op].clip_lo) for op in ("qkv", "qk", "av")}
    record("T4", "per-op clipped conversions, GPU surrogate vs simulator (informational)", True,
           **{op: f"{v['sur']} vs {v['sim']}" for op, v in clip_cmp.items()})
    bmax = float(np.abs(b1 - lg_sur[:first.shape[0]]).max())
    record("T4", "GPU batch-size invariance: batch 1 vs batch 32 (informational)", True,
           max_abs_logit_diff=f"{bmax:.3e}")
    return dict(hw_effect=s_hw, gpu_vs_sim=s_dev, floor_1ulp=s_floor, cpu_batch_order=s_reord,
                fp32_gpu_vs_cpu=s_fp, ratio_to_floor=ratio, cpu_batch1_exact_images=n_exact,
                clip_counts=clip_cmp, batch_invariance_max_abs=bmax)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--frozen-gains", default=None,
                    help="frozen_gains.json (T2/T3/T4 use these gains; default W8A8 23.77/8.41/2.92)")
    ap.add_argument("--precision", default="W8A8")
    ap.add_argument("--paths", default=None, help="configs/paths.json (real images for T3/T4)")
    ap.add_argument("--n-real", type=int, default=500)
    ap.add_argument("--n-real-t3", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="results/surrogate_validation")
    ap.add_argument("--skip-grid", action="store_true")
    ap.add_argument("--force-t4", action="store_true", help="debug: run T4 code path on CPU")
    a = ap.parse_args()
    configure_determinism(True, tf32=False)
    from cim_qat.hw_config import HWConfig, load_frozen
    wb, ab = (8, 8) if a.precision.upper() == "W8A8" else (4, 4)
    if a.frozen_gains:
        hw, _ = load_frozen(a.frozen_gains, wb, ab)
    else:
        hw = HWConfig(wb, ab, (23.77, 8.41, 2.92) if wb == 8 else (6.5, 2.4, 0.9))
        print(f"NOTE: no --frozen-gains; using placeholder gains {hw.gains} (contract test only)")
    dev = select_device(a.device)
    print(f"device {dev}; gains {hw.gain_dict}; W{wb}A{ab}")
    t0 = time.time()

    devices = [("cpu", torch.device("cpu"), False)]
    if dev.type == "cuda":
        devices += [("cuda-tf32off", dev, False), ("cuda-tf32on", dev, True)]
    for label, d, tf32 in devices:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        print(f"\n== T1/T1b per-matmul on {label}")
        if not a.skip_grid:
            t1_grid(d, label)
        t1_boundary(d, label)
    torch.backends.cuda.matmul.allow_tf32 = False

    print("\n== T2 attention modules")
    t2_attention(hw, torch.device("cpu"), True, "cpu")
    if dev.type == "cuda":
        t2_attention(hw, dev, False, "cuda")

    print("\n== T3 full model (CPU)")
    import run_tinyvit
    t3_full(hw, run_tinyvit.synthetic_images(3, seed=0).float(), "synthetic")
    splits = None
    if a.paths:
        from cim_qat.data import DataPaths, EvalImages, Splits
        splits = Splits(DataPaths.load(a.paths).splits_dir, purpose="surrogate_check")
        ds = EvalImages(splits, "val", indices=list(range(0, len(splits.rows("val")),
                                                         max(1, len(splits.rows("val")) // a.n_real_t3)))[:a.n_real_t3])
        t3_full(hw, torch.stack([ds[i][0] for i in range(len(ds))]), "real VAL")

    t4 = None
    if (dev.type == "cuda" or a.force_t4) and splits is not None:
        print("\n== T4 GPU surrogate vs CPU simulator (real images)")
        t4 = t4_gpu_vs_sim(hw, dev, splits, a.n_real, a.workers, a.out)

    exact_fail = [r for r in RESULTS if not r["ok"] and (r["tier"] in ("T1", "T1b", "T2", "T3")
                                                         or r["name"].startswith("CPU surrogate batch 1"))]
    rep = dict(time_utc=now_iso(), device=str(dev), hw=hw.to_dict(), results=RESULTS, t4=t4,
               wall_s=time.time() - t0, exact_tiers_pass=not exact_fail)
    write_json(Path(a.out) / f"surrogate_validation_{dev.type}.json", rep)
    print(f"\n{sum(r['ok'] for r in RESULTS)}/{len(RESULTS)} checks passed "
          f"({len(exact_fail)} exact-tier failures); report -> {a.out}")
    sys.exit(1 if exact_fail else 0)


if __name__ == "__main__":
    main()
