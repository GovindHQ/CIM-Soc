"""
cim_qat/calibration.py — per-operation pre-ADC gain calibration (Stage A).

METHOD (pre-declared; reuses cim_exp/calibration.py's probe and arithmetic)
--------------------------------------------------------------------------
1. Run the ACTUAL simulator network (pretrained TinyViT-5M, cim_sim
   CIMAttention + cim_matmul tiling) on the CAL images with the exact probe
   (cim_exp.calibration.ProbeChain) in place of the analog chain: every
   32-row depth block returns its exact integer column sum P_b, so the network
   sees operand quantization only (no ADC, no gain -> no circularity).
   One probe per attention module, so histograms of |P_b| exist per op AND
   per layer (exact, integer bins).
2. Pooled over the 10 layers (one shared gain per op, three gain registers):
       FS_op(level) = smallest integer FS with count(|P| > FS) <= (1-q) * n
       G_op(level)  = (V_REF_ADC / 2) / (K_MAC_op * FS_op),  rounded to 0.01
   level p99.99 is the PRIMARY objective (clip target <= 0.01 % per op);
   p99.9 / p99.999 / p100 are computed for the sensitivity check.
3. Diagnostics from the same histograms (predictions, no simulation):
   predicted clip % per op and per layer; predicted ADC code use (|d|);
   the ADC-output-MSE-optimal FS per op and the per-conversion SQNR of each
   level (how far each level is from the clipping/rounding optimum).
4. CAL-check: the actual analog simulator (PTQ, pretrained) on the 200
   CALCHECK images at p99.9 / p99.99 / p99.999, plus the ADC-off floor and the
   FP32 reference: KL(FP32 || PTQ), agreement, logit error, NLL, measured
   clip % and ADC utilisation per op.
5. Material trade-off flags (pre-declared, see FLAG_RULES); gains are frozen
   only by the explicit `freeze` command after review.
"""

from __future__ import annotations

import copy
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .common import configure_threads, now_iso, read_json, sha256_file, write_json
from .hw_config import OP_SIGNED, OPS, HWConfig

LEVELS = {"p99.9": 99.9, "p99.99": 99.99, "p99.999": 99.999, "p100": 100.0}
PRIMARY = "p99.99"
CHECK_LEVELS = ("p99.9", "p99.99", "p99.999")
CLIP_TARGET_PCT = {k: round(100.0 - q, 6) for k, q in LEVELS.items()}
FLAG_RULES = dict(
    measured_clip_excess=2.0,       # CALCHECK clip % > 2x the level's target (cim_exp MATERIAL_EXCESS)
    kl_rel_improvement=0.10,        # another level lowers mean KL(FP32||PTQ) by >=10 % with CI excl. 0
    mse_fs_ratio=2.0,               # MSE-optimal FS differs from the primary FS by > 2x
    layer_clip_excess=10.0,         # one layer's predicted clip % > 10x target
    min_median_code=2,              # median |d| <= 2 codes for an op
)


def k_mac(hw: HWConfig, op: str) -> float:
    from cim_sim.analog import AnalogChain
    ch = AnalogChain(hw.analog_config(), hw.rows, hw.weight_bits, hw.act_bits)
    ch.set_operation(op, act_signed=OP_SIGNED[op])
    return float(ch.k_mac)


def hist_max(hw: HWConfig) -> int:
    from cim_exp.calibration import hist_range
    return hist_range(hw.weight_bits, hw.act_bits, hw.rows)


# --------------------------------------------------------------------------- #
# 1. observation (actual simulator, exact probe, per layer)
# --------------------------------------------------------------------------- #

_P = {}


def _obs_init(hw_d: dict, ckpt: str):
    from .common import die_with_parent, limit_blas_threads
    die_with_parent()
    configure_threads(1)
    limit_blas_threads(1)
    import torch
    from cim_exp.calibration import ProbeChain
    from cim_sim import CIMArray
    from cim_sim.attention import CIMAttention

    from .model import build_fp32
    torch.set_grad_enabled(False)
    hw = HWConfig(**hw_d)
    m = build_fp32(ckpt)
    probes = {}

    def walk(parent, prefix):
        for name, child in list(parent.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if type(child).__name__ == "Attention" and hasattr(child, "key_dim"):
                arr = CIMArray(hw.cim_config())
                pr = ProbeChain(hist_max(hw))
                arr.chain = pr                       # exact MAC, observation only
                setattr(parent, name, CIMAttention(child, array=arr, name=path))
                probes[path] = pr
            else:
                walk(child, path)
    walk(m, "")
    if len(probes) != 10:
        raise RuntimeError(f"expected 10 attention modules, got {len(probes)}")
    _P.update(model=m, probes=probes)


def _obs_shard(paths):
    from .data import load_uint8_eval, to_model_input
    t0 = time.time()
    for p in paths:
        _P["model"](to_model_input(load_uint8_eval(p)[None]))
        for pr in _P["probes"].values():
            pr.flush()
    out = {}
    for layer, pr in _P["probes"].items():
        for op, h in pr.hist.items():
            nz = np.nonzero(h)[0]
            out[(layer, op)] = (nz.astype(np.int64), h[nz])
        pr.hist.clear()
        pr.maxabs.clear()
    return out, len(paths), time.time() - t0


def observe(hw: HWConfig, ckpt: str, paths: list, workers: int) -> dict:
    """-> {"per_layer": {(layer, op): dense hist}, "pooled": {op: dense hist}, ...}"""
    H = hist_max(hw)
    shards = [paths[i::workers] for i in range(workers)]
    shards = [s for s in shards if s]
    t0 = time.time()
    per = {}
    ctx = get_context("fork")
    with ctx.Pool(len(shards), initializer=_obs_init, initargs=(hw.job_dict(), ckpt)) as pool:
        n_done = 0
        for out, n, _ in pool.imap_unordered(_obs_shard, shards):
            n_done += n
            for key, (nz, cnt) in out.items():
                h = per.setdefault(key, np.zeros(H + 1, np.int64))
                h[nz] += cnt
            print(f"    observed {n_done}/{len(paths)} images ({time.time() - t0:.0f}s)", flush=True)
    pooled = {op: sum(h for (l, o), h in per.items() if o == op) for op in OPS}
    return dict(per_layer=per, pooled=pooled, n_images=len(paths), wall_s=time.time() - t0)


def save_hists(obs: dict, path: Path):
    arrs = {}
    for (layer, op), h in obs["per_layer"].items():
        nz = np.nonzero(h)[0]
        arrs[f"L|{layer}|{op}|idx"] = nz
        arrs[f"L|{layer}|{op}|cnt"] = h[nz]
    np.savez_compressed(path, **arrs)


def load_hists(path: Path, H: int) -> dict:
    per = {}
    with np.load(path) as z:
        for k in z.files:
            if k.endswith("|idx"):
                _, layer, op, _ = k.split("|")
                h = np.zeros(H + 1, np.int64)
                h[z[k]] = z[k[:-3] + "cnt"]
                per[(layer, op)] = h
    pooled = {op: sum(h for (l, o), h in per.items() if o == op) for op in OPS}
    return dict(per_layer=per, pooled=pooled)


# --------------------------------------------------------------------------- #
# 2-3. gains and histogram diagnostics
# --------------------------------------------------------------------------- #


def _code_of_abs(v: np.ndarray, lsb: float) -> np.ndarray:
    """|d| predicted from |P| (sign-agnostic; saturates at 128 codes)."""
    return np.minimum(np.floor(v / lsb + 0.5), 128)


def hist_stats(h: np.ndarray, gain: float, kmac: float, adc_vref: float = 0.6) -> dict:
    """Predicted clip %, code use and per-conversion SQNR for one histogram."""
    from cim_exp.calibration import predicted_clip_pct
    v = np.nonzero(h)[0].astype(np.float64)
    c = h[v.astype(np.int64)].astype(np.float64)
    n = c.sum()
    lsb = (adc_vref / 2 ** 8) / (gain * kmac)
    code = _code_of_abs(v, lsb)
    q = np.minimum(code, 127) * lsb                       # positive-side reconstruction
    err = ((q - v) ** 2 * c).sum()
    sig = (v ** 2 * c).sum()
    order = np.argsort(code, kind="stable")
    cum = np.cumsum(c[order])
    med = code[order][np.searchsorted(cum, 0.5 * n)]
    p999 = code[order][min(np.searchsorted(cum, 0.999 * n), len(cum) - 1)]
    return dict(pred_clip_pct=predicted_clip_pct(h, gain, kmac, adc_vref),
                lsb_p=lsb, fs=128 * lsb,
                median_abs_code=float(med), p999_abs_code=float(p999),
                util_p999_pct=100 * float(p999) / 128,
                frac_codes_le1_pct=100 * float(c[code <= 1].sum() / n),
                sqnr_db=10 * np.log10(sig / err) if err > 0 else float("inf"),
                mse_per_conv=float(err / n))


def mse_optimal_fs(h: np.ndarray, n_grid: int = 240) -> float:
    """FS minimising sum h(v) (Q(v)-v)^2 for the 8-bit mid-tread ADC (|P| bins)."""
    v = np.nonzero(h)[0].astype(np.float64)
    c = h[v.astype(np.int64)].astype(np.float64)
    if v.size == 0:
        return 1.0
    lo = max(1.0, np.percentile(np.repeat(v, np.minimum(c, 50).astype(int)), 50))
    grid = np.geomspace(lo, v.max() * 1.05, n_grid)
    best, best_fs = np.inf, grid[-1]
    for fs in grid:
        lsb = fs / 128
        q = np.minimum(np.floor(v / lsb + 0.5), 127) * lsb
        e = ((q - v) ** 2 * c).sum()
        if e < best:
            best, best_fs = e, fs
    return float(best_fs)


def compute_gains(hw0: HWConfig, obs: dict) -> dict:
    from cim_exp.calibration import full_scale, gain_for_fs
    out = dict(kmac={op: k_mac(hw0, op) for op in OPS}, levels={}, mse_optimal={},
               n_conversions={op: int(obs["pooled"][op].sum()) for op in OPS},
               max_abs_P={op: int(np.nonzero(obs["pooled"][op])[0].max()) for op in OPS})
    for lvl, q in LEVELS.items():
        fs = {op: full_scale(obs["pooled"][op], q) for op in OPS}
        g = {op: round(gain_for_fs(fs[op], out["kmac"][op], hw0.adc_vref), 2) for op in OPS}
        diag = {op: hist_stats(obs["pooled"][op], g[op], out["kmac"][op], hw0.adc_vref) for op in OPS}
        layers = {}
        for (layer, op), h in sorted(obs["per_layer"].items()):
            st = hist_stats(h, g[op], out["kmac"][op], hw0.adc_vref)
            layers.setdefault(layer, {})[op] = dict(pred_clip_pct=st["pred_clip_pct"],
                                                    median_abs_code=st["median_abs_code"],
                                                    util_p999_pct=st["util_p999_pct"])
        out["levels"][lvl] = dict(quantile_pct=q, clip_target_pct=CLIP_TARGET_PCT[lvl],
                                  fs=fs, gains=g, pooled=diag, per_layer=layers)
    for op in OPS:
        fs = mse_optimal_fs(obs["pooled"][op])
        g = round(gain_for_fs(fs, out["kmac"][op], hw0.adc_vref), 2)
        st = hist_stats(obs["pooled"][op], g, out["kmac"][op], hw0.adc_vref)
        st.update(fs_unrounded=fs, gain=g)        # st["fs"] is the FS of the rounded gain
        out["mse_optimal"][op] = st
    return out


# --------------------------------------------------------------------------- #
# 4. CAL-check on the actual analog simulator
# --------------------------------------------------------------------------- #


def calcheck(hw0: HWConfig, gains: dict, ckpt: str, items: list, labels: np.ndarray,
             out_dir: Path, workers: int) -> dict:
    from . import metrics as M
    from . import sim_eval
    res = {}
    fp = sim_eval.run(dict(kind="fp32", ckpt=ckpt), items, out_dir / "fp32", workers=workers)
    ref = fp["logits"]
    off = sim_eval.run(dict(kind="adc_off", ckpt=ckpt, hw=hw0.job_dict(), with_own_fp32=False),
                       items, out_dir / "adc_off", workers=workers)
    pis = {"adc_off": M.per_image(off["logits"], labels, ref)}
    res["adc_off"] = dict(summary=M.summarize(pis["adc_off"]))
    for lvl in CHECK_LEVELS:
        g = gains["levels"][lvl]["gains"]
        hw = HWConfig(hw0.weight_bits, hw0.act_bits, (g["qkv"], g["qk"], g["av"]))
        r = sim_eval.run(dict(kind="analog", ckpt=ckpt, hw=hw.job_dict(), with_own_fp32=False),
                         items, out_dir / lvl, workers=workers)
        pis[lvl] = M.per_image(r["logits"], labels, ref)
        a = r["summary"]["analog_summary"]
        res[lvl] = dict(gains=g, summary=M.summarize(pis[lvl]),
                        measured={op: dict(clip_pct=a[op]["clip_pct"], clip_hi=a[op]["clip_hi"],
                                           clip_lo=a[op]["clip_lo"], n=a[op]["n"],
                                           util_median_pct=a[op].get("util_median_pct"),
                                           util_p999_pct=a[op].get("util_p999_pct"),
                                           util_max_pct=a[op]["util_max_pct"])
                                  for op in OPS})
    for lvl in CHECK_LEVELS:
        if lvl != PRIMARY:
            res[lvl]["vs_primary"] = M.compare(pis[PRIMARY], pis[lvl])
    res["fp32"] = dict(summary=M.summarize(M.per_image(ref, labels)))
    return res


# --------------------------------------------------------------------------- #
# 5. flags and report
# --------------------------------------------------------------------------- #


def flags(gains: dict, check: dict | None) -> list:
    F = []
    prim = gains["levels"][PRIMARY]
    for op in OPS:
        r = gains["mse_optimal"][op]["fs"] / prim["fs"][op]
        if r > FLAG_RULES["mse_fs_ratio"] or r < 1 / FLAG_RULES["mse_fs_ratio"]:
            F.append(f"{op}: ADC-MSE-optimal FS is {r:.2f}x the P99.99 FS")
        if prim["pooled"][op]["median_abs_code"] <= FLAG_RULES["min_median_code"]:
            F.append(f"{op}: median |code| = {prim['pooled'][op]['median_abs_code']:.0f} at P99.99 "
                     f"(most conversions use very few ADC codes)")
    for layer, d in prim["per_layer"].items():
        for op, st in d.items():
            if st["pred_clip_pct"] > FLAG_RULES["layer_clip_excess"] * CLIP_TARGET_PCT[PRIMARY]:
                F.append(f"{layer} {op}: predicted clip {st['pred_clip_pct']:.4f}% > "
                         f"{FLAG_RULES['layer_clip_excess']:g}x target with the shared gain")
    if check:
        for op in OPS:
            m = check[PRIMARY]["measured"][op]["clip_pct"]
            if m > FLAG_RULES["measured_clip_excess"] * CLIP_TARGET_PCT[PRIMARY]:
                F.append(f"{op}: CALCHECK measured clip {m:.4f}% > "
                         f"{FLAG_RULES['measured_clip_excess']:g}x the 0.01% target")
        base = check[PRIMARY]["summary"]["kl_ref"]
        for lvl in CHECK_LEVELS:
            if lvl == PRIMARY:
                continue
            d = check[lvl]["vs_primary"]["kl_ref_diff"]
            if d["ci95"][1] < 0 and -d["mean"] >= FLAG_RULES["kl_rel_improvement"] * base:
                F.append(f"{lvl} lowers mean KL(FP32||PTQ) on CALCHECK by "
                         f"{-d['mean']:.4f} ({100 * -d['mean'] / base:.0f}%, CI {d['ci95'][0]:.4f}.."
                         f"{d['ci95'][1]:.4f}) vs P99.99")
    return F


def render_md(rep: dict) -> str:
    g = rep["gains"]
    L = [f"# Gain calibration — {rep['precision']} (ADC 8 b, capacitive DAC)\n",
         f"- created {rep['created_utc']}; code fingerprint `{rep['code_fingerprint'][:16]}`",
         f"- checkpoint: pretrained TinyViT-5M, sha256 `{rep['checkpoint_sha256'][:16]}`",
         f"- CAL: {rep['n_cal']} ImageNet-train images (manifest `{rep['cal_manifest_sha256'][:16]}`); "
         f"CALCHECK: {rep.get('n_calcheck', 0)} images (manifest `{rep.get('calcheck_manifest_sha256', '')[:16]}`)",
         f"- conversions observed: " + ", ".join(f"{op} {g['n_conversions'][op]:,}" for op in OPS),
         f"- K_MAC (V/unit): " + ", ".join(f"{op} {g['kmac'][op]:.4e}" for op in OPS) + "\n",
         "## Candidate gains (shared per operation)\n",
         "| level | clip target % | G qkv | G qk | G av | FS qkv | FS qk | FS av | pred. clip % qkv/qk/av | median abs code qkv/qk/av | SQNR dB qkv/qk/av |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for lvl, d in g["levels"].items():
        p = d["pooled"]
        L.append(f"| {lvl}{' (primary)' if lvl == PRIMARY else ''} | {d['clip_target_pct']} | "
                 + " | ".join(f"{d['gains'][o]:.2f}" for o in OPS) + " | "
                 + " | ".join(str(d['fs'][o]) for o in OPS) + " | "
                 + "/".join(f"{p[o]['pred_clip_pct']:.4f}" for o in OPS) + " | "
                 + "/".join(f"{p[o]['median_abs_code']:.0f}" for o in OPS) + " | "
                 + "/".join(f"{p[o]['sqnr_db']:.1f}" for o in OPS) + " |")
    L.append("| ADC-MSE-optimal (diagnostic) | — | "
             + " | ".join(f"{g['mse_optimal'][o]['gain']:.2f}" for o in OPS) + " | "
             + " | ".join(f"{g['mse_optimal'][o]['fs']:.0f}" for o in OPS) + " | "
             + "/".join(f"{g['mse_optimal'][o]['pred_clip_pct']:.4f}" for o in OPS) + " | "
             + "/".join(f"{g['mse_optimal'][o]['median_abs_code']:.0f}" for o in OPS) + " | "
             + "/".join(f"{g['mse_optimal'][o]['sqnr_db']:.1f}" for o in OPS) + " |")
    L += ["\n## Per-layer predicted clip % at P99.99 (shared gains)\n",
          "| layer | qkv | qk | av | median abs code qkv/qk/av |", "|---|---|---|---|---|"]
    for layer, d in g["levels"][PRIMARY]["per_layer"].items():
        L.append(f"| {layer} | " + " | ".join(f"{d[o]['pred_clip_pct']:.4f}" for o in OPS) + " | "
                 + "/".join(f"{d[o]['median_abs_code']:.0f}" for o in OPS) + " |")
    c = rep.get("calcheck")
    if c:
        L += ["\n## CAL-check: actual analog simulator, PTQ (pretrained), CALCHECK images\n",
              "Reference = FP32 pretrained on the same images. KL = mean KL(FP32 || model).\n",
              "| config | KL | agree % | rel logit RMS % | NLL | top-1 % | clip % qkv/qk/av | util p99.9 % qkv/qk/av | util median % qkv/qk/av |",
              "|---|---|---|---|---|---|---|---|---|"]
        s = c["fp32"]["summary"]
        L.append(f"| FP32 | 0 | 100 | 0 | {s['nll']:.4f} | {s['top1_pct']:.1f} | — | — | — |")
        s = c["adc_off"]["summary"]
        L.append(f"| ADC off (operand quantization only) | {s['kl_ref']:.4f} | {s['agree_ref_pct']:.1f} | "
                 f"{s['rel_logit_rms_pct']:.2f} | {s['nll']:.4f} | {s['top1_pct']:.1f} | — | — | — |")
        for lvl in CHECK_LEVELS:
            s, m = c[lvl]["summary"], c[lvl]["measured"]
            L.append(f"| analog {lvl} | {s['kl_ref']:.4f} | {s['agree_ref_pct']:.1f} | "
                     f"{s['rel_logit_rms_pct']:.2f} | {s['nll']:.4f} | {s['top1_pct']:.1f} | "
                     + "/".join(f"{m[o]['clip_pct']:.4f}" for o in OPS) + " | "
                     + "/".join(f"{(m[o]['util_p999_pct'] or 0):.0f}" for o in OPS) + " | "
                     + "/".join(f"{(m[o]['util_median_pct'] or 0):.1f}" for o in OPS) + " |")
        for lvl in CHECK_LEVELS:
            if lvl != PRIMARY:
                d = c[lvl]["vs_primary"]["kl_ref_diff"]
                L.append(f"\n- {lvl} minus P99.99, paired mean KL difference: {d['mean']:+.4f} "
                         f"(95% CI {d['ci95'][0]:+.4f} .. {d['ci95'][1]:+.4f}, n={d['n']})")
    L += ["\n## Flags (pre-declared rules)\n"]
    L += [f"- {f}" for f in rep["flags"]] or ["- none"]
    L.append(f"\n**Status:** {'REVIEW REQUIRED before freezing' if rep['flags'] else 'no material trade-off flagged'}. "
             "Gains are frozen only by `qat_calibrate.py freeze` (see QAT_RUNBOOK.md).")
    return "\n".join(L)


def freeze(cal_dir: Path, precision: tuple, level: str, force: bool = False) -> dict:
    from .hw_config import FROZEN_SCHEMA
    rep_p = cal_dir / "calibration_report.json"
    rep = read_json(rep_p)
    if (rep["weight_bits"], rep["act_bits"]) != precision:
        raise RuntimeError("report precision mismatch")
    if level not in rep["gains"]["levels"]:
        raise ValueError(f"unknown level {level}")
    out = cal_dir / "frozen_gains.json"
    if out.exists() and not force:
        raise FileExistsError(f"{out} exists; gains are frozen. --force only before any training used them.")
    g = rep["gains"]["levels"][level]["gains"]
    rec = dict(schema=FROZEN_SCHEMA, weight_bits=rep["weight_bits"], act_bits=rep["act_bits"],
               adc_bits=8, gains=g, level=level, primary_level=PRIMARY,
               deviates_from_primary=(level != PRIMARY), flags_at_freeze=rep["flags"],
               calibration_report=str(rep_p), calibration_report_sha256=sha256_file(rep_p),
               cal_manifest_sha256=rep["cal_manifest_sha256"],
               calcheck_manifest_sha256=rep.get("calcheck_manifest_sha256"),
               data_used=["cal", "calcheck"], test_split_used=False,
               checkpoint_sha256=rep["checkpoint_sha256"], code_fingerprint=rep["code_fingerprint"],
               frozen_utc=now_iso())
    write_json(out, rec)
    return rec
