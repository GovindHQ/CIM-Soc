"""
engine.py — executes FP32 references and CIM jobs. Deterministic, resumable.

Determinism: every model forward runs with batch size 1 and 1 intra-op thread
(torch and BLAS), so results do not depend on batching or core count.
Parallelism comes from independent worker PROCESSES, each owning whole
(job, chunk) units.

Resumability: a CIM job is split into chunks of `chunk_size` manifest images.
Each chunk is written atomically (tmp file + os.replace) as
    cache/cim/<job_key>/chunk_00000.npz + .json      (json written last)
and skipped on restart if present. When all chunks exist the job is finalized
into final.npz + final.json (merged logits, per-image counters, merged analog
statistics, consistency checks); chunk files are then removed unless
--keep-chunks is given.

Per-image attribution: because each image is its own forward pass, the analog
counters are snapshotted around every image, so "conversions clipped" and
"images with >= 1 clipped conversion" are both exact.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

COUNTER_FIELDS = ("n", "clip_hi", "clip_lo", "rail_hi", "rail_lo", "amp_sat")


def configure_threads(n: int = 1) -> None:
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = str(n)
    import torch
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(n)
    except RuntimeError:
        pass


def _setup_paths(repo_root: Path) -> None:
    for p in (str(repo_root / "_shim"), str(repo_root)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=_json_default))
    os.replace(tmp, path)


def _atomic_savez(path: Path, **arrays) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


# --------------------------------------------------------------------------- #
# FP32 reference
# --------------------------------------------------------------------------- #


def run_reference(repo_root: Path, ckpt_path: Path, inputs_path: Path,
                  out_npz: Path, meta: dict) -> None:
    _setup_paths(repo_root)
    configure_threads(1)
    import torch
    import run_tinyvit
    from .imagenette import to_model_input
    model = run_tinyvit.build_model(ckpt_path)
    u8 = np.load(inputs_path, mmap_mode="r")
    out = np.empty((u8.shape[0], 1000), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(u8.shape[0]):
            out[i] = model(to_model_input(u8[i:i + 1])).numpy()[0]
    meta = dict(meta, wall_s=time.time() - t0, n_images=int(u8.shape[0]),
                execution_path="fp32_software")
    _atomic_savez(out_npz, logits=out)
    _atomic_write_json(out_npz.with_suffix(".json"), meta)


# --------------------------------------------------------------------------- #
# CIM chunk (one work unit)
# --------------------------------------------------------------------------- #


def _snapshot(stats, ops) -> np.ndarray:
    s = np.zeros((len(ops), len(COUNTER_FIELDS)), dtype=np.int64)
    for i, op in enumerate(ops):
        b = stats.b.get(op)
        if b is not None:
            s[i] = [b[f] for f in COUNTER_FIELDS]
    return s


def run_cim_chunk(unit: dict) -> dict:
    """Execute one (job, chunk) unit. Safe to call in a worker process."""
    repo_root = Path(unit["repo_root"])
    _setup_paths(repo_root)
    configure_threads(1)
    import torch
    import run_tinyvit
    from cim_sim import CIMArray
    from cim_sim.analog import AnalogStats
    from cim_sim.attention import convert_attention, set_cim_ops
    from .hardware import OPS, HardwareConfig
    from .imagenette import to_model_input

    hw = HardwareConfig.from_dict(unit["hw"])
    job_dir = Path(unit["job_dir"])
    c = unit["chunk"]
    npz_path = job_dir / f"chunk_{c:05d}.npz"
    json_path = job_dir / f"chunk_{c:05d}.json"
    if json_path.exists() and npz_path.exists():
        return dict(job_key=unit["job_key"], chunk=c, skipped=True, wall_s=0.0)

    t0 = time.time()
    stats = AnalogStats(keep_frac=unit["stats_keep_frac"], cap=unit["stats_cap"],
                        seed=_seed(unit["job_key"], c))
    model = run_tinyvit.build_model(Path(unit["ckpt_path"]))
    cfg = hw.cim_config()
    arrays = [CIMArray(cfg, analog_stats=stats) for _ in range(hw.num_arrays)]
    replaced = convert_attention(model, arrays=arrays, cim_out_proj=hw.cim_out_proj)
    set_cim_ops(model, hw.cim_ops)
    # ---- the analog path must really be the one executing --------------- #
    if any(a.chain is None for a in arrays) or cfg.adc_enabled:
        raise RuntimeError("analog chain not active (or legacy ADC enabled)")
    if len(replaced) != 10:
        raise RuntimeError(f"expected 10 TinyViT-5M attention modules, got {len(replaced)}")

    u8 = np.load(unit["inputs_path"], mmap_mode="r")
    i0, i1 = unit["i0"], unit["i1"]
    logits = np.empty((i1 - i0, 1000), dtype=np.float32)
    per_image = np.zeros((i1 - i0, len(OPS), len(COUNTER_FIELDS)), dtype=np.int64)
    with torch.no_grad():
        for j, i in enumerate(range(i0, i1)):
            before = _snapshot(stats, OPS)
            logits[j] = model(to_model_input(u8[i:i + 1])).numpy()[0]
            per_image[j] = _snapshot(stats, OPS) - before

    tot = {k: 0 for k in ("adc_conversions", "adc_saturations", "array_ops",
                          "weight_tile_writes", "macs_useful", "macs_issued")}
    for a in arrays:
        for k in tot:
            tot[k] += getattr(a.stats, k)
    scal, arrs = stats.state_dict()
    ch = arrays[0].chain
    ch.set_operation("qkv")
    meta = dict(job_key=unit["job_key"], chunk=c, i0=i0, i1=i1,
                wall_s=time.time() - t0, cim_stats=tot, analog_scalars=scal,
                replaced_modules=replaced,
                chain=dict(dac=ch.dac.describe(), adc=ch.adc.describe(),
                           dac_alignment=ch.dac_alignment,
                           dac_code_scale=ch.dac_code_scale,
                           amp_saturation_modeled=ch.amp.saturation_modeled,
                           k_mac_qkv=ch.k_mac, lsb_p_qkv=ch.lsb_p))
    _atomic_savez(npz_path, logits=logits, per_image=per_image,
                  **{f"stats__{k}": v for k, v in arrs.items()})
    _atomic_write_json(json_path, meta)        # written last => chunk complete
    return dict(job_key=unit["job_key"], chunk=c, skipped=False, wall_s=meta["wall_s"])


# --------------------------------------------------------------------------- #
# finalize one job
# --------------------------------------------------------------------------- #


def finalize_job(job_dir: Path, n_chunks: int, job_meta: dict, stats_cap: int,
                 keep_chunks: bool = False) -> dict:
    from cim_sim.analog import AnalogStats
    from .hardware import OPS

    logits, per_image, wall = [], [], 0.0
    merged = AnalogStats(cap=stats_cap, seed=0)
    tot = None
    first_meta = None
    for c in range(n_chunks):
        meta = json.loads((job_dir / f"chunk_{c:05d}.json").read_text())
        z = np.load(job_dir / f"chunk_{c:05d}.npz")
        logits.append(z["logits"])
        per_image.append(z["per_image"])
        arrs = {k[len("stats__"):]: z[k] for k in z.files if k.startswith("stats__")}
        merged.merge(AnalogStats.from_state(meta["analog_scalars"], arrs, cap=stats_cap))
        wall += meta["wall_s"]
        if tot is None:
            tot = dict(meta["cim_stats"])
            first_meta = meta
        else:
            for k, v in meta["cim_stats"].items():
                tot[k] += v
    logits = np.concatenate(logits)
    per_image = np.concatenate(per_image)
    adc = first_meta["chain"]["adc"]
    summ = merged.summary(adc_vref=adc["vref"], adc_bits=adc["bits"])

    # ---- consistency checks (recorded, and failures mark the job invalid) -- #
    conv_ops = sum(v["n"] for v in summ.values())
    clip_ops = sum(v["clip_hi"] + v["clip_lo"] for v in summ.values())
    per_img_conv = per_image[:, :, 0].sum(axis=1)
    checks = {
        "conversions_match_cimstats": conv_ops == tot["adc_conversions"],
        "clips_match_cimstats": clip_ops == tot["adc_saturations"],
        "per_image_sums_match": int(per_image[:, :, 0].sum()) == conv_ops
                                and int(per_image[:, :, 1:3].sum()) == clip_ops,
        "constant_conversions_per_image": bool(np.all(per_img_conv == per_img_conv[0])),
        "all_ops_present": all(op in summ for op in OPS),
    }
    clipped_any = (per_image[:, :, 1] + per_image[:, :, 2]).sum(axis=1) > 0
    img_clip_per_op = {op: int(((per_image[:, i, 1] + per_image[:, i, 2]) > 0).sum())
                       for i, op in enumerate(OPS)}
    nonfinite = int(np.count_nonzero(~np.isfinite(logits).all(axis=1)))
    out = dict(job_meta, n_images=int(logits.shape[0]), wall_s_total=wall,
               cim_stats=tot, analog_summary=summ, consistency=checks,
               consistency_ok=all(checks.values()),
               images_with_any_clipping=int(clipped_any.sum()),
               images_with_clipping_per_op=img_clip_per_op,
               conversions_per_image=int(per_img_conv[0]),
               n_nonfinite_images=nonfinite,
               chain=first_meta["chain"], replaced_modules=first_meta["replaced_modules"],
               status=("INVALID_NONFINITE" if nonfinite else
                       "INVALID_CONSISTENCY" if not all(checks.values()) else "OK"))
    _atomic_savez(job_dir / "final.npz", logits=logits, per_image=per_image)
    _atomic_write_json(job_dir / "final.json", out)
    if not keep_chunks:
        for c in range(n_chunks):
            for ext in (".npz", ".json"):
                p = job_dir / f"chunk_{c:05d}{ext}"
                if p.exists():
                    try:
                        p.unlink()
                    except PermissionError:
                        # Preserve the completed job outputs if Windows
                        # still has a temporary chunk file open.
                        pass
    return out
