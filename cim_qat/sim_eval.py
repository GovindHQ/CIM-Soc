"""
cim_qat/sim_eval.py — run the ACTUAL simulator (cim_sim, NumPy) on images.

Used for: the calibration check (PTQ at candidate gains), the per-epoch
simulator spot check during training, the surrogate-vs-simulator GPU check,
and the final hardware-aware evaluation. Nothing here goes through the
surrogate.

Execution conventions (same as cim_exp): batch 1, one thread per process,
parallelism by processes, deterministic. Work is split into chunks; each chunk
is written atomically to disk, so an interrupted run resumes where it stopped
and finished chunks are never recomputed.

Job kinds
  "analog"   cim_sim.attention.CIMAttention on the analog chain
             (capacitive DAC -> C2C -> per-op gain -> SAR ADC), gains frozen
  "adc_off"  same operand quantization and tiling, exact integer MAC, no ADC
             (legacy path with adc_enabled=False): the operand-quantization floor
  "fp32"     plain FP32 TinyViT (no CIM)
  "legacy_adc" the LEGACY functional ADC (cim_sim CIMConfig adc_enabled=True,
             8 b, worst-case full scale) = the operating point the OLD QAT
             checkpoints were trained against (diagnostic only; QAT_DESIGN.md §9)

Every chunk records, per image, the SHA-256 of the float32 model-input tensor
actually fed to the network (`input_sha`) and, for CIM kinds, the per-image
ADC conversion / saturation counts. The job records the preprocessing id
(cim_qat.data.preprocess_id). qat_evaluate.py's report refuses to compare runs
whose image indices, labels, input hashes or preprocessing differ.

Option input_perturb="ulp_up" moves every input element by one float32 ulp.
Comparing such a run with the unperturbed one measures the hardware model's
own sensitivity to float-level noise (the "discretization avalanche"), the
yardstick for judging GPU-vs-CPU discrepancies (QAT_DESIGN.md §4.4).
"""

from __future__ import annotations

import copy
import json
import os
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .common import configure_threads, read_json, savez_atomic, write_json

_W = {}          # per-process worker state


def _init_worker(job: dict):
    from .common import die_with_parent, limit_blas_threads
    die_with_parent()                    # never outlive the parent (no orphaned workers)
    configure_threads(1)
    bt = limit_blas_threads(1)           # the RUNNING BLAS pool (env vars come too late after fork)
    import torch
    from cim_sim import CIMArray, CIMConfig
    from cim_sim.analog import AnalogStats
    from cim_sim.attention import convert_attention

    from .hw_config import HWConfig
    from .model import build_fp32
    torch.set_grad_enabled(False)
    fp32 = build_fp32(job["ckpt"])
    _W.clear()
    _W["job"] = job
    _W["blas_threads"] = max(bt.values()) if bt else None
    _W["fp32"] = fp32
    if job["kind"] in ("analog", "adc_off", "legacy_adc"):
        hw = HWConfig(**job["hw"])
        cim = copy.deepcopy(fp32)
        if job["kind"] == "legacy_adc":
            arr = CIMArray(CIMConfig(rows=hw.rows, cols=hw.cols, weight_bits=hw.weight_bits,
                                     act_bits=hw.act_bits, adc_enabled=True, adc_bits=8, adc_vref=0.6))
            assert arr.chain is None and arr.cfg.adc_enabled
            _W["stats"] = None
        elif job["kind"] == "analog":
            stats = AnalogStats(seed=0)
            arr = CIMArray(hw.cim_config(), analog_stats=stats)
            if arr.chain is None:
                raise RuntimeError("analog chain not active")
            _W["stats"] = stats
        else:
            arr = CIMArray(CIMConfig(rows=hw.rows, cols=hw.cols, weight_bits=hw.weight_bits,
                                     act_bits=hw.act_bits))
            assert arr.chain is None and not arr.cfg.adc_enabled
            _W["stats"] = None
        n = len(convert_attention(cim, arrays=[arr]))
        if n != 10:
            raise RuntimeError(f"converted {n} attention modules, expected 10")
        _W["cim"] = cim
        _W["arr"] = arr


def _run_chunk(task):
    ci, items, out_path = task
    from cim_sim.array import CIMStats

    from .data import load_uint8_eval, to_model_input
    import hashlib
    job = _W["job"]
    t0 = time.time()
    idx, lg, lg_fp, ish, conv_i, sat_i = [], [], [], [], [], []
    cim_kind = job["kind"] in ("analog", "adc_off", "legacy_adc")
    if cim_kind:
        _W["arr"].stats = CIMStats()
        if _W["stats"] is not None:
            _W["stats"].reset()
    for (i, path) in items:
        x = to_model_input(load_uint8_eval(path, job.get("interpolation", "bicubic"))[None])
        if job.get("input_perturb") == "ulp_up":
            # self-sensitivity floor: every input element moved by one float32 ulp
            import torch
            x = torch.nextafter(x, torch.full_like(x, float("inf")))
        ish.append(hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest())
        if job["kind"] == "fp32":
            lg.append(_W["fp32"](x)[0].numpy())
        else:
            st = _W["arr"].stats
            c0, s0 = st.adc_conversions, st.adc_saturations
            lg.append(_W["cim"](x)[0].numpy())
            conv_i.append(st.adc_conversions - c0)
            sat_i.append(st.adc_saturations - s0)
            if job.get("with_own_fp32", True):
                lg_fp.append(_W["fp32"](x)[0].numpy())
        idx.append(i)
    arrays = dict(idx=np.array(idx, np.int64), logits=np.stack(lg).astype(np.float32),
                  input_sha=np.array(ish, dtype="S64"))
    if conv_i:
        arrays["adc_conversions"] = np.array(conv_i, np.int64)
        arrays["adc_saturations"] = np.array(sat_i, np.int64)
    if lg_fp:
        arrays["logits_own_fp32"] = np.stack(lg_fp).astype(np.float32)
    meta = dict(chunk=ci, n=len(idx), wall_s=time.time() - t0, pid=os.getpid(),
                blas_threads=_W.get("blas_threads"))
    if cim_kind:
        st = _W["arr"].stats
        meta["cim_stats"] = dict(adc_conversions=st.adc_conversions,
                                 adc_saturations=st.adc_saturations,
                                 array_ops=st.array_ops)
    if job["kind"] == "analog":
        scal, arrs = _W["stats"].state_dict()
        meta["analog_scalars"] = scal
        for k, v in arrs.items():
            arrays["as__" + k] = v
    savez_atomic(out_path.with_suffix(".npz"), **arrays)
    write_json(out_path.with_suffix(".json"), meta)       # written last = complete
    return ci, meta["wall_s"]


def run(job: dict, items: list, out_dir, workers: int = 8, chunk: int = 25,
        progress: bool = True) -> dict:
    """items: [(manifest_idx, absolute_path)]. Returns the merged result dict.

    `out_dir` holds job.json, chunks/, and result.npz + summary.json when done.
    Re-running with the same job resumes; a different job in the same dir is
    refused.
    """
    from .data import preprocess_id
    job = dict(job, preprocess_id=preprocess_id(job.get("interpolation", "bicubic")))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jpath = out_dir / "job.json"
    jfull = dict(job=job, n_items=len(items), chunk=chunk,
                 items_sha=_items_sha(items))
    if jpath.exists():
        old = read_json(jpath)
        if json.dumps(old, sort_keys=True) != json.dumps(json.loads(json.dumps(jfull)), sort_keys=True):
            raise RuntimeError(f"{out_dir} holds a different job; use a new directory")
    else:
        write_json(jpath, jfull)
    cdir = out_dir / "chunks"
    cdir.mkdir(exist_ok=True)
    tasks = []
    for ci in range(0, (len(items) + chunk - 1) // chunk):
        p = cdir / f"chunk_{ci:05d}"
        if p.with_suffix(".json").exists() and p.with_suffix(".npz").exists():
            continue
        tasks.append((ci, items[ci * chunk:(ci + 1) * chunk], p))
    n_chunks = (len(items) + chunk - 1) // chunk
    t0 = time.time()
    if tasks:
        if progress:
            print(f"  [{job['kind']}] {len(tasks)}/{n_chunks} chunks to run on {workers} workers "
                  f"({len(items)} images)", flush=True)
        # "fork" is unsafe once the parent has initialised CUDA (training
        # process); use "spawn" then. Workers never touch CUDA either way.
        # Spawned workers load BLAS afresh: give them one thread from the start.
        # The caller's environment is restored afterwards (e.g. the training process).
        from .common import THREAD_ENV
        saved_env = {v: os.environ.get(v) for v in THREAD_ENV}
        for v in THREAD_ENV:
            os.environ[v] = "1"
        try:
            import torch
            ctx = get_context("spawn" if torch.cuda.is_available() and torch.cuda.is_initialized()
                              else "fork")
            with ctx.Pool(max(1, min(workers, len(tasks))), initializer=_init_worker,
                          initargs=(job,)) as pool:
                done = 0
                for ci, wall in pool.imap_unordered(_run_chunk, tasks):
                    done += 1
                    if progress and (done % max(1, len(tasks) // 20) == 0 or done == len(tasks)):
                        el = time.time() - t0
                        print(f"    {done}/{len(tasks)} chunks  {el:.0f}s elapsed  "
                              f"~{el / done * (len(tasks) - done):.0f}s left", flush=True)
        finally:
            for v, old in saved_env.items():
                if old is None:
                    os.environ.pop(v, None)
                else:
                    os.environ[v] = old
    return merge(out_dir, n_chunks, job)


def _items_sha(items) -> str:
    import hashlib
    h = hashlib.sha256()
    for i, p in items:
        h.update(f"{i}|{p}\n".encode())
    return h.hexdigest()


def merge(out_dir: Path, n_chunks: int, job: dict) -> dict:
    cdir = out_dir / "chunks"
    idx, lg, lgf, ish, ci_, si_, conv, sat, ops, wall = [], [], [], [], [], [], 0, 0, 0, 0.0
    blas = set()
    for ci in range(n_chunks):
        p = cdir / f"chunk_{ci:05d}"
        meta = read_json(p.with_suffix(".json"))
        with np.load(p.with_suffix(".npz")) as z:
            idx.append(z["idx"])
            lg.append(z["logits"])
            ish.append(z["input_sha"])
            if "adc_conversions" in z.files:
                ci_.append(z["adc_conversions"])
                si_.append(z["adc_saturations"])
            if "logits_own_fp32" in z:
                lgf.append(z["logits_own_fp32"])
        wall += meta["wall_s"]
        blas.add(meta.get("blas_threads"))
        if "cim_stats" in meta:
            conv += meta["cim_stats"]["adc_conversions"]
            sat += meta["cim_stats"]["adc_saturations"]
            ops += meta["cim_stats"]["array_ops"]
    idx = np.concatenate(idx)
    order = np.argsort(idx, kind="stable")
    res = dict(idx=idx[order], logits=np.concatenate(lg)[order], input_sha=np.concatenate(ish)[order])
    if ci_:
        res["adc_conversions"] = np.concatenate(ci_)[order]
        res["adc_saturations"] = np.concatenate(si_)[order]
    if lgf:
        res["logits_own_fp32"] = np.concatenate(lgf)[order]
    summ = dict(job=job, n=int(len(idx)), cpu_seconds=wall,
                cpu_seconds_per_image=wall / max(1, len(idx)),
                blas_threads_per_worker=sorted(blas, key=lambda b: (b is None, b or 0)))
    import hashlib
    summ["input_sha_digest"] = hashlib.sha256(b"".join(res["input_sha"].tolist())).hexdigest()
    summ["idx_digest"] = hashlib.sha256(res["idx"].tobytes()).hexdigest()
    if job["kind"] in ("analog", "adc_off", "legacy_adc"):
        summ.update(adc_conversions=conv, adc_saturations=sat, array_ops=ops)
        if ci_:
            summ["images_with_any_saturation_pct"] = 100.0 * float((res["adc_saturations"] > 0).mean())
    if job["kind"] == "analog":
        # v2.7: chunk statistics are re-merged at one common keep fraction
        # (cim_qat.analog_merge); AnalogStats.merge() concentrates the weight of a
        # long chunked run on a few samples, which made the percentile columns unreliable.
        from .analog_merge import METHOD, summary_from_chunks
        hw = job["hw"]
        summ["analog_summary"] = summary_from_chunks(out_dir, adc_vref=hw.get("adc_vref", 0.6),
                                                     adc_bits=hw.get("adc_bits", 8))
        summ["analog_summary_method"] = METHOD
    savez_atomic(out_dir / "result.npz", **res)
    write_json(out_dir / "summary.json", summ)
    res["summary"] = summ
    return res
