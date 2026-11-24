#!/usr/bin/env python3
"""
qat_env_check.py — runbook stage 1/2 on the remote machine: environment, GPU selection,
CUDA numerics, dataset paths and disk space. Changes nothing; writes a JSON.

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python qat_env_check.py \
      --device cuda [--paths configs/paths.json] [--out results/env_check.json]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cim_qat.common import (code_fingerprint, configure_determinism, describe_device,  # noqa: E402
                            environment_info, select_device, write_json)

OK, PROBLEMS = [], []


def note(ok, msg):
    (OK if ok else PROBLEMS).append(msg)
    print(f"  [{'OK' if ok else 'PROBLEM'}] {msg}")


def benchmark(dev, n, bs, precision):
    """Placeholder gains (NOT calibrated) only set the ADC operating point for
    timing; the cost of the surrogate does not depend on the gain values."""
    import statistics
    import time

    import torch

    from cim_qat.hw_config import HWConfig
    from cim_qat.model import build_student, build_teacher, set_train_mode, to_device
    from cim_qat.trainer import kd_loss
    wb, ab = (8, 8) if precision.upper() == "W8A8" else (4, 4)
    hw = HWConfig(wb, ab, (23.77, 8.41, 2.92) if wb == 8 else (6.5, 2.4, 0.9))
    st, te = to_device(build_student(hw), dev), to_device(build_teacher(), dev)
    opt = torch.optim.AdamW(st.parameters(), lr=1e-5, weight_decay=1e-4)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(bs, 3, 224, 224, generator=g).to(dev)
    cuda = dev.type == "cuda"
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    ts, tt = [], []
    for i in range(n):
        sync()
        t0 = time.time()
        set_train_mode(st, True)
        with torch.no_grad():
            t = te(x)
        sync()
        t1 = time.time()
        loss = kd_loss(st(x), t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sync()
        ts.append(time.time() - t0)
        tt.append(t1 - t0)
    w = ts[3:] if n > 6 else ts
    step = statistics.median(w)
    spe = 1269167 // bs
    rep = dict(precision=precision, batch_size=bs, steps=n, step_s_median=step,
               teacher_share=statistics.median(tt[3:] if n > 6 else tt) / step,
               peak_gpu_mem_gb=torch.cuda.max_memory_allocated() / 1e9 if cuda else float("nan"),
               est_train_hours_per_epoch_excluding_data_and_eval=spe * step / 3600)
    print(f"  benchmark {precision} bs {bs}: {step:.3f} s/step (teacher {100 * rep['teacher_share']:.0f}%), "
          f"peak mem {rep['peak_gpu_mem_gb']:.2f} GB -> ~{rep['est_train_hours_per_epoch_excluding_data_and_eval']:.1f} "
          f"h/epoch for {spe} steps (data loading and evaluation not included)")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--paths", default=None)
    ap.add_argument("--out", default="results/env_check.json")
    ap.add_argument("--benchmark-steps", type=int, default=0,
                    help="data-free GPU benchmark: N real KD training steps on fixed random "
                         "tensors (batch --batch-size) -> step time and peak memory")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--precision", default="W8A8")
    a = ap.parse_args()
    import torch
    env = environment_info()
    print("environment:")
    for k in ("python", "torch", "torch_cuda", "cudnn", "torchvision", "numpy", "pillow", "cpu_count",
              "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"):
        print(f"  {k}: {env.get(k)}")
    for l in env.get("nvidia_smi", []):
        print(f"  nvidia-smi: {l}")
    note(env["torch"].startswith("2.14"), f"torch {env['torch']} (old QAT logs: 2.14.1+cu130)")
    note(env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID",
         "CUDA_DEVICE_ORDER=PCI_BUS_ID (so CUDA indices match nvidia-smi indices)")
    configure_determinism(True, tf32=False)
    dev = select_device(a.device)
    d = describe_device(dev)
    print(f"selected device: {d}")
    if dev.type == "cuda":
        note(torch.cuda.device_count() == 1,
             f"exactly one GPU visible to PyTorch (count={torch.cuda.device_count()}); set CUDA_VISIBLE_DEVICES")
        smi = [l for l in env.get("nvidia_smi", []) if d.get("uuid", "~~") in l]
        print(f"  this is nvidia-smi GPU: {smi[0] if smi else 'UUID not matched - compare manually'}")
        free, total = torch.cuda.mem_get_info()
        note(free / 1e9 > 12, f"free GPU memory {free / 1e9:.1f} / {total / 1e9:.1f} GB")
        # integer-valued float32 matmul must be exact (surrogate correctness premise)
        g = torch.Generator().manual_seed(0)
        for tf32 in (False, True):
            torch.backends.cuda.matmul.allow_tf32 = tf32
            X = torch.randint(-255, 256, (512, 32), generator=g).float()
            W = torch.randint(-127, 128, (32, 512), generator=g).float()
            exact = (X.double() @ W.double())
            got = (X.to(dev) @ W.to(dev)).cpu().double()
            note(torch.equal(exact, got), f"CUDA float32 integer matmul exact (TF32={tf32})")
        torch.backends.cuda.matmul.allow_tf32 = False
        x = torch.randn(4, 3, 224, 224, device=dev)
        try:
            torch.nn.functional.conv2d(x, torch.randn(8, 3, 3, 3, device=dev)).sum().item()
            note(True, "deterministic-algorithms mode runs a conv on this GPU")
        except RuntimeError as e:
            note(False, f"deterministic mode error: {e}")
    bench = None
    if a.benchmark_steps:
        bench = benchmark(dev, a.benchmark_steps, a.batch_size, a.precision)
    fp = code_fingerprint()
    print(f"code fingerprint: all {fp['all'][:16]}  simulator {fp['simulator'][:16]}")
    du = shutil.disk_usage(Path.cwd())
    note(du.free / 1e9 > 2, f"free disk at the code folder {Path.cwd()}: {du.free / 1e9:.1f} GB (needs ~2 GB)")
    tmpd = os.environ.get("TMPDIR", "/tmp")
    note(shutil.disk_usage(tmpd).free / 1e9 > 5, f"free disk at TMPDIR {tmpd}: "
         f"{shutil.disk_usage(tmpd).free / 1e9:.1f} GB (set TMPDIR to the work disk, runbook stage 2)")
    if a.paths:
        from cim_qat.data import DataPaths
        try:
            dp = DataPaths.load(a.paths)
            note(True, f"dataset paths exist: train {dp.train_roots}, val {dp.val_root}")
            wr = Path(dp.work_root)
            wr.mkdir(parents=True, exist_ok=True)
            duw = shutil.disk_usage(wr)
            note(duw.free / 1e9 > 50, f"free disk at work_root {wr}: {duw.free / 1e9:.1f} GB "
                                       "(splits, hash caches, calibration, runs, eval need ~30-50 GB)")
            note(os.access(wr, os.W_OK), f"work_root {wr} is writable")
            if dp.imagenetv2_root:
                note(Path(dp.imagenetv2_root).is_dir(), f"ImageNetV2 folder exists: {dp.imagenetv2_root}")
        except Exception as e:
            note(False, f"paths: {e}")
    write_json(a.out, dict(environment=env, device=d, code_fingerprint=fp["all"], benchmark=bench,
                           simulator_fingerprint=fp["simulator"], ok=OK, problems=PROBLEMS))
    print(f"\n{len(OK)} OK, {len(PROBLEMS)} problem(s) -> {a.out}")
    sys.exit(1 if PROBLEMS else 0)


if __name__ == "__main__":
    main()
