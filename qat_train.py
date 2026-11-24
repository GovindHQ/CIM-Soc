#!/usr/bin/env python3
"""
qat_train.py — hardware-aware QAT of TinyViT-5M (surrogate forward + STE).

  # runbook stage 7: smoke test + measured throughput/memory (separate *_smoke run dir)
  python qat_train.py --config configs/qat_w8a8_kd.json --device cuda --smoke-steps 60

  # runbook stage 10: the run (only after stages 1-9 passed and were reviewed)
  python qat_train.py --config configs/qat_w8a8_kd.json --device cuda

  # resume an interrupted run (refuses if the config or gains differ, or if the run finished)
  python qat_train.py --config configs/qat_w8a8_kd.json --device cuda --resume

  # decision D2: complete the schedule of an EARLY-STOPPED run with early stopping off
  python qat_train.py --config configs/qat_w4a4_kd_full.json --device cuda --fork-from w4a4_kd_pilot10ep

Output: <work_root>/runs/<run_name>[ _smoke]/ — config_resolved.json, provenance,
train.log, metrics_steps.csv, evals.jsonl, val_perimage/, spot/,
checkpoints/{last.pth, best_model.pth, final_model.pth, checkpoints.json},
summary.json (and throughput.json in smoke mode).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cim_qat.common import tag_job  # noqa: E402
tag_job("train")                     # lets qat_stop.py list (never stop) training runs
from cim_qat.common import (describe_device, now_iso, read_json, select_device,  # noqa: E402
                            write_json, write_provenance)


class Tee:
    def __init__(self, path):
        self.f = open(path, "a")

    def __call__(self, msg):
        line = f"[{now_iso()}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def throughput(run, log, n_steps: int) -> dict:
    """Time n_steps real training steps and one VAL evaluation; extrapolate."""
    import numpy as np
    import torch
    dev = run.dev
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    step_t, data_t = [], []
    t_data = time.time()
    loader = run.train_loader(0, 0)
    for i, (x, y, _) in enumerate(loader):
        dd = time.time() - t_data
        t0 = time.time()
        m = run.step(x, y)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        step_t.append(time.time() - t0)
        data_t.append(dd)
        if i % 10 == 0:
            log(f"  throughput step {i}: {step_t[-1]:.3f}s (data wait {dd:.3f}s) loss {m['loss']:.4f}")
        if len(step_t) >= n_steps:
            break
        t_data = time.time()
    warm = 5 if len(step_t) > 10 else 1
    st, dt = step_t[warm:], data_t[warm:]
    peak = torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else float("nan")
    t0 = time.time()
    ev = run.evaluate("throughput_val", with_fp32=False)
    t_eval = time.time() - t0
    n_val_eval = len(run.val_indices())
    n_val_full = len(run.splits.rows("val"))
    t0 = time.time()
    sc = run.spot_check("throughput_val")
    t_spot = time.time() - t0
    step_s = statistics.median(st)
    wait_s = statistics.median(dt)
    per_step = step_s + max(0.0, wait_s)
    spe = run.steps_per_epoch
    val_full_s = t_eval * n_val_full / n_val_eval          # one full-VAL surrogate pass
    # per epoch: training + one full-VAL eval + the student-FP32 VAL pass (~same cost) + spot check
    epoch_s = spe * per_step + 2 * val_full_s + t_spot
    rep = dict(device=describe_device(dev), batch_size=run.cfg.batch_size, loss=run.cfg.loss,
               steps_timed=len(st), step_s_median=step_s, step_s_p90=float(np.percentile(st, 90)),
               data_wait_s_median=wait_s, data_bound=wait_s > 0.1 * step_s,
               images_per_s=run.cfg.batch_size / per_step, peak_gpu_mem_gb=peak,
               steps_per_epoch=spe, train_images=len(run.train_ds),
               val_eval_s_measured=t_eval, val_eval_images=n_val_eval, val_full_images=n_val_full,
               val_eval_s_full_est=val_full_s,
               spot_check_s=t_spot, spot_check=sc,
               est_epoch_hours=epoch_s / 3600)
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fork-from", default=None, metavar="RUN_NAME",
                    help="decision D2: copy the history of this EARLY-STOPPED run into the new run folder "
                         "(config run_name) and continue its schedule with early stopping off")
    ap.add_argument("--smoke-steps", type=int, default=None,
                    help="Stage D: run N steps on a separate *_smoke run dir, measure throughput")
    a = ap.parse_args()

    from cim_qat.data import DataPaths, Splits
    from cim_qat.trainer import Run, TrainConfig, config_hash
    over = {}
    if a.smoke_steps:
        over = dict(max_steps=a.smoke_steps, val_limit=256, spot_check_n=16)
    cfg = TrainConfig.load(a.config, **over)
    dp = DataPaths.load(cfg.paths)
    name = cfg.run_name + ("_smoke" if a.smoke_steps else "")
    import os
    os.environ["CIM_QAT_RUN"] = name                     # recorded in the data access log
    splits = Splits(dp.splits_dir, purpose="smoke" if a.smoke_steps else "train")   # no TEST access
    out = Path(cfg.out_root or Path(dp.work_root) / "runs") / name
    ckpt = out / "checkpoints" / "last.pth"
    fork_info = fork_state = None
    if a.fork_from:
        if a.resume or a.smoke_steps:
            sys.exit("--fork-from cannot be combined with --resume or --smoke-steps "
                     "(resume an interrupted fork with --resume alone)")
        from cim_qat.trainer import prepare_fork
        try:
            fork_info, fork_state = prepare_fork(out.parent / a.fork_from, out, cfg)
        except (RuntimeError, FileExistsError) as e:
            sys.exit(f"fork refused: {e}")
    if (out / "summary.json").exists() and not a.smoke_steps:
        sys.exit(f"{out} is a FINISHED run (summary.json exists); choose a new run_name")
    if ckpt.exists() and not a.resume and not a.smoke_steps and not fork_info:
        sys.exit(f"{out} already has a checkpoint; pass --resume to continue it, or choose a new run_name")
    if a.smoke_steps and out.exists():
        import shutil
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    log = Tee(out / "train.log")
    dev = select_device(a.device)
    log(f"device: {describe_device(dev)}")

    run = Run(cfg, dev, splits, out)
    gains_rec = run.gains_rec
    resolved = dict(config=cfg.to_dict(), config_file=str(Path(a.config).resolve()),
                    hardware=run.hw.to_dict(), frozen_gains_record=gains_rec,
                    splits=dict(dir=str(splits.dir), seed=splits.info["seed"],
                                manifest_sha256=splits.info["manifest_sha256"],
                                counts=splits.info["counts"]),
                    steps_per_epoch=run.steps_per_epoch, total_steps=run.total_steps,
                    warmup_steps=run.warmup, eval_steps=run.eval_steps,
                    pretrained_sha256=__import__("cim_qat.model", fromlist=["x"]).check_pretrained(),
                    selection_rules=dict(
                        selection_split="val (full ImageNet-1k val, 50,000 images)",
                        primary="best VAL hardware top-1 under the surrogate; ties -> lower VAL NLL (best_model.pth)",
                        secondary="last checkpoint (final_model.pth)",
                        early_stopping=f"stop after {cfg.early_stopping_patience} epoch(s) without VAL improvement",
                        plateau=f"objective ({'KL' if cfg.loss == 'kd' else 'NLL'}) improvement over the last "
                                f"epoch < {cfg.plateau_threshold_nats} nats AND |paired top-1 change| <= "
                                f"{cfg.plateau_se_mult} paired SE",
                        test_split="ImageNetV2; never opened by training, calibration or selection"))
    resolved["config_hash"] = config_hash(cfg, dict(gains=gains_rec["gains"], splits=resolved["splits"]))
    if fork_info:
        from cim_qat.common import torch_save_atomic
        from cim_qat.trainer import FORK_ALLOWED
        run.load_state(fork_state, allow_changes=FORK_ALLOWED)
        run.stopped = None                               # early stopping off from here on (D2)
        run.fork = fork_info
        resolved["fork"] = fork_info
        resolved["selection_rules"]["early_stopping"] = (
            f"OFF after the fork (patience {cfg.early_stopping_patience} >= {cfg.epochs} epochs); "
            f"source {fork_info['source_run']} early-stopped at step {fork_info['forked_at_step']}")
        write_json(out / "config_resolved.json", resolved)
        write_provenance(out, dict(stage="fork", run_name=name, fork=fork_info))
        torch_save_atomic(run.state(), ckpt)            # the fork's own resumable state (not stopped)
        import json as _json
        with open(out / "evals.jsonl", "a") as f:
            f.write(_json.dumps(dict(event="forked", **fork_info)) + "\n")
        with open(out / "metrics_steps.csv", "a") as f:
            f.write(f"# forked from {fork_info['source_run']} at global_step {run.global_step} {now_iso()}\n")
        log(f"FORKED (decision D2) from {fork_info['source_run']} at global step {run.global_step}/"
            f"{run.total_steps}: early stopping OFF (patience {cfg.early_stopping_patience}); source last.pth "
            f"sha256 {fork_info['source_last_sha256'][:16]}; config hash {resolved['config_hash']}")
    elif ckpt.exists() and a.resume:
        old = read_json(out / "config_resolved.json")
        if old["config_hash"] != resolved["config_hash"]:
            sys.exit(f"resume refused: config hash {resolved['config_hash']} != {old['config_hash']}")
        import torch
        # load on the CPU: load_state_dict copies weights to the model's device, the optimizer
        # moves its state to each parameter's device, and RNG states must stay CPU uint8
        run.load_state(torch.load(ckpt, map_location="cpu", weights_only=False))
        log(f"RESUMED at global step {run.global_step}/{run.total_steps}; config hash {resolved['config_hash']}")
        import json as _json
        with open(out / "evals.jsonl", "a") as f:     # marker: rows after this may repeat steps
            f.write(_json.dumps(dict(event="resumed", step=run.global_step, time_utc=now_iso())) + "\n")
        with open(out / "metrics_steps.csv", "a") as f:
            f.write(f"# resumed at global_step {run.global_step} {now_iso()}\n")
    else:
        write_json(out / "config_resolved.json", resolved)
        write_provenance(out, dict(stage="smoke" if a.smoke_steps else "train", run_name=name))
    log(f"run {name}: W{run.hw.weight_bits}A{run.hw.act_bits} gains {run.hw.gain_dict} loss {cfg.loss} "
        f"epochs {cfg.epochs} steps {run.total_steps} (per epoch {run.steps_per_epoch}) "
        f"bs {cfg.batch_size} lr {cfg.lr} BN frozen {cfg.bn_frozen} amp {cfg.amp} tf32 {cfg.tf32} "
        f"config hash {resolved['config_hash']}")

    if a.smoke_steps:
        rep = throughput(run, log, a.smoke_steps)
        rep["est_full_run_hours"] = rep["est_epoch_hours"] * read_json(a.config).get("epochs", cfg.epochs)
        rep["time_utc"] = now_iso()
        write_json(out / "throughput.json", rep)
        log(f"THROUGHPUT: {rep['step_s_median']:.3f} s/step (p90 {rep['step_s_p90']:.3f}), data wait "
            f"{rep['data_wait_s_median']:.3f} s, {rep['images_per_s']:.0f} img/s, peak GPU mem "
            f"{rep['peak_gpu_mem_gb']:.2f} GB; full VAL ({rep['val_full_images']}) eval ~{rep['val_eval_s_full_est'] / 60:.1f} min; "
            f"spot check {rep['spot_check_s']:.0f}s; est. {rep['est_epoch_hours']:.2f} h/epoch, "
            f"{rep['est_full_run_hours']:.1f} h for the configured budget")
        sc = rep["spot_check"]
        log(f"spot check ({sc['n']} VAL imgs): CPU surrogate batch-1 bit-exact on {sc['cpu_batch1_bit_exact_images']}"
            f"/{sc['n']}; device surrogate vs simulator KL {sc['kl_sim_to_surrogate']:.2e} "
            f"(simulator 1-ulp self-sensitivity floor {sc['kl_sim_self_sensitivity_floor']:.2e}), "
            f"agree {sc['agree_pct']:.1f}%")
        return

    summ = run.fit(log)
    p = summ.get("plateau")
    b = summ["best_eval"]
    log(f"DONE ({summ['stopped_reason']}). epoch-0 PTQ VAL hw top1 {summ['epoch0']['hw_val']['top1_pct']:.2f}%; "
        f"best {b['tag'] if b else None} (epoch {b['epoch'] if b else float('nan'):.2f}) "
        f"{b['hw_val']['top1_pct'] if b else float('nan'):.2f}%; final {summ['final']['hw_val']['top1_pct']:.2f}% "
        "[surrogate, full VAL; simulator numbers come from qat_evaluate.py]")
    if p:
        log(f"plateau check: {p['recommendation']} ({p['objective_metric']} improved "
            f"{p['objective_improvement_nats']:.4f} nats; top-1 change {p['top1_change_pct']:+.2f} "
            f"+- {p['top1_paired_se_pct']:.2f} pp)")


if __name__ == "__main__":
    main()
