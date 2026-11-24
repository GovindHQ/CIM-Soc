#!/usr/bin/env python3
"""Checkpoint-selection and early-stopping rules of cim_qat.trainer.Run, with
scripted VAL results (the real training/eval code paths, evaluations mocked).

  S1 epoch-0 (PTQ) is never selected, even if better than every trained epoch
  S2 best = max VAL top-1; equal top-1 -> lower NLL wins
  S3 early stop after `patience` consecutive epochs without improvement
  S4 no early stop while improving; the budget end is not an early stop
  S5 a resumed run continues the patience count (state round trip)
  S6 W8A8 decision (patience 5 = 5-epoch budget): never stops early, best still selected
  S7 config hash ignores num_workers / spot_check_workers / notes, but not patience

  python tests_qat/test_selection.py --paths <paths.json> --frozen-gains <frozen_gains.json>
  (the precision is taken from the frozen-gains file; the rules themselves do not depend on it)
"""
import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402

R = []


def check(n, ok, **kw):
    R.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {n} " + " ".join(f"{k}={v}" for k, v in kw.items()))


def make(a, out, patience, epochs=10):
    from cim_qat.data import DataPaths, Splits
    from cim_qat.trainer import Run, TrainConfig
    Path(out).mkdir(parents=True, exist_ok=True)
    import json
    g = json.loads(Path(a.frozen_gains).read_text())
    prec = f"W{g['weight_bits']}A{g['act_bits']}"        # follow the gains file (W8A8 or W4A4)
    cfg = TrainConfig(run_name="sel", precision=prec, frozen_gains=a.frozen_gains, paths=a.paths,
                      epochs=epochs, batch_size=2, num_workers=0, early_stopping_patience=patience,
                      student_fp32_eval_every_epoch=False)
    run = Run(cfg, torch.device("cpu"), Splits(DataPaths.load(a.paths).splits_dir, purpose="unit_test"), out)
    saved = []
    run.save_model_only = lambda name: saved.append((name, run.global_step)) or "x"
    return run, saved


def script(run, seq):
    """seq: [(top1, nll) at epoch 0, 1, 2, ...]; returns the epochs evaluated."""
    def ev(tag, with_fp32):
        e = run.global_step // run.steps_per_epoch
        t, n = seq[e]
        return dict(tag=tag, step=run.global_step, epoch=float(e), split="val", val_manifest_sha256="x",
                    n_val=1, val_limited=False, hw_val=dict(top1_pct=t, nll=n, kl_ref=0.0, agree_ref_pct=0.0),
                    op_stats={o: dict(clip_pct_sim=0.0) for o in ("qkv", "qk", "av")})
    run.evaluate = ev
    done = []
    for e in range(len(seq)):
        run.global_step = e * run.steps_per_epoch
        run._eval_point(lambda m: None, spot=False)
        done.append(e)
        if run.stopped:
            break
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", required=True)
    ap.add_argument("--frozen-gains", required=True)
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp())
    run, saved = make(a, tmp / "a", patience=2)
    done = script(run, [(90, 1.0), (60, 2.0), (61, 1.9), (61, 1.8), (60, 1.7), (59, 1.6), (70, 1.0)])
    check("S1 epoch-0 PTQ never selected", run.best["step"] > 0)
    check("S2 best = max top-1, ties -> lower NLL (epoch 3)", run.best["step"] == 3 * run.steps_per_epoch,
          best_epoch=run.best["step"] // run.steps_per_epoch)
    check("S3 stopped after 2 epochs without improvement (at epoch 5)", run.stopped is not None
          and done[-1] == 5, stopped=run.stopped and run.stopped["epoch"])
    check("S3 best_model.pth written only at improvements", [s for _, s in saved] ==
          [e * run.steps_per_epoch for e in (1, 2, 3)])
    run2, _ = make(a, tmp / "b", patience=2, epochs=4)
    done2 = script(run2, [(50, 1.0), (60, 1.0), (61, 1.0), (62, 1.0), (63, 1.0)])
    check("S4 improving run never early-stops", run2.stopped is None and done2[-1] == 4)
    # S5: stop/resume mid-way keeps the patience counter
    run3, _ = make(a, tmp / "c", patience=2)
    seq = [(50, 1.0), (60, 1.0), (59, 1.0), (58, 1.0), (70, 1.0)]
    script(run3, seq[:3])                       # epochs 0..2 (one epoch without improvement)
    st = run3.state()
    run4, _ = make(a, tmp / "d", patience=2)
    run4.load_state(st)

    def ev(tag, with_fp32):
        e = run4.global_step // run4.steps_per_epoch
        t, n = seq[e]
        return dict(tag=tag, step=run4.global_step, epoch=float(e), split="val", val_manifest_sha256="x",
                    n_val=1, val_limited=False, hw_val=dict(top1_pct=t, nll=n, kl_ref=0.0, agree_ref_pct=0.0),
                    op_stats={o: dict(clip_pct_sim=0.0) for o in ("qkv", "qk", "av")})
    run4.evaluate = ev
    run4.global_step = 3 * run4.steps_per_epoch
    run4._eval_point(lambda m: None, spot=False)
    check("S5 resumed run continues the patience count (stops at epoch 3)", run4.stopped is not None
          and run4.stopped["epoch"] == 3.0)
    run6, _ = make(a, tmp / "e", patience=5, epochs=5)
    done6 = script(run6, [(50, 1.0), (60, 1.0), (59, 1.0), (58, 1.0), (57, 1.0), (56, 1.0)])
    check("S6 patience 5 with a 5-epoch budget never early-stops; best = epoch 1",
          run6.stopped is None and done6[-1] == 5 and run6.best["step"] == run6.steps_per_epoch)
    from dataclasses import replace
    from cim_qat.trainer import config_hash
    c = run6.cfg
    h0 = config_hash(c, {})
    check("S7 config hash: worker counts / notes exempt, patience not",
          h0 == config_hash(replace(c, num_workers=3, spot_check_workers=5, notes="x"), {})
          and h0 != config_hash(replace(c, early_stopping_patience=4), {}))
    print(f"{sum(R)}/{len(R)} passed")
    sys.exit(0 if all(R) else 1)


if __name__ == "__main__":
    main()
