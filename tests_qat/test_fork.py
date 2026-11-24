#!/usr/bin/env python3
"""Decision D2 fork (qat_train.py --fork-from), end to end on real training steps.

  F1 source run A (patience 3) early-stops at epoch 4 of 6 (scripted VAL results)
  F2 reference run B: same config without early stopping (patience 6), all 6 epochs
  F3 refusals: fork from a run that did not early-stop; a config that changes anything
     else (lr); early stopping not switched off (patience < epochs); an existing fork folder
  F4 the CLI fork of A completes epochs 5-6 (real evaluations and spot checks)
  F5 the fork's final weights are BIT-IDENTICAL to B's (= a run that never early-stopped)
  F6 the fork inherits A's history and best checkpoint; its summary records the fork;
     the source folder is unchanged (SHA-256 of every file)
  F7 a finished fork cannot be resumed or re-forked

Runs A and B use scripted (mocked) VAL evaluations so that A stops at a known epoch;
training steps, optimizer, schedule, RNG and checkpoints are the real code. The mocked
evaluations cannot influence the weights (no evaluation feeds back into training).

  python tests_qat/test_fork.py --paths <paths.json> --frozen-gains <frozen_gains.json> \
      --out-root <runs dir> [--workers 2]
The source/fork use the D2 run names (w4a4_kd_pilot10ep / w4a4_kd_pilot10ep_full), so
point --out-root at a TEST work root only (the self-test's mini work root).
"""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import numpy as np  # noqa: E402
import torch  # noqa: E402

R = []
SRC, FORK, REF = "w4a4_kd_pilot10ep", "w4a4_kd_pilot10ep_full", "fork_reference_noES"


def check(n, ok, **kw):
    R.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {n} " + " ".join(f"{k}={v}" for k, v in kw.items()), flush=True)


def tree_sha(d: Path) -> dict:
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.rglob("*")) if p.is_file()}


def base_cfg(a, prec):
    return dict(precision=prec, frozen_gains=a.frozen_gains, paths=a.paths, loss="kd", epochs=6,
                batch_size=8, num_workers=0, val_batch_size=8, spot_check_n=2, spot_check_workers=a.workers,
                log_every=1, out_root=a.out_root, student_fp32_eval_every_epoch=False)


def mocked_run(a, name, patience, seq):
    from cim_qat.common import write_json
    from cim_qat.common import savez_atomic
    from cim_qat.data import DataPaths, Splits
    from cim_qat.trainer import Run, TrainConfig, config_hash
    g = json.loads(Path(a.frozen_gains).read_text())
    prec = f"W{g['weight_bits']}A{g['act_bits']}"
    cfg = TrainConfig(run_name=name, early_stopping_patience=patience, **base_cfg(a, prec))
    out = Path(a.out_root) / name
    out.mkdir(parents=True)
    sp = Splits(DataPaths.load(a.paths).splits_dir, purpose="unit_test")
    run = Run(cfg, torch.device("cpu"), sp, out)
    n_val = len(sp.rows("val"))

    def ev(tag, with_fp32):
        e = run.global_step // run.steps_per_epoch
        t, nll = seq[e]
        c = np.zeros(n_val, bool)
        c[: int(round(t / 100 * n_val))] = True
        savez_atomic(out / "val_perimage" / f"{tag}.npz", correct1=c, nll=np.full(n_val, nll, np.float32),
                     kl=np.full(n_val, 0.01, np.float32), pred=np.zeros(n_val, np.int16))
        return dict(tag=tag, step=run.global_step, epoch=run.global_step / run.steps_per_epoch, split="val",
                    val_manifest_sha256=sp.sha("val"), n_val=n_val, val_limited=False,
                    hw_val=dict(top1_pct=t, nll=nll, kl_ref=0.01, agree_ref_pct=99.0),
                    op_stats={o: dict(clip_pct_sim=0.0) for o in ("qkv", "qk", "av")})
    run.evaluate = ev
    write_json(out / "config_resolved.json", dict(
        config=cfg.to_dict(), frozen_gains_record=run.gains_rec,
        config_hash=config_hash(cfg, dict(gains=run.gains_rec["gains"]))))
    summ = run.fit(log=lambda m: None, spot=False)
    return run, summ, cfg


def cli(a, cfg_dict, fork_from=None, resume=False, expect=0):
    p = Path(tempfile.mkdtemp()) / "cfg.json"
    p.write_text(json.dumps(cfg_dict))
    cmd = [sys.executable, str(REPO / "qat_train.py"), "--config", str(p), "--device", "cpu"]
    cmd += ["--fork-from", fork_from] if fork_from else []
    cmd += ["--resume"] if resume else []
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if r.returncode != expect:
        print("\n".join((r.stdout + r.stderr).splitlines()[-25:]))
    return r.returncode, r.stdout + r.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", required=True)
    ap.add_argument("--frozen-gains", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    seq = [(50, 1.0), (60, 1.0), (59, 1.0), (58, 1.0), (57, 1.0), (61, 0.9), (62, 0.8)]
    A, sA, cfgA = mocked_run(a, SRC, 3, seq)
    check("F1 source early-stops at epoch 4 of 6", sA["stopped_early"] is not None
          and sA["steps_done"] == 4 * A.steps_per_epoch, stopped=sA["stopped_reason"][:40])
    B, sB, _ = mocked_run(a, REF, 6, seq)
    check("F2 reference without early stopping runs all 6 epochs", sB["stopped_early"] is None
          and sB["steps_done"] == 6 * B.steps_per_epoch)
    src_before = tree_sha(Path(a.out_root) / SRC)
    fork_cfg = dict(base_cfg(a, cfgA.precision), run_name=FORK, early_stopping_patience=10)
    rc, out = cli(a, dict(fork_cfg, run_name="fork_bad_src"), fork_from=REF, expect=1)
    rc2, out2 = cli(a, dict(fork_cfg, lr=2e-5), fork_from=SRC, expect=1)
    rc3, out3 = cli(a, dict(fork_cfg, early_stopping_patience=4), fork_from=SRC, expect=1)
    check("F3 refused: source did not early-stop / other setting changed / early stopping not off",
          rc == 1 and "completed its scheduled budget" in out and rc2 == 1 and "may change only" in out2
          and rc3 == 1 and "must be >= epochs" in out3)
    rc, out = cli(a, fork_cfg, fork_from=SRC)
    check("F4 CLI fork completes the schedule", rc == 0 and "FORKED (decision D2)" in out, exit=rc)
    F = Path(a.out_root) / FORK
    pf = torch.load(F / "checkpoints" / "final_model.pth", weights_only=False)
    pb = torch.load(Path(a.out_root) / REF / "checkpoints" / "final_model.pth", weights_only=False)
    check("F5 fork final weights bit-identical to the never-early-stopped reference",
          pf.keys() == pb.keys() and all(torch.equal(pf[k], pb[k]) for k in pf))
    sF = json.loads((F / "summary.json").read_text())
    ev_src = [l for l in (Path(a.out_root) / SRC / "evals.jsonl").read_text().splitlines() if '"tag"' in l]
    ev_f = [l for l in (F / "evals.jsonl").read_text().splitlines() if '"tag"' in l]
    reg = json.loads((F / "checkpoints" / "checkpoints.json").read_text())
    regA = json.loads((Path(a.out_root) / SRC / "checkpoints" / "checkpoints.json").read_text())
    check("F6 history inherited, fork recorded, best kept, source unchanged",
          ev_f[:len(ev_src)] == ev_src and len(ev_f) == len(ev_src) + 2
          and sF["forked_from"]["source_run"] == SRC and sF["stopped_early"] is None
          and sF["steps_done"] == 6 * A.steps_per_epoch and sF["best"]["tag"] == sA["best"]["tag"]
          and reg["best_model.pth"]["sha256"] == regA["best_model.pth"]["sha256"]
          and tree_sha(Path(a.out_root) / SRC) == src_before,
          best=sF["best"]["tag"])
    rc, out = cli(a, fork_cfg, resume=True, expect=1)
    rc2, out2 = cli(a, fork_cfg, fork_from=SRC, expect=1)
    check("F7 finished fork: resume and re-fork refused", rc == 1 and "FINISHED" in out
          and rc2 == 1 and "exists" in out2)
    print(f"{sum(R)}/{len(R)} passed")
    sys.exit(0 if all(R) else 1)


if __name__ == "__main__":
    main()
