#!/usr/bin/env python3
"""
test_training_mechanics.py — Stage C: the gradient path and training mechanics,
exercised through the SAME cim_qat.trainer.Run used for training.

  C0  every model tensor on the run device (incl. TinyViT's cached `ab`); teacher runs at once
  C1  custom backward == independent autograd STE (float64), incl. saturation
  C2  one real step: loss/grad finite; every parameter has a gradient; fraction
      of non-zero gradient entries per group; ADC-saturated (gradient-blocked)
      fraction per op and per layer
  C3  BatchNorm: modules in eval mode during training, running stats unchanged
      after optimizer steps, affine params receive gradients and change
  C4  teacher: frozen (no grads, eval), unchanged, output == fresh FP32 model
  C5  KD sanity: student with CIM disabled and teacher weights -> KD loss == 0
  C6  optimizer step: trainable params with non-zero grad change; buffers do not
  C7  determinism: two fresh runs, same seed/batches -> identical parameters
  C8  save/resume: 2+2 steps through a checkpoint == 4 uninterrupted steps (loaded on the CPU,
      as qat_train.py does; C8b: loaded straight onto the run device)
  C9  overfit: one fixed small batch, CE, 60 steps -> loss collapses
  C10 a saved student loads into run_tinyvit.build_model (the simulator's
      builder) and reproduces the student's FP32 output

  python test_training_mechanics.py --device cuda --paths configs/paths.json \
      --frozen-gains <work_root>/calibration/W8A8/frozen_gains.json --precision W8A8
Exit code 1 on any failure.
"""

from __future__ import annotations

import argparse
import copy
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cim_qat.common import now_iso, select_device, write_json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

RES = []


def check(name, ok, **kw):
    RES.append(dict(name=name, ok=bool(ok), **kw))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  " + "  ".join(f"{k}={v}" for k, v in kw.items()))


def ref_ste(X, W, spec):
    """Independent autograd implementation of the documented STE rules."""
    qa = (2 ** (spec.act_bits - 1) - 1) if spec.act_signed else (2 ** spec.act_bits - 1)
    qw = 2 ** (spec.weight_bits - 1) - 1
    sx = ((X.abs() if spec.act_signed else X).amax(-1, keepdim=True).clamp(min=1e-12) / qa).detach()
    sw = (W.abs().amax(-2, keepdim=True).clamp(min=1e-12) / qw).detach()
    u, v = X / sx, W / sw
    xq = torch.round(u.detach()) + (u - u.detach())
    wq = torch.round(v.detach()) + (v - v.detach())
    acc = 0
    for b in range((X.shape[-1] + 31) // 32):
        P = xq[..., b * 32:(b + 1) * 32] @ wq[..., b * 32:(b + 1) * 32, :]
        t = P / spec.lsb_p
        Fv = torch.floor(t.detach() + 0.5 + spec.guard)
        inr = ((Fv >= spec.code_lo) & (Fv <= spec.code_hi)).to(t.dtype)
        acc = acc + torch.clamp(Fv, spec.code_lo, spec.code_hi) + (t - t.detach()) * inr
    return acc * spec.lsb_p * sx * sw


def c1_backward(dev):
    from cim_qat.hw_config import HWConfig, op_specs
    from cim_qat.surrogate import cim_matmul_surrogate
    torch.manual_seed(0)
    worst = 0.0
    for hw in [HWConfig(8, 8, (23.77, 8.41, 2.92)), HWConfig(4, 4, (60.0, 30.0, 10.0)),
               HWConfig(8, 8, (300.0, 300.0, 300.0))]:
        sp = op_specs(hw)
        for op in ("qkv", "qk", "av"):
            X = torch.randn(2, 3, 49, 70, dtype=torch.float64)
            if op == "av":
                X = X.softmax(-1)
            W = torch.randn(2, 3, 70, 40, dtype=torch.float64)
            g = torch.randn(2, 3, 49, 40, dtype=torch.float64)
            X1, W1 = X.clone().to(dev).float().requires_grad_(), W.clone().to(dev).float().requires_grad_()
            X2, W2 = X.float().double().requires_grad_(), W.float().double().requires_grad_()
            (cim_matmul_surrogate(X1, W1, sp[op]) * g.to(dev).float()).sum().backward()
            (ref_ste(X2, W2, sp[op]) * g).sum().backward()
            for a, b in ((X1.grad.cpu().double(), X2.grad), (W1.grad.cpu().double(), W2.grad)):
                worst = max(worst, float((a - b).abs().max() / b.abs().max().clamp(min=1e-30)))
    check("C1 custom backward == autograd STE reference (incl. heavy saturation)", worst < 1e-5,
          max_rel_diff=f"{worst:.2e}")


def make_run(args, out, **over):
    from cim_qat.data import DataPaths, Splits
    from cim_qat.trainer import Run, TrainConfig
    cfg = TrainConfig(run_name="mechanics_test", precision=args.precision, frozen_gains=args.frozen_gains,
                      paths=args.paths, loss=over.get("loss", "kd"), epochs=1.0,
                      batch_size=over.get("batch_size", args.batch_size), num_workers=args.num_workers,
                      val_limit=32, spot_check_n=8, spot_check_workers=2, lr=over.get("lr", 1e-5),
                      warmup_epochs=0.0)
    sp = Splits(DataPaths.load(args.paths).splits_dir, purpose="unit_test")
    return Run(cfg, select_device(args.device), sp, out)


def batches(run, n, epoch=0, start=0):
    out = []
    for x, y, _ in run.train_loader(epoch, start):
        out.append((x, y))
        if len(out) == n:
            break
    return out


def params(m):
    return {k: v.detach().clone() for k, v in m.named_parameters()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--paths", required=True)
    ap.add_argument("--frozen-gains", required=True)
    ap.add_argument("--precision", default="W8A8")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--out", default="results/mechanics")
    a = ap.parse_args()
    t0 = time.time()
    dev = select_device(a.device)
    tmp = Path(tempfile.mkdtemp(prefix="qat_mech_"))

    print("== C1 backward rule")
    c1_backward(dev)

    print("== C2-C6 one real training step")
    from cim_qat.attention import set_collect_stats, surrogate_modules
    from cim_qat.model import bn_modules, bn_state, build_fp32, set_cim_enabled
    from cim_qat.trainer import kd_loss
    run = make_run(a, tmp / "r1")
    from cim_qat.model import assert_on_device
    try:
        for m in (run.student, run.teacher, run.ref_teacher):
            assert_on_device(m, dev)
        placed, why = True, ""
    except RuntimeError as e:
        placed, why = False, str(e)[:120]
    check("C0 student, teacher and reference teacher entirely on the run device "
          "(incl. TinyViT's cached attention-bias tensor)", placed, device=str(dev), detail=why)
    (x, y), = batches(run, 1)
    with torch.no_grad():
        run.teacher(x[:1].to(dev))           # the teacher must run right after construction (no eval() call)
    check("C0 teacher forward immediately after Run() on the device", True)
    p0, bn0 = params(run.student), bn_state(run.student)
    t_state0 = {k: v.clone() for k, v in run.teacher.state_dict().items()}
    set_collect_stats(run.student, True)
    from cim_qat.model import set_train_mode
    set_train_mode(run.student, True)
    s = run.student(x.to(dev))
    with torch.no_grad():
        t = run.teacher(x.to(dev))
    loss = kd_loss(s, t)
    run.opt.zero_grad(set_to_none=True)
    loss.backward()
    sat = {}
    for m in surrogate_modules(run.student):
        for op, st in m.stats.items():
            sat[(m.name, op)] = 100.0 * st.sat_codes / max(st.conversions, 1)
    set_collect_stats(run.student, False)
    named = dict(run.student.named_parameters())
    nograd = [k for k, p in named.items() if p.grad is None]
    nonfinite = [k for k, p in named.items() if p.grad is not None and not torch.isfinite(p.grad).all()]
    check("C2 loss finite", torch.isfinite(loss).item(), kd_loss=f"{loss.item():.4e}")
    check("C2 every parameter has a finite gradient", not nograd and not nonfinite,
          n_params=len(named), missing=nograd[:3], nonfinite=nonfinite[:3])
    groups = {"attn.qkv.weight": [], "attention_biases": [], "mlp": [], "patch_embed": [], "bn": [],
              "head": [], "norm(LayerNorm)": []}
    for k, p in named.items():
        g = groups["attn.qkv.weight"] if k.endswith("attn.qkv.weight") else \
            groups["attention_biases"] if "attention_biases" in k else \
            groups["mlp"] if ".mlp." in k else groups["patch_embed"] if k.startswith("patch_embed") else \
            groups["head"] if k.startswith("head") else \
            groups["bn"] if any(k.startswith(b) for b in [n for n, mm in run.student.named_modules()
                                                          if isinstance(mm, torch.nn.BatchNorm2d)]) else \
            groups["norm(LayerNorm)"] if ".norm" in k or k.startswith("norm") else None
        if g is not None:
            g.append(float((p.grad != 0).float().mean()))
    frac = {k: f"{np.mean(v):.3f}" for k, v in groups.items() if v}
    check("C2 non-zero gradient fraction per group (informational)", True, **frac)
    worst = max(sat.items(), key=lambda kv: kv[1])
    by_op = {op: f"{np.mean([v for (l, o), v in sat.items() if o == op]):.4f}%" for op in ("qkv", "qk", "av")}
    check("C2 ADC-saturated (gradient-blocked) conversions (informational)", True,
          mean_by_op=by_op, worst=f"{worst[0][0]} {worst[0][1]} {worst[1]:.4f}%")
    write_json(Path(a.out) / "saturation_per_layer.json", {f"{l}|{o}": v for (l, o), v in sat.items()})
    run.opt.step()
    bnm = bn_modules(run.student)
    check("C3 BatchNorm modules stay in eval mode in training mode",
          all(not m.training for m in bnm) and run.student.training, n_bn=len(bnm))
    for _ in range(2):
        (x2, y2), = batches(run, 1, epoch=1)
        run.step(x2, y2)
    bn1 = bn_state(run.student)
    check("C3 BatchNorm running statistics unchanged after 3 optimizer steps",
          all(torch.equal(bn0[k], bn1[k]) for k in bn0), n_buffers=len(bn0))
    p1 = params(run.student)
    bn_aff = [k for k in named if any(k.startswith(n + ".") for n, mm in run.student.named_modules()
                                      if isinstance(mm, torch.nn.BatchNorm2d))]
    check("C3 BatchNorm affine parameters are trained", all(not torch.equal(p0[k], p1[k]) for k in bn_aff),
          n_affine=len(bn_aff))
    t_state1 = run.teacher.state_dict()
    check("C4 teacher frozen: no grads, eval mode, unchanged",
          all(not p.requires_grad and p.grad is None for p in run.teacher.parameters())
          and not run.teacher.training and all(torch.equal(t_state0[k], t_state1[k]) for k in t_state0))
    fresh = build_fp32().to(dev).eval()
    with torch.no_grad():
        eq = torch.equal(fresh(x.to(dev)), run.teacher(x.to(dev)))
    check("C4 teacher output == fresh FP32 pretrained model (bit-exact, same device)", eq)
    st = copy.deepcopy(run.teacher)
    from cim_qat.attention import convert_attention_surrogate
    convert_attention_surrogate(st, run.student.layers[1].blocks[0].attn.specs)
    set_cim_enabled(st, False)
    with torch.no_grad():
        z = kd_loss(st(x.to(dev)), run.teacher(x.to(dev))).item()
    check("C5 KD loss == 0 for an identical FP32 student", abs(z) < 1e-6, kd=f"{z:.2e}")
    changed = [k for k in named if not torch.equal(p0[k], p1[k])]
    zero_grad_params = [k for k in named if named[k].grad is not None and (named[k].grad == 0).all()]
    check("C6 optimizer updated every parameter with a non-zero gradient",
          len(changed) + len(zero_grad_params) >= len(named), changed=len(changed), of=len(named))

    print("== C7 determinism (two fresh runs, 3 identical steps)")
    finals = []
    for r in range(2):
        rr = make_run(a, tmp / f"det{r}")
        for xb, yb in batches(rr, 3):
            rr.step(xb, yb)
        finals.append(params(rr.student))
    check("C7 bit-identical parameters after 3 steps in two fresh runs",
          all(torch.equal(finals[0][k], finals[1][k]) for k in finals[0]))

    print("== C8 save/resume")
    ra = make_run(a, tmp / "resA")
    for xb, yb in batches(ra, 4):
        ra.step(xb, yb)
    rb = make_run(a, tmp / "resB")
    for xb, yb in batches(rb, 2):
        rb.step(xb, yb)
    from cim_qat.common import torch_save_atomic
    torch_save_atomic(rb.state(), tmp / "resB" / "last.pth")
    rc = make_run(a, tmp / "resC")
    rc.load_state(torch.load(tmp / "resB" / "last.pth", map_location="cpu", weights_only=False))   # as qat_train
    for xb, yb in batches(rc, 2, epoch=0, start=2):
        rc.step(xb, yb)
    pa, pc = params(ra.student), params(rc.student)
    check("C8 2 steps + save + load + 2 steps == 4 uninterrupted steps",
          all(torch.equal(pa[k], pc[k]) for k in pa) and rc.global_step == ra.global_step,
          lr_equal=ra.opt.param_groups[0]["lr"] == rc.opt.param_groups[0]["lr"])

    # C8b: the same resume, loading the checkpoint straight onto the run device (map_location=dev),
    # which moves the RNG states to the GPU; load_state must still restore them (v2.3 fix)
    rd = make_run(a, tmp / "resD")
    rd.load_state(torch.load(tmp / "resB" / "last.pth", map_location=dev, weights_only=False))
    for xb, yb in batches(rd, 2, epoch=0, start=2):
        rd.step(xb, yb)
    pd = params(rd.student)
    check("C8b resume also exact when the checkpoint is loaded with map_location=<run device>",
          all(torch.equal(pa[k], pd[k]) for k in pa) and rd.global_step == ra.global_step)

    print("== C9 overfit one small batch (CE)")
    ro = make_run(a, tmp / "ovf", loss="ce", lr=1e-4, batch_size=4)
    (xo, yo), = batches(ro, 1)
    for g in ro.opt.param_groups:
        g["lr"] = 1e-4
    ro.sched = torch.optim.lr_scheduler.LambdaLR(ro.opt, lambda s: 1.0)
    losses = [ro.step(xo, yo)["loss"] for _ in range(60)]
    check("C9 loss collapses on one fixed batch", losses[-1] < max(0.05, 0.1 * losses[0]),
          first=f"{losses[0]:.3f}", last=f"{losses[-1]:.4f}")

    print("== C10 saved student loads into the simulator's builder")
    h = run.save_model_only("c10_model.pth")
    m2 = build_fp32(run.out / "checkpoints" / "c10_model.pth").to(dev).eval()
    set_cim_enabled(run.student, False)
    run.student.eval()
    with torch.no_grad():
        same = torch.equal(m2(x.to(dev)), run.student(x.to(dev)))
    set_cim_enabled(run.student, True)
    check("C10 run_tinyvit.build_model(student checkpoint) == student FP32 forward", same, sha=h[:12])

    shutil.rmtree(tmp, ignore_errors=True)
    nfail = sum(not r["ok"] for r in RES)
    write_json(Path(a.out) / f"mechanics_{dev.type}.json",
               dict(time_utc=now_iso(), device=str(dev), results=RES, wall_s=time.time() - t0))
    print(f"\n{len(RES) - nfail}/{len(RES)} checks passed ({time.time() - t0:.0f}s)")
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
