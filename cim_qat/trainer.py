"""
cim_qat/trainer.py — the QAT training core (used by qat_train.py AND by
test_training_mechanics.py, so the tests exercise the code that trains).

Approved design (QAT_DESIGN.md v2):
  * student = pretrained TinyViT-5M, attention on the differentiable surrogate,
    frozen per-op gains; ALL parameters trainable; BatchNorm running stats
    frozen (BN modules in eval mode during training).
  * loss "kd": forward KL(p_teacher || p_student), T = 1, teacher = frozen FP32
    pretrained model on the same augmented batch. "ce": cross-entropy (pilot).
  * AdamW lr 1e-5, wd 1e-4, per-step linear warm-up (0.1 epoch) then cosine to
    0 over the scheduled budget; batch 64; fp32, TF32 off, deterministic.
  * VAL = the FULL ImageNet-1k val set (50,000), evaluated with the surrogate
    at step 0 (epoch-0 PTQ) and at the end of every epoch; ACTUAL-simulator
    spot check on 128 fixed VAL images at the same points.
  * checkpoint selection (as in the old QAT): PRIMARY = best VAL checkpoint
    (max hardware top-1 under the surrogate, ties -> lower VAL NLL), among the
    trained epochs; SECONDARY (pre-declared) = the last checkpoint.
  * early stopping (as in the old QAT): stop when VAL has not improved for
    `early_stopping_patience` consecutive epochs (default 2; old QAT used 8
    of a 40-epoch budget). Plateau/extension rule reported at the end.
  * ImageNetV2 (TEST) is never opened here: the Splits object is created
    without TEST access (cim_qat.data.TestAccessError).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import metrics as M
from .attention import merged_stats, set_collect_stats
from .common import (canonical, configure_determinism, now_iso, read_json, savez_atomic,
                     seed_everything, sha256_file, sha256_text, torch_save_atomic, write_json)
from .hw_config import OPS, load_frozen
from .model import (build_student, build_teacher, set_cim_enabled, set_train_mode,
                    student_state_for_simulator, to_device)

CHECKPOINT_SCHEMA = "cim_qat.train_state.v1"


@dataclass
class TrainConfig:
    run_name: str
    precision: str                      # "W8A8" | "W4A4"
    frozen_gains: str                   # path to frozen_gains.json
    paths: str = "configs/paths.json"
    loss: str = "kd"                    # "kd" | "ce"
    kd_temperature: float = 1.0
    epochs: float = 5.0
    batch_size: int = 64
    lr: float = 1e-5
    weight_decay: float = 1e-4
    warmup_epochs: float = 0.1
    bn_frozen: bool = True
    amp: bool = False
    tf32: bool = False
    deterministic: bool = True
    seed: int = 0
    num_workers: int = 8
    val_batch_size: int = 128
    early_stopping_patience: int = 2    # epochs without VAL improvement before stopping
    student_fp32_eval_every_epoch: bool = True
    spot_check_n: int = 128
    spot_check_workers: int = 16
    plateau_threshold_nats: float = 0.002
    plateau_se_mult: float = 2.0
    log_every: int = 50
    max_steps: int | None = None        # smoke tests only
    val_limit: int | None = None        # smoke tests only (VAL subset)
    out_root: str | None = None         # default <work_root>/runs
    notes: str = ""

    @property
    def bits(self):
        p = self.precision.upper()
        if p not in ("W8A8", "W4A4"):
            raise ValueError("precision must be W8A8 or W4A4 in this phase")
        return (8, 8) if p == "W8A8" else (4, 4)

    @classmethod
    def load(cls, path, **overrides) -> "TrainConfig":
        d = read_json(path)
        d.update({k: v for k, v in overrides.items() if v is not None})
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"{path}: unknown config keys {sorted(unknown)}")
        if "{work_root}" in d.get("frozen_gains", ""):
            wr = read_json(d.get("paths", "configs/paths.json"))["work_root"]
            d["frozen_gains"] = d["frozen_gains"].replace("{work_root}", wr)
        return cls(**d)

    def to_dict(self):
        return asdict(self)


# settings that cannot change any number of a run (data order and augmentation are
# per-sample seeded; simulator workers are batch-1 deterministic) -> excluded from the
# config hash, so they may be changed when resuming (e.g. fewer loader workers when two
# runs share the machine)
HASH_EXEMPT = ("num_workers", "spot_check_workers", "notes")


def config_hash(cfg: TrainConfig, extra: dict) -> str:
    d = {k: v for k, v in cfg.to_dict().items() if k not in HASH_EXEMPT}
    return sha256_text(canonical(dict(cfg=d, **extra)))[:16]


# --------------------------------------------------------------------------- #
# losses and schedule
# --------------------------------------------------------------------------- #


def kd_loss(student_logits, teacher_logits, T: float = 1.0):
    """Forward KL(p_T || p_S), batch mean, x T^2 (Hinton); T = 1 is plain KL."""
    s = F.log_softmax(student_logits.float() / T, dim=-1)
    t = F.log_softmax(teacher_logits.float() / T, dim=-1)
    return F.kl_div(s, t, log_target=True, reduction="batchmean") * (T * T)


def lr_factor(step: int, total: int, warmup: int) -> float:
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    if total <= warmup:
        return 1.0
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * prog))


def _cpu_u8(t) -> torch.Tensor:
    """RNG states must be CPU uint8 tensors. torch.load(map_location=<cuda>) moves
    them to the GPU, and set_rng_state then raises "RNG state must be a
    torch.ByteTensor"; bring them back whatever device/dtype they arrive as."""
    return torch.as_tensor(t).detach().to(device="cpu", dtype=torch.uint8).contiguous()


def restore_rng(rng: dict) -> None:
    """Restore the python / numpy / torch-CPU / torch-CUDA generator states saved by Run.state()."""
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(_cpu_u8(rng["torch"]))
    cuda = rng.get("cuda")
    if cuda is not None and torch.cuda.is_available():
        n = torch.cuda.device_count()
        if len(cuda) != n:
            print(f"WARNING: checkpoint has {len(cuda)} CUDA RNG state(s), {n} device(s) visible; "
                  f"restoring the first {min(n, len(cuda))}")
        for i, t in enumerate(cuda[:n]):
            torch.cuda.set_rng_state(_cpu_u8(t), i)


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #


class Run:
    """Holds models, optimizer and data for one training run."""

    def __init__(self, cfg: TrainConfig, device: torch.device, splits, out_dir: Path):
        self.cfg, self.dev, self.splits, self.out = cfg, device, splits, Path(out_dir)
        wb, ab = cfg.bits
        self.hw, self.gains_rec = load_frozen(cfg.frozen_gains, wb, ab)
        configure_determinism(cfg.deterministic, cfg.tf32)
        seed_everything(cfg.seed)
        # to_device (not .to): also moves TinyViT's cached attention-bias tensor `ab` (model.py)
        self.student = to_device(build_student(self.hw), device)
        self.teacher = to_device(build_teacher(), device) if cfg.loss == "kd" else None
        self.ref_teacher = self.teacher if self.teacher is not None else to_device(build_teacher(), device)
        if cfg.loss not in ("kd", "ce"):
            raise ValueError(cfg.loss)
        params = [p for p in self.student.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        from .data import TrainImages
        self.train_ds = TrainImages(splits, "train", seed=cfg.seed)
        self.steps_per_epoch = len(self.train_ds) // cfg.batch_size
        self.total_steps = max(1, int(round(cfg.epochs * self.steps_per_epoch)))
        if cfg.max_steps is not None:
            self.total_steps = min(self.total_steps, cfg.max_steps)
        self.warmup = int(round(cfg.warmup_epochs * self.steps_per_epoch))
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda s: lr_factor(s, self.total_steps, self.warmup))
        # evaluation points: step 0 (epoch-0 PTQ), every epoch end, and the last step
        self.eval_steps = sorted(set(list(range(0, self.total_steps, self.steps_per_epoch))
                                     + [self.total_steps]))
        self.global_step = 0
        self.history = []
        self.best = None
        self.stopped = None
        self.fork = None             # set when this run is a D2 fork (see prepare_fork)

    # -- one optimisation step ------------------------------------------- #

    def step(self, x, y) -> dict:
        cfg = self.cfg
        set_train_mode(self.student, cfg.bn_frozen)
        x, y = x.to(self.dev, non_blocking=True), y.to(self.dev, non_blocking=True)
        ctx = (torch.autocast(self.dev.type, dtype=torch.bfloat16) if cfg.amp
               else torch.autocast(self.dev.type, enabled=False))
        with ctx:
            s = self.student(x)
        if cfg.loss == "kd":
            with torch.no_grad():
                t = self.teacher(x)
            loss = kd_loss(s, t, cfg.kd_temperature)
        else:
            loss = F.cross_entropy(s.float(), y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in self.student.parameters()
                            if p.grad is not None))
        if not torch.isfinite(loss) or not torch.isfinite(gn):
            raise FloatingPointError(f"non-finite loss/grad at step {self.global_step}: "
                                     f"loss={loss.item()} grad_norm={gn.item()}")
        self.opt.step()
        self.sched.step()
        self.global_step += 1
        return dict(loss=float(loss.item()), grad_norm=float(gn.item()),
                    lr=self.opt.param_groups[0]["lr"],
                    acc=float((s.argmax(-1) == y).float().mean().item()))

    # -- evaluation -------------------------------------------------------- #

    def val_loader(self, indices=None):
        from .data import EvalImages
        ds = EvalImages(self.splits, "val", indices=indices)
        return torch.utils.data.DataLoader(ds, batch_size=self.cfg.val_batch_size, shuffle=False,
                                           num_workers=self.cfg.num_workers, pin_memory=True)

    def val_indices(self):
        n = len(self.splits.rows("val"))
        lim = self.cfg.val_limit
        return list(range(n)) if lim is None else list(np.linspace(0, n - 1, lim).astype(int))

    @torch.no_grad()
    def logits_on(self, model, indices, collect=False):
        model.eval()
        if collect:
            set_collect_stats(model, True)
        out, labels, idx = [], [], []
        for x, y, i in self.val_loader(indices):
            out.append(model(x.to(self.dev)).float().cpu())
            labels.append(y)
            idx.append(i)
        stats = None
        if collect:
            ms = merged_stats(model)
            stats = {op: ms[op].as_dict() for op in OPS}
            for op in OPS:
                h = ms[op].abs_code_hist
                if h is not None and h.sum() > 0:
                    c = torch.cumsum(h, 0).double() / h.sum()
                    stats[op]["median_abs_code"] = int(torch.searchsorted(c, torch.tensor(0.5, dtype=c.dtype)))
                    stats[op]["p999_abs_code"] = int(torch.searchsorted(c, torch.tensor(0.999, dtype=c.dtype)))
            set_collect_stats(model, False)
        return torch.cat(out).numpy(), torch.cat(labels).numpy(), torch.cat(idx).numpy(), stats

    def teacher_val_logits(self):
        p = self.out / "teacher_val_logits.npz"
        ind = self.val_indices()
        if p.exists():
            with np.load(p) as z:
                if np.array_equal(z["idx"], np.array(ind)):
                    return z["logits"], z["labels"]
        lg, lab, idx, _ = self.logits_on(self.ref_teacher, ind)
        savez_atomic(p, logits=lg, labels=lab, idx=np.array(ind))
        return lg, lab

    def evaluate(self, tag: str, with_fp32: bool) -> dict:
        t0 = time.time()
        ind = self.val_indices()
        t_lg, labels = self.teacher_val_logits()
        lg, lab, idx, stats = self.logits_on(self.student, ind, collect=True)
        assert np.array_equal(lab, labels)
        pi = M.per_image(lg, lab, t_lg)
        rec = dict(tag=tag, step=self.global_step, epoch=self.global_step / self.steps_per_epoch,
                   split="val", val_manifest_sha256=self.splits.sha("val"), n_val=len(ind),
                   val_limited=self.cfg.val_limit is not None,
                   hw_val=M.summarize(pi), op_stats=stats, time_utc=now_iso())
        savez_atomic(self.out / "val_perimage" / f"{tag}.npz", correct1=pi["correct1"],
                     nll=pi["nll"].astype(np.float32), kl=pi["kl_ref"].astype(np.float32),
                     pred=pi["pred"].astype(np.int16))
        sp = self.spot_indices()
        pos = np.searchsorted(np.array(ind), sp)
        savez_atomic(self.out / "spot" / f"{tag}_surrogate_logits.npz", logits=lg[pos], idx=np.array(sp))
        if with_fp32:
            set_cim_enabled(self.student, False)
            lg2, _, _, _ = self.logits_on(self.student, ind)
            set_cim_enabled(self.student, True)
            rec["student_fp32_val"] = M.summarize(M.per_image(lg2, lab, t_lg))
        rec["eval_seconds"] = time.time() - t0
        return rec

    def spot_indices(self):
        ind = self.val_indices()
        n = min(self.cfg.spot_check_n, len(ind))
        return [ind[i] for i in np.linspace(0, len(ind) - 1, n).astype(int)]

    def spot_check(self, tag: str) -> dict:
        """The ACTUAL simulator on fixed VAL images, three comparisons:
          exact   : CPU surrogate, batch 1, current weights == simulator (bit for bit)
          device  : training-device surrogate (batched) vs simulator, KL / agreement
          floor   : simulator vs simulator with a 1-ulp input change (self-sensitivity)
        """
        import copy as _copy

        from . import sim_eval
        from .data import load_uint8_eval, to_model_input
        rows = self.splits.rows("val")
        sp = self.spot_indices()
        tmp = self.out / "spot" / f"{tag}_model.pth"
        torch.save(student_state_for_simulator(self.student), tmp)
        items = [(rows[i]["idx"], str(self.splits.path(rows[i]))) for i in sp]
        job = dict(kind="analog", ckpt=str(tmp), hw=self.hw.job_dict(), with_own_fp32=False)
        r = sim_eval.run(job, items, self.out / "spot" / f"{tag}_sim",
                         workers=self.cfg.spot_check_workers, progress=False)
        rp = sim_eval.run(dict(job, input_perturb="ulp_up"), items, self.out / "spot" / f"{tag}_sim_ulp",
                          workers=self.cfg.spot_check_workers, progress=False)
        from .common import single_thread
        cpu = _copy.deepcopy(self.student).cpu().eval()
        with torch.no_grad(), single_thread():
            exact = np.stack([cpu(to_model_input(load_uint8_eval(p)[None]))[0].numpy() for _, p in items])
        del cpu
        with np.load(self.out / "spot" / f"{tag}_surrogate_logits.npz") as z:
            sur = z["logits"]
        labels = np.array([rows[i]["label"] for i in sp])
        s_dev = M.summarize(M.per_image(sur, labels, r["logits"]))
        s_flo = M.summarize(M.per_image(rp["logits"], labels, r["logits"]))
        tmp.unlink()
        n_exact = int((exact == r["logits"]).all(1).sum())
        return dict(n=len(sp), cpu_batch1_bit_exact_images=n_exact, cpu_batch1_exact=n_exact == len(sp),
                    kl_sim_to_surrogate=s_dev["kl_ref"], agree_pct=s_dev["agree_ref_pct"],
                    rel_logit_rms_pct=s_dev["rel_logit_rms_pct"],
                    kl_sim_self_sensitivity_floor=s_flo["kl_ref"],
                    sim_clip_pct={op: r["summary"]["analog_summary"][op]["clip_pct"] for op in OPS})

    # -- checkpoints -------------------------------------------------------- #

    def state(self) -> dict:
        return dict(schema=CHECKPOINT_SCHEMA, global_step=self.global_step,
                    model=self.student.state_dict(), optimizer=self.opt.state_dict(),
                    scheduler=self.sched.state_dict(), history=self.history, best=self.best,
                    stopped=self.stopped, fork=self.fork,
                    rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                             torch=torch.get_rng_state(),
                             cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
                    config=self.cfg.to_dict(), hw=self.hw.to_dict(),
                    frozen_gains_sha256=sha256_file(self.cfg.frozen_gains))

    def load_state(self, st: dict, allow_changes: tuple = ()) -> None:
        """Resume from a training checkpoint. `allow_changes` is used ONLY by the D2
        fork (prepare_fork), which may change run_name and early_stopping_patience."""
        if st.get("schema") != CHECKPOINT_SCHEMA:
            raise RuntimeError("not a cim_qat training checkpoint")
        if st["frozen_gains_sha256"] != sha256_file(self.cfg.frozen_gains):
            raise RuntimeError("frozen gains file differs from the one this run started with")
        diff = {k: (st["config"].get(k), v) for k, v in self.cfg.to_dict().items()
                if k not in ("max_steps",) + HASH_EXEMPT + tuple(allow_changes)
                and st["config"].get(k) != v}
        if diff:
            raise RuntimeError(f"resume refused: config differs from the checkpoint: {diff}")
        self.student.load_state_dict(st["model"])
        self.opt.load_state_dict(st["optimizer"])
        self.sched.load_state_dict(st["scheduler"])
        self.global_step = st["global_step"]
        self.history, self.best = st["history"], st["best"]
        self.stopped = st.get("stopped")
        self.fork = st.get("fork")
        restore_rng(st["rng"])

    def save_model_only(self, name: str) -> str:
        p = self.out / "checkpoints" / name
        torch_save_atomic(student_state_for_simulator(self.student), p)
        h = sha256_file(p)
        reg = self.out / "checkpoints" / "checkpoints.json"
        d = read_json(reg) if reg.exists() else {}
        d[name] = dict(sha256=h, global_step=self.global_step, time_utc=now_iso())
        write_json(reg, d)
        return h

    # -- train loader ----------------------------------------------------- #

    def train_loader(self, epoch: int, start_batch: int):
        from .data import EpochSampler
        sampler = EpochSampler(len(self.train_ds), self.cfg.seed, epoch, start_batch * self.cfg.batch_size)
        return torch.utils.data.DataLoader(self.train_ds, batch_size=self.cfg.batch_size, sampler=sampler,
                                           num_workers=self.cfg.num_workers, pin_memory=True,
                                           drop_last=True, persistent_workers=False,
                                           prefetch_factor=4 if self.cfg.num_workers > 0 else None)

    # -- the loop ------------------------------------------------------------ #

    def fit(self, log=print, spot=True) -> dict:
        cfg = self.cfg
        csv_p = self.out / "metrics_steps.csv"
        new = not csv_p.exists()
        fcsv = open(csv_p, "a", newline="")
        wr = csv.writer(fcsv)
        if new:
            wr.writerow(["global_step", "epoch", "loss", "acc", "grad_norm", "lr", "step_s", "data_s",
                         "img_per_s", "gpu_mem_gb"])
        if self.global_step == 0 and not self.history:
            self._eval_point(log, spot)            # epoch-0 PTQ (before any update)
        t_data = time.time()
        while self.global_step < self.total_steps and not self.stopped:
            epoch, b0 = divmod(self.global_step, self.steps_per_epoch)
            for x, y, _ in self.train_loader(epoch, b0):
                dt_data = time.time() - t_data
                t0 = time.time()
                m = self.step(x, y)
                if self.dev.type == "cuda":
                    torch.cuda.synchronize()
                dt = time.time() - t0
                if self.global_step % cfg.log_every == 0 or self.global_step == 1:
                    mem = torch.cuda.max_memory_allocated() / 1e9 if self.dev.type == "cuda" else 0.0
                    wr.writerow([self.global_step, f"{self.global_step / self.steps_per_epoch:.4f}",
                                 f"{m['loss']:.6f}", f"{m['acc']:.4f}", f"{m['grad_norm']:.4e}",
                                 f"{m['lr']:.4e}", f"{dt:.4f}", f"{dt_data:.4f}",
                                 f"{cfg.batch_size / max(dt + dt_data, 1e-9):.1f}", f"{mem:.2f}"])
                    fcsv.flush()
                    log(f"step {self.global_step}/{self.total_steps} ep {self.global_step / self.steps_per_epoch:.3f} "
                        f"loss {m['loss']:.4f} acc {m['acc']:.3f} gn {m['grad_norm']:.3e} lr {m['lr']:.2e} "
                        f"step {dt:.3f}s data {dt_data:.3f}s")
                if self.global_step in self.eval_steps:
                    self._eval_point(log, spot)
                t_data = time.time()
                if self.global_step >= self.total_steps or self.stopped:
                    break
            if self.stopped:
                break
        fcsv.close()
        self.save_model_only("final_model.pth")
        summ = self.summary()
        write_json(self.out / "summary.json", summ)
        return summ

    def _eval_point(self, log, spot):
        cfg = self.cfg
        tag = f"step{self.global_step:07d}"
        at_epoch_end = self.global_step % self.steps_per_epoch == 0
        rec = self.evaluate(tag, with_fp32=cfg.student_fp32_eval_every_epoch)
        if spot:
            rec["spot_check"] = self.spot_check(tag)
            if not rec["spot_check"]["cpu_batch1_exact"]:
                raise RuntimeError(f"spot check at {tag}: CPU surrogate != simulator on "
                                   f"{rec['spot_check']['n'] - rec['spot_check']['cpu_batch1_bit_exact_images']} "
                                   "images. Stopping: the surrogate no longer reproduces the simulator.")
        h = rec["hw_val"]
        key = (h["top1_pct"], -h["nll"])
        if self.global_step > 0 and (self.best is None or key > tuple(self.best["key"])):
            self.best = dict(tag=tag, step=self.global_step, epoch=rec["epoch"], key=list(key))
            rec["new_best"] = True
            self.save_model_only("best_model.pth")
        # early stopping (epoch-end evaluations after the best one, as in the old QAT)
        if self.global_step > 0 and at_epoch_end:
            since = sum(1 for r in self.history if r["step"] > self.best["step"]
                        and r["step"] % self.steps_per_epoch == 0) + (0 if rec.get("new_best") else 1)
            rec["epochs_since_best"] = since
            if (since >= cfg.early_stopping_patience and self.global_step < self.total_steps):
                self.stopped = dict(step=self.global_step, epoch=rec["epoch"],
                                    reason=f"VAL early stopping: no improvement for {since} epoch(s) "
                                           f"(patience {cfg.early_stopping_patience})")
                rec["early_stop"] = self.stopped
        self.history.append(rec)
        with open(self.out / "evals.jsonl", "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        torch_save_atomic(self.state(), self.out / "checkpoints" / "last.pth")
        log(f"[eval {tag}] VAL hw top1 {h['top1_pct']:.2f}% nll {h['nll']:.4f} KL(T||S) {h['kl_ref']:.4f} "
            f"agree {h['agree_ref_pct']:.2f}% clip% " +
            "/".join(f"{rec['op_stats'][o]['clip_pct_sim']:.4f}" for o in OPS) +
            (f" | spot: CPU-exact {rec['spot_check']['cpu_batch1_bit_exact_images']}/{rec['spot_check']['n']} "
             f"KL(sim||sur) {rec['spot_check']['kl_sim_to_surrogate']:.2e} vs 1-ulp floor "
             f"{rec['spot_check']['kl_sim_self_sensitivity_floor']:.2e}" if "spot_check" in rec else "") +
            (f" | student FP32 top1 {rec['student_fp32_val']['top1_pct']:.2f}%" if "student_fp32_val" in rec else "")
            + (" | NEW BEST" if rec.get("new_best") else
               (f" | {rec['epochs_since_best']} epoch(s) since best" if "epochs_since_best" in rec else "")))
        if self.stopped and rec.get("early_stop"):
            log(f"EARLY STOP at step {self.global_step}: {self.stopped['reason']}")

    def summary(self) -> dict:
        """Final vs one epoch earlier: the pre-declared plateau/extension rule."""
        cfg = self.cfg
        evs = self.history
        fin = evs[-1]
        prev = None
        for r in evs[::-1]:
            if r["step"] <= fin["step"] - self.steps_per_epoch:
                prev = r
                break
        reg = read_json(self.out / "checkpoints" / "checkpoints.json")
        best_rec = next((r for r in evs if self.best and r["tag"] == self.best["tag"]), None)
        out = dict(run_name=cfg.run_name, final=fin, best=self.best, best_eval=best_rec, epoch0=evs[0],
                   total_steps=self.total_steps, steps_per_epoch=self.steps_per_epoch,
                   steps_done=self.global_step, stopped_early=self.stopped, forked_from=self.fork,
                   stopped_reason=(self.stopped["reason"] if self.stopped else "completed the scheduled budget"),
                   selection=dict(split="val", val_manifest_sha256=self.splits.sha("val"),
                                  n_val=fin["n_val"], val_limited=fin["val_limited"],
                                  primary=dict(file="best_model.pth", **reg.get("best_model.pth", {}),
                                               rule="max VAL hardware top-1 (surrogate), ties -> lower VAL NLL"),
                                  secondary=dict(file="final_model.pth", **reg.get("final_model.pth", {}),
                                                 rule="last checkpoint (end of budget or early stop)"),
                                  early_stopping_patience=cfg.early_stopping_patience),
                   test_split_used=False, finished_utc=now_iso())
        if prev is not None:
            a = np.load(self.out / "val_perimage" / f"{prev['tag']}.npz")
            b = np.load(self.out / "val_perimage" / f"{fin['tag']}.npz")
            metric = "kl" if cfg.loss == "kd" else "nll"
            d_obj = float(a[metric].astype(np.float64).mean() - b[metric].astype(np.float64).mean())
            dtop = M.paired_diff(a["correct1"], b["correct1"])
            conv = (d_obj < cfg.plateau_threshold_nats and
                    abs(dtop["mean"]) <= cfg.plateau_se_mult * dtop["se"])
            out["plateau"] = dict(compared=[prev["tag"], fin["tag"]], objective_metric=metric,
                                  objective_improvement_nats=d_obj,
                                  top1_change_pct=100 * dtop["mean"], top1_paired_se_pct=100 * dtop["se"],
                                  threshold_nats=cfg.plateau_threshold_nats, converged=bool(conv),
                                  recommendation=("early-stopped on VAL (no extension)" if self.stopped else
                                                  "converged within the pilot budget" if conv else
                                                  "NOT converged: recommend a fresh run at 2x the budget "
                                                  "from the pretrained checkpoint (QAT_DESIGN.md §7.3)"))
        return out


# --------------------------------------------------------------------------- #
# D2 fork: complete the schedule of an EARLY-STOPPED run with early stopping off
# --------------------------------------------------------------------------- #

FORK_ALLOWED = ("run_name", "early_stopping_patience")   # + HASH_EXEMPT keys
FORK_COPY = ("train.log", "metrics_steps.csv", "evals.jsonl", "teacher_val_logits.npz",
             "val_perimage", "spot")


def prepare_fork(src_dir: Path, dst_dir: Path, cfg: TrainConfig) -> tuple[dict, dict]:
    """Decision D2 (QAT_DESIGN.md §0.4). Check that `src_dir` is a finished,
    EARLY-STOPPED run and that `cfg` differs from its config only in run_name and
    early_stopping_patience (which must switch early stopping off: patience >=
    epochs), then create `dst_dir` with the source's history (logs, evaluations,
    per-image VAL arrays, spot checks, cached teacher logits, best checkpoint).

    The source's saved state at the stop point is returned for Run.load_state; with
    every random stream, the optimizer and the schedule restored, continuing from it
    is bit-identical to a run of the same config that never had early stopping
    (tests_qat/test_fork.py). The source folder is only read.
    -> (fork_info, source_state)
    """
    import shutil
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    if dst_dir.exists():
        raise FileExistsError(f"{dst_dir} exists: a fork is made once, into a new run folder")
    summ_p, last_p = src_dir / "summary.json", src_dir / "checkpoints" / "last.pth"
    if not summ_p.exists() or not last_p.exists():
        raise RuntimeError(f"{src_dir}: not a finished run (summary.json / checkpoints/last.pth missing)")
    summ = read_json(summ_p)
    if not summ.get("stopped_early"):
        raise RuntimeError(f"{src_dir.name} completed its scheduled budget (no early stop): "
                           "D2 forks only early-stopped runs")
    st = torch.load(last_p, map_location="cpu", weights_only=False)
    if not st.get("stopped") or st["global_step"] != summ["steps_done"]:
        raise RuntimeError(f"{last_p} is not the early-stop state recorded in {summ_p}")
    new = cfg.to_dict()
    old = st["config"]
    changed = {k: (old.get(k), v) for k, v in new.items()
               if k not in ("max_steps",) + HASH_EXEMPT and old.get(k) != v}
    bad = sorted(set(changed) - set(FORK_ALLOWED))
    if bad:
        raise RuntimeError(f"a D2 fork may change only {FORK_ALLOWED}; this config also changes "
                           f"{ {k: changed[k] for k in bad} }")
    if "run_name" not in changed or new["run_name"] != dst_dir.name:
        raise RuntimeError("the fork needs its own run_name (= the new run folder)")
    if cfg.early_stopping_patience < cfg.epochs:
        raise RuntimeError(f"D2 switches early stopping off: early_stopping_patience must be >= epochs "
                           f"({cfg.epochs}); got {cfg.early_stopping_patience}")
    dst_dir.mkdir(parents=True)
    for name in FORK_COPY:
        p = src_dir / name
        if p.is_dir():
            shutil.copytree(p, dst_dir / name)
        elif p.exists():
            shutil.copy2(p, dst_dir / name)
    (dst_dir / "checkpoints").mkdir()
    reg = read_json(src_dir / "checkpoints" / "checkpoints.json")
    reg.pop("final_model.pth", None)                     # the fork writes its own final model
    if (src_dir / "checkpoints" / "best_model.pth").exists():
        shutil.copy2(src_dir / "checkpoints" / "best_model.pth", dst_dir / "checkpoints" / "best_model.pth")
    write_json(dst_dir / "checkpoints" / "checkpoints.json", reg)
    src_cr = src_dir / "config_resolved.json"
    info = dict(decision="D2 (QAT_DESIGN.md §0.4)", source_run=src_dir.name, source_dir=str(src_dir),
                source_last_sha256=sha256_file(last_p), source_summary_sha256=sha256_file(summ_p),
                source_config_hash=read_json(src_cr).get("config_hash") if src_cr.exists() else None,
                source_stopped=st["stopped"], forked_at_step=st["global_step"],
                changed={k: list(v) for k, v in changed.items()},
                copied=[n for n in FORK_COPY if (src_dir / n).exists()] + ["checkpoints/best_model.pth"],
                time_utc=now_iso())
    return info, st
