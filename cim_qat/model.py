"""
cim_qat/model.py — model construction for QAT and evaluation.

* All weights are loaded through the SIMULATOR's own builder
  (run_tinyvit.build_model: TinyViT-5M config, strict key check, eval mode).
* The student is that model with every Attention replaced by
  CIMAttentionSurrogate. The replacement re-uses the same submodules, so the
  student's state_dict has exactly the original TinyViT keys and a saved
  student loads back into run_tinyvit.build_model unchanged (tested).
* BatchNorm policy (approved): running statistics FROZEN — every BatchNorm
  module stays in eval mode even while the rest of the model is in train mode;
  its affine weight/bias remain trainable. TinyViT-5M has no dropout and
  drop_path_rate = 0, so BatchNorm is the only train/eval difference.
* Teacher (KD): a separate copy of the pretrained FP32 model, eval mode,
  requires_grad False, never converted to CIM.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .attention import convert_attention_surrogate, surrogate_modules
from .common import REPO_ROOT, sha256_file
from .hw_config import HWConfig, op_specs

PRETRAINED = REPO_ROOT / "tinyvit5m.pth"
PRETRAINED_SHA256 = "e4894bab91be2d8f1b5e9a2147b57f35ae84be46e195d6586308864fb06fad3d"


def build_fp32(path=None) -> nn.Module:
    """TinyViT-5M FP32 via the simulator's builder (eval mode)."""
    import run_tinyvit
    return run_tinyvit.build_model(str(path or PRETRAINED))


def check_pretrained(path=None) -> str:
    h = sha256_file(path or PRETRAINED)
    if (path is None or Path(path) == PRETRAINED) and h != PRETRAINED_SHA256:
        raise RuntimeError(f"pretrained checkpoint sha256 {h} != registry {PRETRAINED_SHA256}")
    return h


def to_device(model: nn.Module, device) -> nn.Module:
    """model.to(device) PLUS the tensors .to() does not move.

    Upstream TinyViT's Attention caches `self.ab = attention_biases[:, idxs]` as a
    plain attribute whenever it is put in eval mode (tiny_vit.py, Attention.train).
    `.to(device)` moves parameters and buffers only, so a model built (and put in
    eval mode) on the CPU and then moved to CUDA keeps a CPU `ab` and fails at
    `q @ k^T * scale + ab` ("found at least two devices"). This re-derives `ab`
    on the new device exactly as Attention.train(False) does (no_grad, same
    indexing), without changing any train/eval mode, then asserts that no tensor
    attribute is left on another device.
    """
    model.to(device)
    with torch.no_grad():
        for m in model.modules():
            if "ab" in vars(m) and hasattr(m, "attention_biases") and hasattr(m, "attention_bias_idxs"):
                m.ab = m.attention_biases[:, m.attention_bias_idxs]
    assert_on_device(model, device)
    return model


def assert_on_device(model: nn.Module, device) -> None:
    """Every parameter, buffer and plain tensor attribute is on `device` (type and index)."""
    want = torch.device(device)

    def same(t):
        return t.device.type == want.type and (want.index is None or t.device.index in (None, want.index))
    bad = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if not same(t)]
    for name, m in model.named_modules():
        for k, v in vars(m).items():
            if isinstance(v, torch.Tensor) and not same(v):
                bad.append(f"{name}.{k} (plain attribute)")
    if bad:
        raise RuntimeError(f"tensors not on {want}: {bad[:5]} ({len(bad)} total)")


def build_student(hw: HWConfig, path=None) -> nn.Module:
    m = build_fp32(path)
    replaced = convert_attention_surrogate(m, op_specs(hw))
    if len(replaced) != 10:
        raise RuntimeError(f"expected 10 attention modules, replaced {len(replaced)}")
    return m


def build_teacher(path=None) -> nn.Module:
    t = build_fp32(path)
    t.eval()
    for p in t.parameters():
        p.requires_grad_(False)
    return t


def bn_modules(model: nn.Module):
    return [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]


def set_train_mode(model: nn.Module, bn_frozen: bool = True) -> None:
    """model.train(), then BatchNorm back to eval (frozen running statistics)."""
    model.train()
    if bn_frozen:
        for m in bn_modules(model):
            m.eval()


def bn_state(model: nn.Module) -> dict:
    """Snapshot of every BatchNorm buffer (for the frozen-statistics check)."""
    return {k: v.detach().clone() for k, v in model.state_dict().items()
            if any(k.endswith(s) for s in ("running_mean", "running_var", "num_batches_tracked"))}


def student_state_for_simulator(model: nn.Module) -> dict:
    """state_dict loadable by run_tinyvit.build_model (attention_bias_idxs dropped)."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if not k.endswith("attention_bias_idxs")}


def set_cim_enabled(model: nn.Module, on: bool) -> None:
    for m in surrogate_modules(model):
        m.cim_enabled = on


# --------------------------------------------------------------------------- #
# model registry for evaluation (hash-checked; never substituted)
# --------------------------------------------------------------------------- #

REGISTRY = {
    "pretrained": (PRETRAINED, PRETRAINED_SHA256),
    "old_int8": (REPO_ROOT / "old_qat_checkpoints" / "best_model_int8_qat.pth",
                 "48f0e484d3083359266c5912b5831af57dbb07655d5aaba7958c58fd6cca9e90"),
    "old_int4": (REPO_ROOT / "old_qat_checkpoints" / "best_model_int4_qat.pth",
                 "a903663b2a8e2c7d6bd8c1fb80d6a974495191678b981f3043bd5dfab6a22b56"),
}


def resolve_model(name_or_path: str) -> tuple:
    """-> (label, path, sha256). Registry names are hash-checked."""
    if name_or_path in REGISTRY:
        p, want = REGISTRY[name_or_path]
        h = sha256_file(p)
        if h != want:
            raise RuntimeError(f"{p}: sha256 {h} != registry {want}; refusing to evaluate")
        return name_or_path, Path(p), h
    p = Path(name_or_path)
    if not p.is_file():
        raise FileNotFoundError(p)
    return p.parent.parent.name + "__" + p.stem, p, sha256_file(p)
