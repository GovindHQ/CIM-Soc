"""
checkpoints.py — explicit, hash-checked checkpoint registry.

Three model references, never substituted for one another:
    pretrained : original TinyViT-5M-224 (22k->1k distilled), FP32
    qat_int4   : QAT-INT4 checkpoint, loaded as trained
    qat_int8   : QAT-INT8 checkpoint, loaded as trained

Paths and expected SHA-256 hashes come from configs/checkpoints.json (copy
configs/checkpoints.template.json and edit). If an expected hash is set and the
file differs, loading is REFUSED. If a file is missing, loading is REFUSED —
except the pretrained checkpoint, which may be downloaded from its official
URL on request and is then hash-checked.

FINDINGS ON THE SUPPLIED QAT FILES (inspected Oct 2026)
------------------------------------------------------
Both are plain PyTorch archives (torch.save zip format; internal name
"best_model_only") holding an OrderedDict of 296 tensors: the same keys and
shapes as the pretrained TinyViT-5M (1000-class head), 269 float32 + 27 int64
(BatchNorm num_batches_tracked). No quantizer state (scales, zero points,
observers, LSQ step sizes) is stored, and the weights do NOT lie on any
symmetric int4/int8 grid: they are the FP32 latent ("shadow") weights of QAT.
Running them in FP32 is therefore "the QAT-trained network without its
training-time fake quantizers", and the CIM path re-quantizes them with the
simulator's own scheme (per-token activations, per-output-column weights,
max-abs symmetric). Whether that matches the training-time scheme is UNKNOWN
— it is the main caveat on any "QAT benefit" measured here.
"""

from __future__ import annotations

import json
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch

from .imagenette import sha256_file

MODEL_IDS = ("pretrained", "qat_int4", "qat_int8")

PRETRAINED_URL = ("https://github.com/wkcn/TinyViT-model-zoo/releases/download/"
                  "checkpoints/tiny_vit_5m_22kto1k_distill.pth")
# SHA-256 of the file served at PRETRAINED_URL (measured Oct 2026).
PRETRAINED_SHA256 = "e4894bab91be2d8f1b5e9a2147b57f35ae84be46e195d6586308864fb06fad3d"
# SHA-256 of the QAT files exactly as supplied in the Oct 2026 conversation.
SUPPLIED_QAT_SHA256 = {
    "qat_int4": "a903663b2a8e2c7d6bd8c1fb80d6a974495191678b981f3043bd5dfab6a22b56",
    "qat_int8": "9f8060d4a7464ea9c0873ec3140eacb86cbc090214dfd77628db9eec556dfa70",
}


@dataclass
class CheckpointSpec:
    model_id: str
    path: Path
    expected_sha256: str | None
    trained_weight_bits: int | None      # None for the pretrained model
    description: str


def load_registry(config_path: Path, repo_root: Path) -> dict[str, CheckpointSpec]:
    if not config_path.is_file():
        raise FileNotFoundError(
            f"{config_path} not found. Copy configs/checkpoints.template.json to "
            f"{config_path} and set the three checkpoint paths.")
    cfg = json.loads(config_path.read_text())
    reg = {}
    for mid in MODEL_IDS:
        if mid not in cfg:
            raise KeyError(f"{config_path}: missing entry {mid!r}")
        e = cfg[mid]
        p = Path(e["path"])
        if not p.is_absolute():
            p = repo_root / p
        reg[mid] = CheckpointSpec(model_id=mid, path=p,
                                  expected_sha256=(e.get("expected_sha256") or None),
                                  trained_weight_bits=e.get("trained_weight_bits"),
                                  description=e.get("description", ""))
    if reg["qat_int4"].trained_weight_bits != 4 or reg["qat_int8"].trained_weight_bits != 8:
        raise ValueError("qat_int4 / qat_int8 must declare trained_weight_bits 4 / 8")
    return reg


def ensure_present(spec: CheckpointSpec, allow_download: bool) -> None:
    if spec.path.is_file():
        return
    if spec.model_id == "pretrained" and allow_download:
        spec.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = spec.path.with_suffix(spec.path.suffix + ".part")
        print(f"downloading pretrained TinyViT-5M -> {spec.path}")
        with urllib.request.urlopen(PRETRAINED_URL) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.replace(spec.path)
        return
    raise FileNotFoundError(
        f"checkpoint {spec.model_id!r} not found at {spec.path}. Set its path in "
        f"configs/checkpoints.json. No other checkpoint is substituted.")


def verify_hash(spec: CheckpointSpec) -> str:
    h = sha256_file(spec.path)
    if spec.expected_sha256 and h.lower() != spec.expected_sha256.lower():
        raise ValueError(
            f"checkpoint {spec.model_id!r} at {spec.path}: SHA-256 {h} does not "
            f"match expected {spec.expected_sha256}. Refusing to load. If this "
            f"file is intentionally different, update expected_sha256.")
    return h


def inspect(spec: CheckpointSpec, reference_sd: dict | None = None) -> dict:
    """Format/architecture report for one checkpoint file (no model built)."""
    obj = torch.load(spec.path, map_location="cpu", weights_only=True)
    top_keys = list(obj.keys())[:5] if isinstance(obj, dict) else []
    sd = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    if not isinstance(sd, dict):
        raise TypeError(f"{spec.path}: expected a state dict, got {type(sd)}")
    dtypes: dict[str, int] = {}
    for v in sd.values():
        if torch.is_tensor(v):
            dtypes[str(v.dtype)] = dtypes.get(str(v.dtype), 0) + 1
    quantish = [k for k in sd if any(s in k.lower() for s in
                ("scale", "zero_point", "observer", "fake_quant", "quant", "lsq",
                 "step_size", "min_val", "max_val"))]
    rep = dict(model_id=spec.model_id, path=str(spec.path),
               file_bytes=spec.path.stat().st_size,
               container="dict['model']" if sd is not obj else "state_dict",
               top_level_keys=top_keys, n_tensors=len(sd), dtypes=dtypes,
               head_shape=list(sd["head.weight"].shape) if "head.weight" in sd else None,
               quantizer_keys=quantish[:20], n_quantizer_keys=len(quantish))
    if reference_sd is not None:
        a = {k for k in sd if not k.endswith("attention_bias_idxs")}
        b = {k for k in reference_sd if not k.endswith("attention_bias_idxs")}
        rep["same_keys_as_pretrained"] = a == b
        rep["shape_mismatches"] = [k for k in a & b if sd[k].shape != reference_sd[k].shape][:10]
        diffs = [((sd[k].double() - reference_sd[k].double()).norm()
                  / (reference_sd[k].double().norm() + 1e-30)).item()
                 for k in a & b if torch.is_floating_point(sd[k])]
        rep["median_rel_param_diff_vs_pretrained"] = float(torch.tensor(diffs).median()) if diffs else 0.0
        rep["n_identical_float_tensors_vs_pretrained"] = int(sum(d == 0 for d in diffs))
    # grid test on attention qkv weights: on a per-output-row symmetric grid?
    on_grid = {}
    for bits in (4, 8):
        res = []
        for k in [k for k in sd if k.endswith("attn.qkv.weight")]:
            w = sd[k].double()
            s = w.abs().amax(1, keepdim=True) / (2 ** (bits - 1) - 1)
            r = w / s
            res.append((r - r.round()).abs().mean().item())
        on_grid[f"int{bits}_mean_offgrid"] = float(sum(res) / len(res)) if res else None
    rep["qkv_grid_test"] = on_grid
    rep["qkv_grid_test_note"] = ("mean |w/s - round(w/s)|: ~0 => weights already on "
                                 "that grid; ~0.25 => not quantized (latent FP32)")
    return rep


def build(spec: CheckpointSpec):
    """TinyViT-5M with exactly this checkpoint's weights (strict key check)."""
    import run_tinyvit
    return run_tinyvit.build_model(spec.path)
