"""
imagenette.py — official FastAI Imagenette: download, safe extraction, class
mapping onto TinyViT's 1000 ImageNet-1k outputs, deterministic manifests, and
TinyViT evaluation preprocessing.

DATASET
-------
Official source (fast.ai): https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
Docs: https://docs.fast.ai/data.external.html , https://github.com/fastai/imagenette
Layout after extraction: <root>/imagenette2-160/{train,val}/<wnid>/*.JPEG
Only the `val` split is used for evaluation. Imagenette covers 10 ImageNet
classes — it is NOT the 1000-class ImageNet validation set.

The "-160" variant has its SHORTEST SIDE resized to 160 px. TinyViT-5M-224's
evaluation transform resizes the short side to 256 and centre-crops 224, so
these images are UPSAMPLED 1.6x. Absolute accuracy is therefore expected to be
lower than on full-resolution images; comparisons between models/paths on the
same images remain valid. `imagenette2-320` (or full size) avoids upsampling and
can be selected with --variant.

CLASS MAPPING (verified, not assumed)
-------------------------------------
Imagenette folder names are WordNet IDs. Nine of them are the ImageNet-1k
synsets of the same class. ONE IS NOT: the "parachute" folder is named
n03888605, which in ImageNet-1k is "parallel bars" (index 702); ImageNet's
parachute is n03888257 (index 701). Mapping folders by wnid alone would score
every parachute image as wrong. The table below therefore maps by CLASS, and
`verify_mapping_tables()` checks each target index against the bundled
ImageNet-1k synset list (cim_exp/data/imagenet_synsets.txt, from timm,
Apache-2.0). The mapping is additionally checked EMPIRICALLY on real images
(pretrained FP32 modal prediction per class) by run_experiment.py.

Alphabetical folder order is used ONLY for the 10-way Imagenette label index
(0..9), never as a 1000-class model index.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import sys
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

VARIANTS = {
    "imagenette2-160": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz",
    "imagenette2-320": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
    "imagenette2": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz",
}
DEFAULT_VARIANT = "imagenette2-160"
EVAL_SPLIT = "val"


@dataclass(frozen=True)
class ImagenetteClass:
    folder_wnid: str        # directory name in the Imagenette archive
    name: str               # human-readable class
    imagenet_wnid: str      # ImageNet-1k synset of that class
    imagenet_index: int     # TinyViT output index (sorted-synset order)


# 10-way label index = position in this tuple (= sorted folder order).
IMAGENETTE_CLASSES: tuple[ImagenetteClass, ...] = (
    ImagenetteClass("n01440764", "tench", "n01440764", 0),
    ImagenetteClass("n02102040", "English springer", "n02102040", 217),
    ImagenetteClass("n02979186", "cassette player", "n02979186", 482),
    ImagenetteClass("n03000684", "chain saw", "n03000684", 491),
    ImagenetteClass("n03028079", "church", "n03028079", 497),
    ImagenetteClass("n03394916", "French horn", "n03394916", 566),
    ImagenetteClass("n03417042", "garbage truck", "n03417042", 569),
    ImagenetteClass("n03425413", "gas pump", "n03425413", 571),
    ImagenetteClass("n03445777", "golf ball", "n03445777", 574),
    # Folder wnid != ImageNet wnid: n03888605 is "parallel bars" (702) in
    # ImageNet-1k. The images are parachutes -> ImageNet n03888257, index 701.
    ImagenetteClass("n03888257", "parachute", "n03888257", 701),
)
IMAGENET_INDICES = np.array([c.imagenet_index for c in IMAGENETTE_CLASSES])
FOLDER_TO_10WAY = {c.folder_wnid: i for i, c in enumerate(IMAGENETTE_CLASSES)}

# --------------------------------------------------------------------------- #
# TinyViT evaluation preprocessing (microsoft/Cream TinyViT data/build.py,
# TEST.CROP=True, DATA.INTERPOLATION='bicubic', IMG_SIZE=224):
#   Resize(int(256/224*224)=256, bicubic) -> CenterCrop(224) -> ToTensor ->
#   Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
# Implemented with PIL to match torchvision's PIL code path without requiring
# torchvision: short side -> 256 with long side int(256*long/short), PIL
# BICUBIC (antialiased), crop offsets int(round((dim-224)/2)).
# --------------------------------------------------------------------------- #
RESIZE_SHORT = 256
CROP = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
PREPROCESS_ID = (f"cream_tinyvit_eval_v1:resize_short={RESIZE_SHORT}:bicubic_pil:"
                 f"center_crop={CROP}:mean={MEAN}:std={STD}")


# --------------------------------------------------------------------------- #
# mapping verification
# --------------------------------------------------------------------------- #


def load_imagenet_synsets() -> list[str]:
    path = HERE / "data" / "imagenet_synsets.txt"
    syn = path.read_text(encoding="ascii").split()
    if len(syn) != 1000 or syn != sorted(syn):
        raise RuntimeError(f"{path}: expected 1000 sorted ImageNet-1k synsets")
    return syn


def verify_mapping_tables() -> list[str]:
    """Static check of IMAGENETTE_CLASSES against the ImageNet-1k synset list.
    Returns a list of human-readable findings; raises on an inconsistency."""
    syn = load_imagenet_synsets()
    notes = []
    for i, c in enumerate(IMAGENETTE_CLASSES):
        if syn[c.imagenet_index] != c.imagenet_wnid:
            raise RuntimeError(
                f"class {c.name}: index {c.imagenet_index} is {syn[c.imagenet_index]} "
                f"in ImageNet-1k, expected {c.imagenet_wnid}")
        if c.folder_wnid != c.imagenet_wnid:
            fi = syn.index(c.folder_wnid) if c.folder_wnid in syn else None
            notes.append(f"{c.name}: folder {c.folder_wnid} is ImageNet index {fi} "
                        f"(a different class); mapped to {c.imagenet_wnid} -> "
                        f"{c.imagenet_index} by class name")
        elif c.name == "parachute":
            notes.append(
                "parachute: original Imagenette folder n03888605 refers to "
                "ImageNet 'parallel bars' (702); actual downloaded folder "
                "n03888257 maps to parachute (701)"
            )
    folders = [c.folder_wnid for c in IMAGENETTE_CLASSES]
    if folders != sorted(folders):
        raise RuntimeError("IMAGENETTE_CLASSES must be in sorted folder order")
    return notes


# --------------------------------------------------------------------------- #
# download + extraction (Windows-compatible, path-safe)
# --------------------------------------------------------------------------- #


def sha256_file(path: Path, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download(variant: str, data_root: Path) -> Path:
    url = VARIANTS[variant]
    data_root.mkdir(parents=True, exist_ok=True)
    dst = data_root / f"{variant}.tgz"
    if dst.exists() and dst.stat().st_size > 0:
        print(f"archive already present: {dst} ({dst.stat().st_size:,} bytes)")
        return dst
    tmp = dst.with_suffix(".tgz.part")
    print(f"downloading {url}\n        -> {dst}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            done += len(b)
            if total:
                sys.stdout.write(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB")
                sys.stdout.flush()
    print()
    if total and tmp.stat().st_size != total:
        raise IOError(f"incomplete download: {tmp.stat().st_size} of {total} bytes")
    os.replace(tmp, dst)
    return dst


def extract(archive: Path, data_root: Path, variant: str) -> Path:
    """Extract with a path-traversal guard (works on Windows; no external tar)."""
    target = data_root / variant
    if (target / EVAL_SPLIT).is_dir():
        print(f"already extracted: {target}")
        return target
    root = data_root.resolve()
    print(f"extracting {archive} -> {data_root}")
    with tarfile.open(archive, "r:gz") as tf:
        members = tf.getmembers()
        for m in members:
            dest = (root / m.name).resolve()
            if root not in dest.parents and dest != root:
                raise RuntimeError(f"unsafe path in archive: {m.name}")
            if m.issym() or m.islnk():
                raise RuntimeError(f"links are not expected in the archive: {m.name}")
        try:                                   # Python >= 3.12: safe "data" filter
            tf.extractall(root, members=members, filter="data")
        except TypeError:
            tf.extractall(root, members=members)
    if not (target / EVAL_SPLIT).is_dir():
        raise RuntimeError(f"extraction did not produce {target / EVAL_SPLIT}")
    return target


def write_provenance(data_root: Path, variant: str, archive: Path) -> Path:
    import json
    import time
    p = data_root / f"{variant}.provenance.json"
    info = dict(variant=variant, url=VARIANTS[variant], archive=str(archive),
                archive_bytes=archive.stat().st_size,
                archive_sha256=sha256_file(archive),
                recorded_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                note="fast.ai publishes no checksum; this hash records what was used")
    p.write_text(json.dumps(info, indent=2))
    return p


# --------------------------------------------------------------------------- #
# split listing + manifests
# --------------------------------------------------------------------------- #


def list_split(dataset_dir: Path, split: str = EVAL_SPLIT) -> list[tuple[str, str]]:
    """[(relative posix path, folder_wnid)], sorted, with folder validation."""
    sdir = dataset_dir / split
    if not sdir.is_dir():
        raise FileNotFoundError(
            f"{sdir} not found. Run:  python run_experiment.py download-data "
            f"(no synthetic fallback is used).")
    found = sorted(d.name for d in sdir.iterdir() if d.is_dir())
    expected = [c.folder_wnid for c in IMAGENETTE_CLASSES]
    if found != expected:
        raise RuntimeError(f"unexpected class folders in {sdir}:\n  found    {found}\n"
                           f"  expected {expected}")
    items = []
    for w in expected:
        files = sorted(p.name for p in (sdir / w).iterdir()
                       if p.is_file() and p.suffix.lower() in (".jpeg", ".jpg", ".png"))
        items += [(f"{split}/{w}/{f}", w) for f in files]
    return items


MANIFEST_FIELDS = ["idx", "rel_path", "folder_wnid", "class_name",
                   "label_10way", "label_1000way", "file_sha256"]


def build_manifest(dataset_dir: Path, stage: str, per_class: int = 10,
                   seed: int = 0, split: str = EVAL_SPLIT) -> list[dict]:
    """stage 'smoke': `per_class` images per class, chosen by a seeded RNG over
    the sorted file list (deterministic, not biased to filename order).
    stage 'full': the whole split, sorted. `split` is 'val' for evaluation and
    'train' ONLY for the Experiment B calibration set (never evaluated on)."""
    items = list_split(dataset_dir, split)
    if stage == "smoke":
        rng = np.random.default_rng(seed)
        chosen = []
        for c in IMAGENETTE_CLASSES:
            cls_items = [it for it in items if it[1] == c.folder_wnid]
            if len(cls_items) < per_class:
                raise RuntimeError(f"class {c.name} has only {len(cls_items)} images")
            pick = sorted(rng.choice(len(cls_items), size=per_class, replace=False))
            chosen += [cls_items[i] for i in pick]
        items = chosen
    elif stage != "full":
        raise ValueError(f"unknown stage {stage!r}")
    rows = []
    for i, (rel, w) in enumerate(items):
        c = IMAGENETTE_CLASSES[FOLDER_TO_10WAY[w]]
        rows.append(dict(idx=i, rel_path=rel, folder_wnid=w, class_name=c.name,
                         label_10way=FOLDER_TO_10WAY[w],
                         label_1000way=c.imagenet_index,
                         file_sha256=sha256_file(dataset_dir / rel)))
    return rows


def filename_provenance(rows: list[dict]) -> dict:
    """Count manifest files by ImageNet naming convention. ImageNet-1k training
    images are named <wnid>_<n>.JPEG, validation images ILSVRC2012_val_<n>.JPEG.
    Reported, not assumed: Imagenette2's 70/30 re-split means most of its val
    images are expected to be ImageNet-1k TRAINING images."""
    import re
    c = {"imagenet_train_style (<wnid>_<n>)": 0,
         "imagenet_val_style (ILSVRC2012_val_*)": 0, "other": 0}
    for r in rows:
        name = r["rel_path"].rsplit("/", 1)[-1]
        if name.startswith("ILSVRC2012_val_"):
            c["imagenet_val_style (ILSVRC2012_val_*)"] += 1
        elif re.match(r"^n\d{8}_\d+\.", name):
            c["imagenet_train_style (<wnid>_<n>)"] += 1
        else:
            c["other"] += 1
    return c


def write_manifest(rows: list[dict], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    data = buf.getvalue().encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def read_manifest(path: Path) -> tuple[list[dict], str]:
    data = path.read_bytes()
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
    for r in rows:
        for k in ("idx", "label_10way", "label_1000way"):
            r[k] = int(r[k])
    return rows, hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- #
# preprocessing
# --------------------------------------------------------------------------- #


def load_uint8(path: Path) -> np.ndarray:
    """File -> [224, 224, 3] uint8 after resize + centre crop (pre-normalise)."""
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        if w <= h:
            nw, nh = RESIZE_SHORT, int(RESIZE_SHORT * h / w)
        else:
            nw, nh = int(RESIZE_SHORT * w / h), RESIZE_SHORT
        im = im.resize((nw, nh), Image.BICUBIC)
        top = int(round((nh - CROP) / 2.0))
        left = int(round((nw - CROP) / 2.0))
        im = im.crop((left, top, left + CROP, top + CROP))
        return np.asarray(im, dtype=np.uint8).copy()


def to_model_input(u8: np.ndarray):
    """[n, 224, 224, 3] uint8 -> normalised float32 torch tensor [n, 3, 224, 224]
    (ToTensor + Normalize; exact and deterministic)."""
    import torch
    x = torch.from_numpy(np.array(u8, dtype=np.uint8, copy=True)).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor(MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(STD, dtype=torch.float32).view(1, 3, 1, 1)
    return (x - mean) / std


def build_input_cache(dataset_dir: Path, rows: list[dict], path: Path) -> None:
    """Decode every manifest image once into a uint8 .npy (memory-mappable)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".part.npy")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8,
                                    shape=(len(rows), CROP, CROP, 3))
    for i, r in enumerate(rows):
        arr[i] = load_uint8(dataset_dir / r["rel_path"])
        if (i + 1) % 250 == 0 or i + 1 == len(rows):
            sys.stdout.write(f"\r  preprocessed {i + 1}/{len(rows)}")
            sys.stdout.flush()
    print()
    arr.flush()
    del arr
    os.replace(tmp, path)
