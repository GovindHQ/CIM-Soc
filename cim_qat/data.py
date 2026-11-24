"""
cim_qat/data.py — dataset discovery, the fixed QAT splits, integrity checks,
the TEST (ImageNetV2) access guard, and the datasets used for training and
evaluation.

SPLITS (protocol v2, 25 Oct 2026; fixed by seed before any calibration/training)
-------------------------------------------------------------------------------
  CAL        2,000 ImageNet-1k TRAIN images, 2 per class          gain calibration only
  CALCHECK     200 ImageNet-1k TRAIN images, 1 in each of 200     calibration check only
               seeded classes, disjoint from CAL
  TRAIN      ImageNet-1k train minus CAL minus CALCHECK            gradient updates
             (minus any train file whose CONTENT duplicates a CAL/CALCHECK file,
             when --hash-train is used; every such file is listed)
  VAL        the FULL official ImageNet-1k val set (50,000)       per-epoch monitoring,
                                                                   checkpoint selection,
                                                                   early stopping
  TEST       ImageNetV2 matched-frequency (10,000, 10 per class)  final evaluation ONLY

Class index = position in cim_exp/data/imagenet_synsets.txt (1000 sorted WordNet
ids) = the order of TinyViT's 1000-way head = the ImageNet-2012 index order.
ImageNetV2's folders are named by that index ("0" .. "999"), so the TEST label
is int(folder name). NOT torchvision ImageFolder: it sorts folder names as
strings ("0","1","10","100",...) and would assign wrong labels to 988 of the
1000 classes. tests_qat/test_cim_qat_data.py pins this.

TEST GUARD: Splits(..., allow_test=False) refuses to return TEST rows. Only
qat_evaluate.py with --final (purpose "final_eval") and the label-integrity
check (purpose "v2_integrity") open it. Every split access by every process is
appended to <work_root>/access_log.jsonl; qat_audit.py checks that log, the
configs and the run logs (QAT_DESIGN.md §8).

PREPROCESSING
-------------
* Evaluation (all models, all splits): cim_exp.imagenette.load_uint8 +
  to_model_input, imported and NOT re-implemented: resize short side to 256
  with PIL BICUBIC, centre crop 224, ImageNet mean/std. `preprocess_id()` names
  it; the simulator evaluation records it and a SHA-256 of every model input
  tensor, and the report refuses to compare runs whose inputs differ.
  `interpolation="bilinear"` exists only for the FP32 sensitivity check.
* Training: torchvision RandomResizedCrop(224, scale (0.08, 1), BICUBIC) +
  RandomHorizontalFlip + ToTensor + Normalize, seeded per (run seed, epoch,
  sample index).
"""

from __future__ import annotations

import csv
import json
import os
import socket
import sys
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .common import derive_seed, now_iso, read_json, sha256_file, write_json

IMG_EXTS = (".jpeg", ".jpg", ".png")
SPLIT_SCHEMA = "cim_qat.splits.v2"
DEFAULT_SPLIT_SEED = 20261022
CAL_PER_CLASS = 2
CALCHECK_N = 200
SPLITS = ("cal", "calcheck", "train", "val", "test")
MANIFEST_FIELDS = ["idx", "split", "root", "rel_path", "wnid", "label"]
EXPECTED = dict(cal=2000, calcheck=200, val=50000, test=10000)
V2_DIRNAME = "imagenetv2-matched-frequency-format-val"
V2_PER_CLASS = 10
# purposes allowed to open TEST (everything else is refused by construction)
TEST_PURPOSES = ("final_eval", "v2_integrity", "audit", "prepare", "unit_test")

# ---- ImageNetV2 extension variants (v2.8; QAT_DESIGN.md §0.4 D3) -------------------
# Registered in splits/extra_tests.json (splits.json is never rewritten). Guarded like TEST.
EXT_TESTS = {"test_th07": "imagenetv2-threshold0.7-format-val",
             "test_top": "imagenetv2-top-images-format-val"}
EXT_REGISTRY = "extra_tests.json"
EXT_RULE = ("declared 29 Oct 2026, before any extension image was read: an image of a new ImageNetV2 "
            "variant whose file content equals a CAL, CALCHECK, VAL or TRAIN image is removed from that "
            "variant (counted and listed in excluded_<split>.csv). Duplicates of matched-frequency TEST "
            "images, and between variants, stay in each variant; the pooled report counts every distinct "
            "image once (first occurrence in the order matched-frequency, threshold-0.7, top-images).")
POOL_ORDER = ("test", "test_th07", "test_top")
FREEZE_GLOB = "FINAL_SHA256SUMS_*.txt"
ACCESS_LOG, ACCESS_LOG_EXT = "access_log.jsonl", "access_log_ext.jsonl"


def is_test_split(split: str) -> bool:
    return split == "test" or split in EXT_TESTS


class FrozenFileError(PermissionError):
    pass


def frozen_files(work_root) -> dict:
    """{path relative to work_root: sha256} from every FINAL_SHA256SUMS_*.txt in work_root."""
    out = {}
    for f in sorted(Path(work_root).glob(FREEZE_GLOB)):
        for line in f.read_text().splitlines():
            parts = line.rstrip("\n").split(None, 1)
            if len(parts) != 2:
                continue
            h, rel = parts[0], parts[1].lstrip("*")
            out[rel[2:] if rel.startswith("./") else rel] = h
    return out


def current_access_log(work_root) -> str:
    """A frozen access log (listed in a FINAL_SHA256SUMS file) is never appended to again:
    new entries go to the first unfrozen of access_log.jsonl, access_log_ext.jsonl,
    access_log_ext2.jsonl, ... (the audit reads every access_log*.jsonl)."""
    fz = frozen_files(work_root)
    for name in (ACCESS_LOG, ACCESS_LOG_EXT, *(f"access_log_ext{k}.jsonl" for k in range(2, 100))):
        if name not in fz:
            return name
    raise RuntimeError("no unfrozen access log name left")


def read_access_logs(work_root) -> list:
    """Every entry of access_log.jsonl then access_log_ext*.jsonl, in that order."""
    names = [ACCESS_LOG] + sorted((p.name for p in Path(work_root).glob("access_log_ext*.jsonl")),
                                  key=lambda n: (len(n), n))
    out = []
    for n in names:
        p = Path(work_root) / n
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    try:
                        out.append(dict(json.loads(line), _log=n))
                    except json.JSONDecodeError:
                        out.append(dict(_bad_line=line[:80], _log=n))
    return out


def assert_not_frozen(work_root, *paths) -> None:
    """Refuse to (over)write a file listed in a checksum freeze file."""
    fz = frozen_files(work_root)
    for p in paths:
        rel = os.path.relpath(os.path.abspath(p), os.path.abspath(work_root))
        if rel in fz:
            raise FrozenFileError(f"{p} is frozen (listed in {FREEZE_GLOB}); refusing to overwrite it")


def synsets() -> list[str]:
    from cim_exp.imagenette import load_imagenet_synsets
    s = load_imagenet_synsets()
    if len(s) != 1000 or len(set(s)) != 1000 or s != sorted(s):
        raise RuntimeError("cim_exp/data/imagenet_synsets.txt is not 1000 unique sorted wnids")
    return s


# --------------------------------------------------------------------------- #
# paths configuration
# --------------------------------------------------------------------------- #


@dataclass
class DataPaths:
    train_roots: list            # one or more class-folder roots (<root>/<wnid>/*.JPEG)
    val_root: str                # flat (with LOC_val_solution.csv) or class folders
    val_labels_csv: str | None   # Kaggle LOC_val_solution.csv for a flat val dir
    imagenetv2_root: str | None  # .../imagenetv2-matched-frequency-format-val (None until downloaded)
    work_root: str               # splits, calibration, runs, eval, access log
    imagenetv2_extra_roots: dict | None = None   # {test_th07|test_top: folder} (v2.8 extension)

    @classmethod
    def load(cls, path, require_v2: bool = False) -> "DataPaths":
        d = read_json(path)
        roots = d.get("imagenet_train_roots") or [d["imagenet_train_root"]]
        dp = cls(train_roots=[str(r) for r in roots], val_root=str(d["imagenet_val_root"]),
                 val_labels_csv=d.get("imagenet_val_labels_csv"),
                 imagenetv2_root=d.get("imagenetv2_root"),
                 work_root=str(d["work_root"]),
                 imagenetv2_extra_roots=dict(d.get("imagenetv2_extra_roots") or {}))
        need = dp.train_roots + [dp.val_root]
        if dp.imagenetv2_root or require_v2:
            need.append(str(dp.imagenetv2_root))
        bad = sorted(set(dp.imagenetv2_extra_roots) - set(EXT_TESTS))
        if bad:
            raise ValueError(f"{path}: imagenetv2_extra_roots keys must be {sorted(EXT_TESTS)}, got {bad}")
        need += [str(v) for v in dp.imagenetv2_extra_roots.values()]
        for r in need:
            if "REPLACE" in r or not Path(r).is_dir():
                raise FileNotFoundError(f"{path}: directory not found: {r!r} "
                                        "(edit configs/paths.json, QAT_RUNBOOK.md stage 5)")
        if dp.val_labels_csv and not Path(dp.val_labels_csv).is_file():
            raise FileNotFoundError(f"{path}: labels CSV not found: {dp.val_labels_csv!r}")
        return dp

    @property
    def splits_dir(self) -> Path:
        return Path(self.work_root) / "splits"


# --------------------------------------------------------------------------- #
# listing (symlink-aware: DirEntry.is_file()/is_dir() follow symlinks; the real
# target of every file is recorded for the path-level overlap check)
# --------------------------------------------------------------------------- #


def _scan_files(d: Path, prefix: str) -> tuple[list, list, int]:
    """-> ([(rel_path, real_path)], [non-image names], n_symlinks) in sorted order."""
    files, other, nsym = [], [], 0
    real_d = os.path.realpath(d)
    with os.scandir(d) as it:
        ents = sorted(it, key=lambda e: e.name)
    for e in ents:
        if not e.is_file():                      # follows symlinks; broken links -> False
            if e.is_symlink():
                other.append(e.name + " (broken symlink)")
            continue
        if not e.name.lower().endswith(IMG_EXTS):
            other.append(e.name)
            continue
        if e.is_symlink():
            nsym += 1
            real = os.path.realpath(e.path)
        else:
            real = os.path.join(real_d, e.name)
        files.append((f"{prefix}{e.name}", real))
    return files, other, nsym


def _subdirs(root: Path) -> list:
    with os.scandir(root) as it:
        return sorted(e.name for e in it if e.is_dir())


def list_train(train_roots) -> tuple[dict, dict]:
    """{wnid: [(root_index, rel_path, real_path)]} over the union of roots, + diagnostics."""
    syn = synsets()
    wn = set(syn)
    per = {w: [] for w in syn}
    diag = dict(roots=[], symlinks=0, non_image_files=0, extra_folders=[])
    for ri, r in enumerate(train_roots):
        dirs = _subdirs(Path(r))
        extra = [d for d in dirs if d not in wn]
        diag["extra_folders"] += [f"train{ri}/{d}" for d in extra]
        n = 0
        for d in dirs:
            if d not in wn:
                continue
            files, other, nsym = _scan_files(Path(r) / d, f"{d}/")
            per[d].extend((ri, rel, real) for rel, real in files)
            diag["symlinks"] += nsym
            diag["non_image_files"] += len(other)
            n += len(files)
        diag["roots"].append(dict(root=r, class_folders=len(dirs) - len(extra), files=n))
    missing = [w for w in syn if not per[w]]
    if missing:
        raise RuntimeError(f"ImageNet train: {len(missing)} of 1000 classes have no images "
                           f"(e.g. {missing[:5]}). Is this the ImageNet-1k train split?")
    diag["files"] = sum(len(v) for v in per.values())
    diag["per_class_min"] = min(len(v) for v in per.values())
    diag["per_class_max"] = max(len(v) for v in per.values())
    return per, diag


def list_val(val_root, val_labels_csv=None) -> tuple[dict, dict]:
    """{wnid: [(rel_path, real_path)]} for flat+CSV or class-folder layouts, + diagnostics.
    For a flat folder every image must appear in the CSV exactly once and vice versa."""
    syn = synsets()
    wn = set(syn)
    root = Path(val_root)
    diag = dict(root=str(root))
    if val_labels_csv:
        files, other, nsym = _scan_files(root, "")
        by_stem = {}
        for rel, real in files:
            by_stem.setdefault(Path(rel).stem, []).append((rel, real))
        dup_stems = [s for s, v in by_stem.items() if len(v) > 1]
        if dup_stems:
            raise RuntimeError(f"{root}: several files share an ImageId, e.g. {dup_stems[:3]}")
        per = {w: [] for w in syn}
        seen = set()
        n_rows = 0
        with open(val_labels_csv, newline="") as f:
            rd = csv.DictReader(f)
            if rd.fieldnames is None or "ImageId" not in rd.fieldnames or "PredictionString" not in rd.fieldnames:
                raise RuntimeError(f"{val_labels_csv}: expected columns ImageId,PredictionString")
            for row in rd:
                n_rows += 1
                img = row["ImageId"].strip()
                toks = row["PredictionString"].split()
                wnid = toks[0]
                # LOC_val_solution lists one or more boxes, all of the same class
                ids = set(toks[0::5])
                if wnid not in wn or len(ids) != 1:
                    raise RuntimeError(f"{val_labels_csv}: bad label {row['PredictionString'][:40]!r} for {img}")
                if img in seen:
                    raise RuntimeError(f"{val_labels_csv}: ImageId {img} listed twice")
                seen.add(img)
                if img not in by_stem:
                    raise FileNotFoundError(f"{root}: no image file for CSV ImageId {img}")
                per[wnid].append(by_stem[img][0])
        unlabelled = sorted(set(by_stem) - seen)
        if unlabelled:
            raise RuntimeError(f"{root}: {len(unlabelled)} images have no CSV label, e.g. {unlabelled[:3]}")
        diag.update(layout="flat+csv", csv=str(val_labels_csv), csv_rows=n_rows, files=len(files),
                    symlinks=nsym, non_image_files=len(other))
    else:
        dirs = _subdirs(root)
        if len([d for d in dirs if d in wn]) != 1000:
            raise RuntimeError(
                f"{root}: expected 1000 class sub-folders (found {len([d for d in dirs if d in wn])}). "
                "For a FLAT val folder set imagenet_val_labels_csv to Kaggle's LOC_val_solution.csv.")
        per = {w: [] for w in syn}
        nsym = nother = 0
        for d in dirs:
            if d in wn:
                files, other, ns = _scan_files(root / d, f"{d}/")
                per[d] = files
                nsym += ns
                nother += len(other)
        diag.update(layout="class_folders", files=sum(len(v) for v in per.values()),
                    symlinks=nsym, non_image_files=nother)
    for w in per:
        per[w].sort()
    diag["per_class_min"] = min(len(v) for v in per.values())
    diag["per_class_max"] = max(len(v) for v in per.values())
    return per, diag


def find_v2_root(path) -> Path | None:
    """The folder holding the 1000 numeric class folders, given it or a parent."""
    p = Path(path)
    for cand in (p, p / V2_DIRNAME):
        if cand.is_dir():
            names = set(_subdirs(cand))
            if {"0", "999"} <= names:
                return cand
    return None


def list_v2(v2_root) -> tuple[dict, dict]:
    """{label: [(rel_path, real_path)]} for ImageNetV2 (folder name = class index), + diagnostics.
    Structural requirements: exactly the folders "0".."999" (no others, no zero
    padding), images only. Images per class are reported (expected 10)."""
    root = Path(v2_root)
    dirs = _subdirs(root)
    want = [str(i) for i in range(1000)]
    extra = sorted(set(dirs) - set(want))
    missing = sorted(set(want) - set(dirs), key=int)
    if extra or missing:
        raise RuntimeError(f"{root}: not an ImageNetV2 folder: missing class folders {missing[:5]} "
                           f"({len(missing)}), unexpected folders {extra[:5]} ({len(extra)}). "
                           f"Point imagenetv2_root at the '{V2_DIRNAME}' folder.")
    per, nsym, nother = {}, 0, 0
    for i in range(1000):
        files, other, ns = _scan_files(root / str(i), f"{i}/")
        per[i] = files
        nsym += ns
        nother += len(other)
    counts = [len(per[i]) for i in range(1000)]
    diag = dict(root=str(root), files=sum(counts), per_class_min=min(counts), per_class_max=max(counts),
                classes_not_10=[i for i in range(1000) if counts[i] != V2_PER_CLASS][:20],
                symlinks=nsym, non_image_files=nother)
    return per, diag


def imagefolder_lexicographic_labels() -> dict:
    """What torchvision.datasets.ImageFolder would assign to the V2 folders
    (sorted as strings). Used only to TEST that we do not do this."""
    names = sorted(str(i) for i in range(1000))
    return {int(n): k for k, n in enumerate(names)}


def v2_mapping_report(test_rows, val_rows) -> dict:
    """Automated class-index checks for TEST (ImageNetV2) vs the ImageNet-1k index."""
    syn = synsets()
    ok_label = all(r["label"] == int(Path(r["rel_path"]).parts[0]) for r in test_rows)
    ok_wnid = all(r["wnid"] == syn[r["label"]] for r in test_rows)
    ok_val = all(r["wnid"] == syn[r["label"]] for r in val_rows)
    test_classes = sorted({r["label"] for r in test_rows})
    val_classes = sorted({r["label"] for r in val_rows})
    lex = imagefolder_lexicographic_labels()
    lex_fixed = sum(1 for k, v in lex.items() if k == v)
    return dict(label_is_int_folder=ok_label, wnid_matches_synset_index=ok_wnid,
                val_wnid_matches_synset_index=ok_val,
                synsets_sorted_unique_1000=True,              # synsets() raises otherwise
                same_1000_classes_as_val=(test_classes == val_classes == list(range(1000))),
                imagefolder_trap_classes_that_would_be_mislabelled=1000 - lex_fixed,
                ok=bool(ok_label and ok_wnid and ok_val and test_classes == val_classes == list(range(1000))))


# --------------------------------------------------------------------------- #
# split construction
# --------------------------------------------------------------------------- #


def list_all(dp: DataPaths) -> dict:
    tr, tr_diag = list_train(dp.train_roots)
    va, va_diag = list_val(dp.val_root, dp.val_labels_csv)
    v2, v2_diag = (list_v2(dp.imagenetv2_root) if dp.imagenetv2_root else (None, None))
    return dict(tr=tr, tr_diag=tr_diag, va=va, va_diag=va_diag, v2=v2, v2_diag=v2_diag)


def build_splits(dp: DataPaths, seed: int = DEFAULT_SPLIT_SEED, eligible=None,
                 listing: dict | None = None) -> tuple[dict, dict, dict]:
    """Deterministic, class-stratified splits. -> (rows per split, info, real paths per split).

    Draw (unchanged since v1): one seeded choice of the 200 CALCHECK classes, then one
    seeded permutation of each class's sorted train files. Walking that permutation,
    the first CAL_PER_CLASS ELIGIBLE files become CAL, the next eligible one becomes
    CALCHECK (in CALCHECK classes); every other file -- including ineligible ones --
    is TRAIN. `eligible(root_index, rel_path, real_path) -> (ok, reason)` is the
    held-out eligibility rule (prepare(): the file's content must not equal any VAL
    or TEST image or an already selected CAL/CALCHECK image). The random draws do not
    depend on eligibility, so a class without ineligible candidates gets exactly the
    same CAL/CALCHECK files as with no rule at all.
    """
    syn = synsets()
    label = {w: i for i, w in enumerate(syn)}
    L = listing or list_all(dp)
    tr, tr_diag, va, va_diag, v2, v2_diag = (L["tr"], L["tr_diag"], L["va"], L["va_diag"],
                                             L["v2"], L["v2_diag"])
    skipped = []

    rng_t = np.random.default_rng(derive_seed("cim_qat.splits.train", seed))
    check_classes = set(int(c) for c in rng_t.choice(1000, CALCHECK_N, replace=False))

    rows = {s: [] for s in SPLITS}
    for ci, w in enumerate(syn):
        files = tr[w]
        perm = rng_t.permutation(len(files))
        need = CAL_PER_CLASS + (1 if ci in check_classes else 0)
        if len(files) < need + 1:
            raise RuntimeError(f"class {w}: only {len(files)} train images")
        n_cal = n_chk = 0
        for k, j in enumerate(perm):
            ri, rel, real = files[j]
            want = "cal" if n_cal < CAL_PER_CLASS else (
                "calcheck" if (ci in check_classes and n_chk < 1) else None)
            s = "train"
            if want is not None:
                ok, why = eligible(ri, rel, real) if eligible else (True, "")
                if ok:
                    s = want
                    n_cal += want == "cal"
                    n_chk += want == "calcheck"
                else:
                    skipped.append(dict(class_index=ci, wnid=w, root=f"train{ri}", rel_path=rel,
                                        would_have_been=want, permutation_position=k, reason=why))
            rows[s].append(dict(split=s, root=f"train{ri}", rel_path=rel, wnid=w, label=label[w], _real=real))
        if n_cal < CAL_PER_CLASS or (ci in check_classes and n_chk < 1):
            raise RuntimeError(f"class {w}: not enough eligible train images for CAL/CALCHECK")
        for rel, real in va[w]:
            rows["val"].append(dict(split="val", root="val", rel_path=rel, wnid=w, label=label[w], _real=real))
    if v2 is not None:
        for i in range(1000):
            for rel, real in v2[i]:
                rows["test"].append(dict(split="test", root="v2", rel_path=rel, wnid=syn[i], label=i, _real=real))
    for s in SPLITS:          # canonical order inside each manifest
        rows[s].sort(key=lambda r: (r["label"], r["root"], r["rel_path"]))
        for i, r in enumerate(rows[s]):
            r["idx"] = i
    real = {s: [r.pop("_real") for r in rows[s]] for s in SPLITS}
    info = dict(schema=SPLIT_SCHEMA, seed=seed, created_utc=now_iso(),
                train_roots=list(dp.train_roots), val_root=dp.val_root,
                val_labels_csv=dp.val_labels_csv, imagenetv2_root=dp.imagenetv2_root,
                counts={s: len(rows[s]) for s in SPLITS},
                listing=dict(train=tr_diag, val=va_diag, test=v2_diag),
                calcheck_classes=sorted(check_classes),
                heldout_eligibility_rule=("CAL/CALCHECK candidates whose content equals a VAL or TEST "
                                          "image (or an already selected CAL/CALCHECK image) are skipped "
                                          "and stay in TRAIN" if eligible else "none"),
                heldout_skipped=skipped,
                test_available=v2 is not None)
    return rows, info, real


def write_manifest(rows: list, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r[k] for k in MANIFEST_FIELDS})
    os.replace(tmp, path)
    return sha256_file(path)


def read_manifest(path) -> list:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["idx"] = int(r["idx"])
        r["label"] = int(r["label"])
    return rows


def resolve(dp_or_info, row) -> Path:
    """Absolute path of a manifest row."""
    if isinstance(dp_or_info, DataPaths):
        tr, vr, v2 = dp_or_info.train_roots, dp_or_info.val_root, dp_or_info.imagenetv2_root
        v2x = dp_or_info.imagenetv2_extra_roots or {}
    else:
        tr, vr, v2 = dp_or_info["train_roots"], dp_or_info["val_root"], dp_or_info.get("imagenetv2_root")
        v2x = dp_or_info.get("imagenetv2_extra_roots") or {}
    if row["root"] == "val":
        return Path(vr) / row["rel_path"]
    if row["root"] == "v2":
        return Path(v2) / row["rel_path"]
    if row["root"].startswith("v2x:"):
        return Path(v2x[row["root"][4:]]) / row["rel_path"]
    return Path(tr[int(row["root"][5:])]) / row["rel_path"]


class TestAccessError(PermissionError):
    pass


def log_access(work_root, split: str, purpose: str, manifest_sha: str | None, extra=None):
    """Append one line to <work_root>/access_log.jsonl (never raises)."""
    try:
        rec = dict(time_utc=now_iso(), split=split, purpose=purpose, manifest_sha256=manifest_sha,
                   pid=os.getpid(), host=socket.gethostname(), argv=sys.argv,
                   run=os.environ.get("CIM_QAT_RUN", ""), **(extra or {}))
        p = Path(work_root) / current_access_log(work_root)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError as e:          # logging must not break a run, but say so
        print(f"WARNING: could not write access log: {e}", file=sys.stderr)


class Splits:
    """Read-only access to a prepared split directory.

    Verifies manifest hashes, refuses TEST unless allow_test=True with a TEST
    purpose, and logs every split access with its purpose.
    """

    def __init__(self, splits_dir, purpose: str = "unspecified", allow_test: bool = False):
        self.dir = Path(splits_dir)
        self.info = read_json(self.dir / "splits.json")
        if self.info.get("schema") != SPLIT_SCHEMA:
            raise RuntimeError(f"{self.dir}: not a {SPLIT_SCHEMA} directory (re-run qat_prepare_data.py)")
        if allow_test and purpose not in TEST_PURPOSES:
            raise TestAccessError(f"purpose {purpose!r} may not open TEST")
        self.purpose, self.allow_test = purpose, allow_test
        self._rows = {}
        self.work_root = self.dir.parent
        reg = self.dir / EXT_REGISTRY
        self.ext = read_json(reg) if reg.exists() else {}
        if self.ext:   # in memory only: splits.json itself is never rewritten
            self.info = dict(self.info, imagenetv2_extra_roots={k: v["root"] for k, v in self.ext.items()})

    def manifest_path(self, split):
        return self.dir / f"manifest_{split}.csv"

    def rows(self, split) -> list:
        if is_test_split(split) and not self.allow_test:
            raise TestAccessError("TEST (ImageNetV2) is closed: only the final evaluation "
                                  "(qat_evaluate.py ... --final) may read it (QAT_DESIGN.md §8)")
        if split not in self._rows:
            want = self.sha(split) if (split in self.info["manifest_sha256"] or split in self.ext) else None
            if want is None:
                raise RuntimeError(f"split {split!r} was not prepared (TEST needs imagenetv2_root; "
                                   "see qat_prepare_data.py add-test / add-test-variant)")
            p = self.manifest_path(split)
            h = sha256_file(p)
            if h != want:
                raise RuntimeError(f"{p} changed since prepare (sha256 {h[:12]} != {want[:12]}); refusing.")
            self._rows[split] = read_manifest(p)
            log_access(self.work_root, split, self.purpose, h)
        return self._rows[split]

    def path(self, row) -> Path:
        return resolve(self.info, row)

    def sha(self, split) -> str:
        if split in self.ext:
            return self.ext[split]["manifest_sha256"]
        return self.info["manifest_sha256"][split]


# --------------------------------------------------------------------------- #
# content hashing and disjointness
# --------------------------------------------------------------------------- #


def _hash_one(p):
    try:
        return sha256_file(p)
    except OSError as e:
        return f"ERROR:{e}"


def hash_paths(paths, workers: int = 8, chunksize: int = 64, progress: str = "") -> list:
    paths = [str(p) for p in paths]
    if workers <= 1:
        return [_hash_one(p) for p in paths]
    out = []
    with Pool(workers) as pool:
        for k, h in enumerate(pool.imap(_hash_one, paths, chunksize=chunksize)):
            out.append(h)
            if progress and (k + 1) % 50000 == 0:
                print(f"    {progress}: {k + 1}/{len(paths)} hashed", flush=True)
    return out


def load_hash_caches(d: Path) -> dict:
    """All hashcache_*.csv in a folder -> {root|rel_path: sha256} (errors dropped)."""
    out = {}
    for cache in sorted(Path(d).glob("hashcache_*.csv")):
        with open(cache) as f:
            for line in f:
                k, _, h = line.rstrip("\n").rpartition(",")
                if k and h and not h.startswith("ERROR:"):
                    out[k] = h
    return out


def verify_cache_sample(known: dict, dp, n: int = 200, seed: int = 0) -> dict:
    """Re-hash a seeded random sample of cached entries. The caches assume the dataset
    files never change; a mismatch means a stale cache (delete hashcache_*.csv)."""
    keys = sorted(known)
    if not keys:
        return dict(checked=0, mismatches=[])
    pick = np.random.default_rng(seed).choice(len(keys), min(n, len(keys)), replace=False)
    bad = []
    for i in pick:
        root, _, rel = keys[i].partition("|")
        h = _hash_one(resolve(dp, dict(root=root, rel_path=rel)))
        if h != known[keys[i]]:
            bad.append(keys[i])
    return dict(checked=len(pick), mismatches=bad)


def hash_rows_cached(rows: list, dp, cache: Path, workers: int, progress: str, known: dict | None = None) -> list:
    """SHA-256 of every row's file; resumable through a CSV cache keyed by root|rel_path.
    `known` (shared dict of already known hashes) is consulted and updated."""
    done = known if known is not None else {}
    if cache.exists():
        with open(cache) as f:
            for line in f:
                k, _, h = line.rstrip("\n").rpartition(",")
                if k and not h.startswith("ERROR:"):
                    done[k] = h
    keys = [f"{r['root']}|{r['rel_path']}" for r in rows]
    todo = [i for i, k in enumerate(keys) if k not in done]
    if todo:
        print(f"  sha256 of {len(todo)} {progress} files ({len(rows) - len(todo)} cached) ...", flush=True)
        B = 100_000
        with open(cache, "a") as f:
            for b0 in range(0, len(todo), B):
                part = todo[b0:b0 + B]
                hs = hash_paths([resolve(dp, rows[i]) for i in part], workers=workers, progress="")
                for i, h in zip(part, hs):
                    f.write(f"{keys[i]},{h}\n")
                    if not h.startswith("ERROR:"):
                        done[keys[i]] = h
                f.flush()
                print(f"    {progress}: {min(b0 + B, len(todo))}/{len(todo)} hashed", flush=True)
    return [done.get(k, "ERROR:unreadable") for k in keys]


# keys are "a|b" with a before b in SPLITS order
CRITICAL_PAIRS = ("cal|calcheck", "cal|train", "calcheck|train", "cal|val", "cal|test",
                  "calcheck|val", "calcheck|test", "val|test")


def disjointness_report(hashes: dict) -> dict:
    """hashes: {split: [sha256 per row]} -> duplicate counts within/between splits."""
    rep = dict(within={}, between={}, errors={}, examples={})
    sets = {}
    for s, hs in hashes.items():
        errs = [h for h in hs if h.startswith("ERROR:")]
        rep["errors"][s] = len(errs)
        good = [h for h in hs if not h.startswith("ERROR:")]
        rep["within"][s] = len(good) - len(set(good))
        sets[s] = set(good)
    names = list(hashes)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            key = "|".join(sorted((a, b), key=SPLITS.index))
            inter = sets[a] & sets[b]
            rep["between"][key] = len(inter)
            if inter:
                rep["examples"][key] = sorted(inter)[:5]
    crit = [k for k in CRITICAL_PAIRS if k in rep["between"]]
    within_crit = [s for s in ("cal", "calcheck") if s in rep["within"]]
    rep["critical_pairs"] = crit
    rep["critical_within"] = within_crit
    rep["not_checked"] = [k for k in CRITICAL_PAIRS if k not in rep["between"]]
    rep["ok"] = (all(rep["between"][k] == 0 for k in crit)
                 and all(rep["within"][s] == 0 for s in within_crit)
                 and all(v == 0 for v in rep["errors"].values()))
    return rep


def path_overlap_report(real: dict) -> dict:
    """Overlap of RESOLVED paths (symlinks followed) within and between splits."""
    within, between = {}, {}
    for s in SPLITS:
        within[s] = len(real[s]) - len(set(real[s]))
    for i, a in enumerate(SPLITS):
        sa = set(real[a])
        for b in SPLITS[i + 1:]:
            between[f"{a}|{b}"] = len(sa & set(real[b])) if real[b] else 0
    return dict(within=within, between=between,
                ok=all(v == 0 for v in within.values()) and all(v == 0 for v in between.values()))


# --------------------------------------------------------------------------- #
# preprocessing
# --------------------------------------------------------------------------- #


def preprocess_id(interpolation: str = "bicubic") -> str:
    from cim_exp import imagenette as IM
    return (f"cim_exp.imagenette.load_uint8+to_model_input|interp={interpolation}|"
            f"resize_short={IM.RESIZE_SHORT}|crop={IM.CROP}|mean={[float(v) for v in IM.MEAN]}|std={[float(v) for v in IM.STD]}")


def load_uint8_eval(path, interpolation: str = "bicubic") -> np.ndarray:
    """[224,224,3] uint8. "bicubic" IS cim_exp.imagenette.load_uint8."""
    from cim_exp import imagenette as IM
    if interpolation == "bicubic":
        return IM.load_uint8(Path(path))
    if interpolation != "bilinear":
        raise ValueError(interpolation)
    from PIL import Image
    return _load_uint8_with(path, Image.BILINEAR)


def _load_uint8_with(path, resample) -> np.ndarray:
    """Geometry copied from cim_exp.imagenette.load_uint8; only the filter is a
    parameter. test_cim_qat_data.py asserts that resample=BICUBIC reproduces
    load_uint8 bit for bit, so the bilinear variant differs ONLY in the filter."""
    from cim_exp import imagenette as IM
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        if w <= h:
            nw, nh = IM.RESIZE_SHORT, int(IM.RESIZE_SHORT * h / w)
        else:
            nw, nh = int(IM.RESIZE_SHORT * w / h), IM.RESIZE_SHORT
        im = im.resize((nw, nh), resample)
        top = int(round((nh - IM.CROP) / 2.0))
        left = int(round((nw - IM.CROP) / 2.0))
        im = im.crop((left, top, left + IM.CROP, top + IM.CROP))
        return np.asarray(im, dtype=np.uint8).copy()


def to_model_input(u8):
    from cim_exp.imagenette import to_model_input as f
    return f(u8)


def _torch_dataset_base():
    from torch.utils.data import Dataset
    return Dataset


class EvalImages(_torch_dataset_base()):
    """(normalised [3,224,224] float32, label, manifest idx) — cim_exp protocol."""

    def __init__(self, splits: Splits, split: str, interpolation: str = "bicubic",
                 indices=None):
        self.splits = splits
        rows = splits.rows(split)
        self.rows = rows if indices is None else [rows[i] for i in indices]
        self.interpolation = interpolation
        self.preprocess_id = preprocess_id(interpolation)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        u8 = load_uint8_eval(self.splits.path(r), self.interpolation)
        return to_model_input(u8[None])[0], r["label"], r["idx"]


EPOCH_STRIDE = 10_000_000


class TrainImages(_torch_dataset_base()):
    """Bicubic RandomResizedCrop + flip; per-sample seeded augmentation.

    Index keys are epoch * EPOCH_STRIDE + i (see EpochSampler), so the crop and
    flip of sample i in epoch e are a pure function of (seed, e, i).
    """

    def __init__(self, splits: Splits, split: str = "train", seed: int = 0):
        import torchvision.transforms as T
        from cim_exp.imagenette import MEAN, STD
        if split != "train":
            raise ValueError("TrainImages is for the TRAIN split only")
        self.splits = splits
        self.rows = splits.rows(split)
        self.seed = seed
        self.tf = T.Compose([
            T.RandomResizedCrop(224, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        import torch
        from PIL import Image
        epoch, i = divmod(int(key), EPOCH_STRIDE)
        r = self.rows[i]
        torch.manual_seed(derive_seed("aug", self.seed, epoch, i))
        with Image.open(self.splits.path(r)) as im:
            x = self.tf(im.convert("RGB"))
        return x, r["label"], r["idx"]


class EpochSampler:
    """Seeded permutation per epoch, resumable from any position."""

    def __init__(self, n: int, seed: int, epoch: int, start: int = 0):
        self.n, self.seed, self.epoch, self.start = n, seed, epoch, start

    def order(self):
        return np.random.default_rng(derive_seed("perm", self.seed, self.epoch)).permutation(self.n)

    def __iter__(self):
        base = self.epoch * EPOCH_STRIDE
        for i in self.order()[self.start:]:
            yield base + int(i)

    def __len__(self):
        return self.n - self.start


# --------------------------------------------------------------------------- #
# prepare (writes the split directory)
# --------------------------------------------------------------------------- #


def _write_hashes(path: Path, rows, hs):
    with open(path, "w") as f:
        f.write("idx,sha256\n")
        for r, h in zip(rows, hs):
            f.write(f"{r['idx']},{h}\n")


def read_hashes(path) -> list:
    with open(path) as f:
        next(f)
        return [line.rstrip("\n").split(",", 1)[1] for line in f]


def prepare(dp: DataPaths, seed: int = DEFAULT_SPLIT_SEED, workers: int = 8,
            hash_train: bool = False, force: bool = False) -> dict:
    out = dp.splits_dir
    if (out / "splits.json").exists() and not force:
        raise FileExistsError(f"{out}/splits.json exists. The splits are FIXED once made; "
                              "use --force only if you are certain nothing has used them.")
    out.mkdir(parents=True, exist_ok=True)
    print(f"listing data (train roots {dp.train_roots}, val {dp.val_root}, "
          f"ImageNetV2 {dp.imagenetv2_root}) ...", flush=True)
    L = list_all(dp)
    syn = synsets()
    known = load_hash_caches(out)            # every hash computed by any earlier attempt
    if known:
        vc = verify_cache_sample(known, dp)
        print(f"  {len(known):,} file hashes already cached in {out}; re-hashed a sample of "
              f"{vc['checked']}: {len(vc['mismatches'])} mismatch(es)", flush=True)
        if vc["mismatches"]:
            raise RuntimeError(f"stale hash cache in {out} (files changed since hashing, e.g. "
                               f"{vc['mismatches'][:3]}): delete {out}/hashcache_*.csv and run make again")
    # 1. evaluation-set hashes FIRST, so CAL/CALCHECK can avoid their content
    vrows = [dict(root="val", rel_path=rel) for w in syn for rel, _ in L["va"][w]]
    hv = hash_rows_cached(vrows, dp, out / "hashcache_val.csv", workers, "val", known)
    ht = []
    if L["v2"] is not None:
        trows = [dict(root="v2", rel_path=rel) for i in range(1000) for rel, _ in L["v2"][i]]
        ht = hash_rows_cached(trows, dp, out / "hashcache_test.csv", workers, "test", known)
    if hash_train:
        allt = [dict(root=f"train{ri}", rel_path=rel) for w in syn for ri, rel, _ in L["tr"][w]]
        hash_rows_cached(allt, dp, out / "hashcache_train.csv", workers, "train (all)", known)
    val_set, test_set, held = set(hv), set(ht), set()
    cand_cache = out / "hashcache_candidates.csv"

    def eligible(ri, rel, real):
        key = f"train{ri}|{rel}"
        h = known.get(key)
        if h is None:                         # without --hash-train: hash candidates on demand
            h = _hash_one(real)
            with open(cand_cache, "a") as f:
                f.write(f"{key},{h}\n")
            if h.startswith("ERROR:"):
                return False, f"unreadable ({h[6:60]})"
            known[key] = h
        if h in val_set:
            return False, "content duplicate of a VAL image"
        if h in test_set:
            return False, "content duplicate of a TEST image"
        if h in held:
            return False, "content duplicate of an already selected CAL/CALCHECK image"
        held.add(h)
        return True, ""
    rows, info, real = build_splits(dp, seed, eligible, L)
    print("  counts:", info["counts"], flush=True)
    for sk in info["heldout_skipped"]:
        print(f"  held-out candidate skipped (stays in TRAIN): class {sk['class_index']} {sk['root']}/"
              f"{sk['rel_path']} would have been {sk['would_have_been'].upper()}: {sk['reason']}", flush=True)

    # 2. content hashes of every split (from the cache; TRAIN with --hash-train)
    hs = {}
    for s in ("cal", "calcheck", "val", "test"):
        if rows[s]:
            hs[s] = hash_rows_cached(rows[s], dp, out / f"hashcache_{s}.csv", workers, s, known)
    excluded = []
    if hash_train:
        htr = hash_rows_cached(rows["train"], dp, out / "hashcache_train.csv", workers, "train", known)
        held = set(hs["cal"]) | set(hs["calcheck"])
        keep = [i for i, h in enumerate(htr) if h not in held]
        excluded = [dict(rows["train"][i], sha256=htr[i], reason="content duplicate of a CAL/CALCHECK file")
                    for i, h in enumerate(htr) if h in held]
        if excluded:
            print(f"  excluding {len(excluded)} TRAIN files whose content duplicates CAL/CALCHECK", flush=True)
        rows["train"] = [rows["train"][i] for i in keep]
        real["train"] = [real["train"][i] for i in keep]
        hs["train"] = [htr[i] for i in keep]
        for i, r in enumerate(rows["train"]):
            r["idx"] = i
    info["counts"] = {s: len(rows[s]) for s in SPLITS}
    info["manifest_sha256"] = {s: write_manifest(rows[s], out / f"manifest_{s}.csv")
                               for s in SPLITS if s != "test" or rows["test"]}
    for s, h in hs.items():
        _write_hashes(out / f"hashes_{s}.csv", rows[s], h)
    if excluded:
        with open(out / "train_excluded.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS + ["sha256", "reason"], lineterminator="\n")
            wr.writeheader()
            for r in excluded:
                wr.writerow({k: r[k] for k in MANIFEST_FIELDS + ["sha256", "reason"]})

    n_disk = info["listing"]["train"]["files"]
    acc = dict(train_files_on_disk=n_disk, cal=len(rows["cal"]), calcheck=len(rows["calcheck"]),
               train_excluded_content_duplicates=len(excluded), train=len(rows["train"]),
               formula="TRAIN = train_files_on_disk - CAL - CALCHECK - train_excluded_content_duplicates",
               identity_holds=(n_disk == len(rows["cal"]) + len(rows["calcheck"]) + len(rows["train"]) + len(excluded)),
               val_files_on_disk=info["listing"]["val"]["files"], val=len(rows["val"]),
               test_files_on_disk=(info["listing"]["test"] or {}).get("files"), test=len(rows["test"]))
    info["accounting"] = acc
    rep = disjointness_report(hs)
    rep["train_hashed"] = hash_train
    po = path_overlap_report(real)
    rep["path_overlap"] = po
    info["v2_mapping"] = v2_mapping_report(rows["test"], rows["val"]) if rows["test"] else None
    info["size_check"] = {k: dict(expected=v, actual=len(rows[k]), ok=len(rows[k]) == v)
                          for k, v in EXPECTED.items() if k != "test" or rows["test"]}
    rep["ok"] = bool(rep["ok"] and po["ok"] and acc["identity_holds"]
                     and (info["v2_mapping"] is None or info["v2_mapping"]["ok"]))
    info["disjointness"] = rep
    write_json(out / "splits.json", info)
    log_access(dp.work_root, "all", "prepare", None, dict(counts=info["counts"]))
    return info


SPLIT_OUTPUTS = ("splits.json", "train_excluded.csv") + tuple(
    f"{k}_{s}.csv" for k in ("manifest", "hashes") for s in SPLITS)
SAFE_PURPOSES_BEFORE_USE = ("prepare", "audit")


def archive_failed(dp: DataPaths) -> Path:
    """Move the outputs of a FAILED `make` into splits/failed_<time>/ so that `make`
    can run again. Refuses unless (a) the stored splits failed their critical checks
    and (b) the access log shows no use of them other than prepare/audit. Hash caches
    (hashcache_*.csv) stay in place and are reused."""
    import shutil
    out = dp.splits_dir
    info = read_json(out / "splits.json")
    if info.get("disjointness", {}).get("ok", True):
        raise RuntimeError("the existing splits PASSED their checks; they are fixed and may not be replaced")
    created = info.get("created_utc", "")
    used = []
    logp = Path(dp.work_root) / "access_log.jsonl"
    if logp.exists():
        for line in logp.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("time_utc", "") >= created and e.get("purpose") not in SAFE_PURPOSES_BEFORE_USE:
                used.append(f"{e.get('time_utc')} {e.get('purpose')} {e.get('split')}")
    if used:
        raise RuntimeError("the failed splits were already USED (access log): " + "; ".join(used[:5])
                           + ". Report this before regenerating anything.")
    dst = out / f"failed_{created.replace(':', '').replace('-', '').replace('+0000', 'Z')}"
    dst.mkdir(parents=True, exist_ok=False)
    for name in SPLIT_OUTPUTS:
        if (out / name).exists():
            shutil.move(str(out / name), str(dst / name))
    log_access(dp.work_root, "all", "prepare", None, dict(event="archived failed splits", to=str(dst)))
    return dst


def add_test(dp: DataPaths, workers: int = 8) -> dict:
    """Add the TEST (ImageNetV2) manifest to an existing split directory without
    touching CAL/CALCHECK/TRAIN/VAL (for when V2 is downloaded after prepare)."""
    out = dp.splits_dir
    info = read_json(out / "splits.json")
    if info.get("schema") != SPLIT_SCHEMA:
        raise RuntimeError("not a v2 split directory")
    if "test" in info["manifest_sha256"]:
        raise FileExistsError("TEST manifest already exists")
    if not dp.imagenetv2_root:
        raise RuntimeError("set imagenetv2_root in configs/paths.json first")
    v2, diag = list_v2(dp.imagenetv2_root)
    syn = synsets()
    rows = [dict(split="test", root="v2", rel_path=rel, wnid=syn[i], label=i)
            for i in range(1000) for rel, _ in v2[i]]
    real = [rl for i in range(1000) for _, rl in v2[i]]
    rows.sort(key=lambda r: (r["label"], r["root"], r["rel_path"]))
    for i, r in enumerate(rows):
        r["idx"] = i
    info["imagenetv2_root"] = dp.imagenetv2_root
    info["listing"]["test"] = diag
    info["test_available"] = True
    info["counts"]["test"] = len(rows)
    info["manifest_sha256"]["test"] = write_manifest(rows, out / "manifest_test.csv")
    ht = hash_rows_cached(rows, dp, out / "hashcache_test.csv", workers, "test")
    _write_hashes(out / "hashes_test.csv", rows, ht)
    hs = {s: read_hashes(out / f"hashes_{s}.csv") for s in ("cal", "calcheck", "val", "train")
          if (out / f"hashes_{s}.csv").exists()}
    hs["test"] = ht
    rep = disjointness_report(hs)
    rep["train_hashed"] = "train" in hs
    # path overlap of TEST against all others (resolved paths)
    other = set()
    for s in ("cal", "calcheck", "train", "val"):
        for r in read_manifest(out / f"manifest_{s}.csv"):
            other.add(os.path.realpath(resolve(info, r)))
    po = dict(info["disjointness"]["path_overlap"])
    po["within"] = dict(po["within"], test=len(real) - len(set(real)))
    po["between"] = dict(po["between"], **{f"{s}|test": None for s in ("cal", "calcheck", "train", "val")})
    po["between"]["any|test"] = len(other & set(real))
    po["ok"] = po["ok"] and po["between"]["any|test"] == 0 and po["within"]["test"] == 0
    rep["path_overlap"] = po
    val_rows = read_manifest(out / "manifest_val.csv")
    info["v2_mapping"] = v2_mapping_report(rows, val_rows)
    info["size_check"]["test"] = dict(expected=EXPECTED["test"], actual=len(rows), ok=len(rows) == EXPECTED["test"])
    info["accounting"].update(test_files_on_disk=diag["files"], test=len(rows))
    rep["ok"] = bool(rep["ok"] and po["ok"] and info["accounting"]["identity_holds"] and info["v2_mapping"]["ok"])
    info["disjointness"] = rep
    info["test_added_utc"] = now_iso()
    write_json(out / "splits.json", info)
    log_access(dp.work_root, "test", "prepare", info["manifest_sha256"]["test"])
    return info


def find_variant_root(path, split) -> Path | None:
    """The folder holding the 1000 numeric class folders of a variant: the given folder,
    its <EXT_TESTS[split]> subfolder, or its only subfolder that has them."""
    p = Path(path)
    cands = [p, p / EXT_TESTS[split]] + ([c for c in sorted(p.iterdir()) if c.is_dir()] if p.is_dir() else [])
    for cand in cands:
        if cand.is_dir() and {"0", "999"} <= set(_subdirs(cand)):
            return cand
    return None


def add_test_variant(dp: DataPaths, split: str, workers: int = 8, archive_sha256: str = "",
                     archive_name: str = "") -> dict:
    """Register one extra ImageNetV2 variant (v2.8 extension) WITHOUT touching splits.json
    or any existing file: writes manifest_<split>.csv, hashes_<split>.csv,
    excluded_<split>.csv, extcache_<split>.csv and the <split> entry of extra_tests.json."""
    if split not in EXT_TESTS:
        raise ValueError(f"unknown variant split {split!r}; one of {sorted(EXT_TESTS)}")
    out = dp.splits_dir
    info = read_json(out / "splits.json")
    if info.get("schema") != SPLIT_SCHEMA or "test" not in info["manifest_sha256"]:
        raise RuntimeError("needs the v2 split directory with the matched-frequency TEST registered")
    reg_p = out / EXT_REGISTRY
    reg = read_json(reg_p) if reg_p.exists() else {}
    if split in reg or (out / f"manifest_{split}.csv").exists():
        raise FileExistsError(f"{split} is already registered; it is registered once")
    given = (dp.imagenetv2_extra_roots or {}).get(split)
    if not given:
        raise RuntimeError(f"set imagenetv2_extra_roots.{split} in configs/paths.json first")
    root = find_variant_root(given, split)
    if root is None:
        raise RuntimeError(f"{given}: no folder with the class folders 0..999 found")
    targets = [out / f"{n}_{split}.csv" for n in ("manifest", "hashes", "excluded", "extcache")] + [reg_p]
    assert_not_frozen(dp.work_root, *targets)
    need = {s: out / f"hashes_{s}.csv" for s in ("cal", "calcheck", "val", "train", "test")}
    missing = [str(v) for v in need.values() if not v.exists()]
    if missing:
        raise RuntimeError(f"content hashes needed for the leakage check are missing: {missing}")
    per, diag = list_v2(root)
    syn = synsets()
    rows = [dict(split=split, root=f"v2x:{split}", rel_path=rel, wnid=syn[i], label=i)
            for i in range(1000) for rel, _ in per[i]]
    real = [rl for i in range(1000) for _, rl in per[i]]
    order = sorted(range(len(rows)), key=lambda k: (rows[k]["label"], rows[k]["rel_path"]))
    rows, real = [rows[k] for k in order], [real[k] for k in order]
    dpx = DataPaths(train_roots=dp.train_roots, val_root=dp.val_root, val_labels_csv=dp.val_labels_csv,
                    imagenetv2_root=dp.imagenetv2_root, work_root=dp.work_root,
                    imagenetv2_extra_roots={**(dp.imagenetv2_extra_roots or {}), split: str(root)})
    hs = hash_rows_cached(rows, dpx, out / f"extcache_{split}.csv", workers, split)
    errs = [r["rel_path"] for r, h in zip(rows, hs) if h.startswith("ERROR:")]
    if errs:
        raise RuntimeError(f"{len(errs)} unreadable files in {root}, e.g. {errs[:3]}")
    sets = {s: set(read_hashes(p)) for s, p in need.items()}
    for s2, e in reg.items():
        sets[s2] = set(read_hashes(out / f"hashes_{s2}.csv"))
    overlap = {s: sum(h in sets[s] for h in hs) for s in sets}
    seen_split = [next((s for s in ("cal", "calcheck", "val", "train") if h in sets[s]), None) for h in hs]
    keep = [k for k, w in enumerate(seen_split) if w is None]
    excl = [dict(rows[k], sha256=hs[k], reason=f"content equals a {seen_split[k].upper()} image")
            for k in range(len(rows)) if seen_split[k] is not None]
    krows = [dict(rows[k], idx=i) for i, k in enumerate(keep)]
    khs = [hs[k] for k in keep]
    # path-level: no file shared with any registered split (resolved real paths)
    other = set()
    for s in ("cal", "calcheck", "train", "val", "test"):
        for r in read_manifest(out / f"manifest_{s}.csv"):
            other.add(os.path.realpath(resolve(info, r)))
    for s2, e in reg.items():
        for r in read_manifest(out / f"manifest_{s2}.csv"):
            other.add(os.path.realpath(Path(e["root"]) / r["rel_path"]))
    path_shared = len(other & set(real))
    counts = [sum(r["label"] == i for r in rows) for i in range(1000)]
    mapping = dict(label_is_int_folder=all(r["label"] == int(Path(r["rel_path"]).parts[0]) for r in rows),
                   wnid_matches_index=all(r["wnid"] == syn[r["label"]] for r in rows),
                   classes=len({r["label"] for r in rows}), per_class_min=min(counts), per_class_max=max(counts))
    mapping["ok"] = bool(mapping["label_is_int_folder"] and mapping["wnid_matches_index"]
                         and mapping["classes"] == 1000)
    msha = write_manifest(krows, out / f"manifest_{split}.csv")
    _write_hashes(out / f"hashes_{split}.csv", krows, khs)
    with open(out / f"excluded_{split}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS + ["sha256", "reason"], lineterminator="\n")
        wr.writeheader()
        for r in excl:
            wr.writerow({k: r.get(k, "") for k in MANIFEST_FIELDS + ["sha256", "reason"]})
    entry = dict(variant=EXT_TESTS[split], root=str(root), archive=archive_name, archive_sha256=archive_sha256,
                 listing=diag, files_on_disk=diag["files"], n=len(krows), excluded=len(excl),
                 within_duplicates=len(hs) - len(set(hs)), content_overlap_before_exclusion=overlap,
                 path_shared_with_registered_splits=path_shared, mapping=mapping,
                 manifest_sha256=msha, hashes_sha256=sha256_file(out / f"hashes_{split}.csv"),
                 rule=EXT_RULE, added_utc=now_iso())
    entry["ok"] = bool(mapping["ok"] and path_shared == 0 and diag["files"] == len(rows))
    reg[split] = entry
    write_json(reg_p, reg)
    log_access(dp.work_root, split, "prepare", msha)
    return entry
