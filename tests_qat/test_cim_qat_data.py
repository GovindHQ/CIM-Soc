#!/usr/bin/env python3
"""Data-layer tests (run anywhere; builds a temporary fake ImageNet + ImageNetV2).
  D1  split construction is a pure function of file lists + seed
  D2  flat val + LOC_val_solution.csv == class-folder val; unlabelled file / missing file rejected
  D3  bilinear loader geometry: resample=BICUBIC reproduces cim_exp load_uint8 bit for bit
  D4  eval dataset == cim_exp to_model_input(load_uint8); preprocess id recorded
  D5  training augmentation is a pure function of (seed, epoch, index), independent of workers
  D6  EpochSampler resume yields exactly the tail of the full order
  D7  ImageNetV2 labels = int(folder); torchvision ImageFolder would mislabel 988 of 1000 classes
  D8  ImageNetV2 structure errors (missing / zero-padded folder) are rejected
  D9  TEST guard: closed by default, closed for training purposes, every access logged
  D10 exact accounting: TRAIN = on disk - CAL - CALCHECK - excluded content duplicates
  D11 content duplicates: train copy of a CAL image excluded; VAL==TEST copy fails the report
  D12 symlinked val files: resolved-path overlap with TRAIN is detected
  D13 add-test after make == make with ImageNetV2 (same TEST manifest)
  D14 held-out eligibility: a CAL pick whose content equals a VAL image, and a CALCHECK pick
      equal to a TEST image, are skipped (stay in TRAIN); every other class keeps its picks
  D15 make --replace-failed: refuses passed splits and used splits; archives a failed,
      unused one, keeps the hash caches, and the new make passes
"""
import csv, json, os, shutil, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from PIL import Image
from cim_qat.common import setup_paths  # noqa: F401
from cim_qat import data as D

R = []
def check(n, ok, **kw):
    R.append(bool(ok)); print(f"  [{'PASS' if ok else 'FAIL'}] {n} " + " ".join(f"{k}={v}" for k, v in kw.items()))

def img(rng, p):
    a = rng.integers(0, 255, (int(rng.integers(40, 90)), int(rng.integers(40, 90)), 3), dtype=np.uint8)
    Image.fromarray(a).save(p)

def fake(tmp: Path, n_train=4, n_val=2, n_v2=2, seed=0):
    rng = np.random.default_rng(seed)
    syn = D.synsets()
    (tmp / "val_flat").mkdir(parents=True)
    rows = []
    for ci, w in enumerate(syn):
        d = tmp / "train" / w; d.mkdir(parents=True)
        for k in range(n_train):
            img(rng, d / f"{w}_{k}.JPEG")
        dv = tmp / "val_cls" / w; dv.mkdir(parents=True)
        for k in range(n_val):
            name = f"ILSVRC2012_val_{ci * n_val + k + 1:08d}"
            img(rng, dv / f"{name}.JPEG")
            shutil.copy(dv / f"{name}.JPEG", tmp / "val_flat" / f"{name}.JPEG")
            rows.append((name, f"{w} 1 2 3 4"))
        d2 = tmp / "v2" / D.V2_DIRNAME / str(ci); d2.mkdir(parents=True)
        for k in range(n_v2):
            img(rng, d2 / f"{ci:04d}{k}abc.jpeg")
    with open(tmp / "LOC_val_solution.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["ImageId", "PredictionString"]); wr.writerows(rows)
    return syn

def dpaths(tmp, work, v2=True, val="flat"):
    return D.DataPaths([str(tmp / "train")], str(tmp / ("val_flat" if val == "flat" else "val_cls")),
                       str(tmp / "LOC_val_solution.csv") if val == "flat" else None,
                       str(tmp / "v2" / D.V2_DIRNAME) if v2 else None, str(work))

tmp = Path(tempfile.mkdtemp(prefix="qat data test "))          # a space in the path on purpose
syn = fake(tmp)
dp = dpaths(tmp, tmp / "w1")
rows1, info1, _ = D.build_splits(dp, 123)
rows2, _, _ = D.build_splits(dp, 123)
check("D1 deterministic splits", all(rows1[s] == rows2[s] for s in D.SPLITS), counts=info1["counts"])
rows3, _, _ = D.build_splits(dp, 124)
check("D1 a different seed gives a different CAL", rows3["cal"] != rows1["cal"])
check("D1 VAL = every val image, TEST = every V2 image",
      len(rows1["val"]) == 2000 and len(rows1["test"]) == 2000 and len(rows1["cal"]) == 2000
      and len(rows1["calcheck"]) == 200 and len(rows1["train"]) == 4000 - 2200)
rowsc, _, _ = D.build_splits(dpaths(tmp, tmp / "w0", val="cls"), 123)
same = [(r["wnid"], Path(r["rel_path"]).name) for r in rowsc["val"]] == \
       [(r["wnid"], Path(r["rel_path"]).name) for r in rows1["val"]]
check("D2 flat val + LOC csv == class-folder val", same)
img(np.random.default_rng(9), tmp / "val_flat" / "ILSVRC2012_val_99999999.JPEG")
try:
    D.list_val(str(tmp / "val_flat"), str(tmp / "LOC_val_solution.csv")); ok = False
except RuntimeError as e:
    ok = "no CSV label" in str(e)
check("D2 an unlabelled file in the flat val folder is rejected", ok)
os.remove(tmp / "val_flat" / "ILSVRC2012_val_99999999.JPEG")
os.rename(tmp / "val_flat" / "ILSVRC2012_val_00000005.JPEG", tmp / "x.JPEG")
try:
    D.list_val(str(tmp / "val_flat"), str(tmp / "LOC_val_solution.csv")); ok = False
except FileNotFoundError:
    ok = True
check("D2 a CSV row without an image file is rejected", ok)
os.rename(tmp / "x.JPEG", tmp / "val_flat" / "ILSVRC2012_val_00000005.JPEG")

from cim_exp import imagenette as IM
p = tmp / "val_flat" / "ILSVRC2012_val_00000011.JPEG"
check("D3 bilinear loader geometry == load_uint8 (BICUBIC)", np.array_equal(D._load_uint8_with(p, Image.BICUBIC), IM.load_uint8(p)))
check("D3 bilinear differs from bicubic", not np.array_equal(D.load_uint8_eval(p, "bilinear"), IM.load_uint8(p)))

info = D.prepare(dp, 123, workers=1, hash_train=True)
sp = D.Splits(dp.splits_dir, purpose="unit_test")
ds = D.EvalImages(sp, "val")
x, y, i = ds[7]
r = sp.rows("val")[7]
import torch
check("D4 EvalImages == cim_exp to_model_input(load_uint8)",
      torch.equal(x, IM.to_model_input(IM.load_uint8(sp.path(r))[None])[0]) and y == r["label"])
check("D4 preprocess id names the protocol", "interp=bicubic" in ds.preprocess_id and "resize_short=256" in ds.preprocess_id)
tr = D.TrainImages(sp, "train", seed=5)
k = 3 * D.EPOCH_STRIDE + 11
a1, a2 = tr[k][0], tr[k][0]
torch.manual_seed(999); a3 = tr[k][0]
check("D5 augmentation deterministic per (seed, epoch, idx)", torch.equal(a1, a2) and torch.equal(a1, a3))
check("D5 different epoch -> different augmentation", not torch.equal(tr[11][0], a1))
x0 = torch.cat([b[0] for b in torch.utils.data.DataLoader(tr, batch_size=8, sampler=D.EpochSampler(len(tr), 5, 2), num_workers=0)][:3])
x2 = torch.cat([b[0] for b in torch.utils.data.DataLoader(tr, batch_size=8, sampler=D.EpochSampler(len(tr), 5, 2), num_workers=2)][:3])
check("D5 identical batches with 0 vs 2 loader workers", torch.equal(x0, x2))
full = list(D.EpochSampler(100, 1, 4)); tail = list(D.EpochSampler(100, 1, 4, start=37))
check("D6 EpochSampler resume == tail of full order", tail == full[37:])

# D7 ImageNetV2 mapping
spt = D.Splits(dp.splits_dir, purpose="unit_test", allow_test=True)
te = spt.rows("test")
check("D7 TEST label == int(folder) and wnid == synsets[label]",
      all(r["label"] == int(r["rel_path"].split("/")[0]) and r["wnid"] == syn[r["label"]] for r in te))
import torchvision
ifo = torchvision.datasets.ImageFolder(str(tmp / "v2" / D.V2_DIRNAME))
wrong = sum(1 for name, k in ifo.class_to_idx.items() if int(name) != k)
lex = D.imagefolder_lexicographic_labels()
exp_wrong = sum(1 for k, v in lex.items() if k != v)
check("D7 torchvision ImageFolder WOULD mislabel (trap pinned)", wrong == exp_wrong >= 980 and ifo.class_to_idx["10"] == 2
      and all(ifo.class_to_idx[str(k)] == v for k, v in lex.items()),
      imagefolder_wrong_classes=wrong)
check("D7 v2_mapping_report ok", info["v2_mapping"]["ok"] and info["v2_mapping"]["imagefolder_trap_classes_that_would_be_mislabelled"] == exp_wrong)
# D8 structure errors
os.rename(tmp / "v2" / D.V2_DIRNAME / "7", tmp / "v2" / D.V2_DIRNAME / "007")
try:
    D.list_v2(tmp / "v2" / D.V2_DIRNAME); ok = False
except RuntimeError as e:
    ok = "missing class folders" in str(e)
check("D8 zero-padded / missing V2 folder rejected", ok)
os.rename(tmp / "v2" / D.V2_DIRNAME / "007", tmp / "v2" / D.V2_DIRNAME / "7")
check("D8 find_v2_root accepts the parent folder", D.find_v2_root(tmp / "v2") == tmp / "v2" / D.V2_DIRNAME)
# D9 guard + access log
try:
    sp.rows("test"); ok = False
except D.TestAccessError:
    ok = True
check("D9 TEST closed by default", ok)
try:
    D.Splits(dp.splits_dir, purpose="train", allow_test=True); ok = False
except D.TestAccessError:
    ok = True
check("D9 a training purpose cannot open TEST", ok)
log = [json.loads(l) for l in (Path(dp.work_root) / "access_log.jsonl").read_text().splitlines()]
check("D9 accesses logged with purpose", any(e["split"] == "test" and e["purpose"] == "unit_test" for e in log)
      and any(e["split"] == "val" and e["purpose"] == "unit_test" for e in log), entries=len(log))
# D10 accounting
acc = info["accounting"]
check("D10 TRAIN = on disk - CAL - CALCHECK - excluded", acc["identity_holds"] and acc["train"] == 4000 - 2000 - 200 - acc["train_excluded_content_duplicates"], accounting=acc["train"])
check("D10 clean fake data passes every critical check", info["disjointness"]["ok"], not_checked=info["disjointness"]["not_checked"])
# D11 planted duplicates
cal0 = sp.path(sp.rows("cal")[0])
tr_victim = sp.path(sp.rows("train")[0])
shutil.copy(cal0, tr_victim)                                         # a TRAIN file == a CAL file
v0 = sp.path(sp.rows("val")[3]); t0 = spt.path(te[5]); shutil.copy(v0, t0)   # a TEST file == a VAL file
info2 = D.prepare(dpaths(tmp, tmp / "w2"), 123, workers=1, hash_train=True)
ex = info2["accounting"]["train_excluded_content_duplicates"]
check("D11 TRAIN copy of a CAL image is excluded from TRAIN", ex == 1 and info2["accounting"]["identity_holds"]
      and info2["counts"]["train"] == 4000 - 2200 - 1, excluded=ex)
check("D11 VAL==TEST content duplicate fails the report", (not info2["disjointness"]["ok"])
      and info2["disjointness"]["between"]["val|test"] == 1)
# D12 symlinked val pointing into train
vt = tmp / "val_flat" / "ILSVRC2012_val_00000001.JPEG"
os.remove(vt); os.symlink(tr_victim, vt)
_, _, real = D.build_splits(dpaths(tmp, tmp / "w3"), 123)
po = D.path_overlap_report(real)
check("D12 symlinked val file resolving into TRAIN is detected", (not po["ok"]) and po["between"]["train|val"] == 1)
# D13 add-test
shutil.rmtree(tmp); tmp = Path(tempfile.mkdtemp(prefix="qat data test ")); fake(tmp)
ia = D.prepare(dpaths(tmp, tmp / "wa"), 123, workers=1, hash_train=False)
ib = D.prepare(dpaths(tmp, tmp / "wb", v2=False), 123, workers=1, hash_train=False)
ib2 = D.add_test(dpaths(tmp, tmp / "wb"), workers=1)
check("D13 add-test == prepare with V2", ia["manifest_sha256"] == ib2["manifest_sha256"] and ib2["disjointness"]["ok"])
check("D13 without --hash-train, TRAIN content pairs are reported as not checked",
      set(ia["disjointness"]["not_checked"]) == {"cal|train", "calcheck|train"})
# D14 held-out eligibility rule
shutil.rmtree(tmp); tmp = Path(tempfile.mkdtemp(prefix="qat data test ")); fake(tmp)
base = D.prepare(dpaths(tmp, tmp / "w0"), 123, workers=1, hash_train=True)
s0 = D.Splits(dpaths(tmp, tmp / "w0").splits_dir, purpose="unit_test", allow_test=True)
cal0, chk0 = s0.rows("cal"), s0.rows("calcheck")
victim_cal = cal0[10]                                    # a CAL pick of class 5
victim_chk = chk0[3]
val_target = s0.path(s0.rows("val")[500])                # some val image (another class)
test_target = s0.path(s0.rows("test")[1500])
shutil.copy(s0.path(victim_cal), val_target)             # VAL image := content of a CAL pick
shutil.copy(s0.path(victim_chk), test_target)            # TEST image := content of a CALCHECK pick
i14 = D.prepare(dpaths(tmp, tmp / "w1"), 123, workers=1, hash_train=True)
s1 = D.Splits(dpaths(tmp, tmp / "w1").splits_dir, purpose="unit_test")
cal1, chk1 = s1.rows("cal"), s1.rows("calcheck")
key = lambda r: (r["root"], r["rel_path"])
reasons = sorted(x["reason"] for x in i14["heldout_skipped"])
check("D14 skipped exactly the two colliding picks, with reasons",
      reasons == ["content duplicate of a TEST image", "content duplicate of a VAL image"], skipped=len(reasons))
check("D14 CAL/CALCHECK free of VAL/TEST content; report ok",
      i14["disjointness"]["between"]["cal|val"] == 0 and i14["disjointness"]["between"]["calcheck|test"] == 0
      and i14["disjointness"]["ok"] and i14["accounting"]["identity_holds"])
moved = {victim_cal["label"], victim_chk["label"]}
same_cal = [key(r) for r in cal0 if r["label"] not in moved] == [key(r) for r in cal1 if r["label"] not in moved]
same_chk = [key(r) for r in chk0 if r["label"] not in moved] == [key(r) for r in chk1 if r["label"] not in moved]
check("D14 every other class keeps exactly the same CAL/CALCHECK files", same_cal and same_chk)
tr1 = {key(r) for r in s1.rows("train")}
check("D14 skipped files stay in TRAIN (reported as train|val / train|test duplicates)",
      key(victim_cal) in tr1 and key(victim_chk) in tr1
      and i14["disjointness"]["between"]["train|val"] == 1 and i14["disjointness"]["between"]["train|test"] == 1)
# D15 replace-failed
dpf = dpaths(tmp, tmp / "w0")                             # w0 was made BEFORE the plant -> now stale
try:
    D.archive_failed(dpf); ok = False
except RuntimeError as e:
    ok = "PASSED" in str(e)
check("D15 refuses to replace splits that passed", ok)
v_dup = s0.path(s0.rows("val")[7]); t_dup = s0.path(s0.rows("test")[8]); shutil.copy(v_dup, t_dup)   # VAL==TEST
dp2 = dpaths(tmp, tmp / "w2")
i15 = D.prepare(dp2, 123, workers=1, hash_train=True)
check("D15 a VAL==TEST duplicate makes make FAIL", not i15["disjointness"]["ok"])
with open(Path(dp2.work_root) / "access_log.jsonl", "a") as f:
    f.write(json.dumps(dict(time_utc="2999-01-01T00:00:00+00:00", split="train", purpose="train")) + "\n")
try:
    D.archive_failed(dp2); ok = False
except RuntimeError as e:
    ok = "already USED" in str(e)
check("D15 refuses to replace failed splits that were used", ok)
lines = (Path(dp2.work_root) / "access_log.jsonl").read_text().splitlines()[:-1]
(Path(dp2.work_root) / "access_log.jsonl").write_text("\n".join(lines) + "\n")
n_cache = len(D.load_hash_caches(dp2.splits_dir))
dst = D.archive_failed(dp2)
check("D15 archives the failed outputs and keeps the hash caches",
      (dst / "splits.json").exists() and not (dp2.splits_dir / "splits.json").exists()
      and len(D.load_hash_caches(dp2.splits_dir)) == n_cache)
img(np.random.default_rng(77), t_dup)                    # remove the VAL==TEST duplicate (file changed)
vc = D.verify_cache_sample(D.load_hash_caches(dp2.splits_dir), dp2, n=10**9)
check("D15 the cache check detects the changed file", len(vc["mismatches"]) == 1, mismatches=vc["mismatches"])
for c in dp2.splits_dir.glob("hashcache_*.csv"):         # a changed dataset needs fresh hashes
    c.unlink()
i15b = D.prepare(dp2, 123, workers=1, hash_train=True)
check("D15 make after the fix passes", i15b["disjointness"]["ok"])
shutil.rmtree(tmp)
print(f"{sum(R)}/{len(R)} passed"); sys.exit(0 if all(R) else 1)
