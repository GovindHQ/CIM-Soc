#!/usr/bin/env python3
"""
test_experiment_pipeline.py — offline tests of the experiment pipeline logic
(cim_exp/*). No dataset download is needed: a small stand-in folder with the
Imagenette layout is generated with random JPEGs. These images are used ONLY to
test manifest / preprocessing / reporting mechanics; they never enter a
results table.

Run:  python test_experiment_pipeline.py
"""
import io
import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "_shim"))
sys.path.insert(0, str(HERE))

import numpy as np

from cim_exp import hardware as HWM
from cim_exp import imagenette as IM
from cim_exp import metrics as M
from cim_exp import reporting as RP

PASS, SKIP = [], []


def check(name, cond, detail=""):
    PASS.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def skip(name, why):
    SKIP.append(name)
    print(f"  [SKIP] {name}  ({why})")


TMP = Path(tempfile.mkdtemp(prefix="cimexp_test_"))

# --------------------------------------------------------------------------- #
print("\nTest 1 — class mapping (static)")
# --------------------------------------------------------------------------- #
notes = IM.verify_mapping_tables()
syn = IM.load_imagenet_synsets()
check("table consistent with the ImageNet-1k synset list", True, f"{len(syn)} synsets")
check("parachute folder n03888257 maps to ImageNet class 701",
      syn.index("n03888605") == 702 and syn[701] == "n03888257")
par = [c for c in IM.IMAGENETTE_CLASSES if c.name == "parachute"][0]
check("parachute mapped by class to n03888257 -> 701, and flagged in notes",
      par.imagenet_index == 701 and any("parachute" in n for n in notes))
check("folder wnid == ImageNet wnid for all ten classes",
      sum(c.folder_wnid == c.imagenet_wnid for c in IM.IMAGENETTE_CLASSES) == 10)
check("10-way label order = sorted folder order",
      [c.folder_wnid for c in IM.IMAGENETTE_CLASSES]
      == sorted(c.folder_wnid for c in IM.IMAGENETTE_CLASSES))

# --------------------------------------------------------------------------- #
print("\nTest 2 — metric definitions")
# --------------------------------------------------------------------------- #
lg = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0], [1.0, 1.0, 5.0]])
y = np.array([0, 2, 2])
ce_hand = np.mean([np.log(np.exp(lg[i]).sum()) - lg[i, y[i]] for i in range(3)])
check("top-1 vs true labels", abs(M.top1_pct(lg, y) - 200 / 3) < 1e-12)
check("cross-entropy from raw logits (natural log)",
      abs(M.cross_entropy(lg, y) - ce_hand) < 1e-12, f"{ce_hand:.6f}")
p = np.exp(lg) / np.exp(lg).sum(1, keepdims=True)
check("CE == -mean log(softmax(logits))[y]: softmax applied exactly once",
      abs(M.cross_entropy(lg, y) - np.mean(-np.log(p[np.arange(3), y]))) < 1e-12)
p2 = np.exp(p) / np.exp(p).sum(1, keepdims=True)          # a double softmax
check("a double softmax would give a different value (guard)",
      abs(np.mean(-np.log(p2[np.arange(3), y])) - ce_hand) > 1e-3)
check("CE stable for huge logits",
      np.isfinite(M.cross_entropy(lg * 1e4, y)))
check("agreement compares argmaxes against a named reference",
      M.agreement_pct(lg, lg) == 100.0 and abs(M.agreement_pct(lg, lg[[0, 2, 1]]) - 100 / 3) < 1e-12)
check("relative RMS = 100*||t-r||/||r|| over the whole tensor",
      abs(M.rel_rms_pct(lg * 1.1, lg) - 10.0) < 1e-9)
bad = lg.copy()
bad[1, 2] = np.nan
check("non-finite images counted", M.nonfinite_images(bad) == 1)
try:
    M.agreement_pct(lg, lg[:2])
    check("shape mismatch rejected", False)
except ValueError:
    check("shape mismatch rejected", True)
big = np.zeros((2, 1000))
big[0, 701] = 5
big[1, 3] = 9
mm = M.model_metrics(big, np.array([701, 0]), np.array([9, 0]), IM.IMAGENET_INDICES)
check("10-way restriction uses the mapped indices",
      mm["top1_1000way_pct"] == 50.0 and mm["top1_10way_pct"] == 100.0,
      f"1000-way {mm['top1_1000way_pct']}%, 10-way {mm['top1_10way_pct']}%")

# --------------------------------------------------------------------------- #
print("\nTest 3 — manifests on a stand-in Imagenette folder")
# --------------------------------------------------------------------------- #
from PIL import Image

ds = TMP / "imagenette2-160"
rng = np.random.default_rng(0)
for c in IM.IMAGENETTE_CLASSES:
    d = ds / "val" / c.folder_wnid
    d.mkdir(parents=True)
    for k in range(13):
        w, h = (213, 160) if k % 2 else (160, 240)
        Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(
            d / f"img_{k:03d}.JPEG", quality=90)
rows1 = IM.build_manifest(ds, "smoke", per_class=10, seed=0)
rows2 = IM.build_manifest(ds, "smoke", per_class=10, seed=0)
sha1 = IM.write_manifest(rows1, TMP / "m1.csv")
sha2 = IM.write_manifest(rows2, TMP / "m2.csv")
check("smoke manifest is deterministic (same SHA-256 twice)", sha1 == sha2)
check("smoke manifest: 100 images, 10 per class",
      len(rows1) == 100 and all(sum(r["label_10way"] == i for r in rows1) == 10 for i in range(10)))
check("labels follow the verified mapping",
      all(r["label_1000way"] == IM.IMAGENETTE_CLASSES[r["label_10way"]].imagenet_index
          for r in rows1))
check("smoke selection is not simply the first 10 files",
      any(int(r["rel_path"][-8:-5]) >= 10 for r in rows1))
rows3 = IM.build_manifest(ds, "smoke", per_class=10, seed=1)
check("a different seed selects a different subset", rows3 != rows1)
full = IM.build_manifest(ds, "full")
check("full manifest = whole split, sorted", len(full) == 130
      and [r["rel_path"] for r in full] == sorted(r["rel_path"] for r in full))
back, shab = IM.read_manifest(TMP / "m1.csv")
check("manifest round trip", shab == sha1 and back[5]["label_1000way"] == rows1[5]["label_1000way"])
(ds / "val" / "n99999999").mkdir()
try:
    IM.list_split(ds)
    check("unexpected class folder rejected", False)
except RuntimeError:
    check("unexpected class folder rejected", True)
(ds / "val" / "n99999999").rmdir()
try:
    IM.list_split(TMP / "nope")
    check("missing dataset -> error, no synthetic fallback", False)
except FileNotFoundError:
    check("missing dataset -> error, no synthetic fallback", True)

# --------------------------------------------------------------------------- #
print("\nTest 4 — preprocessing (Cream TinyViT eval transform)")
# --------------------------------------------------------------------------- #
img = Image.fromarray(rng.integers(0, 256, (160, 213, 3), dtype=np.uint8))
img.save(TMP / "p.png")
u = IM.load_uint8(TMP / "p.png")
ref = img.resize((int(256 * 213 / 160), 256), Image.BICUBIC)
top, left = int(round((256 - 224) / 2)), int(round((ref.size[0] - 224) / 2))
ref = np.asarray(ref.crop((left, top, left + 224, top + 224)))
check("resize short side 256 (bicubic) + centre crop 224", u.shape == (224, 224, 3)
      and np.array_equal(u, ref))
x = IM.to_model_input(np.full((1, 224, 224, 3), 255, np.uint8))
exp = (1.0 - np.array(IM.MEAN)) / np.array(IM.STD)
check("ToTensor + ImageNet normalisation", tuple(x.shape) == (1, 3, 224, 224)
      and np.allclose(x[0, :, 0, 0].numpy(), exp, atol=1e-6))
cache = TMP / "inputs.npy"
IM.build_input_cache(ds, rows1[:5], cache)
arr = np.load(cache, mmap_mode="r")
check("uint8 input cache equals direct decoding",
      arr.shape == (5, 224, 224, 3) and np.array_equal(arr[3], IM.load_uint8(ds / rows1[3]["rel_path"])))

# --------------------------------------------------------------------------- #
print("\nTest 5 — hardware matrix, job planning, cache keys")
# --------------------------------------------------------------------------- #
mat = HWM.primary_matrix()
check("24 hardware configurations, unique labels",
      len(mat) == 24 and len({h.label for h in mat}) == 24)
check("matrix = 4 precisions x ADC {8,10,12} x 2 gain sets",
      {(h.weight_bits, h.act_bits) for h in mat} == set(HWM.PRECISIONS)
      and {h.adc_bits for h in mat} == {8, 10, 12}
      and {h.gain_set for h in mat} == {"gain1", "cal_w8a8"})
check("calibrated gains are 23.77 / 8.41 / 2.92",
      HWM.GAIN_SETS["cal_w8a8"] == {"qkv": 23.77, "qk": 8.41, "av": 2.92})
check("fixed: capacitive 8-bit DAC @ 0.6 V, ADC Vref 0.6 V, MSB alignment",
      all(h.dac_type == "CAPACITIVE" and h.dac_bits == 8 and h.dac_vref == 0.6
          and h.adc_vref == 0.6 and h.dac_alignment == "msb" for h in mat))
jm, jb = HWM.planned_cim_jobs("matched"), HWM.planned_cim_jobs("both")
check("matched: 48 CIM jobs; QAT chosen by weight bits",
      len(jm) == 48 and all((m == "pretrained") or (m == HWM.qat_model_for(h.weight_bits))
                            for m, h in jm))
check("both: 72 CIM jobs", len(jb) == 72)
cfg = mat[0].cim_config()
check("hardware config enables the analog chain, legacy ADC off",
      cfg.analog.enabled and not cfg.adc_enabled and cfg.analog.dac_bits == 8)
ctx = HWM.RunContext("t", "m" * 64, IM.PREPROCESS_ID, "c" * 64, "2.x", "2.x")
k0 = HWM.cim_key(ctx, "qat_int4", "a" * 64, mat[0])
check("cache key is stable", k0 == HWM.cim_key(ctx, "qat_int4", "a" * 64, mat[0]))
variants = [HWM.cim_key(ctx, "qat_int8", "a" * 64, mat[0]),
            HWM.cim_key(ctx, "qat_int4", "b" * 64, mat[0]),
            HWM.cim_key(ctx, "qat_int4", "a" * 64, mat[1])]
import dataclasses
for f, v in (("adc_vref", 0.5), ("dac_vref", 0.5), ("dac_bits", 10), ("dac_alignment", "lsb"),
             ("token_block", 8), ("amp_swing_limit_v", 0.3), ("cim_out_proj", True)):
    variants.append(HWM.cim_key(ctx, "qat_int4", "a" * 64, dataclasses.replace(mat[0], **{f: v})))
for f, v in (("manifest_sha256", "x"), ("preprocess_id", "y"), ("code_sha256", "z"),
             ("torch_version", "9"), ("chunk_size", 7), ("threads", 2), ("batch_size", 4)):
    variants.append(HWM.cim_key(dataclasses.replace(ctx, **{f: v}), "qat_int4", "a" * 64, mat[0]))
check("every output-affecting setting changes the cache key",
      len(set(variants + [k0])) == len(variants) + 1, f"{len(variants)} variants")
fp = HWM.code_fingerprint(HERE)
check("code fingerprint is CRLF-insensitive (normalised)", len(fp) == 64)

# --------------------------------------------------------------------------- #
print("\nTest 6 — extraction guard")
# --------------------------------------------------------------------------- #
evil = TMP / "evil.tgz"
with tarfile.open(evil, "w:gz") as tf:
    data = b"x"
    ti = tarfile.TarInfo("../escape.txt")
    ti.size = len(data)
    tf.addfile(ti, io.BytesIO(data))
try:
    IM.extract(evil, TMP / "ex", "imagenette2-160")
    check("path traversal in archive refused", False)
except RuntimeError:
    check("path traversal in archive refused", not (TMP / "escape.txt").exists())

# --------------------------------------------------------------------------- #
print("\nTest 7 — five-case reporting on fabricated cached outputs (logic only)")
# --------------------------------------------------------------------------- #
rd = TMP / "run"
(rd / "cache" / "reference").mkdir(parents=True)
IM.write_manifest(rows1, rd / "input_manifest.csv")
_, msha = IM.read_manifest(rd / "input_manifest.csv")
run_cfg = dict(stage="smoke", variant="imagenette2-160", manifest_sha256=msha,
               preprocess_id=IM.PREPROCESS_ID, n_images=100)
hashes = {"pretrained": "p" * 64, "qat_int4": "q" * 64, "qat_int8": "r" * 64}
tb = {"pretrained": None, "qat_int4": 4, "qat_int8": 8}
ctx = HWM.RunContext("smoke", msha, IM.PREPROCESS_ID, "c" * 64, "t", "n")
ref_keys = {m: HWM.reference_key(ctx, m, hashes[m]) for m in hashes}
base = rng.standard_normal((100, 1000)).astype(np.float32)
for m, k in ref_keys.items():
    np.savez(rd / "cache" / "reference" / f"{k}.npz",
             logits=base + 0.1 * rng.standard_normal((100, 1000)).astype(np.float32))
    (rd / "cache" / "reference" / f"{k}.json").write_text(json.dumps(
        dict(model=m, checkpoint_sha256=hashes[m])))
cim_keys = {}
zero_ops = {op: dict(n=1000, clip_hi=1, clip_lo=0, rail_hi=1, rail_lo=0, amp_sat=0,
                     clip_pct=0.1) for op in HWM.OPS}
for m, hw in HWM.planned_cim_jobs("matched"):
    k = HWM.cim_key(ctx, m, hashes[m], hw)
    cim_keys[(m, hw.label)] = (k, hw)
    jd = rd / "cache" / "cim" / k
    jd.mkdir(parents=True)
    np.savez(jd / "final.npz", logits=base + 0.3 * rng.standard_normal((100, 1000)).astype(np.float32))
    (jd / "final.json").write_text(json.dumps(dict(
        n_images=100, cim_stats=dict(adc_conversions=3000, adc_saturations=3),
        analog_summary=zero_ops, images_with_any_clipping=2,
        images_with_clipping_per_op={op: 1 for op in HWM.OPS}, conversions_per_image=30,
        consistency_ok=True, wall_s_total=1.0, status="OK",
        chain=dict(amp_saturation_modeled=False))))
R = RP.Results(rd, run_cfg, hashes, tb, ref_keys, cim_keys)
t = RP.build_tables(R, "matched")
cov = RP.coverage(R, t, "matched")
check("coverage complete with expected rows per case",
      cov["complete"] and cov["counts"] == {"CASE1": 24, "CASE2": 2, "CASE2b": 1,
                                            "CASE3": 24, "CASE4": 24, "CASE5": 24},
      f"{cov['counts']}")
check("raw metrics: 3 FP32 + 48 CIM rows", len(t["raw"]) == 51)
one_named = all(sum(bool(r[c] != "") for c in RP.AGREE_COLS) == 1
                and sum(bool(r[c] != "") for c in RP.ERR_COLS) == 1 for r in t["pair"])
check("each comparison fills exactly ONE named agreement and ONE named error column",
      one_named)
check("no reference-free agreement column exists",
      not any(k == "agreement_pct" or k == "agreement" for r in t["pair"] for k in r))
c3 = [r for r in t["pair"] if r["case"] == "CASE3"]
check("CASE3 reference = same QAT checkpoint FP32",
      all(r["reference_model"] == r["tested_model"] and r["reference_execution_path"] == "fp32_software"
          and r["tested_execution_path"] == "cim_analog" for r in c3))
c5 = [r for r in t["pair"] if r["case"] == "CASE5"]
check("CASE5 reference = pretrained FP32, tested = QAT CIM",
      all(r["reference_model"] == "pretrained" and r["tested_model"].startswith("qat") for r in c5))
c1 = [r for r in t["pair"] if r["case"] == "CASE1"]
check("CASE1 = pretrained CIM vs QAT CIM under the SAME hardware label",
      all(r["reference_execution_path"] == r["tested_execution_path"] == "cim_analog"
          and r["reference_run_id"].split("_W")[1].rsplit("_", 1)[0]
          == r["tested_run_id"].split("_W")[1].rsplit("_", 1)[0] for r in c1))
c3r = c3[0]
lgq = np.load(rd / "cache" / "reference" / f"{ref_keys[c3r['reference_model']]}.npz")["logits"]
lgc = np.load(rd / "cache" / "cim" / c3r["tested_run_id"] / "final.npz")["logits"]
check("CASE3 metric recomputed independently matches the table",
      abs(c3r["rel_logit_rms_err_pct_vs_own_fp32"] - M.rel_rms_pct(lgc, lgq)) < 1e-9
      and abs(c3r["agreement_vs_own_qat_fp32_pct"] - M.agreement_pct(lgc, lgq)) < 1e-9)
check("every row traceable (run ids, checkpoint hashes, manifest)",
      not cov["untraceable_rows"])
env = dict(code_sha256="c" * 64, torch="t", numpy="n")
cov2 = RP.write_all(R, rd / "out", "matched", env, {m: {"sha256": h} for m, h in hashes.items()},
                    [dict(run_id="x")])
files = {p.name for p in (rd / "out").iterdir()}
need = {"run_manifest.csv", "raw_metrics.csv", "pairwise_comparisons.csv",
        "accuracy_summary.csv", "cross_entropy_summary.csv", "agreement_summary.csv",
        "logit_error_summary.csv", "clipping_summary.csv", "environment.json", "REPORT.md"}
check("all required result files written", need <= files, f"missing {need - files}")
class _FakeR:                     # minimal stand-in for Results (mapping check only)
    pass


fr = _FakeR()
fr.y10 = np.repeat(np.arange(10), 5)
lg_map = rng.standard_normal((50, 1000)) * 0.1
for i in range(50):
    lg_map[i, IM.IMAGENET_INDICES[fr.y10[i]]] = 5.0          # correct class wins the 10-way
lg_map[fr.y10 == 2, 848] = 9.0       # 'tape player' wins the 1000-way for cassette players
fr.refs = {"pretrained": {"logits": lg_map}}
cm = RP.class_mapping_check(fr)
check("mapping check passes when a near-synonym wins the 1000-way argmax",
      all(r["mapping_confirmed"] for r in cm)
      and cm[2]["modal_prediction_1000way_info"] == 848)
lg_bad = lg_map.copy()
lg_bad[fr.y10 == 9, 701] = -5.0      # images of 'parachute' do not score on 701 ...
lg_bad[fr.y10 == 9, 0] = 6.0         # ... (as if the index were wrong)
cm_bad = RP.class_mapping_check(type("R", (), {"y10": fr.y10,
                                              "refs": {"pretrained": {"logits": lg_bad}}})())
check("mapping check fails when a class's mapped index is wrong",
      not cm_bad[9]["mapping_confirmed"] and all(r["mapping_confirmed"] for r in cm_bad[:9]))
shutil.rmtree(rd / "cache" / "cim" / cim_keys[("qat_int8", "W8A8_ADC12_gain1")][0])
R2 = RP.Results(rd, run_cfg, hashes, tb, ref_keys, cim_keys)
cov3 = RP.coverage(R2, RP.build_tables(R2, "matched"), "matched")
check("a missing job makes coverage incomplete", not cov3["complete"]
      and "qat_int8:W8A8_ADC12_gain1" in cov3["missing_jobs"])

# --------------------------------------------------------------------------- #
print("\nTest 8 — engine end-to-end on 1 image (needs the pretrained checkpoint)")
# --------------------------------------------------------------------------- #
ck = HERE / "tinyvit5m.pth"
if not ck.exists():
    skip("engine end-to-end", "tinyvit5m.pth not present")
else:
    from cim_exp import engine as EN
    jd = TMP / "job"
    jd.mkdir()
    hw = HWM.HardwareConfig(weight_bits=4, act_bits=4, adc_bits=8, gain_set="cal_w8a8")
    unit = dict(repo_root=str(HERE), job_key="t8", job_dir=str(jd), hw=hw.as_dict(),
                ckpt_path=str(ck), chunk=0, i0=0, i1=1, inputs_path=str(cache),
                stats_keep_frac=2e-4, stats_cap=400_000)
    EN.run_cim_chunk(unit)
    out = EN.finalize_job(jd, 1, {}, 400_000)
    check("engine: finalized OK, consistency checks pass, analog active",
          out["status"] == "OK" and out["consistency_ok"] and out["cim_stats"]["adc_conversions"] > 0
          and out["chain"]["dac"]["bits"] == 8, f"{out['consistency']}")
    import torch
    import run_tinyvit
    from cim_sim import CIMArray
    from cim_sim.attention import convert_attention, set_cim_ops
    xin = IM.to_model_input(np.load(cache)[0:1])
    with torch.no_grad():
        r = run_tinyvit.build_model(ck)(xin)
        m = run_tinyvit.build_model(ck)
        convert_attention(m, CIMArray(hw.cim_config()))
        set_cim_ops(m, set())
        o = m(xin)
    check("converted model with every CIM site off == FP32 model, bit-exact",
          torch.equal(o, r))

# --------------------------------------------------------------------------- #
print("\nTest 9 — amplifier option, custom gains, per-op reporting columns")
# --------------------------------------------------------------------------- #
hw_a = HWM.HardwareConfig(weight_bits=8, act_bits=8, adc_bits=8, gain_set="cal_w8a8")
check("primary default: amplifier swing limit None (not modelled), label unchanged",
      hw_a.amp_swing_limit_v is None and hw_a.label == "W8A8_ADC8_cal_w8a8"
      and hw_a.cim_config().analog.gain_non.v_swing_limit is None)
hw_l = dataclasses.replace(hw_a, amp_swing_limit_v=0.25)
check("configurable swing limit reaches the amplifier and the job label",
      hw_l.cim_config().analog.gain_non.v_swing_limit == 0.25 and hw_l.label.endswith("_amp0.25V")
      and all(h.amp_swing_limit_v == 0.25 for _, h in HWM.planned_cim_jobs("matched", 0.25)))
pc = RP.per_op_columns({op: dict(n=200, clip_hi=1, clip_lo=1, amp_sat=0, util_p999_pct=50.0,
                                 util_max_pct=90.0, vadc_absp999=0.1, voa_absp999=0.01)
                        for op in HWM.OPS}, amp_modeled=False)
check("per-op columns: clip % per operation; amplifier saturation 'not_modeled' (not 0)",
      pc["adc_clip_pct_qkv"] == 1.0 and pc["amp_saturation_pct_av"] == "not_modeled"
      and abs(pc["vadc_absp999_mv_qk"] - 100.0) < 1e-9)
hw_c = HWM.HardwareConfig(weight_bits=4, act_bits=4, adc_bits=8, gain_set="calB_x",
                          custom_gains=(10.0, 5.0, 2.0), gain_note="test")
check("custom gains: values, dict round trip, distinct cache key",
      hw_c.gains == {"qkv": 10.0, "qk": 5.0, "av": 2.0}
      and HWM.HardwareConfig.from_dict(hw_c.as_dict()) == hw_c
      and HWM.cim_key(ctx, "pretrained", "p" * 64, hw_c)
      != HWM.cim_key(ctx, "pretrained", "p" * 64, dataclasses.replace(hw_c, custom_gains=(10.0, 5.0, 2.5))))
check("Experiment A matrix never contains custom gains",
      all(h.custom_gains is None and h.gain_set in HWM.GAIN_SETS for h in HWM.primary_matrix()))

# --------------------------------------------------------------------------- #
print("\nTest 10 — Experiment B calibration arithmetic and job planning")
# --------------------------------------------------------------------------- #
from cim_exp import calibration as CB
h = np.zeros(1001, np.int64)
h[:1000] = 10                      # 10 000 colsums, |P| uniform on 0..999
h[1000] = 1                        # one outlier at 1000
n = h.sum()
fs99 = CB.full_scale(h, 99.0)
check("FS(q) = smallest FS with count(|P| > FS) <= n(1-q/100)",
      (h[fs99 + 1:].sum() <= np.floor(n * 0.01)) and (h[fs99:].sum() > np.floor(n * 0.01)),
      f"FS99={fs99}")
check("FS(p100) = largest observed |P|", CB.full_scale(h, 100.0) == 1000)
km = CB.k_mac(8, 8, "qkv")
check("K_MAC(W8A8) matches CLAUDE.md (2.8610e-07)", abs(km - 2.8610e-07) / 2.861e-07 < 1e-3)
check("K_MAC(W4A4, MSB-aligned) = 0.6/16/(32*16)", abs(CB.k_mac(4, 4, "av") - 0.6 / 16 / 512) < 1e-15)
g = CB.gain_for_fs(fs99, km)
check("gain formula G = (Vref/2)/(K_MAC*FS); predicted clip at that gain <= target",
      abs(g * km * fs99 - 0.3) < 1e-12 and CB.predicted_clip_pct(h, g, km) <= 1.0)
check("higher gain predicts more clipping (monotone)",
      CB.predicted_clip_pct(h, 2 * g, km) > CB.predicted_clip_pct(h, g, km))
obs = dict(hist={op: h for op in HWM.OPS}, maxabs={op: 1000 for op in HWM.OPS}, n_images=1)
cal1 = CB.calibrate(obs, 8, 8)
check("calibrate(): 4 levels, gains rounded to 0.01, fixed-set prediction present",
      set(cal1["levels"]) == set(CB.LEVELS)
      and all(round(v, 2) == v for v in cal1["levels"]["p99.99"]["gains"].values())
      and "cal_w8a8" in cal1["predicted_clip_pct"] and "gain1" in cal1["predicted_clip_pct"])
check("primary objective is pre-declared P99.99 (clip target 0.01 %)",
      CB.PRIMARY_LEVEL == "p99.99" and CB.LEVEL_TARGET_PCT["p99.99"] == 0.01)
cal_all = {}
for src in ("pretrained", "qat"):
    for (w, a_) in HWM.PRECISIONS:
        c = CB.calibrate(obs, w, a_)
        c["cal_model"] = CB.model_for(src, w)
        cal_all[(src, f"W{w}A{a_}")] = c
plan = CB.plan_jobs(cal_all, [8], list(CB.LEVELS), "c" * 64)
kinds = {}
for t_, hw_, pv in plan:
    kinds[pv["kind"]] = kinds.get(pv["kind"], 0) + 1
check("plan: per source x precision 4 levels matched, 3 precision transfers, A baselines",
      kinds == {"experiment_A_baseline": 16, "calibrated_matched": 32,
                "calibrated_precision_transfer": 6}, f"{kinds}")
check("plan: calibrated gains applied only at their own precision (except _xfer rows)",
      all(hw_.gain_set.endswith("_xfer") or hw_.gain_set in HWM.GAIN_SETS
          or f"W{hw_.weight_bits}A{hw_.act_bits}" in hw_.gain_set for _, hw_, _ in plan))
check("plan: A-baselines identical to Experiment A configs (shared cache keys)",
      {h_.label for _, h_, p in plan if p["kind"] == "experiment_A_baseline"}
      <= {h_.label for h_ in HWM.primary_matrix()})
plan_x = CB.plan_jobs(cal_all, [8], ["p99.99"], "c" * 64, cross_apply=True)
check("cross-apply adds primary-level checkpoint-transfer rows",
      any(p["kind"] == "calibrated_cross_checkpoint" for _, _, p in plan_x))

# --------------------------------------------------------------------------- #
print("\nTest 11 — exact calibration probe == legacy exact path (needs checkpoint)")
# --------------------------------------------------------------------------- #
if not ck.exists():
    skip("probe exactness", "tinyvit5m.pth not present")
else:
    import torch
    import run_tinyvit
    from cim_sim import CIMArray, CIMConfig
    from cim_sim.attention import convert_attention
    xin = IM.to_model_input(np.load(cache)[0:1])
    with torch.no_grad():
        m1 = run_tinyvit.build_model(ck)
        convert_attention(m1, CIMArray(CIMConfig(weight_bits=4, act_bits=8)))
        r1 = m1(xin)
        m2 = run_tinyvit.build_model(ck)
        arr = CIMArray(HWM.HardwareConfig(4, 8, 8, "gain1").cim_config())
        arr.chain = CB.ProbeChain(CB.hist_range(4, 8))
        convert_attention(m2, arr)
        r2 = m2(xin)
    arr.chain.flush()
    check("probe network output == legacy ADC-off output, bit-exact; all ops histogrammed",
          torch.equal(r1, r2) and set(arr.chain.hist) == set(HWM.OPS)
          and sum(int(v.sum()) for v in arr.chain.hist.values()) == arr.stats.adc_conversions)

shutil.rmtree(TMP, ignore_errors=True)
print(f"\n{sum(PASS)}/{len(PASS)} checks passed, {len(SKIP)} skipped")
sys.exit(0 if all(PASS) else 1)
