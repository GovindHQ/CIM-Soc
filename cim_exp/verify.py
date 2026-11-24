"""
verify.py — the pre-flight validation suite (run_experiment.py verify).

Each check reports PASS / FAIL / SKIP with a detail string. Nothing is hidden:
a failing check fails the command.

  unit:*                     verify_array.py, test_analog.py, test_multi_array.py,
                             test_experiment_pipeline.py (subprocesses, exit codes)
  stage3_smoke               run_tinyvit.py --stage3-only (legacy path) runs
  dac_transfer_equation      unsigned 8-bit CDAC: V = 0.6*c/256, bit by bit
  signed_input_handling      offset binary on the unsigned DAC; AV unsigned
  checkpoint_loading         3 checkpoints: hash, strict build, inspection
  fp32_inference_finite      3 FP32 models on 4 real images: finite logits
  analog_path_active         chain present, conversions > 0, output differs from
                             the legacy digital path and from FP32
  smoke_W4A4 / smoke_W8A8    2 real images, gain1 + calibrated, consistent stats
  gain1_vs_calibrated        LSB_P scales as 1/G per operation, outputs differ
  adc_conversion_count       counted conversions == scheduler geometry
  determinism                same unit twice -> bit-identical logits + counters
  metric_definitions         hand-computed metric values
  reference_identity         cached references carry the registry hashes
  class_mapping_static       table vs bundled ImageNet-1k synset list
  class_mapping_empirical    pretrained FP32 10-way modal prediction per class
                             (1000-way modal reported for information)
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def _res(status, detail="", **extra):
    return dict(status=status, detail=detail, **extra)


def _run_script(repo: Path, args: list[str], timeout=3600):
    t = time.time()
    p = subprocess.run([sys.executable] + args, cwd=repo, capture_output=True,
                       text=True, timeout=timeout)
    tail = "\n".join((p.stdout + p.stderr).strip().splitlines()[-4:])
    return p.returncode, tail, time.time() - t


def run_all(repo: Path, a, run_cfg: dict, reg: dict, hashes: dict, run_dir: Path) -> dict:
    from . import checkpoints as CK
    from . import engine as EN
    from . import hardware as HWM
    from . import imagenette as IM
    from . import metrics as M

    EN._setup_paths(repo)
    EN.configure_threads(1)
    import torch
    checks = {}

    def record(name, fn):
        print(f"[verify] {name} ...", flush=True)
        try:
            checks[name] = fn()
        except Exception as e:                       # a crash is a FAIL, not a skip
            checks[name] = _res("FAIL", f"{type(e).__name__}: {e}")
        print(f"         {checks[name]['status']}  {checks[name]['detail'][:300]}", flush=True)

    # ---- unit tests ---------------------------------------------------- #
    for script in ("verify_array.py", "test_analog.py", "test_multi_array.py",
                   "test_experiment_pipeline.py"):
        def f(script=script):
            rc, tail, dt = _run_script(repo, [script])
            return _res("PASS" if rc == 0 else "FAIL", f"exit {rc} in {dt:.0f}s | {tail}")
        record(f"unit:{script}", f)

    def stage3():
        rc, tail, dt = _run_script(repo, ["run_tinyvit.py", "--stage3-only"])
        return _res("PASS" if rc == 0 else "FAIL", f"exit {rc} | {tail}")
    record("stage3_smoke", stage3)

    # ---- DAC + signed input ---------------------------------------------- #
    def dac_eq():
        from cim_sim.analog import CapacitiveDAC
        d = CapacitiveDAC(bits=8, vref=0.6)
        c = np.arange(256)
        v = d.vout_from_unsigned_code(c)
        bitsum = np.array([0.6 * sum(((k >> b) & 1) / 2 ** (8 - b) for b in range(8))
                           for k in c])
        ok = (np.allclose(v, bitsum, rtol=0, atol=1e-15) and v[0] == 0
              and abs(v[255] - 0.6 * 255 / 256) < 1e-15 and np.all(np.diff(v) > 0))
        return _res("PASS" if ok else "FAIL",
                    f"V(0)={v[0]} V(255)={v[255]!r} (=0.6*(1-2^-8)), step={v[1]:.6g} V")
    record("dac_transfer_equation", dac_eq)

    def signed():
        from cim_sim.analog import CapacitiveDAC
        d = CapacitiveDAC(bits=8, vref=0.6)
        x = np.arange(-128, 128)
        ok = np.array_equal(d.unsigned_code(x), x + 128) and np.allclose(
            d.transfer(x), d.vout_from_unsigned_code(x + 128) - 0.3, rtol=0, atol=1e-15)
        d.set_input_signed(False)
        u = np.arange(256)
        ok = ok and np.array_equal(d.unsigned_code(u), u) and d.v_common_mode == 0.0
        try:
            d.transfer(np.array([-1]))
            ok = False
        except ValueError:
            pass
        return _res("PASS" if ok else "FAIL",
                    "signed: c = x + 128 then V - V_REF/2 (simulator convention); "
                    "AV unsigned: c = x, no offset; negative code rejected in unsigned domain")
    record("signed_input_handling", signed)

    # ---- checkpoints ------------------------------------------------------ #
    def ck():
        pre = torch.load(reg["pretrained"].path, map_location="cpu", weights_only=True)
        pre = pre.get("model", pre)
        det = []
        ok = True
        for mid, spec in reg.items():
            rep = CK.inspect(spec, None if mid == "pretrained" else pre)
            model = CK.build(spec)
            n_att = sum(type(m).__name__ == "Attention" for m in model.modules())
            ok &= rep["head_shape"] == [1000, 320] and n_att == 10
            if mid != "pretrained":
                ok &= bool(rep["same_keys_as_pretrained"]) and not rep["shape_mismatches"]
            det.append(f"{mid}: sha {hashes[mid][:12]}, {rep['n_tensors']} tensors, "
                       f"quantizer keys {rep['n_quantizer_keys']}, int4 off-grid "
                       f"{rep['qkv_grid_test']['int4_mean_offgrid']:.3f}")
        return _res("PASS" if ok else "FAIL", "; ".join(det))
    record("checkpoint_loading", ck)

    u8 = np.load(run_cfg["inputs_cache"], mmap_mode="r")

    def fp32():
        det, ok = [], True
        for mid, spec in reg.items():
            model = CK.build(spec)
            with torch.no_grad():
                lg = np.stack([model(IM.to_model_input(u8[i:i + 1])).numpy()[0]
                               for i in range(min(4, len(u8)))])
            fin = bool(np.isfinite(lg).all())
            ok &= fin
            det.append(f"{mid}: finite={fin} argmax={lg.argmax(1).tolist()}")
        return _res("PASS" if ok else "FAIL", "; ".join(det))
    record("fp32_inference_finite", fp32)

    # ---- CIM units on 2 real images --------------------------------------- #
    tmp = run_dir / "verify_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    def unit(mid, hw, tag, n=2):
        jd = tmp / tag
        jd.mkdir()
        u = dict(repo_root=str(repo), job_key=tag, job_dir=str(jd), hw=hw.as_dict(),
                 ckpt_path=str(reg[mid].path), chunk=0, i0=0, i1=min(n, len(u8)),
                 inputs_path=run_cfg["inputs_cache"], stats_keep_frac=2e-4,
                 stats_cap=400_000)
        EN.run_cim_chunk(u)
        out = EN.finalize_job(jd, 1, dict(model=mid, hw=hw.as_dict()), 400_000,
                              keep_chunks=True)
        z = np.load(jd / "final.npz")
        return out, z["logits"], z["per_image"]

    H = HWM.HardwareConfig
    smoke = {}

    def smoke_fn(w, a_):
        def f():
            det, ok = [], True
            for g in HWM.GAIN_SETS:
                for mid in ("pretrained", HWM.qat_model_for(w)):
                    hw = H(weight_bits=w, act_bits=a_, adc_bits=8, gain_set=g)
                    out, lg, _ = unit(mid, hw, f"smoke_{mid}_{hw.label}")
                    smoke[(mid, hw.label)] = (out, lg)
                    ok &= out["status"] == "OK" and bool(np.isfinite(lg).all())
                    s = out["analog_summary"]
                    det.append(f"{mid} {hw.label}: status {out['status']}, conv/img "
                               f"{out['conversions_per_image']:,}, clip% " +
                               "/".join(f"{s[o]['clip_pct']:.3f}" for o in HWM.OPS))
            return _res("PASS" if ok else "FAIL", "; ".join(det))
        return f
    record("smoke_W4A4", smoke_fn(4, 4))
    record("smoke_W8A8", smoke_fn(8, 8))

    def active():
        import run_tinyvit
        from cim_sim import CIMArray, CIMConfig
        from cim_sim.attention import convert_attention
        hw = H(weight_bits=8, act_bits=8, adc_bits=8, gain_set="cal_w8a8")
        out, lg = smoke.get(("pretrained", hw.label)) or unit("pretrained", hw, "active")[:2]
        m_leg = run_tinyvit.build_model(reg["pretrained"].path)
        convert_attention(m_leg, CIMArray(CIMConfig(weight_bits=8, act_bits=8)))
        m_fp = run_tinyvit.build_model(reg["pretrained"].path)
        with torch.no_grad():
            x = IM.to_model_input(u8[0:2])
            leg = np.stack([m_leg(x[i:i + 1]).numpy()[0] for i in range(2)])
            fp = np.stack([m_fp(x[i:i + 1]).numpy()[0] for i in range(2)])
        d_leg = float(np.abs(lg - leg).max())
        d_fp = float(np.abs(lg - fp).max())
        ok = (out["cim_stats"]["adc_conversions"] > 0 and d_leg > 0 and d_fp > 0
              and out["chain"]["dac"]["name"] == "capacitive"
              and out["chain"]["dac"]["bits"] == 8 and out["chain"]["dac"]["vref"] == 0.6
              and out["chain"]["adc"]["vref"] == 0.6)
        return _res("PASS" if ok else "FAIL",
                    f"conversions {out['cim_stats']['adc_conversions']:,}; max|analog - legacy "
                    f"digital| = {d_leg:.4g}; max|analog - FP32| = {d_fp:.4g}; DAC "
                    f"{out['chain']['dac']['name']} {out['chain']['dac']['bits']}b "
                    f"{out['chain']['dac']['vref']} V, ADC Vref {out['chain']['adc']['vref']} V")
    record("analog_path_active", active)

    def gains():
        from cim_sim.analog import AnalogChain
        det, ok = [], True
        for w, a_ in HWM.PRECISIONS:
            c1 = H(weight_bits=w, act_bits=a_, adc_bits=8, gain_set="gain1").cim_config()
            c2 = H(weight_bits=w, act_bits=a_, adc_bits=8, gain_set="cal_w8a8").cim_config()
            ch1 = AnalogChain(c1.analog, 32, w, a_)
            ch2 = AnalogChain(c2.analog, 32, w, a_)
            for op in HWM.OPS:
                ch1.set_operation(op, act_signed=op != "av")
                ch2.set_operation(op, act_signed=op != "av")
                r = ch1.lsb_p / ch2.lsb_p
                ok &= abs(r - HWM.GAIN_SETS["cal_w8a8"][op]) < 1e-9
        k1 = ("pretrained", "W8A8_ADC8_gain1")
        k2 = ("pretrained", "W8A8_ADC8_cal_w8a8")
        if k1 in smoke and k2 in smoke:
            diff = float(np.abs(smoke[k1][1] - smoke[k2][1]).max())
            ok &= diff > 0
            det.append(f"W8A8 logits differ between gain sets (max |d| {diff:.4g})")
        det.insert(0, "LSB_P(gain1)/LSB_P(cal) == G_cal for every op and precision")
        return _res("PASS" if ok else "FAIL", "; ".join(det))
    record("gain1_vs_calibrated", gains)

    def conv_count():
        import run_tinyvit
        from cim_sim import CIMArray, TraceLog
        from cim_sim.attention import convert_attention
        det, ok = [], True
        for w, a_ in ((4, 4), (8, 8)):
            hw = H(weight_bits=w, act_bits=a_, adc_bits=8, gain_set="cal_w8a8")
            m = run_tinyvit.build_model(reg["pretrained"].path)
            arr = CIMArray(hw.cim_config())
            log = TraceLog()
            convert_attention(m, arr, log=log)
            with torch.no_grad():
                m(IM.to_model_input(u8[0:1]))
            expected = sum(r.N * r.depth_blocks * r.output_blocks * 32 for r in log.records)
            counted = sum(v["n"] for v in arr.chain.stats.summary().values())
            per_img = smoke.get(("pretrained", hw.label))
            pi = per_img[0]["conversions_per_image"] if per_img else counted
            ok &= expected == counted == arr.stats.adc_conversions == pi
            det.append(f"W{w}A{a_}: geometry {expected:,} = counted {counted:,} = "
                       f"CIMStats {arr.stats.adc_conversions:,} = engine {pi:,}")
        return _res("PASS" if ok else "FAIL", "; ".join(det))
    record("adc_conversion_count", conv_count)

    def determinism():
        hw = H(weight_bits=4, act_bits=8, adc_bits=10, gain_set="cal_w8a8")
        o1, l1, p1 = unit("qat_int4", hw, "det_a")
        o2, l2, p2 = unit("qat_int4", hw, "det_b")
        ok = np.array_equal(l1, l2) and np.array_equal(p1, p2)
        return _res("PASS" if ok else "FAIL",
                    f"qat_int4 {hw.label} twice: logits identical={np.array_equal(l1, l2)}, "
                    f"per-image counters identical={np.array_equal(p1, p2)}")
    record("determinism", determinism)

    def metric_defs():
        lg = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0], [1.0, 1.0, 5.0]])
        y = np.array([0, 2, 2])
        ce_hand = np.mean([-np.log(np.exp(2) / np.exp([2, 1, 0]).sum()),
                           -np.log(np.exp(1) / np.exp([0, 3, 1]).sum()),
                           -np.log(np.exp(5) / np.exp([1, 1, 5]).sum())])
        ok = (abs(M.top1_pct(lg, y) - 200 / 3) < 1e-12
              and abs(M.cross_entropy(lg, y) - ce_hand) < 1e-12
              and abs(M.agreement_pct(lg, lg[[0, 2, 1]]) - 100 / 3) < 1e-9
              and abs(M.rel_rms_pct(lg * 1.1, lg) - 10.0) < 1e-9)
        return _res("PASS" if ok else "FAIL",
                    f"top1 66.67%, CE {ce_hand:.6f} nats (log-softmax once), agreement, rel RMS")
    record("metric_definitions", metric_defs)

    def ref_ident():
        d = run_dir / "cache" / "reference"
        if not d.exists() or not list(d.glob("*.json")):
            return _res("SKIP", "no cached references yet (run `reference` then `report`; "
                                "report re-checks identities)")
        bad = []
        for j in d.glob("*.json"):
            meta = json.loads(j.read_text())
            if meta["model"] in hashes and meta["checkpoint_sha256"] != hashes[meta["model"]]:
                bad.append(j.name)
        return _res("FAIL" if bad else "PASS",
                    f"stale/mismatched: {bad}" if bad else "all cached references match registry hashes")
    record("reference_identity", ref_ident)

    def cmap_static():
        notes = IM.verify_mapping_tables()
        return _res("PASS", "; ".join(notes))
    record("class_mapping_static", cmap_static)

    def cmap_emp():
        from .imagenette import IMAGENET_INDICES, IMAGENETTE_CLASSES, read_manifest
        rows, _ = read_manifest(run_dir / "input_manifest.csv")
        y10 = np.array([r["label_10way"] for r in rows])
        model = CK.build(reg["pretrained"])
        with torch.no_grad():
            lg = np.stack([model(IM.to_model_input(u8[i:i + 1])).numpy()[0]
                           for i in range(len(rows))])
        pred10 = lg[:, IMAGENET_INDICES].argmax(1)
        pred = lg.argmax(1)
        det, ok = [], True
        for i, c in enumerate(IMAGENETTE_CLASSES):
            m = y10 == i
            if not m.any():
                continue
            v10, c10 = np.unique(pred10[m], return_counts=True)
            modal10 = int(v10[np.argmax(c10)])
            v, cn = np.unique(pred[m], return_counts=True)
            ok &= modal10 == i
            det.append(f"{c.name}: 10-way modal {modal10} ({c10.max()}/{m.sum()}) "
                       f"{'OK' if modal10 == i else 'MISMATCH'}; 1000-way modal "
                       f"{int(v[np.argmax(cn)])} (info), mapped {c.imagenet_index}")
        return _res("PASS" if ok else "FAIL", "; ".join(det))
    record("class_mapping_empirical", cmap_emp)

    shutil.rmtree(tmp, ignore_errors=True)
    return dict(checks=checks, run=run_cfg.get("stage"),
                manifest_sha256=run_cfg["manifest_sha256"],
                recorded=time.strftime("%Y-%m-%d %H:%M:%S"))
