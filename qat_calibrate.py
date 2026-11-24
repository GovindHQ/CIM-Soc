#!/usr/bin/env python3
"""
qat_calibrate.py — runbook stage 8: hardware scale check, gain calibration, freeze.

  python qat_calibrate.py scale-check --precision W8A8
      Prints the analog scale chain (K_MAC, LSB_P, FS per op, ADC, DAC) for a
      precision and verifies it against the closed form in QAT_DESIGN.md §1.
      Needs no data.

  python qat_calibrate.py run --paths configs/paths.json --precision W8A8 --workers 24
      observe (exact probe on CAL) -> candidate gains (P99.9/99.99/99.999/100,
      MSE-optimal diagnostic, per-layer) -> CAL-check on the analog simulator
      (CALCHECK images) -> calibration_report.{json,md} with flags.
      Writes NO frozen gains.

  python qat_calibrate.py freeze --paths configs/paths.json --precision W8A8 --level p99.99
      Writes <work_root>/calibration/W8A8/frozen_gains.json (refuses to overwrite).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cim_qat.common import single_thread_env_before_numpy, tag_job  # noqa: E402
single_thread_env_before_numpy()     # BLAS sizes its thread pool when numpy loads: set 1 thread first
tag_job("calib")
from cim_qat.common import (code_fingerprint, now_iso, sha256_file,  # noqa: E402
                            write_json, write_provenance)


def parse_prec(s: str) -> tuple:
    s = s.upper()
    if s not in ("W8A8", "W4A4"):
        raise SystemExit("--precision must be W8A8 or W4A4 in this phase")
    return (8, 8) if s == "W8A8" else (4, 4)


def scale_check(wb, ab) -> bool:
    from cim_qat.hw_config import OPS, OP_SIGNED, HWConfig, op_specs
    from cim_sim.analog import AnalogChain
    hw = HWConfig(wb, ab, (1.0, 1.0, 1.0))
    ok = True
    print(f"Analog scale chain, W{wb}A{ab}, gains = 1 (LSB_P scales as 1/G):")
    for op in OPS:
        ch = AnalogChain(hw.analog_config(), hw.rows, wb, ab)
        ch.set_operation(op, act_signed=OP_SIGNED[op])
        kmac_closed = (0.6 / 256) * 2 ** (8 - ab) / (32 * 2 ** wb)
        lsbp_closed = (0.6 / 256) / kmac_closed
        print(ch.scale_report().replace("\n", "\n    "))
        good = (abs(ch.k_mac / kmac_closed - 1) < 1e-12 and abs(ch.lsb_p / lsbp_closed - 1) < 1e-12
                and ch.dac.name == "capacitive" and ch.dac.bits == 8 and ch.dac.vref == 0.6
                and ch.adc.bits == 8 and ch.adc.vref == 0.6 and ch.macro.m == 32
                and ch.dac.signed_input == OP_SIGNED[op])
        spec = op_specs(hw)[op]
        good &= abs(spec.lsb_p - ch.lsb_p) == 0
        print(f"  [{'PASS' if good else 'FAIL'}] {op}: K_MAC={ch.k_mac:.6e} (closed form {kmac_closed:.6e}), "
              f"LSB_P={ch.lsb_p:.4f}, FS=+-{128 * ch.lsb_p:.1f} column-sum units, "
              f"DAC input {'signed (offset binary)' if OP_SIGNED[op] else 'unsigned'}")
        ok &= good
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s0 = sub.add_parser("scale-check")
    s0.add_argument("--precision", required=True)
    s1 = sub.add_parser("run")
    s1.add_argument("--paths", required=True)
    s1.add_argument("--precision", required=True)
    s1.add_argument("--workers", type=int, default=16)
    s1.add_argument("--skip-check", action="store_true", help="debug only: no CAL-check")
    s2 = sub.add_parser("freeze")
    s2.add_argument("--paths", required=True)
    s2.add_argument("--precision", required=True)
    s2.add_argument("--level", required=True)
    s2.add_argument("--force", action="store_true")
    a = ap.parse_args()

    wb, ab = parse_prec(a.precision)
    if a.cmd == "scale-check":
        sys.exit(0 if scale_check(wb, ab) else 1)

    import numpy as np

    from cim_qat import calibration as C
    from cim_qat.data import DataPaths, Splits
    from cim_qat.hw_config import HWConfig
    from cim_qat.model import PRETRAINED, check_pretrained
    dp = DataPaths.load(a.paths)
    cal_dir = Path(dp.work_root) / "calibration" / f"W{wb}A{ab}"

    if a.cmd == "freeze":
        rec = C.freeze(cal_dir, (wb, ab), a.level, a.force)
        print(f"FROZEN {rec['gains']} (level {a.level}) -> {cal_dir / 'frozen_gains.json'}")
        if rec["deviates_from_primary"]:
            print("NOTE: this deviates from the pre-declared primary level P99.99; record why in "
                  "QAT_EXPERIMENT_LOG.md.")
        return

    if not scale_check(wb, ab):
        sys.exit("scale check failed; not calibrating")
    sp = Splits(dp.splits_dir, purpose="calibration")   # no TEST access
    ck_sha = check_pretrained()
    cal_dir.mkdir(parents=True, exist_ok=True)
    write_provenance(cal_dir, dict(stage="calibration", precision=f"W{wb}A{ab}"))
    hw0 = HWConfig(wb, ab, (1.0, 1.0, 1.0))
    cal_rows = sp.rows("cal")
    t0 = time.time()
    hist_p = cal_dir / "hist_cal.npz"
    if hist_p.exists():
        print(f"loading existing histograms {hist_p}")
        obs = C.load_hists(hist_p, C.hist_max(hw0))
        obs["n_images"] = len(cal_rows)
    else:
        print(f"observing {len(cal_rows)} CAL images with the exact simulator probe ...")
        obs = C.observe(hw0, str(PRETRAINED), [str(sp.path(r)) for r in cal_rows], a.workers)
        C.save_hists(obs, hist_p)
    gains = C.compute_gains(hw0, obs)
    write_json(cal_dir / "gains_candidates.json", gains)
    check = None
    if not a.skip_check:
        rows = sp.rows("calcheck")
        items = [(r["idx"], str(sp.path(r))) for r in rows]
        labels = np.array([r["label"] for r in rows])
        print(f"CAL-check on {len(rows)} CALCHECK images (analog simulator) ...")
        check = C.calcheck(hw0, gains, str(PRETRAINED), items, labels, cal_dir / "calcheck", a.workers)
    rep = dict(precision=f"W{wb}A{ab}", weight_bits=wb, act_bits=ab, adc_bits=8,
               created_utc=now_iso(), code_fingerprint=code_fingerprint()["all"],
               checkpoint_sha256=ck_sha, n_cal=len(cal_rows),
               cal_manifest_sha256=sp.sha("cal"), calcheck_manifest_sha256=sp.sha("calcheck"),
               n_calcheck=len(sp.rows("calcheck")), gains=gains, calcheck=check,
               flag_rules=C.FLAG_RULES, primary_level=C.PRIMARY,
               data_used=["cal", "calcheck"], test_split_used=False,
               wall_s=time.time() - t0)
    rep["flags"] = C.flags(gains, check)
    write_json(cal_dir / "calibration_report.json", rep)
    (cal_dir / "calibration_report.md").write_text(C.render_md(rep))
    print((cal_dir / "calibration_report.md").read_text())
    print(f"\nreport: {cal_dir / 'calibration_report.md'}  (sha256 "
          f"{sha256_file(cal_dir / 'calibration_report.json')[:16]})")
    print("Gains are NOT frozen. Review the report, then run `qat_calibrate.py freeze`.")


if __name__ == "__main__":
    main()
