#!/usr/bin/env python3
"""
exp_analog.py — observe the physical analog datapath.

Per spec 12 this OBSERVES only: it reports the V_OA / V_ADC_IN distributions
each operation actually produces, and lets a gain be set manually. It does not
calibrate the amplifier.

    python exp_analog.py --scales              # scale chain per DAC, no run
    python exp_analog.py --dac-compare         # all DACs, G=1
    python exp_analog.py --stats               # per-op V_OA distributions
    python exp_analog.py --gain-sweep          # manual per-op gain sweep
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "_shim"))
sys.path.insert(0, str(HERE))

import numpy as np
import torch

from cim_sim import CIMArray, CIMConfig
from cim_sim.analog import AnalogConfig, AnalogChain, DAC_TYPES
from cim_sim.attention import convert_attention
from run_tinyvit import build_model, synthetic_images

W_BITS = A_BITS = 8
ROWS = 32
OPS = ["qkv", "qk", "av"]


def run_model(analog_cfg, x, weight_bits=W_BITS, act_bits=A_BITS):
    cfg = CIMConfig(rows=ROWS, cols=ROWS, weight_bits=weight_bits,
                    act_bits=act_bits, analog=analog_cfg)
    m = build_model()
    a = CIMArray(cfg)
    convert_attention(m, a)
    with torch.no_grad():
        out = m(x)
    return out, a


def rms(o, ref):
    return (torch.norm(o - ref) / torch.norm(ref)).item()


def agree(o, ref):
    return 100.0 * (o.argmax(-1) == ref.argmax(-1)).float().mean().item()


def show_scales():
    print(f"{'DAC':>18}{'v_per_unit':>14}{'K_MAC':>14}{'LSB_P @G=1':>13}"
          f"{'signed swing':>14}")
    print("-" * 73)
    for name in DAC_TYPES:
        ch = AnalogChain(AnalogConfig(enabled=True, dac_type=name),
                         rows=ROWS, weight_bits=W_BITS, act_bits=A_BITS)
        ch.set_operation(None)
        print(f"{name:>18}{ch.dac.v_per_unit:>14.4e}{ch.k_mac:>14.4e}"
              f"{ch.lsb_p:>13.1f}{ch.adc.signed_swing_volts*1e3:>11.1f} mV")
    print("\n  K_MAC = dac.v_per_unit / (m * 2^B); LSB_P = ADC_LSB / (G * K_MAC)")
    print("  Offset-binary DACs spend half their codes on negative values, so")
    print("  their slope - and hence K_MAC - is about half that of the natively")
    print("  bipolar ones. That is a real architectural difference, not a")
    print("  normalisation choice.")


def dac_compare(x, ref):
    print(f"{'DAC':>18}{'rms err':>11}{'agreement':>11}{'clip%':>9}"
          f"{'|V_OA| p99.9':>14}{'swing used':>12}")
    print("-" * 75)
    for name in DAC_TYPES:
        an = AnalogConfig(enabled=True, dac_type=name, adc_bits=8)
        out, a = run_model(an, x)
        s = a.chain.stats.summary()
        tot_n = sum(v["n"] for v in s.values())
        clip = sum(v["clip_hi"] + v["clip_lo"] for v in s.values())
        p999 = max(v.get("voa_absp999", 0.0) for v in s.values())
        half = a.chain.adc.signed_swing_volts
        print(f"{name:>18}{rms(out, ref):>10.2%}{agree(out, ref):>10.1f}%"
              f"{100.0*clip/max(1,tot_n):>8.3f}%{p999*1e3:>11.2f} mV"
              f"{100*p999/half:>11.1f}%")


def per_op_stats(x, dac_type="CAPACITIVE", gains=None):
    an = AnalogConfig(enabled=True, dac_type=dac_type, adc_bits=8,
                      gain_per_op=gains or {})
    out, a = run_model(an, x)
    s = a.chain.stats.summary()
    print(f"\nDAC={dac_type}  gains={gains or 'all 1.0'}")
    print(f"{'op':>6}{'G':>7}{'n':>14}{'|V_OA| p50':>13}{'|V_OA| p99.9':>14}"
          f"{'|V_OA| max':>13}{'swing used':>12}{'clip%':>9}")
    print("-" * 88)
    half = a.chain.adc.signed_swing_volts
    for op in OPS:
        v = s.get(op)
        if not v:
            continue
        g = (gains or {}).get(op, 1.0)
        print(f"{op:>6}{g:>7.2f}{v['n']:>14,}"
              f"{v.get('voa_absp50',0)*1e3:>10.3f} mV"
              f"{v.get('voa_absp999',0)*1e3:>11.3f} mV"
              f"{abs(v['voa_max'])*1e3:>10.2f} mV"
              f"{100*g*v.get('voa_absp999',0)/half:>11.1f}%"
              f"{v['clip_pct']:>8.3f}%")
    return out, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=2)
    ap.add_argument("--scales", action="store_true")
    ap.add_argument("--dac-compare", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--gain-sweep", action="store_true")
    ap.add_argument("--dac", default="CAPACITIVE")
    a = ap.parse_args()

    if a.scales:
        show_scales()
        return

    x = synthetic_images(a.images)
    with torch.no_grad():
        ref = build_model()(x)

    if a.dac_compare:
        print("\nDAC ARCHITECTURE COMPARISON (G=1, ADC 8b, w=8b a=8b)")
        dac_compare(x, ref)
    if a.stats:
        print("\nPER-OPERATION ANALOG NODE STATISTICS (G=1 — observation only)")
        out, _ = per_op_stats(x, a.dac)
        print(f"\n  rms={rms(out, ref):.2%}  agreement={agree(out, ref):.1f}%")
    if a.gain_sweep:
        print("\nMANUAL PER-OPERATION GAIN SWEEP (gains are set, not calibrated)")
        for gains in ({}, {"qkv": 4, "qk": 2, "av": 1},
                      {"qkv": 16, "qk": 6, "av": 2},
                      {"qkv": 24, "qk": 8, "av": 3}):
            out, aa = per_op_stats(x, a.dac, gains)
            print(f"  -> rms={rms(out, ref):.2%}  agreement={agree(out, ref):.1f}%")


if __name__ == "__main__":
    main()
