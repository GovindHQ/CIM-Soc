#!/usr/bin/env python3
"""
exp_analog.py — observe the physical analog datapath.

Per spec 12 this OBSERVES only: it reports the V_OA / V_ADC_IN distributions
each operation actually produces, and lets a gain be set manually. It does not
calibrate the amplifier. The input DAC is the hardware-specified unsigned 8-bit
capacitive DAC (the only DAC in the simulator); the former --dac-compare mode
was removed with the other DAC candidates (Oct 2026).

    python exp_analog.py --scales              # capacitive-DAC scale chain per precision, no run
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
from cim_sim.analog import AnalogConfig, AnalogChain
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
    """Scale chain of the capacitive DAC (8 b, MSB alignment) per precision."""
    print(f"{'precision':>10}{'v_per_unit':>14}{'K_MAC':>14}{'LSB_P @G=1':>13}"
          f"{'signed swing':>14}")
    print("-" * 65)
    for wb, ab in ((4, 4), (4, 8), (8, 4), (8, 8)):
        ch = AnalogChain(AnalogConfig(enabled=True),
                         rows=ROWS, weight_bits=wb, act_bits=ab)
        ch.set_operation(None)
        print(f"{'W%dA%d' % (wb, ab):>10}{ch.dac.v_per_unit:>14.4e}{ch.k_mac:>14.4e}"
              f"{ch.lsb_p:>13.1f}{ch.adc.signed_swing_volts*1e3:>11.1f} mV")
    print("\n  capacitive DAC: V = V_REF*c/2^N (8 b, 0.6 V), signed inputs offset binary")
    print("  K_MAC = dac.v_per_unit * align / (m * 2^B); LSB_P = ADC_LSB / (G * K_MAC)")


def per_op_stats(x, gains=None):
    an = AnalogConfig(enabled=True, adc_bits=8, gain_per_op=gains or {})
    out, a = run_model(an, x)
    s = a.chain.stats.summary()
    print(f"\nDAC=capacitive 8b  gains={gains or 'all 1.0'}")
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
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--gain-sweep", action="store_true")
    a = ap.parse_args()

    if a.scales:
        show_scales()
        return

    x = synthetic_images(a.images)
    with torch.no_grad():
        ref = build_model()(x)

    if a.stats:
        print("\nPER-OPERATION ANALOG NODE STATISTICS (G=1 — observation only)")
        out, _ = per_op_stats(x)
        print(f"\n  rms={rms(out, ref):.2%}  agreement={agree(out, ref):.1f}%")
    if a.gain_sweep:
        print("\nMANUAL PER-OPERATION GAIN SWEEP (gains are set, not calibrated)")
        for gains in ({}, {"qkv": 4, "qk": 2, "av": 1},
                      {"qkv": 16, "qk": 6, "av": 2},
                      {"qkv": 24, "qk": 8, "av": 3}):
            out, aa = per_op_stats(x, gains)
            print(f"  -> rms={rms(out, ref):.2%}  agreement={agree(out, ref):.1f}%")


if __name__ == "__main__":
    main()
