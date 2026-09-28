#!/usr/bin/env python3
"""
test_analog.py — tests for the physical analog datapath (spec 29, 51).

Run:  python test_analog.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from cim_sim.analog import (AnalogChain, AnalogConfig, C2CMacro, DAC_TYPES,
                            SARADC, SignedWeightAdapter, make_dac)

rng = np.random.default_rng(0)
PASS = []


def check(name, cond, detail=""):
    PASS.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


# --------------------------------------------------------------------------- #
print("\nTest 1 — DAC transfer functions, hand-computable cases")
# --------------------------------------------------------------------------- #
for name in DAC_TYPES:
    d = make_dac(name, bits=4, vref=0.6)
    z = float(d.transfer(np.array([0]))[0])
    check(f"{name}: zero code -> 0 V signed-equivalent", abs(z) < 1e-15,
          f"got {z:.3e}")
    # monotonic across the full signed range
    xs = np.arange(d.x_min, d.x_max + 1)
    v = d.transfer(xs)
    check(f"{name}: monotonic over [{d.x_min},{d.x_max}]",
          bool(np.all(np.diff(v) > 0)))
    # linearity: v_per_unit must reproduce the transfer exactly in ideal mode
    check(f"{name}: matches v_per_unit*X_q",
          np.allclose(v, xs * d.v_per_unit, atol=1e-12),
          f"v_per_unit={d.v_per_unit:.4e}")

# architectures must NOT all share a slope - that is the point of comparing
slopes = {n: make_dac(n, bits=8, vref=0.6).v_per_unit for n in DAC_TYPES}
check("offset-binary DACs have half the slope of bipolar ones",
      abs(slopes["R2R"] / slopes["IDEAL"] - (127 / 256)) < 1e-9,
      f"R2R={slopes['R2R']:.3e} IDEAL={slopes['IDEAL']:.3e}")

d = make_dac("DIFFERENTIAL", bits=4, vref=0.6)
vp, vn = d.paths(np.array([-5, 0, 5]))
check("DIFFERENTIAL: negative code drives only the V- path",
      vp[0] == 0 and vn[0] > 0 and vp[2] > 0 and vn[2] == 0)

try:
    make_dac("IDEAL", bits=4).transfer(np.array([8]))
    check("DAC rejects out-of-range signed code", False)
except ValueError:
    check("DAC rejects out-of-range signed code", True)

# --------------------------------------------------------------------------- #
print("\nTest 2 + 9 — signed weights never reach the ladder as two's complement")
# --------------------------------------------------------------------------- #
ad = SignedWeightAdapter(4)
sign, mag = ad.split(np.array([-5, -8, 0, 7]))
check("sign/magnitude split", list(sign) == [-1, -1, 0, 1] and list(mag) == [5, 8, 0, 7])
cap = ad.capacitor_code(mag)
check("capacitor code is non-negative", bool(np.all(cap >= 0)), f"{cap}")
check("capacitor code of -5 is 5/16 (NOT 11/16 from bit pattern 1011)",
      abs(cap[0] - 5 / 16) < 1e-12, f"got {cap[0]:.4f}")

# full sign table through the real macro
m = C2CMacro(rows=1, weight_bits=4, m=1)
dac = make_dac("IDEAL", bits=4, vref=0.6)
for xv, wv in [(3, 5), (3, -5), (-3, 5), (-3, -5)]:
    s_, c_ = m.program(np.array([[wv]]))
    v = m.mac(dac.transfer(np.array([[xv]])), s_, c_)
    expect = np.sign(xv * wv)
    check(f"sign of ({xv})x({wv})", np.sign(v[0, 0]) == expect,
          f"V_OA={v[0,0]:+.6e}")

# mixed-sign accumulation / cancellation
m8 = C2CMacro(rows=4, weight_bits=4, m=4)
s_, c_ = m8.program(np.array([[3], [-3], [2], [-2]]))
v = m8.mac(dac.transfer(np.array([[5, 5, 7, 7]])), s_, c_)
check("exact cancellation of mixed signs", abs(v[0, 0]) < 1e-15, f"{v[0,0]:.2e}")

# --------------------------------------------------------------------------- #
print("\nTest 3 — K_MAC equivalence (analog chain vs digital reference)")
# --------------------------------------------------------------------------- #
for dac_type in DAC_TYPES:
    cfg = AnalogConfig(enabled=True, dac_type=dac_type, adc_bits=8, adc_vref=0.6)
    ch = AnalogChain(cfg, rows=32, weight_bits=4, act_bits=4)
    X = rng.integers(-8, 8, size=(20, 32))
    W = rng.integers(-8, 8, size=(32, 32))
    ch.program(W)
    v_ia = ch.dac.transfer(X)
    v_oa = ch.macro.mac(v_ia, *ch._prog)
    P = X @ W                                   # digital reference MAC
    check(f"{dac_type}: V_OA == K_MAC * P",
          np.allclose(v_oa, ch.k_mac * P, rtol=0, atol=1e-15),
          f"K_MAC={ch.k_mac:.4e}")

# --------------------------------------------------------------------------- #
print("\nTest 4 — SAR ADC equals ideal round-to-nearest, and is bit-by-bit")
# --------------------------------------------------------------------------- #
adc = SARADC(bits=8, vref=0.6)
v = np.linspace(-0.3, 0.3, 5001)
d = adc.convert(v)
ideal = np.clip(np.rint(v / adc.lsb_volts), -adc.code_mid, adc.code_mid - 1)
check("ideal SAR search == round-to-nearest quantiser",
      np.array_equal(d, ideal.astype(np.int64)),
      f"max diff {int(np.abs(d-ideal).max())}")
check("zero volts -> code 0 exactly", int(adc.convert(np.array([0.0]))[0]) == 0)
check("full-scale positive saturates at +2^(N-1)-1",
      int(adc.convert(np.array([1.0]))[0]) == 127)
check("full-scale negative saturates at -2^(N-1)",
      int(adc.convert(np.array([-1.0]))[0]) == -128)

# --------------------------------------------------------------------------- #
print("\nTest 5 + 8 — pre-ADC gain and clipping are observable, not hidden")
# --------------------------------------------------------------------------- #
cfg = AnalogConfig(enabled=True, dac_type="CAPACITIVE", adc_bits=8)
ch = AnalogChain(cfg, rows=32, weight_bits=8, act_bits=8)
X = rng.integers(-128, 128, size=(64, 32))
W = rng.integers(-128, 128, size=(32, 32))
ch.program(W)
prev_clip = -1
lsbs = []
for g in (1.0, 4.0, 16.0, 64.0):
    ch.cfg.gain_default = g
    ch.set_operation(None)
    ch.adc.n_clipped_hi = ch.adc.n_clipped_lo = 0
    ch.execute(X)
    clip = ch.adc.n_clipped_hi + ch.adc.n_clipped_lo
    lsbs.append(ch.lsb_p)
    print(f"      G={g:5.1f}  LSB_P={ch.lsb_p:10.2f}  clipped={clip}")
    check(f"G={g}: clipping monotonically non-decreasing", clip >= prev_clip)
    prev_clip = clip
check("higher gain => finer LSB_P (LSB_P ~ 1/G)",
      all(a > b for a, b in zip(lsbs, lsbs[1:]))
      and abs(lsbs[0] / lsbs[-1] - 64.0) < 1e-9)

# --------------------------------------------------------------------------- #
print("\nTest 6 + 7 — partial-sum consistency and per-operation gain")
# --------------------------------------------------------------------------- #
cfg = AnalogConfig(enabled=True, dac_type="CAPACITIVE",
                   gain_per_op={"qkv": 8.0, "qk": 4.0, "av": 1.5})
ch = AnalogChain(cfg, rows=32, weight_bits=4, act_bits=4)
ch.set_operation("qkv")
k1, key1 = ch.lsb_p, ch.domain_key
ch.set_operation("qk")
k2, key2 = ch.lsb_p, ch.domain_key
check("different operations get different gain/LSB_P", k1 != k2,
      f"qkv LSB_P={k1:.2f}  qk LSB_P={k2:.2f}")
check("different operations are different accumulation domains", key1 != key2)

ch.set_operation("qkv")
acc = ch.accumulator((4, 32))
X = rng.integers(-8, 8, size=(4, 32))
for _ in range(3):                    # three depth blocks, same domain
    ch.program(rng.integers(-8, 8, size=(32, 32)))
    acc.add(ch.execute(X), ch.domain_key)
check("same-operation depth blocks accumulate", acc.n_blocks == 3)
ch.set_operation("av")                # different gain => must be rejected
try:
    acc.add(ch.execute(X), ch.domain_key)
    check("accumulator rejects a foreign analog scale domain", False)
except ValueError:
    check("accumulator rejects a foreign analog scale domain", True)

# --------------------------------------------------------------------------- #
print("\nTest 10 — independent precision configuration")
# --------------------------------------------------------------------------- #
cfg = AnalogConfig(enabled=True, dac_type="R2R", dac_bits=4, adc_bits=10)
ch = AnalogChain(cfg, rows=32, weight_bits=8, act_bits=4)
check("act=4b, DAC=4b, W=8b, ADC=10b all independent",
      ch.dac.bits == 4 and ch.weight_bits == 8 and ch.adc.bits == 10,
      f"LSB_P={ch.lsb_p:.3f}")

# --------------------------------------------------------------------------- #
print(f"\n{sum(PASS)}/{len(PASS)} checks passed")
sys.exit(0 if all(PASS) else 1)
