#!/usr/bin/env python3
"""
test_analog.py — tests for the physical analog datapath (spec 29, 51).

Run:  python test_analog.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

import cim_sim.analog as analog_pkg
from cim_sim.analog import (AnalogChain, AnalogConfig, C2CMacro, CapacitiveDAC,
                            SARADC, SignedWeightAdapter)

rng = np.random.default_rng(0)
PASS = []


def check(name, cond, detail=""):
    PASS.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


# --------------------------------------------------------------------------- #
print("\nTest 1 — capacitive input DAC (the only DAC), hand-computable cases")
# --------------------------------------------------------------------------- #
for nb in (4, 8):
    for signed in (True, False):
        d = CapacitiveDAC(bits=nb, vref=0.6)
        d.set_input_signed(signed)
        tag = f"CDAC {nb}b {'signed' if signed else 'unsigned'}"
        z = float(d.transfer(np.array([0]))[0])
        check(f"{tag}: zero code -> 0 V signed-equivalent", abs(z) < 1e-15,
              f"got {z:.3e}")
        xs = np.arange(d.x_min, d.x_max + 1)
        v = d.transfer(xs)
        check(f"{tag}: monotonic over [{d.x_min},{d.x_max}]",
              bool(np.all(np.diff(v) > 0)))
        # linearity: v_per_unit must reproduce the transfer exactly in ideal mode
        check(f"{tag}: transfer == v_per_unit*X_q, v_per_unit == V_REF/2^N",
              np.allclose(v, xs * d.v_per_unit, rtol=0, atol=1e-15)
              and d.v_per_unit == 0.6 / 2 ** nb,
              f"v_per_unit={d.v_per_unit:.4e}")

try:
    CapacitiveDAC(bits=4).transfer(np.array([8]))
    check("DAC rejects out-of-range signed code", False)
except ValueError:
    check("DAC rejects out-of-range signed code", True)

# capacitive-only design: the other DAC candidates and the option are gone
check("no other DAC implementations exported (DAC_TYPES/make_dac/R2R/... removed)",
      not any(hasattr(analog_pkg, n) for n in
              ("DAC_TYPES", "make_dac", "BaseDAC", "IdealLinearSignedDAC",
               "R2RDAC", "CurrentSteeringDAC", "DifferentialDAC")))
try:
    AnalogConfig(enabled=True, dac_type="R2R")
    check("AnalogConfig has no dac_type option", False)
except TypeError:
    check("AnalogConfig has no dac_type option", True)
cfg0 = AnalogConfig(enabled=True)
ch0 = AnalogChain(cfg0, rows=32, weight_bits=4, act_bits=4)
check("default input DAC = capacitive, 8 bits, V_REF 0.6 V",
      isinstance(ch0.dac, CapacitiveDAC) and ch0.dac.bits == 8 and ch0.dac.vref == 0.6)

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
dac = CapacitiveDAC(bits=4, vref=0.6)
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
for dac_bits in (None, 8):                       # 4-bit-wide DAC and the 8-bit CDAC
    for signed in (True, False):
        cfg = AnalogConfig(enabled=True, dac_bits=dac_bits, adc_bits=8, adc_vref=0.6)
        ch = AnalogChain(cfg, rows=32, weight_bits=4, act_bits=4)
        ch.set_operation(None, act_signed=signed)
        X = (rng.integers(-8, 8, size=(20, 32)) if signed
             else rng.integers(0, 16, size=(20, 32)))
        W = rng.integers(-8, 8, size=(32, 32))
        ch.program(W)
        v_ia = ch.dac.transfer(ch.dac_codes(X))
        v_oa = ch.macro.mac(v_ia, *ch._prog)
        P = X @ W                                   # digital reference MAC
        check(f"CDAC {ch.dac.bits}b {'signed' if signed else 'unsigned'}: V_OA == K_MAC * P",
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
cfg = AnalogConfig(enabled=True, adc_bits=8)
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
cfg = AnalogConfig(enabled=True,
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
cfg = AnalogConfig(enabled=True, dac_bits=4, adc_bits=10)
ch = AnalogChain(cfg, rows=32, weight_bits=8, act_bits=4)
check("act=4b, DAC=4b, W=8b, ADC=10b all independent",
      ch.dac.bits == 4 and ch.weight_bits == 8 and ch.adc.bits == 10,
      f"LSB_P={ch.lsb_p:.3f}")

# =========================================================================== #
# Tests 11-18 (added Oct 2026): the hardware-specified unsigned 8-bit CDAC,
# signed-input convention, act->DAC alignment, amplifier, ADC full scale,
# signed accumulation + rescale round trip, statistics.
# =========================================================================== #
from cim_sim.analog import (AnalogStats, GainNonidealities, PreADCGain,
                            CapacitiveDAC)
from cim_sim import CIMArray, CIMConfig, cim_matmul

# --------------------------------------------------------------------------- #
print("\nTest 11 — CDAC transfer == V_REF*(b7/2 + b6/4 + ... + b0/256), unsigned")
# --------------------------------------------------------------------------- #
VREF = 0.6
cd = CapacitiveDAC(bits=8, vref=VREF)


def screenshot_eq(c, n=8, vref=VREF):
    """The hardware equation written exactly as specified: sum_k b_k / 2^(n-k)."""
    bits = [(c >> k) & 1 for k in range(n)]           # b0..b_{n-1}
    return vref * sum(bits[k] / 2 ** (n - k) for k in range(n))


check("zero code -> 0 V", cd.vout_from_unsigned_code(np.array([0]))[0] == 0.0)
vmax = float(cd.vout_from_unsigned_code(np.array([255]))[0])
check("all-ones code -> V_REF*(1 - 2^-8) = 0.59765625 V, NOT V_REF",
      abs(vmax - VREF * (1 - 2 ** -8)) < 1e-15 and vmax < VREF, f"{vmax!r}")
single = cd.vout_from_unsigned_code(2 ** np.arange(8))
check("each bit b_k alone contributes V_REF*2^k/256 (b7 -> V_REF/2)",
      np.allclose(single, VREF * 2.0 ** np.arange(8) / 256, rtol=0, atol=1e-15)
      and abs(single[7] - 0.3) < 1e-15, f"{np.round(single * 1e3, 4)} mV")
mids = np.array([1, 37, 85, 128, 170, 200, 254])
check("intermediate codes match the bit-sum equation",
      all(abs(cd.vout_from_unsigned_code(np.array([c]))[0] - screenshot_eq(int(c)))
          < 1e-15 for c in mids))
allv = cd.vout_from_unsigned_code(np.arange(256))
check("strictly monotonic over all 256 codes, constant step V_REF/256",
      bool(np.all(np.diff(allv) > 0))
      and np.allclose(np.diff(allv), VREF / 256, rtol=0, atol=1e-15))
check("superposition: V(0x0F | 0xF0) == V(0x0F) + V(0xF0)",
      abs(allv[0xFF] - allv[0x0F] - allv[0xF0]) < 1e-15)
for n in (4, 6, 10):
    dn = CapacitiveDAC(bits=n, vref=VREF)
    top = float(dn.vout_from_unsigned_code(np.array([2 ** n - 1]))[0])
    check(f"N={n}: V_out = V_REF*c/2^N, all-ones = V_REF*(1-2^-{n})",
          abs(top - VREF * (1 - 2.0 ** -n)) < 1e-15
          and abs(dn.vout_from_unsigned_code(np.array([3]))[0] - VREF * 3 / 2 ** n) < 1e-15)

# --------------------------------------------------------------------------- #
print("\nTest 12 — signed-input convention on the unsigned CDAC")
# --------------------------------------------------------------------------- #
cd.set_input_signed(True)
xs = np.arange(-128, 128)
c = cd.unsigned_code(xs)
check("signed X_q -> offset-binary unsigned code c = X_q + 128 in [0, 255]",
      np.array_equal(c, xs + 128) and c.min() == 0 and c.max() == 255)
v = cd.transfer(xs)
check("signed-equivalent V_IA = V_out(c) - V_REF/2  (common mode removed)",
      np.allclose(v, cd.vout_from_unsigned_code(c) - 0.3, rtol=0, atol=1e-15))
check("X_q = -128 -> -V_REF/2;  X_q = +127 -> +V_REF/2 - LSB",
      abs(v[0] + 0.3) < 1e-15 and abs(v[-1] - (0.3 - VREF / 256)) < 1e-15)
check("v_common_mode reports the physical V_REF/2 offset (signed domain)",
      cd.v_common_mode == 0.3)
try:
    cd.transfer(np.array([200]))     # 200 == unsigned bit pattern of int8 -56
    check("signed domain rejects an unsigned/two's-complement word (200)", False)
except ValueError:
    check("signed domain rejects an unsigned/two's-complement word (200)", True)
cd.set_input_signed(False)           # post-softmax AV probabilities
xu = np.arange(0, 256)
check("unsigned domain (AV): c = X_q, V_IA = V_out(c), no common mode",
      np.array_equal(cd.unsigned_code(xu), xu)
      and np.allclose(cd.transfer(xu), cd.vout_from_unsigned_code(xu), rtol=0, atol=0)
      and cd.v_common_mode == 0.0)
try:
    cd.transfer(np.array([-1]))
    check("unsigned domain rejects negative codes", False)
except ValueError:
    check("unsigned domain rejects negative codes", True)
cd.set_input_signed(True)

# --------------------------------------------------------------------------- #
print("\nTest 13 — activation narrower than the 8-bit DAC: alignment")
# --------------------------------------------------------------------------- #
base = dict(enabled=True, adc_bits=8)
ch_ref = AnalogChain(AnalogConfig(dac_bits=None, **base),       # 4-bit-wide DAC
                     rows=32, weight_bits=4, act_bits=4)
ch_msb = AnalogChain(AnalogConfig(dac_bits=8, dac_input_alignment="msb", **base),
                     rows=32, weight_bits=4, act_bits=4)
ch_lsb = AnalogChain(AnalogConfig(dac_bits=8, dac_input_alignment="lsb", **base),
                     rows=32, weight_bits=4, act_bits=4)
check("A4 on 8-bit DAC, MSB-aligned: DAC code = 16*X_q",
      np.array_equal(ch_msb.dac_codes(np.array([-8, -1, 0, 7])), [-128, -16, 0, 112]))
check("MSB-aligned 8-bit DAC has the same K_MAC as a 4-bit DAC",
      ch_msb.k_mac == ch_ref.k_mac, f"{ch_msb.k_mac:.6e} vs {ch_ref.k_mac:.6e}")
check("LSB-aligned 8-bit DAC: K_MAC is 16x smaller (uses 1/16 of the swing)",
      abs(ch_msb.k_mac / ch_lsb.k_mac - 16.0) < 1e-12)
X4 = rng.integers(-8, 8, size=(64, 32))
W4 = rng.integers(-8, 8, size=(32, 32))
for ch in (ch_ref, ch_msb):
    ch.set_operation(None)
    ch.program(W4)
check("MSB-aligned 8-bit DAC: ADC codes bit-identical to a 4-bit DAC",
      np.array_equal(ch_ref.execute(X4), ch_msb.execute(X4)))
check("alignment is part of the accumulation domain",
      ch_msb.domain_key != ch_lsb.domain_key)
try:
    AnalogChain(AnalogConfig(dac_bits=4, **base), rows=32, weight_bits=8, act_bits=8)
    check("dac_bits < act_bits is rejected", False)
except ValueError:
    check("dac_bits < act_bits is rejected", True)
ch_u = AnalogChain(AnalogConfig(dac_bits=8, **base), rows=32, weight_bits=4, act_bits=4)
ch_u.set_operation("av", act_signed=False)
check("unsigned A4 (AV) on 8-bit DAC: max code 15 -> DAC code 240 <= 255",
      int(ch_u.dac_codes(np.array([15]))[0]) == 240
      and abs(ch_u.dac.transfer(ch_u.dac_codes(np.array([15])))[0] - VREF * 240 / 256) < 1e-15)

# --------------------------------------------------------------------------- #
print("\nTest 14 — pre-ADC amplifier: gain, saturation modelled only if configured")
# --------------------------------------------------------------------------- #
amp = PreADCGain(23.77)
vo = np.linspace(-0.05, 0.05, 101)
check("V_ADC_IN = G * V_OA exactly", np.array_equal(amp.apply(vo), vo * 23.77))
check("default amplifier: saturation NOT modelled, counter stays 0",
      (not amp.saturation_modeled) and amp.amp_saturated == 0)
amp_l = PreADCGain(10.0, GainNonidealities(v_swing_limit=0.25))
out = amp_l.apply(np.array([0.01, 0.03, -0.03, -0.01]))
check("with a swing limit: saturating samples counted and clipped",
      amp_l.saturation_modeled and amp_l.amp_saturated == 2
      and np.allclose(out, [0.1, 0.25, -0.25, -0.1]))

# --------------------------------------------------------------------------- #
print("\nTest 15 — SAR ADC full scale, clipping and rail counters")
# --------------------------------------------------------------------------- #
for nb in (8, 10, 12):
    a_ = SARADC(bits=nb, vref=0.6)
    check(f"{nb}b: LSB = Vref/2^N, signed swing = Vref/2",
          a_.lsb_volts == 0.6 / 2 ** nb and abs(a_.signed_swing_volts - 0.3) < 1e-15)
    vv = np.array([-0.5, -0.3, -0.299, 0.0, 0.299, 0.3 - a_.lsb_volts, 0.31, 0.7])
    dd = a_.convert(vv)
    check(f"{nb}b: clip_lo counts V_phys < 0, clip_hi counts V_phys > Vref",
          a_.n_clipped_lo == 1 and a_.n_clipped_hi == 2,
          f"lo={a_.n_clipped_lo} hi={a_.n_clipped_hi}")
    check(f"{nb}b: rail counts >= clip counts (rail is an upper bound)",
          a_.n_rail_hi >= a_.n_clipped_hi and a_.n_rail_lo >= a_.n_clipped_lo)
    check(f"{nb}b: codes span [-2^(N-1), 2^(N-1)-1]",
          dd.min() == -2 ** (nb - 1) and dd.max() == 2 ** (nb - 1) - 1)

# --------------------------------------------------------------------------- #
print("\nTest 16 — signed accumulation + single rescale (round trip vs digital MAC)")
# --------------------------------------------------------------------------- #
for wb, ab, adcb in ((8, 8, 8), (4, 4, 8), (4, 8, 10), (8, 4, 12)):
    for signed in (True, False):
        an = AnalogConfig(enabled=True, dac_bits=8,
                          adc_bits=adcb, gain_default=1.0)
        arr = CIMArray(CIMConfig(weight_bits=wb, act_bits=ab, analog=an))
        N_, K_, M_ = 9, 70, 40                         # 3 ragged depth blocks
        Xf = rng.standard_normal((N_, K_))
        if not signed:
            Xf = np.abs(Xf)
        Wf = rng.standard_normal((K_, M_))
        Y = cim_matmul(Xf, Wf, arr, act_signed=signed, op="t16")
        from cim_sim.quant import quantize
        xq, sx = quantize(Xf, bits=ab, axis=1, signed=signed)
        wq, sw = quantize(Wf, bits=wb, axis=0, signed=True)
        P = (xq.astype(np.int64) @ wq.astype(np.int64)).astype(np.float64)
        lsbp = arr.chain.lsb_p
        n_db = -(-K_ // 32)
        err_units = np.abs(Y / (sx * sw) - P)          # in column-sum units
        clipped = arr.stats.adc_saturations
        check(f"W{wb}A{ab} ADC{adcb} {'signed' if signed else 'unsigned'}: "
              f"|y - digital MAC| <= n_db*LSB_P/2 (no clipping)",
              clipped == 0 and float(err_units.max()) <= n_db * 0.5 * lsbp + 1e-9,
              f"max err {err_units.max() / lsbp:.3f} LSB_P over {n_db} blocks")

# --------------------------------------------------------------------------- #
print("\nTest 17 — statistics: exact counts, weights, merge, serialisation")
# --------------------------------------------------------------------------- #
st_all = AnalogStats(keep_frac=0.01, cap=5_000, seed=1)
st_a, st_b = AnalogStats(keep_frac=0.01, cap=5_000, seed=2), AnalogStats(keep_frac=0.01, cap=5_000, seed=3)
blocks = [rng.standard_normal((200, 32)) * 0.01 for _ in range(30)]
for i, blk in enumerate(blocks):
    for st in (st_all, st_a if i < 15 else st_b):
        st.record("qkv", blk, blk * 20, np.rint(blk * 1000).astype(int), clip_hi=i % 3)
st_m = AnalogStats(keep_frac=0.01, cap=5_000)
st_m.merge(st_a)
st_m.merge(st_b)
sa, sm = st_all.summary()["qkv"], st_m.summary()["qkv"]
check("scalar stats of merged chunks == one long run",
      all(sa[k] == sm[k] for k in ("n", "clip_hi", "voa_min", "voa_max", "vadc_min", "vadc_max"))
      and abs(sa["voa_std"] - sm["voa_std"]) < 1e-12)
w_tot = np.concatenate(st_all.b["qkv"]["w"]).sum()
check("subsample weights sum EXACTLY to the conversion count (no thinning)",
      abs(w_tot - sa["n"]) < 1e-6, f"sum w={w_tot:.0f} n={sa['n']}")
st_thin = AnalogStats(keep_frac=0.01, cap=300, seed=4)
for blk in blocks:
    st_thin.record("qkv", blk, blk, np.zeros(blk.size, int))
bt = st_thin.b["qkv"]
w_thin = np.concatenate(bt["w"]).sum()
check("thinning: kept <= cap, keep_frac halved, weights still ~ conversion count",
      bt["n_kept"] <= 300 and bt["kf"] < 0.01 and abs(w_thin / bt["n"] - 1) < 0.15,
      f"kept={bt['n_kept']} kf={bt['kf']:.5f} sum w/n={w_thin / bt['n']:.3f}")
check("thinning keeps samples from the whole run (not only the first calls)",
      sum(np.isin(a, blocks[-1].ravel().astype(np.float32)).sum() for a in bt["voa"]) > 0)
allv = np.abs(np.concatenate(blocks).ravel())
check("weighted |V_OA| p99 within 5% of the exact value",
      abs(sa["voa_absp99"] / np.percentile(allv, 99) - 1) < 0.05,
      f"{sa['voa_absp99']:.4g} vs {np.percentile(allv, 99):.4g}")
scal, arrs = st_m.state_dict()
st_r = AnalogStats.from_state(scal, arrs)
check("state_dict / from_state round trip preserves the summary",
      st_r.summary()["qkv"] == sm)
an = AnalogConfig(enabled=True, dac_bits=8,
                  gain_per_op={"qkv": 23.77})
shared = AnalogStats()
arr = CIMArray(CIMConfig(weight_bits=8, act_bits=8, analog=an), analog_stats=shared)
cim_matmul(rng.standard_normal((50, 96)), rng.standard_normal((96, 64)), arr, op="qkv")
check("shared stats: conversions == CIMStats.adc_conversions, clips == saturations",
      shared.summary()["qkv"]["n"] == arr.stats.adc_conversions
      and shared.summary()["qkv"]["clip_hi"] + shared.summary()["qkv"]["clip_lo"]
      == arr.stats.adc_saturations)

# --------------------------------------------------------------------------- #
print("\nTest 18 — common-mode residue diagnostic (observation only)")
# --------------------------------------------------------------------------- #
ch = AnalogChain(AnalogConfig(enabled=True, dac_bits=8),
                 rows=4, weight_bits=4, act_bits=8)
ch.set_operation("qkv")
Wt = np.array([[7, -3], [7, 1], [-8, 0], [7, 2]])
ch.program(Wt)
d_before = ch.execute(np.zeros((1, 4), dtype=int))
expect = 0.3 * (np.array([7 + 7 - 8 + 7, -3 + 1 + 0 + 2]) / 16) / 4
s18 = ch.stats.summary()["qkv"]
check("CM term = V_CM * sum_j(sign|W|/2^B) / m, hand-computed",
      abs(s18["cm_residue_absmax_v"] - np.abs(expect).max()) < 1e-15,
      f"{s18['cm_residue_absmax_v']:.6f} V")
check("CM term is NOT applied: zero input still gives code 0",
      np.array_equal(d_before, np.zeros((1, 2), dtype=int)))
ch.set_operation("av", act_signed=False)
ch.execute(np.zeros((1, 4), dtype=int))
check("no CM term for the unsigned (AV) domain",
      ch.stats.summary()["av"]["cm_residue_absmax_v"] == 0.0)

# --------------------------------------------------------------------------- #
print(f"\n{sum(PASS)}/{len(PASS)} checks passed")
sys.exit(0 if all(PASS) else 1)
