"""Self-test for the ADC probe path. Pure numpy — no torch, no checkpoint."""
import numpy as np

from cim_sim import CIMArray, CIMConfig, TraceLog, adc_probe
from cim_sim.matmul import cim_matmul

rng = np.random.default_rng(0)

# --- 1. numerics are unchanged when the probe is off vs on ------------------
X = rng.normal(size=(196, 160))
W = rng.normal(size=(160, 480))

cfg = CIMConfig(act_bits=4, weight_bits=4)
adc_probe.disable()
a0 = CIMArray(cfg)
y0 = cim_matmul(X, W, a0, tag="s3.qkv", log=TraceLog())

probe = adc_probe.enable(CIMArray(cfg).adc_full_scale_theoretical)
a1 = CIMArray(cfg)
log = TraceLog()
y1 = cim_matmul(X, W, a1, tag="s3.qkv", log=log)
assert np.array_equal(y0, y1), "probe changed the numerics"
print("1. probe is numerically inert                      OK")

# --- 2. _delta / _snapshot survive a full pass ------------------------------
tags = log.by_tag()
assert set(tags) == {"s3.qkv"}
st = tags["s3.qkv"]
assert st.array_ops == a1.stats.array_ops, (st.array_ops, a1.stats.array_ops)
print("2. per-tag delta bookkeeping intact                OK")

# --- 3. the histogram is exact ---------------------------------------------
counts = probe.counts("qkv")
values = probe.values()
# Reference: recompute the column sums the scheduler produced, exactly.
from cim_sim.quant import quantize
xq, _ = quantize(X, bits=4, axis=1, signed=True)
wq, _ = quantize(W, bits=4, axis=0, signed=True)
n_db, n_ob = (160 + 31) // 32, (480 + 31) // 32
xp = np.zeros((196, n_db * 32), dtype=np.int64); xp[:, :160] = xq
wp = np.zeros((n_db * 32, n_ob * 32), dtype=np.int64); wp[:160, :480] = wq
ref = []
for ob in range(n_ob):
    uc = min(32, 480 - ob * 32)
    for db in range(n_db):
        blk = xp[:, db * 32:(db + 1) * 32] @ wp[db * 32:(db + 1) * 32,
                                                ob * 32:(ob + 1) * 32]
        ref.append(blk[:, :uc].ravel())
ref = np.concatenate(ref)
ref_hist = np.bincount(ref + probe.full_scale, minlength=len(values))
assert np.array_equal(counts, ref_hist), "histogram does not match brute force"
print(f"3. histogram exact over {counts.sum():,} column sums      OK")

# --- 4. percentiles from histogram match numpy on raw samples --------------
from cim_sim.adc_probe import abs_percentiles_from_hist
for q in (50, 90, 99, 99.9):
    h = abs_percentiles_from_hist(counts, values, [q])[0]
    n = np.percentile(np.abs(ref), q, method="inverted_cdf")
    assert abs(h - n) <= 1, (q, h, n)
print("4. histogram percentiles match raw-sample numpy    OK")

# --- 5. offline ADC error reproduces _adc_convert exactly ------------------
from cim_sim.adc_probe import adc_error_from_hist
from cim_sim.array import _adc_convert
for fs in (2048, 300, 97):
    for bits in (4, 6, 8, 10):
        yq, n_sat = _adc_convert(ref, bits=bits, full_scale=fs)
        rmse_live = float(np.sqrt(np.mean((yq - ref) ** 2)))
        clip_live = 100.0 * n_sat / ref.size
        r = adc_error_from_hist(counts, values, fs, bits)
        assert abs(r["rmse"] - rmse_live) < 1e-6, (fs, bits, r["rmse"], rmse_live)
        assert abs(r["clip_pct"] - clip_live) < 1e-9, (fs, bits)
print("5. offline sweep == live _adc_convert, all FS/bits  OK")

# --- 6. reduced full scale really is applied -------------------------------
cfg_fs = CIMConfig(act_bits=4, weight_bits=4, adc_enabled=True, adc_bits=6,
                   adc_full_scale_override=100)
a2 = CIMArray(cfg_fs)
assert a2.adc_full_scale == 100 and a2.adc_full_scale_theoretical == 2048
adc_probe.enable(a2.adc_full_scale_theoretical)
y2 = cim_matmul(X, W, a2, tag="s3.qkv", log=TraceLog())
assert a2.stats.adc_saturations > 0, "expected clipping at a reduced full scale"
print(f"6. FS override applied, {a2.stats.adc_saturations:,} saturations   OK")

# --- 7. multi-array path still works --------------------------------------
adc_probe.enable(2048)
arrs = [CIMArray(CIMConfig(act_bits=4, weight_bits=4)) for _ in range(4)]
log4 = TraceLog()
y4 = cim_matmul(X, W, arrays=arrs, tag="s3.qkv", log=log4)
assert np.allclose(y4, y0), "P=4 diverged from P=1"
print("7. multi-array scheduling unchanged                OK")

# --- 8. padded columns are excluded ---------------------------------------
adc_probe.enable(2048)
Wr = rng.normal(size=(160, 33))          # 33 -> second output tile has 1 useful col
cim_matmul(X, Wr, CIMArray(cfg), tag="t.ragged", log=TraceLog())
pr = adc_probe.get()
n_recorded = int(pr.counts("ragged").sum())
expected = 196 * 5 * 33                  # tokens * depth blocks * useful cols
assert n_recorded == expected, (n_recorded, expected)
print("8. padded columns excluded from histogram          OK")

adc_probe.disable()

# --- 9. unsigned activations get the wider (correct) bound -----------------
a = CIMArray(CIMConfig(act_bits=4, weight_bits=4))
assert a.full_scale_for(True) == 32 * 8 * 8 == 2048
assert a.full_scale_for(False) == 32 * 8 * 15 == 3840
assert a.adc_full_scale_theoretical_max == 3840
print("9. unsigned AV bound is 2x the signed bound        OK")

# --- 10. a worst-case AV column sum overflows the OLD (signed) bound -------
# Flat attention quantizes to the top of the unsigned range on every row. With
# aligned weights this is the largest column sum the model can make, and the
# signed bound cannot represent it.
from cim_sim.array import _adc_convert
aq = np.full((1, 32), 15, dtype=np.int64)          # unsigned 4b, all at max
wq = np.full((32, 1), 7, dtype=np.int64)           # signed 4b, all at max
worst = int((aq @ wq)[0, 0])
assert worst == 3360
_, sat_old = _adc_convert(np.array([worst]), bits=8, full_scale=2048)
_, sat_new = _adc_convert(np.array([worst]), bits=8, full_scale=3840)
assert sat_old == 1 and sat_new == 0
print(f"10. worst-case AV sum {worst}: clipped by old bound 2048, "
      f"fits new bound 3840   OK")

# the probe records the correct per-op bound for an unsigned op
adc_probe.enable(a.adc_full_scale_theoretical_max)
scores = rng.normal(size=(196, 196)) * 0.2
P = np.exp(scores - scores.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
V = rng.normal(size=(196, 32))
cim_matmul(P, V, CIMArray(CIMConfig(act_bits=4, weight_bits=4)),
           tag="s3.av", log=TraceLog(), act_signed=False)
pr = adc_probe.get()
assert pr.op_bound["av"] == 3840, pr.op_bound
print("    probe tagged 'av' with the unsigned bound         OK")

# --- 11. per-operation ADC references resolve independently ----------------
cfg_map = CIMConfig(act_bits=4, weight_bits=4, adc_enabled=True, adc_bits=6,
                    adc_full_scale_override={"qkv": 300, "av": 900})
am = CIMArray(cfg_map)
assert am.adc_full_scale_for(True, "s3.qkv") == 300
assert am.adc_full_scale_for(False, "s3.av") == 900
assert am.adc_full_scale_for(True, "s3.qk_t") == 2048   # unlisted -> theoretical
print("11. per-operation ADC references resolve           OK")

adc_probe.disable()
print("\nall probe self-tests passed (11/11)")
