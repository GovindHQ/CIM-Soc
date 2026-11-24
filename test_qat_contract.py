#!/usr/bin/env python3
"""
test_qat_contract.py — the forward contract a QAT surrogate must reproduce, and
the two simulator fixes made before the new QAT run (Oct 2026).

  1. CLOSED-FORM CONTRACT. With the ideal analog chain, cim_matmul equals, bit
     for bit, the per-depth-block quantizer a differentiable surrogate would
     implement:
         P_b  = Xq[:, block b] @ Wq[block b, :]               (exact integers)
         d_b  = clip(floor(P_b / LSB_P + 1/2), -2^(N-1), 2^(N-1) - 1)
         y    = (sum_b d_b) * LSB_P * S_X * S_W
     with LSB_P = chain.lsb_p after set_operation(op, act_signed). Checked for
     W4A4/W4A8/W8A4/W8A8, signed and unsigned (AV) activations, ragged shapes,
     tie-heavy gains (1, 2, 16), calibrated-style gains, heavy clipping (300),
     and ADC 8/10/12 bits. Exact-threshold ties resolve UPWARD (floor(x+1/2)),
     not half-to-even, so a torch surrogate must use floor(x + 0.5 + guard),
     not torch.round, to match the simulator at ties; guard =
     SARADC.TIE_GUARD_REL * 2^N codes absorbs P / LSB_P landing 1e-15 below an
     exact tie when LSB_P is not exactly representable.

     The input DAC is the hardware-specified unsigned 8-bit capacitive DAC
     (the only DAC in the simulator), checked with MSB and LSB alignment.

  2. TOKEN-BLOCK / ARRAY-COUNT NEUTRALITY ON THE ANALOG PATH at tie-heavy gains
     (CLAUDE.md §11.1/§11.3). Before the SAR tie guard, T=1 vs T=all differed.

  3. GRADIENT GUARD. CIMAttention refuses a training-mode forward with autograd
     enabled (it would silently give qkv.weight and everything upstream zero
     gradient); eval / no_grad use is unchanged, and the float path
     (set_cim_enabled False) still trains normally.

  4. INITIAL QAT OPERATING POINT. The ADC stays configurable, but the default
     analog configuration is the one the initial QAT must use: capacitive DAC
     8 b @ 0.6 V and an 8-bit SAR ADC (QAT_INITIAL_ADC_BITS).

Run:  python test_qat_contract.py      (exit code 1 on any failure)
"""
import itertools
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "_shim"))
sys.path.insert(0, str(HERE))

import numpy as np

from cim_sim import CIMArray, CIMConfig, cim_matmul, quantize
from cim_sim.analog import QAT_INITIAL_ADC_BITS, AnalogConfig, CapacitiveDAC, SARADC

PASS = []


def check(name, cond, detail=""):
    PASS.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def closed_form(X, W, wb, ab, signed, lsb_p, nbits, rows=32):
    """The surrogate contract, in exact integer arithmetic where possible."""
    xq, sx = quantize(X, ab, axis=1, signed=signed)
    wq, sw = quantize(W, wb, axis=0, signed=True)
    N, K = xq.shape
    n_db = -(-K // rows)
    xp = np.zeros((N, n_db * rows), np.int64)
    xp[:, :K] = xq
    wp = np.zeros((n_db * rows, wq.shape[1]), np.int64)
    wp[:K] = wq
    lo, hi = -(2 ** (nbits - 1)), 2 ** (nbits - 1) - 1
    # LSB_P is generally not exactly representable (e.g. 8/75), so P / LSB_P
    # can land 1e-15 below an exact tie; the same guard the SAR uses
    # (TIE_GUARD_REL * V_REF, = TIE_GUARD_REL * 2^N in code units) resolves it.
    guard = SARADC.TIE_GUARD_REL * 2 ** nbits
    acc = np.zeros((N, wq.shape[1]), np.int64)
    for b in range(n_db):
        P = xp[:, b * rows:(b + 1) * rows] @ wp[b * rows:(b + 1) * rows]
        acc += np.clip(np.floor(P / lsb_p + 0.5 + guard), lo, hi).astype(np.int64)
    return acc.astype(np.float64) * lsb_p * sx * sw


rng = np.random.default_rng(2026)

# --------------------------------------------------------------------------- #
print("\nTest 1 — analog cim_matmul == closed-form per-depth-block quantizer")
# --------------------------------------------------------------------------- #
n_cfg = n_bad = 0
worst = ""
for (wb, ab), signed, gain, nb in itertools.product(
        [(4, 4), (4, 8), (8, 4), (8, 8)], [True, False],
        [1.0, 2.0, 16.0, 2.92, 23.77, 300.0], [8, 10, 12]):
    an = AnalogConfig(enabled=True, dac_bits=8, gain_default=gain, adc_bits=nb)
    arr = CIMArray(CIMConfig(weight_bits=wb, act_bits=ab, analog=an))
    X = rng.normal(size=(37, 75))
    W = rng.normal(size=(75, 45))
    if not signed:
        X = np.abs(X) ** 3                       # heavy-tailed, non-negative
    y = cim_matmul(X, W, arr, act_signed=signed)
    ref = closed_form(X, W, wb, ab, signed, arr.chain.lsb_p, nb)
    bad = int(np.count_nonzero(y != ref))
    n_cfg += 1
    if bad:
        n_bad += 1
        worst = worst or f"first failure W{wb}A{ab} signed={signed} G={gain} ADC{nb}: {bad} outputs"
check(f"bit-exact in all {n_cfg} configurations (incl. ties and clipping)",
      n_bad == 0, worst)

# the provisional act->DAC alignment only moves LSB_P; the contract holds
n_bad = 0
for align, dac_bits in [("msb", 8), ("lsb", 8), ("msb", None)]:
    for signed, gain in itertools.product([True, False], [1.0, 2.0, 23.77]):
        an = AnalogConfig(enabled=True, dac_bits=dac_bits,
                          dac_input_alignment=align, gain_default=gain, adc_bits=10)
        arr = CIMArray(CIMConfig(weight_bits=4, act_bits=4, analog=an))
        X = rng.normal(size=(20, 70))
        X = X if signed else np.abs(X)
        W = rng.normal(size=(70, 40))
        y = cim_matmul(X, W, arr, act_signed=signed)
        n_bad += int(np.count_nonzero(y != closed_form(X, W, 4, 4, signed,
                                                       arr.chain.lsb_p, 10)))
check("capacitive DAC, MSB/LSB alignment and act-wide DAC obey the same contract",
      n_bad == 0, f"{n_bad} mismatching outputs")

# exact ties really occur and resolve upward
adc = SARADC(bits=12, vref=0.6)
k = np.arange(-2000, 2000)
v_tie = (k + 0.5) * adc.lsb_volts
check("exact-threshold inputs resolve upward (floor(x+1/2))",
      np.array_equal(adc.convert(v_tie), k + 1))
check("zero volts -> code 0; rails unchanged",
      int(adc.convert(np.array([0.0]))[0]) == 0
      and int(adc.convert(np.array([1.0]))[0]) == 2047
      and int(adc.convert(np.array([-1.0]))[0]) == -2048)

# --------------------------------------------------------------------------- #
print("\nTest 2 — token-block and array-count neutrality on the analog path")
# --------------------------------------------------------------------------- #
X = rng.normal(size=(196, 160))
W = rng.normal(size=(160, 480))
for (wb, ab), gain, nb in [((4, 4), 1.0, 12), ((4, 4), 2.0, 10), ((4, 4), 16.0, 8),
                           ((8, 8), 1.0, 12), ((8, 8), 23.77, 8)]:
    an = AnalogConfig(enabled=True, dac_bits=8, gain_default=gain, adc_bits=nb)
    outs = {}
    for T in (None, 1, 7, 64):
        arr = CIMArray(CIMConfig(weight_bits=wb, act_bits=ab, token_block=T, analog=an))
        outs[T] = cim_matmul(X, W, arr)
    arrs = [CIMArray(CIMConfig(weight_bits=wb, act_bits=ab, analog=an)) for _ in range(3)]
    outs["P3"] = cim_matmul(X, W, arrays=arrs)
    diffs = {t: int(np.count_nonzero(outs[t] != outs[None])) for t in (1, 7, 64, "P3")}
    check(f"W{wb}A{ab} G={gain} ADC{nb}: T=1/7/64 and P=3 bit-identical to T=all",
          not any(diffs.values()), str(diffs))

# --------------------------------------------------------------------------- #
print("\nTest 3 — CIMAttention gradient guard")
# --------------------------------------------------------------------------- #
import torch                                                     # noqa: E402

import run_tinyvit as R                                          # noqa: E402
from cim_sim.attention import CIMAttention, set_cim_enabled      # noqa: E402

model = R.build_model()
ref = [m for m in model.modules() if type(m).__name__ == "Attention"][0]
x = torch.randn(1, 49, ref.norm.normalized_shape[0])
cim = CIMAttention(ref, CIMArray(CIMConfig()))

cim.train()
try:
    cim(x)
    check("train mode + grad enabled raises", False)
except RuntimeError:
    check("train mode + grad enabled raises", True)

cim.eval()
with torch.no_grad():
    out_eval = cim(x)
check("eval + no_grad forward still works", out_eval.shape == x.shape)

cim.train()
with torch.no_grad():
    out_tr = cim(x)
check("train mode under no_grad works and equals eval output",
      torch.equal(out_tr, out_eval))

cim.cim_ops = set()                      # nothing on the array -> pure torch
y = cim(x).sum()
y.backward()
check("no CIM sites selected: train + grad allowed, qkv.weight gets a gradient",
      cim.qkv.weight.grad is not None and float(cim.qkv.weight.grad.abs().sum()) > 0)

cim.cim_ops = {"qkv", "qk", "av"}
cim.zero_grad()
set_cim_enabled(cim, False)              # float path for training
cim(x).sum().backward()
check("set_cim_enabled(False): float path trains normally",
      float(cim.qkv.weight.grad.abs().sum()) > 0)

# --------------------------------------------------------------------------- #
print("\nTest 4 — initial QAT operating point (defaults)")
# --------------------------------------------------------------------------- #
d_an = AnalogConfig(enabled=True)
check("QAT_INITIAL_ADC_BITS == 8 and AnalogConfig defaults to it",
      QAT_INITIAL_ADC_BITS == 8 and d_an.adc_bits == 8 and d_an.adc_vref == 0.6)
d_arr = CIMArray(CIMConfig(weight_bits=4, act_bits=4, analog=d_an))
check("default chain: capacitive DAC 8 b @ 0.6 V, MSB alignment, SAR ADC 8 b",
      isinstance(d_arr.chain.dac, CapacitiveDAC) and d_arr.chain.dac.bits == 8
      and d_arr.chain.dac.vref == 0.6 and d_arr.chain.dac_alignment == "msb"
      and d_arr.chain.adc.bits == 8)
a12 = CIMArray(CIMConfig(analog=AnalogConfig(enabled=True, adc_bits=12)))
check("ADC remains configurable for later studies (12 b accepted)",
      a12.chain.adc.bits == 12)

# --------------------------------------------------------------------------- #
print(f"\n{sum(PASS)}/{len(PASS)} checks passed")
sys.exit(0 if all(PASS) else 1)
