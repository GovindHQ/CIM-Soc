#!/usr/bin/env python3
"""
test_multi_array.py — multi-array scheduling invariants (CLAUDE.md §3, §11, §12).

NOTE: CLAUDE.md referred to a test_multi_array.py that was NOT present in the
delivered repository. This file is a reconstruction (Oct 2026) that checks the
invariants CLAUDE.md states, against the code actually present:

  1. P=1 equivalence: arrays=[a] reproduces array=a bit-for-bit, with
     identical statistics.
  2. Numerical invariance: any P gives bit-identical outputs (legacy exact,
     legacy ADC, and the analog chain).
  3. Physical work invariance: CIMStats.array_ops (summed over arrays) does not
     change with P; parallel cycles = sum over (token block, group, depth
     block) of token-block length.
  4. Group scheduling geometry: parallel_groups = ceil(n_ob / P),
     array_parallel_utilization = n_ob / (groups * P).
  5. Determinism: the same configuration run twice is bit-identical.

Run:  python test_multi_array.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from cim_sim import CIMArray, CIMConfig, TraceLog, cim_matmul
from cim_sim.analog import AnalogConfig

rng = np.random.default_rng(0)
PASS = []


def check(name, cond, detail=""):
    PASS.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def stats_tuple(s):
    return tuple(sorted(vars(s).items()))


CFGS = {
    "legacy exact": dict(weight_bits=4, act_bits=4),
    "legacy ADC 8b": dict(weight_bits=8, act_bits=8, adc_enabled=True, adc_bits=8),
    "analog CDAC": dict(weight_bits=8, act_bits=4,
                        analog=AnalogConfig(enabled=True,
                                            dac_bits=8, adc_bits=8,
                                            gain_per_op={"qkv": 23.77})),
}
SHAPES = [(196, 160, 480), (49, 33, 65), (7, 196, 32)]

# --------------------------------------------------------------------------- #
print("\nTest 1 — P=1 equivalence: arrays=[a] == array=a, outputs and stats")
# --------------------------------------------------------------------------- #
for name, kw in CFGS.items():
    for N, K, M in SHAPES:
        X = rng.standard_normal((N, K))
        W = rng.standard_normal((K, M)) * 0.1
        a1, a2 = CIMArray(CIMConfig(**kw)), CIMArray(CIMConfig(**kw))
        l1, l2 = TraceLog(), TraceLog()
        y1 = cim_matmul(X, W, array=a1, log=l1, op="qkv")
        y2 = cim_matmul(X, W, arrays=[a2], log=l2, op="qkv")
        check(f"{name} {(N, K, M)}: identical output and stats",
              np.array_equal(y1, y2) and stats_tuple(a1.stats) == stats_tuple(a2.stats)
              and l1.total_parallel_cycles() == l2.total_parallel_cycles() == a1.stats.array_ops)

# --------------------------------------------------------------------------- #
print("\nTest 2+3+4 — any P: same numbers, same physical work, correct grouping")
# --------------------------------------------------------------------------- #
for name, kw in CFGS.items():
    N, K, M = 196, 160, 480
    X = rng.standard_normal((N, K))
    W = rng.standard_normal((K, M)) * 0.1
    ref, ref_ops = None, None
    for P in (1, 2, 3, 4, 8):
        for T in (None, 32):
            cfg = CIMConfig(token_block=T, **kw)
            arrs = [CIMArray(cfg) for _ in range(P)]
            log = TraceLog()
            y = cim_matmul(X, W, arrays=arrs, log=log, op="qkv")
            ops = sum(a.stats.array_ops for a in arrs)
            rec = log.records[0]
            n_ob, n_db = -(-M // 32), -(-K // 32)
            groups = -(-n_ob // P)
            exp_cycles = groups * n_db * N
            if ref is None:
                ref, ref_ops = y, ops
            ok = (np.array_equal(y, ref) and ops == ref_ops
                  and rec.parallel_groups == groups
                  and log.total_parallel_cycles() == exp_cycles
                  and abs(rec.array_parallel_utilization - n_ob / (groups * P)) < 1e-12)
            check(f"{name}: P={P} T={T or 'all'}", ok,
                  f"cycles={log.total_parallel_cycles()} ops={ops} groups={groups}")

# --------------------------------------------------------------------------- #
print("\nTest 5 — determinism: same config twice is bit-identical")
# --------------------------------------------------------------------------- #
for name, kw in CFGS.items():
    X = rng.standard_normal((49, 128))
    W = rng.standard_normal((128, 384)) * 0.1
    ys = [cim_matmul(X, W, arrays=[CIMArray(CIMConfig(**kw)) for _ in range(3)], op="qkv")
          for _ in range(2)]
    check(f"{name}: repeat run identical", np.array_equal(ys[0], ys[1]))

print(f"\n{sum(PASS)}/{len(PASS)} checks passed")
sys.exit(0 if all(PASS) else 1)
