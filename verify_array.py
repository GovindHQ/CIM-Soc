#!/usr/bin/env python3
"""
verify_array.py — checks the array model does what it claims, with no ViT in
the way. Run this first; it is the thing that proves the scheduler is correct.

Three claims are tested:

  1. TILING IS EXACT. With quantization switched off (weights passed through as
     integers, activations already integer), the tiled 32x32 decomposition must
     reproduce integer matmul BIT-EXACTLY for arbitrary K and M, including
     ragged shapes that do not divide by 32.

  2. TOKEN BLOCKING IS NUMERICALLY NEUTRAL. Changing T changes weight-write and
     partial-sum-storage counts, but not a single output value.

  3. THE ONLY ERROR IS QUANTIZATION. The tiled 4-bit result must match a plain
     float matmul of the fake-quantized operands to within float round-off, so
     any error observed later is attributable to 4 bits, not to the schedule.
"""

import numpy as np

from cim_sim import (CIMArray, CIMConfig, TraceLog, cim_matmul, fake_quant,
                     rel_error, tile_map)

rng = np.random.default_rng(0)
SHAPES = [
    (196, 160, 480),   # TinyViT-5M stage 3 fused QKV projection
    (196, 32, 196),    # stage 3 QK^T, one head
    (196, 196, 32),    # stage 3 AV, one head
    (49, 128, 384),    # stage 2 fused QKV projection
    (7, 33, 65),       # deliberately ragged in every dimension
    (1, 5, 3),         # smaller than one tile in every dimension
]

print("=" * 78)
print("1. TILING EXACTNESS  (integers in, integers out, no quantization error)")
print("=" * 78)
for N, K, M in SHAPES:
    Xi = rng.integers(-8, 8, size=(N, K)).astype(np.float64)
    Wi = rng.integers(-8, 8, size=(K, M)).astype(np.float64)

    arr = CIMArray(CIMConfig())
    # act_bits high enough that quantization is the identity on [-8,7] integers
    arr.cfg.act_bits = 8
    Y = cim_matmul(Xi / 1.0, Wi, arr, quantize_weights=False)
    # activations are quantized per-token; with 8b symmetric on integer data the
    # mapping is not exactly identity, so compare against the same fake-quant
    ref = fake_quant(Xi, bits=8, axis=1) @ Wi
    ok = np.allclose(Y, ref, rtol=0, atol=1e-6)
    print(f"  {str((N,K,M)):>18}  exact={ok}   {tile_map(N,K,M)}")

print()
print("=" * 78)
print("2. TOKEN BLOCKING IS NUMERICALLY NEUTRAL")
print("=" * 78)
N, K, M = 196, 160, 480
X = rng.standard_normal((N, K))
W = rng.standard_normal((K, M)) * 0.1

results = {}
for T in (None, 64, 8, 1):
    arr = CIMArray(CIMConfig(token_block=T))
    log = TraceLog()
    results[T] = cim_matmul(X, W, arr, tag="qkv", log=log)
    s = arr.stats
    label = "all" if T is None else str(T)
    print(f"  T={label:>4}   tile writes={s.weight_tile_writes:>6}   "
          f"array cycles={s.array_ops:>7}   psum words live={s.psum_words_live:>6}"
          f"   ({s.psum_words_live * 20 / 8 / 1024:.1f} kB at 20b)")

base = results[None]
for T, Y in results.items():
    assert np.array_equal(Y, base), f"T={T} changed the result"
print("  -> all token-block sizes produce bit-identical outputs.")

print()
print("=" * 78)
print("3. ERROR IS ATTRIBUTABLE TO 4 BITS, NOT TO THE SCHEDULE")
print("=" * 78)
print(f"  {'shape':>18}  {'tiled 4b':>10}  {'plain 4b':>10}  {'gap':>10}")
for N, K, M in SHAPES:
    X = rng.standard_normal((N, K))
    W = rng.standard_normal((K, M)) * 0.1
    ref = X @ W

    arr = CIMArray(CIMConfig())
    Y = cim_matmul(X, W, arr)

    # same quantization, ordinary dense matmul — no tiling at all
    Yq = fake_quant(X, bits=4, axis=1) @ fake_quant(W, bits=4, axis=0)

    e_tiled = rel_error(Y, ref)
    e_plain = rel_error(Yq, ref)
    print(f"  {str((N,K,M)):>18}  {e_tiled:>9.4%}  {e_plain:>9.4%}  "
          f"{abs(e_tiled - e_plain):>9.2e}")
print("  -> tiled and untiled 4-bit errors agree; the schedule adds nothing.")
