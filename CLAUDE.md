# CLAUDE.md — CIM / TinyViT Simulator

Persistent engineering specification for this repository. Read this before modifying code.

> **Revision Oct 2026.** Updated after an audit of the delivered code (see `AUDIT.md`)
> and the addition of the real-image experiment pipeline (`run_experiment.py`,
> `cim_exp/`, see `EXPERIMENTS.md`). Changes vs the previous revision: input DAC now
> hardware-specified (unsigned 8-bit CDAC, §6/§9); act->DAC alignment; analog
> statistics rewritten (unbiased, mergeable, V_ADC_IN + rail + amplifier + common-mode
> diagnostics); `run_tinyvit.py --analog`; `test_multi_array.py` reconstructed;
> `verify_array.py` now exits non-zero on failure; Imagenette class-mapping trap (§8).
> Historical numbers in §8 were produced on synthetic inputs, mostly via the LEGACY
> path, and are provisional until reproduced.
> **Second revision (Oct 2026):**
> - QAT context incorporated (§8, §9 Q6, AUDIT.md §10). The QAT checkpoints were
>   trained against the LEGACY functional ADC at W4/A4 and W8/A8, with the simulator's
>   own quantizer.
> - Imagenette training-data-overlap caveat.
> - Experiment A (fixed gains) separated from Experiment B (gain calibration and
>   robustness) and the optional Experiment T (QAT training-point check).
> - Configurable amplifier swing limit for diagnostics; the default remains an ideal
>   amplifier, reported as not modelled.
> **Third revision (21 Oct 2026, pre-QAT):**
> - **Capacitive DAC only.** The four comparison DAC models (IDEAL, R2R, CURRENT_STEERING,
>   DIFFERENTIAL), `DAC_TYPES`, `make_dac`, `AnalogConfig.dac_type`, `run_tinyvit.py
>   --dac-type` and `exp_analog.py --dac-compare` were removed. `AnalogConfig.dac_bits`
>   now defaults to the hardware's 8 (bit-identical to the old `None` default with MSB
>   alignment, verified on the full model).
> - The ADC stays configurable; the initial QAT operating point is an 8-bit ADC
>   (`cim_sim.analog.QAT_INITIAL_ADC_BITS`, the `AnalogConfig.adc_bits` default).
> - SAR exact-tie guard, CIMAttention gradient guard, `test_qat_contract.py` (see CHANGES.md).

This file separates **established facts about the current code** from **proposed future
architecture**. Anything marked `[FUTURE]` is not implemented. Anything marked `[OBSOLETE]`
was tried and deliberately abandoned — do not reintroduce it.

---

## 1. Project purpose

This repository is a **functional software model of an analog Compute-In-Memory (CIM) array**,
used to evaluate whether ViT attention can be executed on that hardware before the silicon
exists.

The thesis goal is a reconfigurable CIM accelerator SoC for Vision Transformer inference. The
intended primitive is a **32×32 analog C2C (capacitor-to-capacitor) macro** with SRAM-held
digital weights, one SAR ADC per column. TinyViT-5M is the reference workload.

The simulator exists to answer questions that must be settled *before* RTL:

- Does attention survive 4-bit / 8-bit operands and a finite-resolution ADC?
- Which of the attention matmuls is the accuracy bottleneck?
- What ADC resolution, full-scale range, and pre-ADC gain does the workload actually require?
- How do array size, array count, and tiling affect latency, utilization, and error?

It is **not** a performance model of PyTorch, and **not** a transistor-level model. It is a
behavioral model whose physical boundaries (DAC / macro / amplifier / ADC) are explicit so that
nonidealities can be added independently later.

---

## 2. Current simulator architecture

### Repository state (important)

The repo contains these **baseline** files (all present and working):

```
run_tinyvit.py        CLI driver, model build, experiment modes
tiny_vit.py           UNMODIFIED upstream microsoft/Cream TinyViT source
_shim/timm/           5-symbol timm stand-in so tiny_vit.py imports without real timm
cim_sim/__init__.py   package exports
cim_sim/array.py      CIMArray, CIMConfig, CIMStats, legacy _adc_convert
cim_sim/matmul.py     the scheduler (depth/output/token blocking, multi-array)
cim_sim/attention.py  CIMAttention nn.Module + model surgery
cim_sim/quant.py      symmetric integer quantization + error metrics
cim_sim/report.py     tables (report(), tile_map())
experiments.py        the four thesis plots
verify_array.py       tiling/blocking correctness harness
test_cim_array.py     tiny ad-hoc smoke script (NOT a regression suite)
test_matmul.py        tiny ad-hoc smoke script (NOT a regression suite)
test_quant.py         tiny ad-hoc smoke script (NOT a regression suite)
```

**Verified present (Oct 2026):** the analog datapath `cim_sim/analog/` (wired into
`array.py` / `matmul.py` / `attention.py`), `test_analog.py`, `exp_analog.py`, and:

```
run_experiment.py            real-image (Imagenette) experiment driver
cim_exp/                     imagenette, checkpoints, hardware, metrics, engine,
                             reporting, verify  (+ data/imagenet_synsets.txt)
configs/checkpoints.template.json
test_multi_array.py          reconstructed (Oct 2026) - the referenced original was absent
test_experiment_pipeline.py  offline tests of cim_exp
EXPERIMENTS.md / AUDIT.md / CHANGES.md
```

**Not present** in the delivered tree: `exp_array_size.py`, `exp_per_op_fs.py`
`[OBSOLETE, see §14]`, `analyze_adc.py`. Results attributed to them cannot be reproduced
from this repository.

**Which path ran?** `run_tinyvit.py` and `experiments.py` run the LEGACY digital-MAC model
unless `run_tinyvit.py --analog` is given (`experiments.py` has no analog option).
`exp_analog.py`, `run_experiment.py` and `test_analog.py` use the analog chain.

### Module responsibilities

**`tiny_vit.py`** — verbatim upstream source. Do not edit it. Model structure facts are read
from here, not from memory.

**`_shim/timm/`** — supplies exactly five symbols (`DropPath`, `trunc_normal_`, `to_2tuple`,
`register_model`, `build_model_with_cfg`). The builder raises `NotImplementedError` on purpose:
`run_tinyvit.py` constructs `TinyViT(**TINYVIT_5M)` directly and calls `load_state_dict`. Real
`timm` is never needed.

**`cim_sim/quant.py`** — `int_range`, `quantize`, `dequantize`, `fake_quant`, `rel_error`,
`cosine_sim`. The boundary between float and the integer world.

**`cim_sim/array.py`** — `CIMArray` models exactly **one** 32×32 macro. Owns the resident weight
tile, the activity counters (`CIMStats`), the legacy ADC, and (when configured) the analog chain.
`CIMConfig` is the single hardware configuration object.

**`cim_sim/matmul.py`** — the scheduler, and the only place that knows about tiling. Turns an
arbitrary `X[N,K] @ W[K,M]` into 32×32 array operations. Owns depth blocking, output blocking,
token blocking, multi-array output-tile parallelism, digital accumulation, and the single
end-of-chain rescale. `TraceLog` / `MatmulTrace` carry per-call statistics.

**`cim_sim/attention.py`** — `CIMAttention` is a drop-in `nn.Module` replacement for TinyViT's
`Attention`, structurally mirroring the upstream `forward`. `convert_attention(model, ...)`
walks the model and swaps every `Attention` (matched by class name, so it works with any TinyViT
copy). `set_cim_enabled` / `set_cim_ops` toggle the CIM path per site.

**`cim_sim/report.py`** — rendering only. `report(log)` produces the op/tile/cycle table;
`tile_map(N,K,M)` produces a one-line tiling description.

**`run_tinyvit.py`** — builds the pretrained model (auto-downloads
`tiny_vit_5m_22kto1k_distill.pth` on first run), generates synthetic inputs, and hosts the
experiment modes.

**`cim_sim/analog/`** (see §6) — `dac.py`, `macro.py`, `adc.py`, `digital.py`, `chain.py`.

**`cim_exp/` + `run_experiment.py`** (see §8 and `EXPERIMENTS.md`) — official Imagenette
download/extraction, verified class mapping, deterministic manifests, TinyViT eval
preprocessing, hash-checked checkpoint registry, the 24-configuration hardware matrix,
FP32 reference caching, resumable/parallel CIM jobs, the five comparison cases, all CSVs
and `REPORT.md`, and the `verify` pre-flight suite.

### Data flow

```
FP32 TinyViT
  └─ CIMAttention.forward
       ├─ LayerNorm                          (float, off-CIM)
       ├─ qkv projection      ─┐
       ├─ QK^T                 ├─ cim_matmul / cim_matmul_batched
       ├─ softmax + biases     │    (float, off-CIM)
       ├─ AV                  ─┘
       └─ output projection                  (float by default)

cim_matmul(X, W, ...)
  1. quantize X per token (axis=1), W per output column (axis=0)
  2. zero-pad K → multiple of 32 (depth blocks), M → multiple of 32 (output blocks)
  3. loop: token block → parallel group → depth block → array slot
  4. CIMArray.mvm() per (depth block, output tile)
  5. accumulate partial sums as integers
  6. ONE end-of-chain rescale back to float
```

---

## 3. CIM architecture (established)

- **32×32 array.** `rows` = contraction depth held by one macro, `cols` = output width.
  Configurable (`CIMConfig.rows/cols`), and 16×16 has been tested, but 32 is the design point.
- **Weight-stationary.** A 32×32 integer weight tile is written into the array and stays
  resident while activations stream through it.
- **Row broadcast, column sum.** One 32-element input vector is broadcast along the rows; each
  column yields the sum of its 32 row-wise products. 32 column sums out per input vector.
- **Depth blocking.** `n_db = ceil(K / rows)`. Partial sums across depth blocks are accumulated
  **digitally**, never in the analog domain.
- **Output blocking.** `n_ob = ceil(M / cols)`. Ragged edges are zero-padded; the padding *is*
  the idle rows/columns of the real array, and shows up in `CIMStats.utilization`.
- **Token blocking `T`** (`CIMConfig.token_block`). Weight tiles stay resident while `T` tokens
  stream through. `T` trades weight-write energy against live partial-sum storage:
  writes = `ceil(N/T)·n_ob·n_db`, psum words = `T·cols`. **All values of `T` produce
  bit-identical results** — only the counters change. This is asserted by `verify_array.py`.
- **Single weight plane.** `CIMConfig.planes` must be 1; `__post_init__` rejects anything else.
  Every distinct tile costs a real write. Four-plane residency is `[FUTURE]`.
- **Batching note.** `CIMArray.mvm(x)` accepts `[T, rows]`. That axis is **T consecutive
  cycles**, not parallel hardware. Counters treat it as T separate array operations.

### Multiple arrays (implemented)

`cim_matmul(..., arrays=[a0, a1, ...])` schedules across P identical macros using
**output-tile parallelism**: the M axis is divided into output tiles, dealt out P at a time
("parallel groups"). Within a group, one activation tile is broadcast to every active array,
each holding a *different* output tile's weights.

```
for token_block:
    for group in ceil(n_ob / P):
        for depth_block:                     ← one hardware timestep
            for slot, ob in group_tiles:
                arrays[slot].load_weight_tile(W[db, ob])
                acc[:, ob] += arrays[slot].mvm(x_tile[db])
```

- P=1 generates the **exact same call sequence in the exact same order** as the original
  single-array nest. `arrays=[a]` ≡ `array=a`, bit-for-bit.
- Ragged last group leaves slots idle; `MatmulTrace.array_parallel_utilization` measures this.
- `CIMConfig.n_arrays` is **descriptive metadata only**. The real array count is `len(arrays)`.
  A single `CIMArray` always models one macro regardless of that field.
- **Cycle model:** `CIMStats.array_ops` (summed across arrays) is *total physical work* and does
  **not** change with P. `TraceLog.total_parallel_cycles()` is wall-clock, incremented once per
  (group, depth block). Speedup = `array_ops / parallel_cycles`.
- `convert_attention` shares **one array pool across every layer** — a real accelerator has
  fixed macros time-multiplexed over the whole forward pass, not one set per layer.

Two different utilization metrics exist and must not be conflated:

| metric | meaning |
|---|---|
| `CIMStats.utilization` | real MACs / issued MACs *inside* a 32×32 tile (padding waste) |
| `MatmulTrace.array_parallel_utilization` | active array slots / total slots across groups |

---

## 4. TinyViT attention mapping

### Model facts (read from `tiny_vit.py`, not assumed)

TinyViT-5M-224: `embed_dims=[64,128,160,320]`, `depths=[2,2,6,2]`, `num_heads=[2,4,5,10]`,
`window_sizes=[7,7,14,7]`, `mlp_ratio=4.0`.

In `TinyViTBlock.__init__`: `head_dim = dim // num_heads`, and
`Attention(dim, head_dim, num_heads, attn_ratio=1, resolution=window_resolution)`.
Therefore `key_dim = head_dim`, `d = attn_ratio·key_dim = key_dim`, `dh = dim`, and the fused
`qkv = nn.Linear(dim, 3·dim)`.

Stage 3 (dim=160, 5 heads, window 14 → N=196): `key_dim = 32`, `qkv = [160 → 480]`.

| site | shape | tiles | array utilization |
|---|---|---|---|
| qkv projection | `[196×160] @ [160×480]` | 5 depth × 15 output = 75 | 100% |
| QK^T (per head) | `[196×32] @ [32×196]` | 1 × 7 = 7 | 87.5% |
| AV (per head) | `[196×196] @ [196×32]` | 7 × 1 = 7 | 87.5% |

### What runs where

| operation | current implementation | notes |
|---|---|---|
| LayerNorm | float, off-CIM | SIMD/LUT unit in hardware |
| **qkv projection** | **CIM** | fused `[B·N, dim] @ [dim, 3·dim]`; batch and token axes merge |
| **QK^T** | **CIM** | per head, via `cim_matmul_batched`; dynamic weights |
| attention-bias add | float, off-CIM | learned biases |
| softmax | float, off-CIM | serialization bubble in hardware |
| **AV** | **CIM** | per head; **activations quantized UNSIGNED** (`act_signed=False`) |
| output projection | float by default | on CIM only with `cim_out_proj=True` |
| MLP | float, never on CIM | out of scope |
| residual adds | float | |

`CIMAttention.cim_ops` (a set drawn from `{"qkv","qk","av"}`) selects which sites use the array;
anything not listed falls back to float torch. This is how the ablation experiment isolates
each site's contribution.

**Not differentiable.** The array path runs in NumPy on detached tensors. Since Oct 2026
`CIMAttention.forward` raises if called in training mode with autograd enabled while any
site is on the array (it would silently give `qkv.weight` and everything upstream zero
gradient). QAT needs a differentiable surrogate validated against this module
(`test_qat_contract.py` states the exact per-depth-block contract).

`[FUTURE]` MLP layers, convolutions via im2col (the scheduler already handles them — it does not
care whether a row is a token or an output pixel), and the output projection as a default CIM site.

---

## 5. Quantization

**Scheme:** symmetric, zero-point free, integer.

```
X_real = S_X · X_q        S_X: one scale per token          (quantize(..., axis=1))
W_real = S_W · W_q        S_W: one scale per output column  (quantize(..., axis=0))
y_int  = Σ_k X_q[n,k] · W_q[k,m]
y_real = y_int · S_X[n] · S_W[m]                             ← ONE rescale at the end
```

**The governing constraint:** for a single end-of-chain rescale to be valid, the scales must be
**constant along the contraction axis K**. That is why activation scale is per-token (shared
across all depth blocks of that token) and weight scale is per-output-column (shared across all
depth blocks feeding that column). A per-depth-block scale would be more accurate but would
force a rescale-and-requantize between every depth block, defeating the cheap integer
accumulator. **This is a hardware-imposed constraint, not a tuning choice.**

**Signedness:**
- Weights: always signed two's complement, `[-2^(b-1), 2^(b-1)-1]`.
- Activations: signed by default. **Exception:** post-softmax AV activations are quantized
  **unsigned** `[0, 2^b-1]` (`act_signed=False`) — probabilities have no negative side and a
  sign bit would be wasted. Anything touching the activation path must respect this; the analog
  DAC carries an explicit input-code-domain flag because of it (see §6).
- The positive limit is used for the scale, so the extra negative code (−8 in 4b) is clipping
  headroom.

**Bit widths:** `CIMConfig.weight_bits` and `act_bits`, independently configurable. 4b and 8b
are both exercised. **Never hard-code INT4.** Bit-slicing to reach 8b from 4b hardware is
`[FUTURE]` and not implemented — `weight_bits=8` means 8-bit cells.

**Reference path:** the FP32 TinyViT forward is the ground truth. `fake_quant` (quantize then
immediately dequantize, no tiling) isolates quantization error from scheduling error.

---

## 6. Analog datapath (`cim_sim/analog/`)

The physical model of the intended inference datapath. **Off by default**
(`CIMConfig.analog = None`). When enabled it replaces the legacy digital-MAC abstraction inside
`CIMArray.mvm`.

```
X_q → InputDAC → V_IA → C2CMacro → V_OA → PreADCGain → V_ADC_IN → SARADC → d
    → DigitalAccumulator (int64) → Requantizer → y
```

Every arrow is a real object boundary carrying a physical quantity. The ideal equations do
collapse algebraically to `V_OA = K_MAC · Σ(X_q·W_q)` — that is acknowledged and reported — but
**the collapse is not the implementation**. `AnalogChain.execute()` really builds volts, hands
them to the macro, scales them through the amplifier, and runs a bit-by-bit SAR search, so that
DAC/macro/amplifier/ADC nonidealities can be added independently later.

### Input DAC (`dac.py`)

**Hardware-specified (Oct 2026):** the input DAC is an **unsigned, binary-weighted
capacitive DAC, 8 bits, V_REF_DAC = 0.6 V**, `V_out = V_REF·(b7/2 + … + b0/256) = V_REF·c/2^N`
(all-ones -> `V_REF·(1−2^-N)`, never `V_REF`). `CapacitiveDAC.vout_from_unsigned_code` is
exactly this equation and is tested bit by bit (test_analog Test 11). **It is the only input
DAC in the simulator** (Oct 2026): the comparison candidates (ideal bipolar, R-2R,
current-steering, differential) were removed once the topology was confirmed, and there is
no `dac_type` option.

**Not specified — PROVISIONAL assumptions, not hardware facts:** (a) signed activations are presented to the
unsigned CDAC in offset binary `c = X_q + 2^(N−1)` and the `V_REF/2` common mode is removed
**ideally** (signed-equivalent domain); (b) an activation narrower than the DAC is
**MSB-aligned** (`AnalogConfig.dac_input_alignment="msb"`, code × 2^(dac_bits−act_bits)),
which gives *exactly* the transfer of an act_bits-wide DAC (verified bit-identical on the
full model). `"lsb"` alignment (code applied as-is, 1/16 of the swing for A4) is available.
`AnalogConfig.dac_bits` defaults to **8** (the hardware). `dac_bits=None` means "follow
act_bits" (the pre-Oct-2026 default); with MSB alignment it is bit-identical to 8 for A4/A8.
`dac_bits < act_bits` is rejected.

| property | capacitive DAC |
|---|---|
| unsigned code `c` | signed inputs: `X_q + 2^(N−1)` (offset binary, provisional); unsigned (AV): `X_q` |
| physical output | `V_REF·c/2^N` (hardware equation) |
| `V_IA` (signed-equivalent) | `V_REF·c/2^N − V_cm`, `V_cm = V_REF/2` (signed) or 0 (unsigned) |
| slope `v_per_unit` | `V_REF/2^N` in both domains |

Because offset binary spends half the codes on negative values, the usable signed swing is
±V_REF/2; a natively bipolar DAC would have twice the slope, so `K_MAC`, and hence the
required amplifier gain, depends on this provisional signed convention (see below).

`V_IA` is returned as a **signed analog-equivalent** voltage (code 0 → 0.0 V);
`v_common_mode` reports the physical offset.

The DAC carries an explicit **input code domain** (`set_input_signed`) because AV's activations are
unsigned. A DAC that assumed one domain would silently reinterpret the other. The domain is part
of `domain_key`.

`DAC_bits` is independent of `act_bits` and of `adc_bits`. `K_MAC` includes the alignment
factor: `K_MAC = v_per_unit · 2^(dac_bits−act_bits if msb) / (m·2^B)`.

### C2C macro (`macro.py`)

Hardware-supplied first-order behavioral relation:

```
V_OA = (1/m) · Σ_j ( V_IA,j · W_j / 2^B )
```

`m` = number of MAC units sharing the output node. It is a **physical** property of the column
(how many capacitors hang on the node), so it equals the macro's row count — **not** the number
of rows carrying real data in a ragged tile. Zero-padded rows still hold a capacitor and still
divide the sum.

Contributions combine by **capacitive charge sharing**, so the node develops the *normalised
(averaged)* sum, not the raw sum. Confirmed by the hardware team. Note: the 1/m normalisation
cancels between signal and full scale, so it does not by itself change any digitized value —
it was verified that a "sum" and an "average" model give bit-identical ADC codes when the full
scale is sized consistently.

### Signed weights — `SignedWeightAdapter`

`W_j / 2^B` in the relation above is a **non-negative binary-weighted capacitor fraction**.
Feeding a two's-complement word into it is a category error: the pattern `1011` means −5 as a
signed 4-bit number but would be read as 11/16 of the input by a capacitor ladder.

The adapter splits the weight **before** the capacitor equation:

```
W_q → (sign = ±1, magnitude = |W_q|)
magnitude → capacitor code |W_q| / 2^B    (non-negative, the ONLY thing the ladder sees)
sign      → applied at an explicit behavioral boundary (C2CMacro._combine)
```

**Sign-magnitude was chosen deliberately** over differential accumulation, positive/negative
sub-columns, and dual paths — those are all plausible *real circuits*, but each asserts physical
structure nobody has told us exists. Sign-magnitude asserts nothing physical: it is a behavioral
statement true of all of them. `SignedWeightAdapter.is_physical_claim = False` is an attribute on
the class so it cannot be quoted as a hardware fact.

**Replacement points:** if the hardware team specifies differential accumulation, replace
`SignedWeightAdapter` + `C2CMacro._combine`. Nothing else in the chain changes.

### Pre-ADC gain (`adc.py: PreADCGain`)

`V_ADC_IN = G · V_OA`. Amplifier saturation is modelled **only** when
`GainNonidealities.v_swing_limit` is set (`PreADCGain.saturation_modeled`); by default it
is ideal and results report amplifier saturation as `not_modeled` — never substitute ADC
clipping for it. **`G` is a hardware configuration parameter, per operation. It is never
derived from the data at inference time** — doing that recreates the arbitrary normalisation
this design exists to remove. A calibration utility may *recommend* a G, but must stay outside
the inference path. `AnalogConfig.gain_per_op` maps `{"qkv","qk","av"} → G`.

This is **not** the ADC full scale. The ADC keeps its own fixed reference.

### SAR ADC (`adc.py: SARADC`, `SARDAC`)

- Real bit-by-bit successive approximation (N comparator decisions against SAR-DAC trial
  voltages), vectorised across samples. Ideal-mode equivalence to round-to-nearest is **asserted
  by the test suite**, not assumed.
- **Exact ties resolve upward** (`d = floor(V/LSB + 1/2)`, clipped). With `G·K_MAC/LSB` a
  simple ratio (G = 1, power-of-two gains) up to ~50% of conversions sit exactly on a
  threshold; before Oct 2026 float noise (dependent on BLAS shape, i.e. on token block)
  decided them, which broke §11.3 on the analog path. `SARADC.TIE_GUARD_REL·V_REF`
  (1e-9·V_REF) makes the rule deterministic; it is not a physical offset. The full
  surrogate contract is pinned by `test_qat_contract.py`.
- `SARDAC` is the ADC's **internal** comparison DAC. It is a different object from the input DAC.
  Never conflate them.
- **LSB convention: `LSB = Vref / 2^N`.** A binary-weighted SAR DAC's trial weights are
  `Vref·2^k/2^N`, so it steps by `Vref/2^N`. Using `Vref/(2^N − 1)` makes the quantiser
  inconsistent with the SAR DAC driving it and produces a 1-code error at full scale.
  **This differs from the legacy `_adc_convert`, which uses the other convention. Do not
  "harmonise" them — they belong to different models.**
- **Offset binary is applied at the ADC boundary**, because that is the only place the unipolar
  constraint physically exists: `V_phys = Vref/2 + V_ADC_IN`, clipped to `[0, Vref]`. The
  unsigned physical code is immediately re-centred to a signed code
  `d = code − 2^(N-1)`, so 0 V → d = 0 and signed partial sums accumulate in a plain integer
  accumulator with no per-block offset bookkeeping. Signed swing = ±Vref/2 = **±0.3 V**.
- Clipping is counted (`n_clipped_hi`, `n_clipped_lo`: physical input outside `[0, V_REF]`)
  and **never silently rescaled away**. `n_rail_hi/lo` count conversions landing on the
  top/bottom code — an upper bound that also contains correctly-rounded rail values and the
  `(V_REF − LSB/2, V_REF]` region that saturates at the top code without being "clipped".
  The amplifier can also saturate before the ADC does (`GainNonidealities.v_swing_limit`).

### Scale chain and accumulation domains (`digital.py`, `chain.py`)

```
K_MAC = dac.v_per_unit / (m · 2^B)          [volts per unit of Σ X_q·W_q]
LSB_P = adc.lsb_volts / (G · K_MAC)         [column-sum units per ADC code]
y_real = (Σ_blocks d) · LSB_P · S_X · S_W
```

Both are derived **once, from architecture**. Neither ever inspects a tensor.

`LSB_P` is constant within an accumulation domain, so ADC codes are summed as plain int64 and
rescaled once at the end. `AnalogChain.domain_key` fingerprints everything that must match
(DAC name/bits/vref/input-domain, m, weight_bits, gain, ADC bits/vref), and
`DigitalAccumulator.add()` **raises** if a block from a different domain is added. This is the
mechanical guard against mixing analog scales.

`set_operation(op, act_signed)` is called **once per matmul**, never per depth block, so every
partial sum in a domain shares one gain and one input domain.

Hardware scale (`LSB_P`) and NN scales (`S_X`, `S_W`) are kept as **separate factors**, never
merged into one opaque constant, so the two error sources stay separately attributable.

### Statistics (`AnalogStats`)

Bucketed by operation label. Exact running scalars (conversions, clip hi/lo, rail hi/lo,
amplifier saturation, V_OA and V_ADC_IN min/max/mean/std) plus a bounded **uniform weighted**
subsample for percentiles: each call keeps `max(1, int(n·keep_frac))` samples weighted
`n/k`; when the cap is exceeded every sample survives with p=½ and its weight doubles, so
the subsample stays uniform over the whole run (the previous version stopped sampling at
the cap, biasing percentiles toward the first images, and spent ~half the runtime in an
O(n) `sum()`). Mergeable (`merge`, `state_dict`, `from_state`) for chunked runs. Also
records the **common-mode residue** `V_CM·Σ_j sign_j|W_j|/2^B / m` that an offset-binary
input adds per column and that the model cancels ideally, plus the counterfactual share of
conversions that would clip if it were not cancelled before the amplifier. A shared
`AnalogStats` can be passed to `CIMArray(cfg, analog_stats=...)`.
**Observation only — it never feeds back into the gain or the outputs.**

### Verified gain values

`G = (Vref/2) / (K_MAC · FS_op)` where `FS_op` is the per-operation full scale in column-sum
units. `K_MAC` appears **exactly once** (it converts a column-sum full scale into volts); this
was verified dimensionally and by a round-trip test showing ≤0.5 LSB error (pure quantization
floor) across gains (earlier also across the since-removed DAC candidates).

At `w=8b, a=8b, m=32`, `FS = {qkv: 44122, qk: 124674, av: 359280}` (P99.99, calibrated on
synthetic inputs):

| DAC | K_MAC | G_qkv | G_qk | G_av |
|---|---|---|---|---|
| capacitive, 8 b, 0.6 V, offset-binary signed inputs | 2.8610e-07 | **23.77** | **8.41** | **2.92** |

These gains are W8/A8-specific; the QAT operating point needs gains calibrated at its own
precision (and with the 8-bit ADC) on held-out data.
Independently confirmed: a manual gain sweep found `{24, 8, 3}` best (4.66% rms, 100% agreement,
0.012% clipping) without reference to the formula.

---

## 7. Legacy ADC model vs new analog datapath

**These are two separate models. Do not conflate or merge them.**

| | **Legacy** (`array.py: _adc_convert`) | **New** (`cim_sim/analog/`) |
|---|---|---|
| enabled by | `CIMConfig.adc_enabled=True`, `adc_bits`, `adc_vref` | `CIMConfig.analog = AnalogConfig(enabled=True, ...)` |
| input | exact integer column sum `y = x @ W` | real volts from a DAC + C2C macro |
| full scale | `rows · 2^(w-1) · 2^(a-1)`, in **column-sum units** | `±Vref/2` in **volts**; range set by amplifier gain |
| LSB | `2·FS / (2^N − 1)` | `Vref / 2^N` |
| returns | column-sum units (decoded back) | **signed ADC codes** (rescaled once by `LSB_P`) |
| physical stages | none — digital MAC + normalised voltage mapping | DAC, macro, amplifier, SAR ADC all explicit |
| default | off | off (`analog=None`) |

In `CIMArray.mvm`, the analog chain takes priority: if `self.chain is not None` it executes and
returns early; otherwise the legacy path runs unchanged. They are mutually exclusive per array.

**The legacy path is retained deliberately** as a fast digital reference and for reproducing all
earlier experiments. Do not delete it. Do not "unify" the two LSB conventions. Two known
properties of it (documented in `array.py`, unchanged on purpose): its full scale does **not**
bound unsigned AV partial sums (32·15·7 = 3360 > 2048 at W4/A4), and it is mid-rise with no
zero code (an exact-zero sum decodes to ≈ +LSB/2). Do not train new QAT against it.

---

## 8. Experiments and metrics

### Real-image experiment (primary, Oct 2026) — `run_experiment.py`

Full runbook in `EXPERIMENTS.md`. Fixed protocol:

- **Data:** official fast.ai Imagenette (`imagenette2-160` by default), `val` split only.
  Stage `smoke` = 100 images (10/class, seeded selection); stage `full` = whole split
  (3,925 images). No synthetic fallback. 10 classes, not ImageNet-1k.
- **Class mapping trap:** folder `n03888605` holds the *parachute* images but is
  "parallel bars" (index 702) in ImageNet-1k; parachute is `n03888257` (701). The mapping
  is by class (`cim_exp/imagenette.py`), checked against the bundled synset list and
  empirically by `verify`/`report`. The empirical check passes when each class's 10-way
  pretrained-FP32 modal prediction is the class itself; a 1000-way check would be fooled
  by near-synonyms such as tape player (848) vs cassette player (482).
- **Preprocessing:** Cream TinyViT eval — resize short side 256 (bicubic), centre crop 224,
  ImageNet mean/std. Images are 160 px short side -> upsampled.
- **Models:** pretrained FP32, QAT-INT4 and QAT-INT8 (FP32 latent weights, no quantizer
  state; hash-checked via `configs/checkpoints.json`; never substituted). Per the QAT
  context document (code not supplied):
  - both were fine-tuned from this pretrained model on ImageNet-1k (verified from BN
    counters);
  - training used the simulator's own quantizer (per-token / per-column symmetric,
    unsigned AV), so operand quantization matches;
  - training used the **legacy functional ADC** (8-bit, worst-case full scale) at
    W4/A4 (INT4) and W8/A8 (INT8). **They were never trained against the analog chain.**

  BN counters show INT8 = 21 epochs vs the documented "best epoch 26" (unresolved).
- **Training-data overlap:** Imagenette2's 70/30 re-split makes most val images ImageNet-1k
  *training* images (seen by both models; QAT also early-stopped on ImageNet val).
  `prepare` counts file-name provenance. Absolute accuracies are optimistic and
  QAT-vs-pretrained differences are confounded; CIM-vs-own-FP32 comparisons are robust.
- **Matrix:** W4/A4, W4/A8, W8/A4, W8/A8 × ADC 8/10/12 b × gains {1,1,1} / {23.77, 8.41,
  2.92} = 24 configs; analog chain ON, CDAC 8 b @ 0.6 V (MSB-aligned A4), ADC V_REF 0.6 V,
  ideal, P=1, sites qkv/qk/av. QAT checkpoint chosen by weight bits. 48 CIM jobs.
- **Experiment A = this matrix.** Gains are FIXED: G = 1 and the W8/A8 P99.99 set
  (derived on synthetic calibration inputs). They are never recalibrated per checkpoint
  or precision. Every CIM run reports clipping per operation (qkv/qk/av, hi/lo, per
  conversion and per image), plus V_OA / V_ADC_IN ranges and ADC utilisation.
- **Amplifier:** ideal linear by default; amplifier saturation is reported as
  `not_modeled`. `--amp-swing-limit V` gives a labelled diagnostic variant (`_amp{V}V`)
  in which saturation is counted separately from ADC clipping.
- **Experiment B** (`prepare-calibration`, `calibrate`, `gain-robustness`):
  - **Calibration data:** Imagenette **train** (seed 4242), disjoint from evaluation by
    SHA-256.
  - **Measurement:** an exact probe (bit-identical to the ADC-off path) histograms every
    per-depth-block column sum |P| per op.
  - **Objective (pre-declared):** P99.99, i.e. a clip target ≤ 0.01% per op;
    G = (V_REF_ADC/2)/(K_MAC·FS).
  - **Sensitivity levels:** P99.9 / P99.999 / P100 are reported only.
  - **Evaluation:** on the held-out val images, against the tested checkpoint's own FP32
    output. It compares fixed vs calibrated gains, precision transfer (W8A8-calibrated
    gains at other precisions) and, optionally, checkpoint transfer.
  - **Validation:** reproduces the historical W8/A8 gains to within 2.5% on the historical
    synthetic calibration inputs (23.19 / 8.28 / 2.92 vs 23.77 / 8.41 / 2.92).
- **Experiment T** (`trainpoint`): each checkpoint at the QAT training operating point
  (legacy ADC, 8-bit, W4/A4 and W8/A8) through the reference simulator.
- **Cases:** CASE1 pretrained-CIM vs QAT-CIM; CASE2 QAT FP32 vs pretrained FP32;
  CASE3 QAT FP32 vs same QAT on CIM; CASE4 pretrained FP32 vs pretrained CIM;
  CASE5 QAT CIM vs pretrained FP32 (combined).
- **Metrics:** top-1 vs true labels (1000-way primary, 10-way secondary), cross-entropy
  (natural log, from raw logits, float64 log-sum-exp), named agreement columns
  (`agreement_vs_own_qat_fp32_pct`, `agreement_vs_pretrained_fp32_pct`,
  `agreement_pretrained_cim_vs_qat_cim_pct`, …), relative logit RMS in percent
  `100·‖t−r‖₂/‖r‖₂`, non-finite counts, clipping per conversion and per image.
- **Determinism:** batch 1, 1 thread per forward; parallelism by processes. Cache keys hash
  checkpoint SHA-256, full hardware config, manifest SHA-256, preprocessing id, a source
  fingerprint of every output-affecting file, torch/numpy versions, chunking.

### Legacy synthetic-input scripts

The subsections below describe `run_tinyvit.py` / `experiments.py` / `exp_analog.py`,
which use **synthetic** inputs and (except `exp_analog.py` / `--analog`) the LEGACY path.

### Metric definitions (do not silently change these)

- **Output agreement (%)** = `100 · (out.argmax(-1) == ref.argmax(-1)).float().mean()`.
  Called **agreement**, never "accuracy" — inputs are synthetic.
- **RMS error (relative)** = `‖cim − ref‖₂ / ‖ref‖₂` on final logits. **This is the trustworthy
  metric**; agreement moves in coarse steps (1/n_images) and can be non-monotonic from sampling
  noise.
- **Latency (cycles)** = `TraceLog.total_parallel_cycles()`. Clock-free array timesteps.
- **Throughput** = `macs_useful / parallel_cycles` (MACs/cycle). **TOPS is deliberately not
  reported** — no clock frequency exists anywhere in this project. Multiply by `f_clk` yourself.
- **Clipping (%)** = `100 · adc_saturations / adc_conversions`.

### Synthetic inputs caveat

`run_tinyvit.synthetic_images(n, seed=0)` generates smooth gradients and blobs, ImageNet-
normalized. Natural images are low-frequency dominated, so this exercises activation ranges far
more realistically than white noise — but it is **not a validation set**. All accuracy-like
numbers are indications of pipeline coherence, not model accuracy. Real ImageNet validation is
outstanding.

### Command reference

```bash
# main driver
python run_tinyvit.py                                   # full model, defaults
python run_tinyvit.py --stage3-only                     # one stage-3 block + tile map
python run_tinyvit.py --ablate                          # which matmul site costs accuracy
python run_tinyvit.py --sweep-bits                      # 2/3/4/6/8 bit operands
python run_tinyvit.py --sweep-token-block               # T sweep (writes vs psum)
python run_tinyvit.py --compare-adc --sweep-adc-bits 8 10 12
python run_tinyvit.py --compare-arrays --sweep-arrays 1 2 4 8
python run_tinyvit.py --num-arrays 4 --adc-enabled --adc-bits 10
python run_tinyvit.py --act-bits 8 --weight-bits 8      # precision is CLI, never hard-coded

# thesis plots (writes results/*.csv + *.png)
python experiments.py --all
python experiments.py --exp {weight|activation|adc|arrays}
python experiments.py --plot-only

# correctness harnesses
python verify_array.py
python test_analog.py          # analog datapath, 98 checks (capacitive DAC only)
python test_multi_array.py     # multi-array scheduling (reconstructed Oct 2026)
python test_experiment_pipeline.py   # cim_exp logic, offline

# real-image experiment (see EXPERIMENTS.md)
python run_experiment.py verify --stage smoke
python run_experiment.py all --stage smoke --workers 4

# analog observation
python exp_analog.py --scales          # capacitive-DAC K_MAC / LSB_P per precision, no model run
python exp_analog.py --stats           # per-op V_OA distributions
python exp_analog.py --gain-sweep      # manual per-op gain sweep
```

### Established results (for orientation; re-measure before quoting)

> **Provisional (Oct 2026).** All numbers in this subsection were produced on synthetic
> inputs, several by scripts not present in this repository, and — except the "Analog
> chain" paragraph — through the LEGACY digital-MAC path. Do not quote them as hardware
> results; the real-image pipeline supersedes them.

**Site ablation** (4 images, 4b/4b) — each row puts one site on the array, rest float:

| sites on array | rms | cos | top-1 agree |
|---|---|---|---|
| qkv only | 44.12% | 0.9004 | 25% |
| QK^T only | **2.94%** | 0.9996 | 100% |
| AV only | 20.73% | 0.9784 | 75% |
| all three | 42.05% | 0.9075 | 25% |

**qkv dominates the error; QK^T at 4-bit is nearly free.** This inverted the prior assumption
that the dynamic-weight attention matmuls were the headline risk.

**Operand bit sweep** (4 images): 2b 102.7% → 3b 79.5% → 4b 42.1% → 6b 13.8% → 8b 3.2% rms.

**Array size** (32 images, 8b/8b, P=1): 16×16 gives ~1.35× better RMS at matched ADC bits
(halved full scale → finer LSB) but **3.9× the cycles and 2× the ADC conversions**. On equal
cell area (one 32×32 = four 16×16) it is still ~20% slower and needs 64 ADCs vs 32.
Not recommended; 32×32 stands.

**Per-operation full scale** (16 eval images, 8b/8b, legacy ADC): common FS at 8b → 0% agreement,
68.17% rms; per-op FS at 8b → 93.75% agreement, **6.89% rms**; per-op at 10b → 3.57% rms,
essentially equal to the ADC-off ideal (3.34%). The ceiling of 93.75% is operand quantization,
not the ADC.

**Analog chain** (2 images, capacitive DAC, 8b/8b): at G=1, qkv occupies **3.2%** of the ±0.3 V
swing (median V_OA 1.54 mV vs 2.34 mV LSB — below one LSB), qk 10.4%, av 22.5%; rms 43.6%.
With gains `{24, 8, 3}`: **4.66% rms, 100% agreement, 0.012% clipping**.

**Environment caveat:** absolute numbers are not comparable across sessions where the PyTorch
version changed. Always re-measure a baseline in the same environment before claiming a
before/after improvement.

### Calibration vs evaluation

Full scales and gains must be derived from a **held-out calibration set** (a different seed),
never from the evaluation inputs. `exp_per_op_fs.py` uses `CAL_SEED = 4242` for calibration and
`seed=0` (the standard set) for evaluation. Preserve this split in any new calibration work.

---

## 9. Hardware architecture context

`[FUTURE]` unless stated otherwise — this section is design context, not simulator behavior.

**Hardware-confirmed (from the company / Aswani; DAC from the Oct 2026 hardware reference):**
- Input DAC: **unsigned binary-weighted capacitive DAC, 8 bits, V_REF = 0.6 V**,
  `V_out = V_REF·c/256`.
- 32×32 analog CIM macro, **one ADC per column** (32 ADCs per macro).
- ADC is **SAR**, currently assumed 8 bits, reference/input range ≈ **0–0.6 V**.
- Multiplication is in the **charge domain** via a C2C capacitor ladder whose taps are selected
  by digital weights held in SRAM.
- MAC units in a column share one output node and combine by **capacitive charge sharing**, so
  the node develops the **averaged** (normalised) sum. There is **no separate current-to-voltage
  conversion stage** between column summation and the ADC.
- First-order behavioral relation: `V_OA = (1/m)·Σ_j(V_IA,j · W_j / 2^B)`.

**Not confirmed / open questions to put to the hardware team:**
1. The constant of proportionality — at full-scale input and full-scale weights, does `V_OA`
   land exactly at the ADC reference, or is there ladder attenuation?
2. **Unipolar DAC vs signed activations.** A 0–0.6 V DAC output with an unsigned `W/2^B`
   weighting makes `V_OA` a sum of non-negative terms, which cannot represent the signed MACs
   attention requires. The sign must be realised somewhere physical — differential columns,
   common-mode referencing, or digital subtraction. **This currently blocks a physically correct
   C2C model.**
3. Do unused / zero-padded rows stay connected to the charge-sharing node? If taps disconnect,
   `m` drops for ragged depth blocks and those tiles get better effective ADC resolution. The
   simulator currently always uses `m = rows`. A small change (`adc_fs_from_active_rows`) would
   test this if confirmed.
4. ~~Input DAC topology~~ **Answered (Oct 2026):** unsigned binary-weighted 8-bit CDAC,
   V_REF 0.6 V. Still open: how SIGNED activations are presented to it (offset binary +
   where the V_REF/2·ΣW common mode is cancelled), and how A4 codes are aligned onto it.
5. Amplifier output swing (needed to model amplifier saturation) and whether a ~24× gain
   at 0.6 V is realisable.
6. ~~QAT training recipe~~ **Largely answered by the QAT context document (AUDIT.md §10):**
   the simulator's quantizer was used, with the legacy functional ADC, at W4/A4 and W8/A8.
   Open: the INT8 epoch discrepancy (21 by BN counters vs 26 documented), and
   reference-simulator accuracy of both checkpoints (Experiment T measures it).

**Earlier design-discussion concepts, all `[FUTURE]`, none in the simulator:**
- **Engine partitioning:** SEngine0 (static W_K), SEngine1 (static W_Q, shared W_V), DEngine
  (dynamic matmuls QK^T then AV) across two pipeline stages. The simulator has no engine concept.
- **Weight planes:** 4 planes per cell selectable by a 1-of-4 select line, making up to four
  tiles resident at zero write cost. **`planes=1` is enforced**; plane residency is unmodeled.
- **Readout modes:** column-sum (the matmul path) and row-sum (orthogonal reduction). Only
  column-sum is modeled. Row-sum is explicitly *not* a free transpose.
- **Inter-tile "systolic" dataflow:** the analog array is *not* internally systolic (inputs are
  broadcast, reduction is analog charge on a shared node, no PE-to-PE movement). Systolic
  behavior, if any, belongs to the **digital inter-tile network**, not inside a macro.
- **Convolution** needs no second primitive: it reduces to GEMM via im2col onto the same
  weight-stationary tile, with an address generator + line buffer feeding it. Depthwise
  convolution cannot be rescued this way (no channel contraction).
- **Scaling philosophy:** scale *out* (more 32×32 tiles + digital reduction), never *up* — long
  analog columns bring parasitic capacitance, slow settling, worse SNR, harder ADC input range.
- **ADC count is the real scaling wall** — dominates macro area and power and caps tile count.
- **Programmable ADC reference / gain registers:** three static per-operation values
  (`ADC_FS_QKV` / `ADC_FS_QKT` / `ADC_FS_AV`, or equivalently three gain registers) selected by
  the operation the controller already sequences. No runtime recalibration needed for this
  workload. A ~24× pre-ADC gain at 0.6 V is a non-trivial analog ask — flag it.
- **No area model exists.** Nothing in this repo defines a per-macro or per-ADC area. Any "area"
  column is a placeholder until the hardware team supplies mm² or an ADC:macro area ratio.

---

## 10. Research methodology

The simulator is a **functional precursor**, used in this order:

1. **Establish the mapping** — confirm a workload tiles onto 32×32 with acceptable utilization
   (`tile_map`, `report`).
2. **Bound the arithmetic** — sweep operand precision with an ideal converter to find the
   floor that quantization alone imposes.
3. **Isolate the bottleneck** — ablate one matmul site at a time. Never attribute an aggregate
   error to a site without this.
4. **Introduce the physical path** — DAC → macro → gain → ADC, ideal first.
5. **Observe distributions before tuning anything** — V_OA / V_ADC_IN percentiles per operation,
   swing utilization, clipping.
6. **Choose hardware parameters from calibration data** on a held-out set, then evaluate.
7. **Add nonidealities independently** once each boundary is trusted.

Ground rules: measure rather than reason about magnitudes; separate calibration from evaluation;
keep the FP32 model as the only ground truth; and state which numbers are hardware-confirmed,
which are simulator assumptions, and which are yet to be supplied.

---

## 11. Important invariants

Do not change these casually — each encodes an architectural decision or an experimental
assumption.

1. **`num_arrays=1` reproduces the single-array path bit-for-bit**, with identical statistics
   and identical call ordering.
2. **Analog disabled reproduces the legacy path bit-for-bit.** `analog=None` must change nothing.
3. **Token blocking is numerically neutral.** Every `T` gives bit-identical outputs.
4. **Tiling is exact.** With quantization neutralised, the tiled decomposition equals plain
   integer matmul, including ragged shapes.
5. **Quantization scales are constant along K** (per-token activations, per-output-column
   weights). Changing this breaks the single end-of-chain rescale.
6. **AV activations are unsigned.** Anything on the activation path must honour `act_signed`.
7. **`K_MAC` / `LSB_P` are derived from configuration, never from data**, and are constant within
   an accumulation domain. `domain_key` enforcement must stay.
8. **Pre-ADC gain is a configuration parameter, never data-derived at inference time.**
9. **`m` = physical row count, not `useful_rows`.** Padded rows still hold capacitors.
10. **The C2C ladder never sees a two's-complement word** — only `|W_q| / 2^B`.
11. **Clipping is observable, never silently rescaled away.**
12. **`CIMArray` models exactly one macro.** Never create `CIMArray1`, `CIMArray2`, …
13. **`planes = 1`.** Multi-plane residency is not modeled.
14. **The two ADC LSB conventions are intentional.** Legacy `2·FS/(2^N−1)`; analog `Vref/2^N`.
15. **`array_ops` is total physical work and does not vary with P.** Only
    `parallel_array_cycles` does.
16. **`tiny_vit.py` stays unmodified.**
17. **MSB-aligned A4 on the 8-bit DAC is bit-identical to a 4-bit DAC** (powers of two);
    the default is `dac_bits=8`, and `dac_bits=None` (follow act_bits) gives identical results.
18. **Analog statistics never change an output** (including the common-mode diagnostic).
19. **The experiment pipeline asserts the analog chain is active** for every CIM job and
    that counted conversions = scheduler geometry = `CIMStats.adc_conversions`.
20. **No synthetic data in the real-image pipeline**, and no checkpoint substitution
    (hash-checked registry).
21. **Experiment A gains are fixed** (G = 1, 23.77 / 8.41 / 2.92) for every precision and
    checkpoint. Calibrated gains live only in Experiment B, under their own gain-set names
    and cache keys, calibrated on data disjoint from evaluation.
22. **Amplifier saturation is `not_modeled` unless a swing limit is set**; ADC clipping is
    always measured separately and never substituted for it.
23. **The input DAC is the capacitive DAC only** (`V_REF·c/2^N`, 8 b, 0.6 V). There is no DAC
    selector; the signed-input convention and alignment remain the provisional knobs.
24. **The initial QAT operating point uses an 8-bit ADC** (`QAT_INITIAL_ADC_BITS`, the
    `AnalogConfig.adc_bits` default). Other widths remain available for later studies only.

---

## 12. Verification rules

Run after **any** change to `array.py`, `matmul.py`, `attention.py`, `quant.py`, or
`cim_sim/analog/`:

```bash
python verify_array.py          # tiling exactness, token-block neutrality, error attribution (exit 1 on failure)
python test_analog.py           # 98 checks: CDAC equation, signed path, alignment, K_MAC, SAR, gain, domains, stats
python test_multi_array.py      # P=1 equivalence, group scheduling, numerical invariance
python test_experiment_pipeline.py   # mapping, metrics, manifests, preprocessing, keys, 5-case logic
python test_qat_contract.py     # 17 checks: surrogate contract; T/P neutrality at ties; grad guard; 8-bit-ADC QAT defaults
python run_tinyvit.py --stage3-only     # end-to-end smoke + known rms/cosine
python run_experiment.py verify --stage smoke   # all of the above + checkpoint/analog/real-image checks
```

Required checks:

- **Regression:** the default path (analog off, legacy ADC off, P=1) must reproduce its previous
  numbers **exactly**. Re-measure the baseline in the *same* environment; do not compare across
  PyTorch versions.
- **Determinism:** run the same config twice and assert `torch.equal`.
- **Reference-vs-CIM:** every accuracy claim compares against the FP32 forward of the *same*
  model on the *same* inputs, computed once.
- **Attribution:** when adding an error source, show it is the *only* new source — e.g. tiled
  4-bit error must match untiled 4-bit error to ~1e-16.
- **Geometry changes:** with the ADC off, two different array sizes must give **bit-identical**
  logits (integer sums are exact, addition is associative). If that fails, the change broke
  something and no downstream number is trustworthy.
- **Analog changes:** round-trip `d · LSB_P` against the digital MAC; error must be ≤0.5 LSB.
- New experiments must print the config they ran under, so results are reproducible from the log.

---

## 13. Current unfinished work

1. **Analog rewrite stages 3–4** — the full post-implementation experiment set and the formal
   engineering report structure. Stages 1–2 (module skeleton, ideal path, DAC family, tests,
   TinyViT integration) are done.
2. ~~`exp_analog.py --dac-compare`~~ removed (Oct 2026): the DAC topology is specified
   (capacitive only).
3. **Nonideality sweeps.** Hooks exist and default to ideal
   (`DACNonidealities`, `MacroNonidealities`, `GainNonidealities`, `ADCNonidealities`,
   `SARDACNonidealities`) but nothing exercises them. Capacitor mismatch and comparator
   offset/noise are the highest-value first additions.
4. **Real-image validation.** Pipeline implemented (`run_experiment.py`, Imagenette);
   results must be produced locally (the dataset could not be downloaded in the
   development sandbox). Full ImageNet-1k validation remains outstanding.
4b. **Per-precision gain calibration**: implemented as Experiment B (`calibrate`,
   `gain-robustness`); results must be produced locally.
4c. **Clean test set:** Imagenette val overlaps ImageNet-1k train, which both models saw.
   An independent set such as ImageNetV2 would remove the confound. Not implemented.
5. **Open hardware questions** (§9) — proportionality constant, unipolar-DAC-vs-signed-activations,
   whether padded rows stay connected.
6. **Area model** — blocked on a per-macro or per-ADC area figure from the hardware team.
6b. **`AnalogStats.merge()` percentile defect (found 29 Oct 2026).** Merging many chunk
   subsamples concentrates the weight on a few samples (see QAT_DESIGN.md §13, v2.7): exact scalars
   are fine, sampled percentiles of long chunked runs are not. `cim_qat.analog_merge` re-merges
   correctly without changing cim_sim; `cim_exp/engine.py` still uses the defective merge.
7. **Repo hygiene** — the local tree has drifted from delivered versions before (`__init__.py`,
   `report.py`, `run_tinyvit.py` were all out of sync at various points). Verify file versions
   before debugging a mysterious failure.

---

## 14. Historical / obsolete approaches

Do not reintroduce these.

- **`[OBSOLETE]` Per-operation ADC full-scale via `CIMAttention.adc_fs` / `set_adc_full_scales`.**
  This was the *digital* precursor of the pre-ADC amplifier. It worked (68% → 6.9% rms at 8b) and
  the calibrated full scales are still used to derive gains, but the mechanism is **superseded by
  `PreADCGain`**, which is the physically meaningful form. `exp_per_op_fs.py` is kept for its
  calibration data only.
- **`[OBSOLETE]` Non-capacitive DAC models** (IDEAL, R2R, CURRENT_STEERING, DIFFERENTIAL,
  `DAC_TYPES`, `make_dac`, `AnalogConfig.dac_type`, `--dac-type`, `--dac-compare`). Removed
  Oct 2026 after the capacitive DAC was confirmed. Do not reintroduce a DAC selector.
- **`[OBSOLETE]` Gains 11.8 / 4.2 / 1.45.** Computed with the *bipolar* `K_MAC`. The default DAC
  is capacitive (offset binary), whose `K_MAC` is half, so the correct set is **23.8 / 8.4 / 2.9**.
- **`[OBSOLETE]` "8-bit ADC is fundamentally insufficient."** The real cause was a single shared
  full scale mismatched to qkv's small dynamic range, not ADC resolution as such.
- **`[OBSOLETE]` `key_dim = 16` and `W_Q = [160×80]` for stage 3.** Upstream sets
  `head_dim = dim // num_heads`, so stage 3 has `key_dim = 32` and a fused `qkv = [160 → 480]`.
  There is **no ragged edge** in the projection, and the "50% depth underutilization on QK^T"
  risk does not exist — `key_dim = 32` fills the array exactly.
- **`[OBSOLETE]` Shrinking ADC full scale below the worst case to gain resolution.** Measured:
  clipping dominates almost immediately (α=0.25 → 20% RMS error). Full scale is not the knob;
  per-operation gain is.
- **`[OBSOLETE]` 16×16 arrays as an accuracy fix.** Real but small benefit (~1.35× RMS) at 3.9×
  cycles and 2× ADC count.
- **`[OBSOLETE]` Treating the "sum vs charge-sharing average" distinction as the cause of ADC
  error.** The 1/m factor cancels between signal and full scale; verified to give bit-identical
  codes.
- **`[OBSOLETE]` Two's-complement bits as physical C2C capacitor codes.** Category error; the
  `SignedWeightAdapter` exists to prevent it.
- **`[OBSOLETE]` `FS = max(observed)` auto-ranging.** Outlier-dominated and data-dependent, which
  is not physical. P99.99 beat P100 on every operation.
- **`[OBSOLETE]` Treating `CIMConfig.n_arrays` as functional.** It is descriptive metadata; the
  array count is `len(arrays)` passed to `cim_matmul`.

---

## 15. Coding rules for Claude Code

1. **Inspect before modifying.** Read the actual file. This project's local tree has drifted from
   delivered versions more than once. Never assume an API, filename, or line from memory.
2. **Preserve the legacy path.** The digital-MAC + legacy-ADC path is a deliberate reference
   model. Additions are opt-in and default to off.
3. **Minimal diffs.** No refactoring beyond what the task requires. Do not rename, reorganise, or
   "clean up" unrelated code. Preserve each file's existing line-ending convention (the repo
   mixes CRLF and LF).
4. **Never silently change experiment methodology.** The agreement metric, reference path,
   synthetic-input generator and seeds, and calibration/evaluation split are fixed. If a change
   is genuinely required, say so explicitly and explain why.
5. **Run the tests in §12 after every change** and report the results, including failures.
6. **Do not overwrite unrelated functionality.** Especially: quantization conventions, tiling,
   token blocking, ragged handling, statistics fields.
7. **Explain architectural changes** before or alongside making them — what boundary moved, what
   becomes replaceable, what assumption is being introduced.
8. **Preserve reproducibility.** Print the configuration with the results. Keep seeds explicit.
   Write raw numbers to CSV, not just to stdout.
9. **Git safety.** Never force-push, never rewrite history, never commit without being asked.
   Branch before non-trivial work. Do not commit the checkpoint (`tinyvit5m.pth`, ~22 MB) or
   `results/` artifacts unless asked.
10. **Do not invent hardware facts.** If a physical value is unknown, make it a configurable
    assumption with a documented default and say so. Never present a modeled candidate as "the
    hardware".
11. **Label every number.** Hardware-confirmed, simulator assumption, or measured — and under
    which configuration. Flag the synthetic-input caveat wherever an accuracy-like number appears.
12. **Report unfavourable results.** If a hypothesis fails, say it failed. Prior sessions
    overturned their own conclusions twice; that is the process working.
