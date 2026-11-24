# Audit of the delivered repository (`CIM-Soc-qat-weights.zip`) — Oct 2026

This audit describes the code **as delivered**, before any change. What was changed in
response is in `CHANGES.md`. "Verified" means executed in the development environment
(Python 3.13, torch 2.14.1+cu130 on CPU, numpy 2.5.3, 2 cores) unless marked otherwise.

Status labels:
- **IMPLEMENTED AND VERIFIED**: present, and checked by execution.
- **IMPLEMENTED BUT UNVERIFIED**: present, but no adequate check existed.
- **MISSING**: not in the delivered tree.
- **OUTDATED DOCUMENTATION**: the documentation contradicts the code.

## 1. Files present and their roles

| file | role | status |
|---|---|---|
| `tiny_vit.py` | upstream microsoft/Cream TinyViT model (Attention forward compared line-by-line with `CIMAttention`) | IMPLEMENTED AND VERIFIED |
| `_shim/timm/*` | 5-symbol timm stand-in | IMPLEMENTED AND VERIFIED |
| `cim_sim/quant.py` | symmetric per-token / per-column quantization | IMPLEMENTED AND VERIFIED (verify_array §3) |
| `cim_sim/array.py` | `CIMArray`, `CIMConfig`, `CIMStats`, legacy `_adc_convert`, analog dispatch | IMPLEMENTED AND VERIFIED |
| `cim_sim/matmul.py` | tiling scheduler, multi-array, single rescale | IMPLEMENTED AND VERIFIED (verify_array; multi-array only after reconstruction, see §7) |
| `cim_sim/attention.py` | `CIMAttention`, `convert_attention`, site toggles | IMPLEMENTED AND VERIFIED (all sites off reproduces FP32 bit-exactly) |
| `cim_sim/report.py` | tables | IMPLEMENTED AND VERIFIED |
| `cim_sim/analog/dac.py` | 5 input-DAC models incl. `CapacitiveDAC` | IMPLEMENTED AND VERIFIED (the hardware equation was not tested explicitly, see §4) |
| `cim_sim/analog/macro.py` | C2C relation + `SignedWeightAdapter` | IMPLEMENTED AND VERIFIED |
| `cim_sim/analog/adc.py` | `PreADCGain`, `SARDAC`, `SARADC` | IMPLEMENTED AND VERIFIED (SAR = round-to-nearest; clipping counted) |
| `cim_sim/analog/digital.py` | accumulator/requantizer reference contract | IMPLEMENTED AND VERIFIED |
| `cim_sim/analog/chain.py` | `AnalogConfig`, `AnalogStats`, `AnalogChain` | IMPLEMENTED BUT UNVERIFIED for statistics (see §5) |
| `run_tinyvit.py` | synthetic-input CLI modes | IMPLEMENTED AND VERIFIED (stage-3 smoke 24.55% / cos 0.9694) — **legacy path only** |
| `exp_analog.py` | analog observation on synthetic inputs | IMPLEMENTED BUT UNVERIFIED (`--dac-compare` documented as never run cleanly) |
| `experiments.py` | four synthetic-input sweeps (legacy path) | IMPLEMENTED BUT UNVERIFIED (not run in this audit) |
| `verify_array.py` | tiling exactness, token blocking, attribution | IMPLEMENTED BUT UNVERIFIED as a gate: sections 1 and 3 only printed and could not fail |
| `test_analog.py` | 45 analog checks | IMPLEMENTED AND VERIFIED (45/45) |
| `test_cim_array.py`, `test_matmul.py`, `test_quant.py` | print-only scripts, no assertions | not regression tests (as documented) |
| `cim_architecture_estimator.xlsx` | separate first-order spreadsheet (W4/A8, 128 arrays, 500 MHz) | not used by the simulator |
| `~$cim_architecture_estimator.xlsx` | Excel lock file, accidentally committed | junk; `.gitignore` allowed it |
| `readme.md` | describes an early version ("INT4, no ADC model") | OUTDATED DOCUMENTATION |

## 2. Files or functionality missing

| item | status |
|---|---|
| `test_multi_array.py` (required by CLAUDE.md §12) | MISSING, reconstructed |
| `exp_per_op_fs.py`, `exp_array_size.py`, `analyze_adc.py` (cited for results and calibration) | MISSING: the cited numbers cannot be reproduced from this repo |
| Real-image data loading, class mapping, preprocessing | MISSING, implemented (`cim_exp/imagenette.py`) |
| Checkpoint registry / hash checks / QAT inspection | MISSING, implemented (`cim_exp/checkpoints.py`) |
| FP32 reference caching, resumable jobs, five comparison cases, CSV/report generation | MISSING, implemented (`cim_exp/`, `run_experiment.py`) |
| Requirements / environment file | MISSING, added `requirements.txt` |
| Analog CLI flags in the main runner | MISSING, added `run_tinyvit.py --analog …` |

## 3. Is the analog path integrated into and activated by the model runner?

- **Integrated:** yes. `CIMArray.mvm` executes `AnalogChain` whenever
  `CIMConfig.analog.enabled`; `cim_matmul` sets the per-operation gain and input domain
  once per matmul and rescales once by `LSB_P`. IMPLEMENTED AND VERIFIED.
- **Activated by `run_tinyvit.py`: NO.** Its `CIMConfig` never receives an
  `AnalogConfig`. Every mode (`--ablate`, `--sweep-bits`, `--compare-adc`,
  `--compare-arrays`, the default run, `--stage3-only`) ran the **legacy digital-MAC**
  model, optionally with the functional legacy ADC.
- **Activated by `experiments.py`: NO** (legacy only).
- **Activated by** `exp_analog.py` and `test_analog.py`: yes.
- Consequence: earlier "ADC" results from `run_tinyvit.py` / `experiments.py` describe
  the legacy normalised-voltage ADC, not the DAC → C2C → gain → SAR chain.

## 4. Capacitive DAC vs the hardware equation

- The `CapacitiveDAC` transfer is `V_phys = Vref·D/2^N` with offset-binary
  `D = X_q + 2^(N−1)` for signed inputs, and `D = X_q` for unsigned (AV). The unsigned
  part matches the specified `V_out = V_REF·c/2^N` exactly (all-ones gives
  `V_REF(1−2^-N)`). IMPLEMENTED AND VERIFIED after adding explicit tests: zero code, max
  code, each bit, intermediate codes, monotonicity, superposition.
- **DAC bit width:** `AnalogConfig.dac_bits=None` makes the DAC follow `act_bits`, which
  gives a 4-bit DAC at A4. The hardware DAC is 8-bit. Mapping A4 onto the 8-bit DAC
  MSB-aligned (×16) is **bit-identical** to the 4-bit-DAC behaviour (verified on the full
  model, all four precisions, both gain sets). So the historical results are equivalent
  to an 8-bit DAC with MSB alignment, but the alignment is an assumption nobody has
  confirmed.
- **Signed inputs on an unsigned DAC:** never two's complement. The offset-binary code is
  physically valid for the unsigned CDAC. The simulator then removes the `V_REF/2` common
  mode **ideally**. Physically this leaves a weight-dependent per-column term
  `V_CM·Σ sign|W|/2^B/m` that must be cancelled somewhere. The hardware documentation
  does not say where. **Simulator convention, not a confirmed circuit** (now measured as
  a diagnostic).
- AV unsigned handling: correct (no offset, separate accumulation domain).

## 5. Clipping, gain and voltage statistics

| item | status |
|---|---|
| ADC clipping hi/lo counted per operation and in `CIMStats.adc_saturations` | IMPLEMENTED AND VERIFIED |
| Gain applied per operation, once per matmul; `LSB_P ∝ 1/G` | IMPLEMENTED AND VERIFIED |
| V_OA min/max/mean/std/percentiles | IMPLEMENTED BUT UNVERIFIED: the subsample **stopped at its cap**, biasing percentiles toward the first images of a run, and its O(n) `sum()` made it **~50% of total runtime** |
| V_ADC_IN statistics | partial (two percentiles only; no min/max/mean/std) |
| Amplifier saturation | hook exists only when `v_swing_limit` is set; **not exposed** in any summary; by default the amplifier cannot saturate |
| Export to CSV | MISSING (stdout only, via `exp_analog.py`) |
| Per-image clipping attribution | MISSING |

## 6. Mixed weight/activation precision

`CIMConfig.weight_bits` and `act_bits` are independent throughout quantization, the
scheduler, the macro (`2^B` uses weight bits) and the DAC. All four configurations run
end-to-end with consistent conversion counts and no out-of-range codes. IMPLEMENTED AND
VERIFIED (the round-trip test now covers W8A8, W4A4, W4A8, W8A4 at ADC 8/10/12, signed and
unsigned). Note that in QK^T the *keys*, and in AV the *values*, are quantized with
**weight_bits**: they are the array-resident operand. So W8/A4 means 8-bit K and V with
4-bit Q and attention probabilities.

## 7. Tests

| test | status |
|---|---|
| `verify_array.py` | runs and passes; sections 1/3 could not fail. Fixed: exits 1 on failure |
| `test_analog.py` | 45/45 as delivered; now 101 checks |
| `test_multi_array.py` | MISSING; reconstructed (42 checks) |
| `run_tinyvit.py --stage3-only` | runs: 24.55% / cos 0.9694 (torch 2.14) |
| Regression of the analog path across code changes | none existed; done in this work (bit-identical logits, 9 configs) |
| Real-image, checkpoint, metric, reporting tests | MISSING; `test_experiment_pipeline.py` (53 checks) + `run_experiment.py verify` |

## 8. Inconsistencies between CLAUDE.md and the code

| CLAUDE.md says | code | status |
|---|---|---|
| analog path files "may or may not be copied into this tree" | they are present and wired in | OUTDATED DOCUMENTATION |
| `test_multi_array.py` exists (§2, §8, §12) | absent | OUTDATED DOCUMENTATION |
| results from `exp_per_op_fs.py`, `exp_array_size.py`, `analyze_adc.py` | scripts absent | OUTDATED DOCUMENTATION |
| "Established results" are presented as orientation numbers | synthetic inputs; mostly legacy path; several from absent scripts | OUTDATED DOCUMENTATION (now marked provisional) |
| §6 "Input DAC topology is not hardware-confirmed" | hardware now specifies an unsigned 8-bit CDAC | OUTDATED DOCUMENTATION (updated) |
| §6 "AnalogStats … memory stays flat … percentiles" | percentiles biased after the cap (see §5) | implementation issue, fixed |
| §8 command reference implies the CLI can exercise the analog chain | `run_tinyvit.py` had no analog option | OUTDATED DOCUMENTATION (option added) |
| `readme.md` "one array, INT4, no ADC model" | multi-array, 4/8-bit, two ADC models | OUTDATED DOCUMENTATION |
| `.gitignore` | ignores `CLAUDE.md`, all configs and data files the new code needs, and allows Excel lock files | fixed |

## 9. QAT checkpoints (as supplied in the conversation)

| property | QAT-INT4 | QAT-INT8 |
|---|---|---|
| file | torch.save zip archive, internal name `best_model_only` | same |
| SHA-256 | `a903663b…6a22b56` | `9f8060d4…ec556dfa70` |
| content | OrderedDict, 296 tensors (269 float32, 27 int64 BN counters) | same |
| keys / shapes vs pretrained | identical, head [1000, 320] | identical |
| quantizer state (scales, observers, LSQ) | **none** | **none** |
| weights on a symmetric int4/int8 grid | **no** (mean off-grid ≈ 0.25 = uniform) | **no** |
| median relative parameter change vs pretrained | ≈ 3.2% (every tensor changed, incl. BN statistics) | ≈ 2.8% |
| strict load into TinyViT-5M | yes | yes |

Interpretation:
- These are FP32 latent weights from QAT. Running them in FP32 is a valid "QAT-trained
  network without its fake quantizers" reference.
- The CIM path re-quantizes them with the simulator's scheme. Whether that matches the
  training-time quantizer **cannot be determined from the files alone**. *Update: the QAT
  context document (§10) states that it does match, but that training used the legacy
  functional ADC, not the analog chain.*
- BatchNorm statistics also changed, so the QAT runs were full fine-tunes on some dataset.
  Differences in CASE2 may therefore include fine-tuning effects beyond QAT.

## 10. QAT context document (`QAT_SIMULATOR_CONTEXT.md`, supplied later in Oct 2026)

Statuses for each claim:
- **CONFIRMED BY FILES**: checked against the checkpoints.
- **DOCUMENTED ONLY**: stated by the document; the training code (`cim_sim/qat.py`,
  `test_qat.py`, `train_qat_tinyvit.py`, `eval_adc_qat.py`, `memory_bank.md`) was not
  supplied, so it cannot be checked here.
- **UNKNOWN**: not established by either source.

### What it establishes

| item | status |
|---|---|
| Both checkpoints fine-tuned from the pretrained `tinyvit5m.pth`, not trained from scratch | CONFIRMED BY FILES: same architecture and keys; ~3% median parameter change; BN counters = pretrained + training batches |
| Data: ImageNet-1k train, batch 64 | CONFIRMED BY FILES: every BN `num_batches_tracked` grew by an exact multiple of 20,018 (= floor(1,281,167 / 64)) |
| INT4 checkpoint = epoch 22 | CONFIRMED BY FILES: Δ = 440,396 = 22 × 20,018 |
| **INT8 checkpoint = best epoch 26** | **CONTRADICTED BY FILES:** Δ = 420,378 = exactly **21** × 20,018. Possible causes (none can be verified here): the table entry is wrong, a different checkpoint was saved, or the run was restarted. **Check `metrics.csv` / `config.json` of the INT8 run** |
| File sizes 21,720,031 bytes | CONFIRMED BY FILES |
| Quantizer = the simulator's `quant.quantize`: symmetric, zero-point-free, per-token activations, per-output-column weights, `scale = amax/qmax`, unsigned post-softmax AV | DOCUMENTED ONLY. Consequence: the CIM path's operand quantization matches training, which resolves the earlier "quantizer mismatch" concern for the operands |
| Rounding by straight-through estimator; scales from detached statistics (no learned step sizes) | DOCUMENTED ONLY. Consistent with no quantizer keys in the state dicts |
| Sites trained under quantization: qkv, QKᵀ, AV only; LayerNorm, softmax, biases, out-projection and MLP in float | DOCUMENTED ONLY |
| **ADC model during training: the LEGACY functional ADC** (`adc_enabled=True`, 8-bit, worst-case full scale `32·2^(w−1)·2^(a−1)`, offset-binary, per 32-row depth block, `(2^N−1)` normalisation). **Not the analog chain** | DOCUMENTED ONLY. Consequence below |
| Operating points: INT4 trained at W4/A4, INT8 at W8/A8, both ADC 8-bit | DOCUMENTED ONLY (an earlier "INT8" run that silently trained at 4/4 is said to have been discarded) |
| Augmentation RandomResizedCrop + flip; val Resize 256 / CenterCrop 224 | DOCUMENTED ONLY. The val transform matches this pipeline's preprocessing |
| Model selection: early stopping on ImageNet-1k **val** top-1 | DOCUMENTED ONLY |
| Reported 79.354% (INT4) / 80.110% (INT8) ImageNet val top-1 | DOCUMENTED ONLY, and **surrogate-path** numbers. The document itself says they were never evaluated through the reference simulator |

### Consequences for the experiments

1. **The primary matrix evaluates the QAT checkpoints on a different ADC model from the one
   they were trained against.**
   - Training used the legacy ADC: a full scale fixed at the worst case, with no gain.
   - The primary matrix uses the analog chain: CDAC, C2C, a per-operation gain, and an SAR
     ADC with LSB = Vref/2^N.
   - So any "QAT benefit" in CASE1/CASE5 measures **transfer**, not the trained-for
     condition.
   - The new optional `trainpoint` command (Experiment T) evaluates every checkpoint at
     its own training operating point (legacy ADC, 8-bit, W4/A4 and W8/A8) through the
     reference NumPy simulator. That is the "row 3 vs row 4" check the document asks for.
   - The cross cells (INT4 at W8/A8, INT8 at W4/A4) also help detect a mislabelled run.
2. **Training-data overlap.**
   - Imagenette2 re-splits about all ImageNet-1k train+val images of its 10 classes 70/30
     (9,469 / 3,925). So most Imagenette *val* images are expected to be ImageNet-1k
     *training* images, which the QAT runs fine-tuned on for 21–22 epochs. The remainder
     come from ImageNet val, which the QAT runs used for early stopping.
   - The pretrained model was trained on ImageNet as well.
   - `prepare` now counts file names by ImageNet convention (`<wnid>_<n>.JPEG` = train,
     `ILSVRC2012_val_*` = val), and REPORT.md states the counts. This could not be verified
     here (the dataset was not reachable).
   - Absolute accuracies are optimistic. QAT-vs-pretrained differences are confounded.
   - CIM-vs-own-FP32 comparisons on identical images (CASE3/4) are much less affected.
   - A clean independent test set (e.g. ImageNetV2) would remove this. It is not
     implemented.

### Still unknown

- The **INT8 epoch discrepancy** (above).
- Whether the documented surrogate was actually used **unchanged** for these two files.
  `test_qat.py` results for these runs were not supplied.
- Accuracy of either checkpoint through the reference simulator. The document marks this
  as open. Experiment T will measure it on Imagenette, which is not ImageNet val.
- Nothing about the **analog** chain was trained. Its provisional assumptions (offset-binary
  signed inputs, ideal common-mode removal, MSB alignment of INT4 codes, ideal amplifier)
  stay provisional.
