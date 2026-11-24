# Changes

## Capacitive-DAC-only design; 8-bit ADC for the initial QAT (21 Oct 2026, second change set)

The input DAC is confirmed as the unsigned binary-weighted 8-bit capacitive DAC,
`V_out = V_REF·(b7/2 + … + b0/256)`, V_REF = 0.6 V. The other DAC models are no longer needed.

| file | change |
|---|---|
| `cim_sim/analog/dac.py` | rewritten: only `CapacitiveDAC` (+ `DACNonidealities`). Removed `IdealLinearSignedDAC`, `R2RDAC`, `CurrentSteeringDAC`, `DifferentialDAC`, the `BaseDAC` base class (folded into `CapacitiveDAC`), `DAC_TYPES`, `make_dac`. Transfer arithmetic unchanged |
| `cim_sim/analog/chain.py` | `AnalogConfig.dac_type` removed (passing it now raises `TypeError`); the chain builds `CapacitiveDAC` directly; `dac_bits` default `None` → **8** (the hardware; bit-identical for A4/A8 with MSB alignment); new `QAT_INITIAL_ADC_BITS = 8`, used as the `adc_bits` default (ADC still configurable) |
| `cim_sim/analog/__init__.py` | exports updated (`QAT_INITIAL_ADC_BITS` added; removed DAC classes no longer exported) |
| `run_tinyvit.py` | `--dac-type` removed; banner says "capacitive DAC" |
| `exp_analog.py` | `--dac-compare` and `--dac` removed; `--scales` now prints the capacitive scale chain per precision |
| `cim_exp/hardware.py` | `HardwareConfig.dac_type` is now a read-only `"CAPACITIVE"` label, not a field; `from_dict` accepts old cached dicts that still carry `dac_type` (and rejects anything but CAPACITIVE). Report columns unchanged. Cache keys change (the hardware dict lost a field), which only means old smoke caches are not reused |
| `test_analog.py` | DAC-family tests replaced by capacitive-only tests in both input domains (4 b and 8 b), plus checks that the other DACs and `dac_type` are gone and that the default DAC is 8 b @ 0.6 V; 98 checks |
| `test_multi_array.py`, `test_qat_contract.py` | `dac_type` arguments removed; contract test covers MSB/LSB alignment and an act-wide DAC instead of DAC families, and adds the 8-bit-ADC QAT-default checks; 17 checks |
| `CLAUDE.md`, `readme.md` | DAC section, gain table, commands, invariants 17/23/24, obsolete list |

Numerical effect: none. Full TinyViT-5M logits (4 precisions × G = 1 / calibrated × ADC 8 / 12,
plus the legacy path) are bit-identical to the previous change set, and default-config runs with
the new `dac_bits = 8` are bit-identical to the old `dac_bits = None` default.

## Pre-QAT simulator review (21 Oct 2026)

Scope: the hardware simulator only (`cim_sim/`). No experiment code, results or
methodology were changed.

| file | change | why |
|---|---|---|
| `cim_sim/analog/adc.py` | `SARADC.TIE_GUARD_REL = 1e-9`; `v_s` gets `+ TIE_GUARD_REL * V_REF` | exact-threshold ties were decided by ~1e-16 V float noise that depends on the BLAS shape; on the analog path `token_block=1` gave different outputs from `token_block=None` (CLAUDE.md §11.3 violated) whenever G·K_MAC/LSB is a simple ratio (G = 1, power-of-two gains). Ties now resolve upward deterministically, as the `decide >= 0` rule intends |
| `cim_sim/attention.py` | `CIMAttention.forward` raises in training mode with autograd enabled while a site is on the array | the NumPy path detaches; training through it silently gave `qkv.weight` and everything upstream zero gradient. Every caller in the repo uses eval + `no_grad`, so nothing else changes |
| `cim_sim/array.py` | docstring only: legacy full scale does NOT bound unsigned AV sums; legacy ADC has no zero code | the previous note ("conservative, never wrong") was false |
| `test_qat_contract.py` | NEW, 14 checks: analog `cim_matmul` == closed-form per-depth-block quantizer (144 configs + all DAC families), T/P neutrality at tie-heavy gains, gradient guard | pins the exact forward contract a QAT surrogate must reproduce |
| `CLAUDE.md`, `CHANGES.md` | tie rule, non-differentiability, legacy-ADC properties, new test in §12 | documentation |

Numerical effect (measured, 1–2 synthetic images, full TinyViT-5M):
- Only conversions that are exact ties in integer arithmetic change
  (|P/LSB_P − (k+½)| = 0 for every changed conversion). Rates per image: W8A8 with gains
  23.77 / 8.41 / 2.92 at ADC 8: 3 AV conversions (P = ±102 400); W4A4 same gains: 0.017% of
  AV conversions (P = ±400, because G = 2.92 makes LSB_P = 800/73); W4A4 at G = 1: 0.003%
  qkv, 0.31% qk, 0.21% AV. qkv/qk at the calibrated gains: none.
- The legacy path is bit-identical (all four precisions).
- Full-model analog logits are nevertheless **not** bit-identical to before. The quantized
  network amplifies single-code changes: adding +1 code to ONE conversion changes the
  W8A8 / ADC 8 / calibrated logits by 2.7–4.2% (the whole CIM-vs-FP32 error there is 4.2%).
  CIM-vs-FP32 error magnitudes are statistically unchanged (e.g. 4.48 / 4.27% before vs
  4.23 / 4.24% after). Earlier cached analog results stay internally consistent but will not
  reproduce bit-for-bit.

## Second update (Oct 2026): QAT context, Experiments A / B / T, amplifier option

Apply the whole ZIP again (EXPERIMENTS.md §1). Differences from the first update:

| file | change |
|---|---|
| `cim_exp/calibration.py` | **NEW** Experiment B. Exact column-sum probe (bit-identical to the ADC-off path), per-op histograms on an Imagenette-train calibration set, pre-declared P99.99 objective plus P99.9 / P99.999 / P100 sensitivity levels, held-out robustness plan (fixed vs calibrated, precision transfer, optional checkpoint transfer), EXPERIMENT_B.md |
| `cim_exp/trainpoint.py` | **NEW** Experiment T. All three checkpoints at the QAT training operating point (legacy functional ADC, 8-bit, W4/A4 and W8/A8), TRAINPOINT.md |
| `run_experiment.py` | new commands `prepare-calibration`, `calibrate`, `gain-robustness`, `trainpoint`; `--amp-swing-limit` (diagnostic; default none); file-name provenance printed and stored at `prepare`; CIM runner factored into `_run_jobs`; per-op clipping in progress output |
| `cim_exp/hardware.py` | `custom_gains` / `gain_note` (Experiment B only), `from_dict`, amplifier limit carried into the job label (`_amp<V>V`), `primary_matrix()` documented as Experiment A |
| `cim_exp/reporting.py` | per-op clipping (total/hi/lo), ADC utilisation P99.9/max, abs V_ADC_IN / V_OA P99.9 and amplifier saturation (`not_modeled` by default) in raw_metrics and the case tables; amplifier-limit column; caveats for the QAT training context, training-data overlap and the provisional DAC assumptions; fixed a variable-shadowing bug introduced in this round (caught by the tests before release) |
| `cim_exp/imagenette.py` | `split` argument (train for calibration only); `filename_provenance()` |
| `cim_exp/engine.py` | uses `HardwareConfig.from_dict` (custom gains) |
| `cim_sim/analog/dac.py`, `chain.py` | comments only: signed-input convention and MSB alignment marked PROVISIONAL (no code or numeric change) |
| `test_experiment_pipeline.py` | 53 → 71 checks: amplifier option, custom gains, per-op columns, calibration arithmetic, job planning, probe exactness |
| `AUDIT.md` | §10: what the QAT context establishes, each item checked against the files where possible (INT8 epoch discrepancy found) |
| `CLAUDE.md`, `EXPERIMENTS.md` | Experiments A/B/T, amplifier policy, QAT context, overlap caveat, exact PowerShell apply commands |
| `readme.md` | one-line pointer to Experiments A / B / T |

Numerical behaviour of Experiment A is unchanged: same simulator, same fixed gains,
same matrix. **Cache keys differ from the first update** (the hardware dictionary gained
fields, and the source fingerprint changed), so nothing computed with the first ZIP is
reused. No real-data runs existed yet.

---

## First update

Base: `CIM-Soc-qat-weights.zip` as received. Line endings preserved (all files were LF).
`tiny_vit.py` is untouched. The legacy digital-MAC/ADC path is untouched. The analog path
gives **bit-identical logits** to the delivered code for every pre-existing
configuration (verified: W8A8, W4A4, W4A8, W8A4 × gain 1 / calibrated, plus legacy W4A4).

## Files to REPLACE in your clone

| file | what changed | why |
|---|---|---|
| `cim_sim/analog/dac.py` | `CapacitiveDAC.unsigned_code()` and `vout_from_unsigned_code()` implement the specified CDAC equation `V = V_REF·c/2^N`; `_ideal_transfer` routes through them (same arithmetic); header records the hardware specification and the remaining assumptions | make the hardware equation explicit and testable; separate the specified unsigned DAC from the simulator's signed-input convention |
| `cim_sim/analog/chain.py` | `AnalogConfig.dac_input_alignment` ("msb" default / "lsb"); `dac_bits < act_bits` rejected; alignment factor in `K_MAC` and `domain_key`; `AnalogStats` rewritten (O(1) recording, uniform weighted reservoir, V_ADC_IN min/max/mean/std/percentiles, rail codes, amplifier saturation, common-mode residue diagnostic, `merge`/`state_dict`/`from_state`); `describe()`/`scale_report()` extended | 8-bit hardware DAC with narrower activations; unbiased percentiles over a whole run; stats were ~50% of runtime; chunked resumable runs need mergeable stats; diagnostics you asked for |
| `cim_sim/analog/adc.py` | `PreADCGain.saturation_modeled`; `SARADC.n_rail_hi/lo` counters | report amplifier saturation honestly as not modelled; rail-code upper bound alongside true clipping. SAR algorithm unchanged |
| `cim_sim/analog/__init__.py` | exports `DAC_ALIGNMENTS` | — |
| `cim_sim/array.py` | `CIMArray(cfg, analog_stats=None)` passes a shared `AnalogStats` to the chain | one statistics object per run / array pool |
| `run_tinyvit.py` | `--analog`, `--dac-type`, `--dac-bits`, `--dac-vref`, `--dac-align`, `--gain-qkv/qk/av`; analog summary printout; legacy-only modes refuse `--analog`; `--analog` + `--adc-enabled` rejected | the main runner could not activate the analog chain at all |
| `verify_array.py` | exits 1 on any failure (sections 1 and 3 used to print only) | a test that cannot fail is not a gate |
| `test_analog.py` | +56 checks (Tests 11-18): CDAC equation bit by bit, signed/unsigned input domains, alignment, amplifier, ADC full scale/clipping/rails at 8/10/12 b, accumulation round trip for all 4 precisions, statistics, common-mode diagnostic | requested validation |
| `CLAUDE.md` | updated to the verified implementation (revision note at top). It was **not inside the ZIP** (the old `.gitignore` excluded it); replace the copy you received separately | it was out of date (see AUDIT.md §8) |
| `readme.md` | current-scope header, run commands | it described the first INT4/no-ADC version |
| `.gitignore` | tracks `*.md`, `requirements.txt`, `configs/*.template.json`, `cim_exp/data/*.txt`; ignores `/data/`, `/results/`, `/checkpoints/`, `*.pth`, `*.npy`, `*.npz`, Excel lock files | the previous rules ignored CLAUDE.md and every file the pipeline needs, and allowed `~$*.xlsx` lock files |

## Files to ADD

| file | purpose |
|---|---|
| `run_experiment.py` | CLI: `download-data`, `inspect-checkpoints`, `prepare`, `verify`, `reference`, `cim`, `status`, `report`, `all` |
| `cim_exp/__init__.py` | package doc |
| `cim_exp/imagenette.py` | official download + safe extraction, verified class mapping (parachute trap), deterministic manifests, Cream TinyViT eval preprocessing, uint8 input cache |
| `cim_exp/checkpoints.py` | hash-checked registry, inspection, strict build; no substitution |
| `cim_exp/hardware.py` | the 24-config matrix, QAT selection, cache keys, source fingerprint |
| `cim_exp/metrics.py` | top-1, cross-entropy, named agreement, relative logit RMS, non-finite |
| `cim_exp/engine.py` | FP32 references; chunked, atomic, resumable CIM jobs; per-image clipping; finalization with consistency checks |
| `cim_exp/reporting.py` | CASE1-5, all CSVs, coverage verification, REPORT.md |
| `cim_exp/verify.py` | pre-flight validation suite |
| `cim_exp/data/imagenet_synsets.txt` | ImageNet-1k synset list (from timm 1.0.30, Apache-2.0) for mapping verification |
| `configs/checkpoints.template.json` | checkpoint paths + expected SHA-256 (copy to `configs/checkpoints.json`) |
| `test_multi_array.py` | reconstruction of the missing multi-array test (42 checks) |
| `test_experiment_pipeline.py` | offline pipeline tests (53 checks) |
| `requirements.txt` | dependencies |
| `EXPERIMENTS.md` | PowerShell runbook |
| `AUDIT.md` | audit of the delivered code |
| `CHANGES.md` | this file |

## Files NOT changed

`tiny_vit.py`, `_shim/**`, `cim_sim/quant.py`, `cim_sim/matmul.py`,
`cim_sim/attention.py`, `cim_sim/report.py`, `cim_sim/__init__.py`,
`cim_sim/analog/macro.py`, `cim_sim/analog/digital.py`, `exp_analog.py`,
`experiments.py`, `test_cim_array.py`, `test_matmul.py`, `test_quant.py`,
`cim_architecture_estimator.xlsx`.

Delete `~$cim_architecture_estimator.xlsx` from the repository. It is an Excel lock file.

## Behavioural notes

- `AnalogConfig.dac_bits=None` still means "follow act_bits". All earlier analog results
  reproduce bit-for-bit. The new pipeline uses `dac_bits=8` with MSB alignment, which is
  bit-identical for these precisions.
- `AnalogStats` percentiles now differ slightly from the old ones: the old subsample was
  biased toward the first images. Its random stream changed too. Statistics never affect
  outputs.
- `run_tinyvit.py` banner text changed: legacy runs now say "legacy, ADC disabled". The
  numbers are unchanged (stage-3 smoke 24.55% / 0.9694 before and after).
