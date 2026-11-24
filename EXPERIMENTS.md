# Real-image CIM experiments — runbook (Windows / PowerShell)

This file is the operating manual for `run_experiment.py`: the Imagenette evaluation of
the original pretrained TinyViT-5M and the QAT-INT4 / QAT-INT8 checkpoints on the modelled
analog CIM datapath. Every command below is run from the repository root in a VS Code
PowerShell terminal.

| experiment | command(s) | purpose | outputs |
|---|---|---|---|
| **A (primary)** | `prepare`, `verify`, `reference`, `cim`, `report` | five comparison cases; 24 fixed-gain hardware configs (G = 1 and the fixed W8/A8 set) | `results\<stage>\` |
| T (optional) | `trainpoint` | each checkpoint at its QAT training operating point (legacy functional ADC) | `results\<stage>\expT\` |
| B (optional, after A) | `prepare-calibration`, `calibrate`, `gain-robustness` | real-image gain calibration and its robustness on held-out images | `results\<stage>\expB\` |

Experiments T and B never modify Experiment A's tables or gains.

Nothing here falls back to synthetic data or to another checkpoint. A missing dataset,
a missing checkpoint or a checkpoint with an unexpected SHA-256 stops the run with an
explanation.

---

## 0. One-time setup

```powershell
# Python 3.10 - 3.12, 64-bit. From the repo root:
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -c "import torch, numpy, PIL; print(torch.__version__, numpy.__version__, PIL.__version__)"
```

A GPU is not used. The CPU wheel is enough.

## 1. Apply the code update

```powershell
cd C:\path\to\your\clone              # the repository root (contains run_tinyvit.py)
git status                              # must be clean before you start
git switch -c qat-imagenette-experiment # work on a branch

# unpack the delivered ZIP to a temporary folder, then merge it over the clone
$zip = "$HOME\Downloads\CIM-Soc-qat-weights-final.zip"   # adjust to where you saved it
$tmp = Join-Path $env:TEMP "cim_update"
Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
Expand-Archive -Path $zip -DestinationPath $tmp -Force
robocopy "$tmp\CIM-Soc-qat-weights" . /E /NFL /NDL /NJH /NP
# robocopy exit codes 0-7 mean success (1 = files copied); 8 or higher is an error.

# the Excel lock file committed by mistake (only if present in your clone)
git rm --cached --ignore-unmatch '~$cim_architecture_estimator.xlsx'
Remove-Item -LiteralPath '~$cim_architecture_estimator.xlsx' -ErrorAction SilentlyContinue

git status                              # review what changed (see CHANGES.md)
git diff --stat
```

`robocopy … /E` adds new files and overwrites changed ones. It deletes nothing, so your
`checkpoints\`, `data\` and `results\` folders are untouched.

`.gitignore` was updated so that `*.md`, `configs/*.json` templates,
`cim_exp/data/*.txt` and `requirements.txt` are tracked, while `data/`, `results/`,
`checkpoints/` and `*.pth` stay ignored. Do not commit checkpoints, datasets or results
unless you intend to.

## 2. Checkpoints

The QAT files you shared are **PyTorch archives**: `torch.save` writes a zip container
with an internal folder named `best_model_only/`. The chat upload showed them as
`.pth.zip`. **Do not unzip them.** Unzipping produces a `best_model_only/` folder that
`torch.load` cannot read. Use the files as `.pth`; if your copies end in `.pth.zip`,
just rename them to `.pth`.

```powershell
New-Item -ItemType Directory -Force checkpoints | Out-Null
# copy/rename your files to:
#   checkpoints\best_model_int4_qat.pth
#   checkpoints\best_model_int8_qat.pth
Get-FileHash checkpoints\best_model_int4_qat.pth -Algorithm SHA256
Get-FileHash checkpoints\best_model_int8_qat.pth -Algorithm SHA256
#   expected (files as shared in our conversation):
#   INT4  A903663B2A8E2C7D6BD8C1FB80D6A974495191678B981F3043BD5DFAB6A22B56
#   INT8  9F8060D4A7464EA9C0873EC3140EACB86CBC090214DFD77628DB9EEC556DFA70

Copy-Item configs\checkpoints.template.json configs\checkpoints.json
code configs\checkpoints.json           # edit paths if yours differ
```

If you intentionally use different files, set their `expected_sha256` to `null` (or to
the new hash). The actual hash is recorded in every result row either way.

The pretrained checkpoint (`tinyvit5m.pth`, ~22 MB) is downloaded from the official
TinyViT release on request and checked against SHA-256
`e4894bab91be2d8f1b5e9a2147b57f35ae84be46e195d6586308864fb06fad3d`:

```powershell
python run_experiment.py inspect-checkpoints --download-pretrained
```

This prints, for each checkpoint, its hash, tensor count, head shape, dtypes, any
quantizer keys, the int4 "off-grid" statistic, and whether it loads strictly into
TinyViT-5M. It also writes `results\checkpoint_inspection.json`.

## 3. Unit tests (no dataset needed)

```powershell
python verify_array.py
python test_analog.py
python test_multi_array.py
python test_experiment_pipeline.py
python run_tinyvit.py --stage3-only
```

Every script exits non-zero on failure. Expected results:
- verify_array: "all checks passed";
- test_analog: 101/101;
- test_multi_array: 42/42;
- test_experiment_pipeline: 71/71 (tests 8 and 11 are skipped if `tinyvit5m.pth` is
  absent);
- stage-3 smoke: about 24.55% / cosine 0.9694. That pair was measured with torch 2.14
  and can differ slightly with another torch version.

## 4. Dataset (official fast.ai Imagenette)

```powershell
python run_experiment.py download-data                 # imagenette2-160, ~94 MB
# -> data\imagenette2-160\{train,val}\<wnid>\*.JPEG
#    data\imagenette2-160.provenance.json  (archive SHA-256 as downloaded)
```

If the download is blocked (proxy or firewall), download
`https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz` in a browser, save it as
`data\imagenette2-160.tgz`, and run the same command again. It will then only extract.

`--variant imagenette2-320` uses the 320 px version, which avoids upsampling for the
224 px model. Use the same variant for every stage of a run.

## 5. Stage 1: smoke test (100 real images)

```powershell
python run_experiment.py prepare   --stage smoke              # manifest + decoded input cache
python run_experiment.py verify    --stage smoke              # validation suite (about 2-4 min)
python run_experiment.py reference --stage smoke              # FP32 logits for the 3 models
python run_experiment.py cim       --stage smoke --workers 4  # 48 CIM jobs x 100 images
python run_experiment.py status    --stage smoke
python run_experiment.py report    --stage smoke
code results\smoke\REPORT.md
```

`verify` must report **0 FAIL** before you trust anything. Its checks are:

- the four unit-test scripts and the stage-3 smoke test;
- the CDAC equation and signed-input handling;
- strict loading of all 3 checkpoints;
- finite FP32 logits;
- that the analog path is really active (output differs from the legacy digital path
  and from FP32);
- W4/A4 and W8/A8 smoke runs at both gain sets;
- that LSB_P scales as 1/G per operation;
- counted conversions equal the scheduler geometry;
- bit-identical repeat runs;
- the metric definitions;
- the class mapping, statically against the bundled synset list and empirically on
  your real images. Empirically, each class's 10-way modal pretrained-FP32 prediction
  must be the class itself. The 1000-way modal prediction is shown for information,
  because ImageNet near-synonyms such as "tape player" can legitimately win it.

**Workers.** Each worker is one process with one thread. Use the number of physical
cores minus one. Measured on the development machine: about 2.2 s per image per job at
an 8-bit ADC, and about 3.6 s at 12-bit, per core. The smoke stage is
48 × 100 image-jobs ≈ 3.5 core-hours, so ≈ 50 min with 4 workers.

Check the smoke results before going further:

1. `class_mapping_check.csv`: every class `mapping_confirmed = True`.
2. `coverage.json`: `"complete": true`, with rows per case
   CASE1 24, CASE2 2, CASE2b 1, CASE3 24, CASE4 24, CASE5 24.
3. `raw_metrics.csv`: every `status = OK`, and `consistency_ok = True` for every CIM row.
4. `REPORT.md`: the tables look sensible.

## 6. Stage 2: full validation split (3,925 images)

Run this only after the smoke stage is validated:

```powershell
python run_experiment.py prepare   --stage full
python run_experiment.py verify    --stage full     # class-mapping check uses all images
python run_experiment.py reference --stage full
python run_experiment.py cim       --stage full --workers 6
python run_experiment.py report    --stage full
```

Cost: 48 × 3,925 image-jobs ≈ 140 core-hours, so about 24 h with 6 workers. Run it
overnight, or prioritise configurations first:

```powershell
python run_experiment.py cim --stage full --workers 6 --only W8A8_ADC10_cal_w8a8 W4A4_ADC10_cal_w8a8
python run_experiment.py report --stage full --allow-incomplete    # partial report, clearly marked
```

## 6a. Experiment T (optional): checkpoints at their QAT training operating point

Per the supplied QAT context, both QAT checkpoints were trained through a surrogate of the
**legacy functional ADC**: 8-bit, worst-case full scale, at W4/A4 (INT4) and W8/A8
(INT8). They were **not** trained against the analog chain used in Experiment A. This
check runs all three checkpoints at both points through the reference simulator, so you
can confirm each QAT checkpoint beats the pretrained model at its own training point
before interpreting its analog results.

```powershell
python run_experiment.py trainpoint --stage smoke --workers 4    # 6 jobs, ~1.2 s/image-job/core
code results\smoke\expT\TRAINPOINT.md
```

Prerequisite: `reference` has run.

## 6b. Experiment B (optional, after A): gain calibration and robustness

This is a separate follow-up that never replaces the fixed gains of Experiment A.

1. **Calibration data:** Imagenette **train** split, 10 images per class (seed 4242).
   `prepare-calibration` verifies by SHA-256 that it shares no file with the evaluation
   manifest.
2. **Measurement:** for each checkpoint (pretrained, matched QAT) and each precision, an
   exact probe records every per-depth-block column sum |P| that an ADC would digitize.
   The probe gives no ADC/gain feedback and is bit-identical to the ADC-off network.
3. **Objective (pre-declared):** **P99.99**. The full scale is the 99.99th percentile of
   |P|, i.e. a clipping target of at most 0.01% of conversions per operation on the
   calibration data. Then G = (V_REF_ADC/2)/(K_MAC·FS), rounded to 0.01. P99.9 (higher
   gain), P99.999 and P100 (no calibration clipping) are reported as sensitivity levels
   and are never used to pick a gain from evaluation results.
4. **Evaluation:** on the held-out val images of the stage, each calibrated gain set is
   compared with G = 1 and the fixed W8/A8 set at the same checkpoint and precision.
   - Experiment A's cached runs are reused automatically.
   - Also included: W8A8-calibrated gains applied at the other precisions (precision
     transfer).
   - Optionally (`--cross-apply`): each checkpoint's gains applied to the other
     checkpoint.

   Reported per row:
   - clipping % per operation, plus the held-out/target ratio;
   - ADC utilisation;
   - relative logit RMS and agreement vs the tested checkpoint's own FP32 output;
   - top-1 and CE;
   - applied gains and full provenance (calibration checkpoint, precision, objective,
     calibration manifest hash).

```powershell
python run_experiment.py prepare-calibration --stage smoke     # 100 train images, disjoint check
python run_experiment.py calibrate --stage smoke               # ~1 s/image x 8 (checkpoint, precision)
python run_experiment.py gain-robustness --stage smoke --workers 4
code results\smoke\expB\EXPERIMENT_B.md
```

`gain-robustness` options:
- `--levels p99.99 p100` to restrict the levels;
- `--b-adc-bits 8 10 12` to add ADC resolutions (default 8 only);
- `--cross-apply`;
- `--no-precision-transfer`.

Cost with defaults: about 54 CIM evaluations × 100 images ≈ 3.3 core-hours, minus what
Experiment A already cached.

## 6c. Amplifier swing (diagnostic only)

No amplifier output-swing specification exists. Experiment A therefore uses an ideal
linear amplifier and reports amplifier saturation as `not_modeled`, never as 0. ADC
clipping is always measured separately.

If a swing figure becomes available, run a labelled diagnostic variant **in its own run
folder**. Its labels end in `_amp<V>V`, and amplifier saturation is then counted per
operation, separately from ADC clipping.

```powershell
python run_experiment.py prepare --stage smoke --run-name smoke_amp0p25
python run_experiment.py reference --stage smoke --run-name smoke_amp0p25
python run_experiment.py cim --stage smoke --run-name smoke_amp0p25 --amp-swing-limit 0.25 --workers 4
python run_experiment.py report --stage smoke --run-name smoke_amp0p25 --amp-swing-limit 0.25
```

## 7. Interrupting, resuming, regenerating

- **Interrupt** with Ctrl+C at any time. Work is saved per chunk (25 images) with atomic
  writes, so at most one chunk per worker is lost.
- **Resume** by re-running exactly the same `cim` command. Completed chunks and jobs are
  skipped.
- **Regenerate** tables and `REPORT.md` with `report --stage <stage>`. It only reads
  cached outputs and never re-runs a model.
- **Code changes invalidate caches automatically.** Every cache key contains a
  fingerprint of all output-affecting source files, plus:
  - checkpoint SHA-256, full hardware config, manifest SHA-256 and preprocessing id;
  - torch/numpy versions, chunk size, threads and batch size.

  After a code edit, results are recomputed, not mixed. `results\<stage>\cache\` can
  be deleted at any time to start clean.
- `--keep-chunks` keeps per-chunk files after a job is finalized (for debugging).
- `--qat-selection both` additionally runs each QAT checkpoint at the other weight
  precision (72 jobs). Those rows carry `precision_matches_training = False`.

## 8. Result files (`results\<stage>\`)

| file | content |
|---|---|
| `input_manifest.csv` | the fixed image list (path, class, 10-way and 1000-way label, file SHA-256) |
| `run_config.json` | stage, variant, manifest SHA-256, preprocessing id, mapping notes |
| `verify.json` | the validation suite results |
| `run_manifest.csv` | every planned job: run id (= cache key), model, checkpoint hash, hardware label, status |
| `raw_metrics.csv` | one row per model × path × config: top-1 (1000-/10-way), CE, non-finite, ADC/clipping totals, images clipped, consistency |
| `pairwise_comparisons.csv` | one row per comparison (CASE1-5): both sides' identities and metrics, named agreement and error columns, full hardware config, manifest |
| `case1_table.csv` … `case5_table.csv` | the same rows split by case |
| `accuracy_summary.csv`, `cross_entropy_summary.csv` | per model/path/config |
| `agreement_summary.csv`, `logit_error_summary.csv` | per comparison, named columns only |
| `clipping_summary.csv` | per CIM job × {qkv, qk, av, ALL} (see the list below) |
| `raw_metrics.csv` / case tables (extra columns) | per-operation ADC clip % (total/hi/lo), ADC utilisation P99.9 and max, abs V_ADC_IN / V_OA P99.9, amplifier saturation (`not_modeled` by default) |
| `expT\trainpoint.csv`, `expT\TRAINPOINT.md` | Experiment T |
| `expB\calibration_manifest.csv`, `calibration_config.json`, `calibration_gains.csv`, `calibration.json`, `gain_robustness.csv`, `EXPERIMENT_B.md` | Experiment B |
| `class_mapping_check.csv` | empirical mapping verification |
| `checkpoints.json`, `environment.json`, `coverage.json` | checkpoint inspection, software environment, coverage verification |
| `REPORT.md` | generated report: caveats, checkpoint table, mapping check, five case tables, precision comparison, ADC/gain comparison, clipping, per-op analog diagnostics |
| `cache\` | raw logits and statistics (reference + one folder per CIM job) |

`clipping_summary.csv` contains, per job and operation:

- conversions; clip hi/lo/total counts and %; images with clipping;
- rail-code counts; amplifier saturation (`not_modeled` unless a swing limit is set);
- V_OA and V_ADC_IN min/max/mean/std, P1/P50/P99 and |·| P50/P99/P99.9;
- swing utilisation; common-mode residue diagnostics; applied gain and both reference
  voltages.

Percentages of conversions use the conversion count of that operation as the
denominator. Percentages of images use the number of images.

## 9. What the numbers mean (read before quoting)

- **Training-data overlap.** Imagenette2 re-split about all ImageNet-1k train+val images
  of its 10 classes 70/30, so most val images are expected to be ImageNet-1k **training**
  images. The pretrained model saw them, and the QAT runs fine-tuned on them; QAT also
  early-stopped on ImageNet val. `prepare` prints the file-name counts. Absolute
  accuracies are optimistic and QAT-vs-pretrained differences are confounded.
  CIM-vs-own-FP32 comparisons (CASE3/4) on identical images are much less affected.
- **QAT was trained on the legacy ADC, not the analog chain.** QAT results in Experiment A
  measure transfer to a different ADC model; Experiment T shows the trained-for
  condition.

- **Agreement is not accuracy.** Accuracy and cross-entropy are always against the true
  Imagenette labels. Agreement is the share of identical top-1 predictions versus the
  reference named in the column.
- **Primary accuracy is 1000-way** (argmax over all TinyViT outputs). 10-way restricts
  the argmax to the 10 Imagenette classes.
- **CASE3 and CASE4** compare a checkpoint's CIM output against its own FP32 software
  output: this is the effect of CIM execution. **CASE1** compares two checkpoints under
  identical simulated hardware. **CASE5** mixes checkpoint, precision and hardware
  effects. **CASE2** has no hardware involved.
- **The calibrated gains were derived for W8/A8** and are held fixed. They are not
  claimed optimal for other precisions.
- **Mixed precision (W4/A8, W8/A4) is inference-only.** No checkpoint was trained that
  way.
- **The CIM path** = digital operand quantization of qkv/QK^T/AV plus the analog chain.
  Everything else runs in FP32. The matrix does not separate DAC, macro, amplifier and
  ADC error. That needs dedicated counterfactual runs (change one stage, hold the rest).

## 10. Troubleshooting

- *"checkpoint … SHA-256 … does not match expected"*: you pointed at a different file.
  Fix the path, or deliberately update `expected_sha256`.
- *"…\val not found"*: run `download-data`. There is no synthetic fallback by design.
- *`verify` FAIL class_mapping_empirical*: do not proceed. Inspect
  `class_mapping_check.csv`, then check the dataset variant and folder layout.
- *Slow or stalled workers*: Windows Defender scanning `results\` can slow writes.
  Excluding the repo folder helps. Each worker needs about 0.5-1 GB RAM.
- *`ModuleNotFoundError: timm`*: run from the repo root. `_shim/` provides the 5 timm
  symbols TinyViT needs.
