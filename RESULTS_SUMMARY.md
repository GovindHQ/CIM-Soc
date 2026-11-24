# CIM QAT protocol v2 — final results (VAL + ImageNetV2 TEST + extension + post hoc), updated 17 Nov 2026

Copied from the remote reports `REPORT_{val,test}_{W8A8,W4A4}.json` (generated 29 Oct 06:55–06:58 UTC
with package v2.7 after `restat`) and the final audit (145 PASS / 3 WARN / 0 FAIL); the extension (§8.4) from `REPORT_test_{th07,top,pooled}_*.json` (package v2.8) and its audit (212 PASS / 6 WARN / 0 FAIL, 1 Nov); the audit re-run with package v2.8.1 after the post hoc analysis (214 PASS / 6 WARN / 0 FAIL, 17 Nov, §8.6). Actual cim_sim
simulator, batch 1, CPU; calibrated gains frozen 25 Oct (W8A8 22.92/8.75/2.72, W4A4 29.05/11.28/3.82).
Caveats: single training seed (CIs are over images, not training runs); VAL selected the checkpoints
and the D2 winner; TEST = ImageNetV2 matched-frequency (different distribution: compare within a table);
train/VAL overlap measured and harmless (below); simulator assumptions (ideal analog, provisional offset-binary input).

**Train/VAL overlap (measured 29 Oct):** 799 VAL images (1.6 %; 797 distinct contents, two of them repeated inside VAL) have an identical-content copy in the TRAIN split. Measured, not assumed (`tools_qat/val_overlap_sensitivity.py`, 29 Oct): every model scores only 35–39 % top-1 on them against 72–81 % overall, so including them LOWERS VAL top-1 by 0.59–0.70 pp for every row rather than inflating it, and every paired comparison changes by at most 0.07 pp (C10 +0.24 → +0.26 pp, CI +0.10 to +0.42). The overlap is therefore not a source of VAL optimism; the optimism of the QAT rows on VAL comes from selecting on VAL. ImageNetV2 (TEST) has no content overlap with TRAIN.

## Headline (plain language)

Accuracy differences are in percentage points (top-1).

- **8-bit weights and activations (W8A8), 8-bit ADC:** with calibrated per-operation ADC gains, the
  pretrained model on the CIM simulator is about 0.1 points below full precision. The new
  hardware-aware training (QAT) neither helps nor hurts there (new QAT vs untrained: −0.03, not
  significant). On the larger pooled ImageNetV2 set (20,683 images) the new W8A8 QAT model is 0.15 points
  below full precision (95 % interval 0.005 to 0.28 below; a check chosen after seeing the results). So
  the gap is at most about 0.3 points.
- **4-bit (W4A4):** almost all of the loss comes from rounding the numbers to 4 bits (−2.7 to −2.8
  points), not from the ADC. The new QAT recovers 2.3–2.6 points over the untrained model and beats the
  old QAT by about 1.8–1.9 points. It stays 0.4–0.6 points below full precision.
- **Gain calibration matters:** without it (gain 1) the ADC costs about 4 points (W8A8) and 6–6.6
  points (W4A4).
- The original ImageNetV2 test (10,000 images) is the pre-declared result. The two extra variants were
  added on request after the results were frozen. No clear result of the 10k reverses on the pooled set.
- **Post hoc check (2 Nov), each trained model against its own float version:** for 4-bit, the new
  QAT model scores 0.45 points more on the CIM simulator than the same weights in plain float. This is
  expected: training optimised the hardware version (4-bit numbers and the 8-bit ADC) to match the
  original model, never the float version. It does not mean the hardware adds accuracy; the model is
  still 0.59 points below the original. For 8-bit everything is within 0.15 points. The old QAT models
  lost their accuracy in the training of the weights, not on the hardware.

## Training runs

| run | precision | GPU | budget / patience | config hash | epoch-0 PTQ VAL (surrogate) | best epoch / VAL top-1 | stopped reason | plateau | best / final SHA |
|---|---|---|---|---|---|---|---|---|---|
| w8a8_kd_pilot5ep | W8A8 | 1 | 5 / 5 | (in `config_resolved.json`; not in summary.json) | 80.670 % | epoch 3 (step 59,949) / 80.668 % | completed the scheduled budget (finished 26 Oct 23:20:21 UTC) | converged: last epoch improved KL by 0.00007 nats (threshold 0.002), top-1 −0.02 pp (paired SE 0.04) | `c69284c9…` (step 59,949) / `6fad049f…` (step 99,915) |
| w4a4_kd_pilot10ep | W4A4 | 0 | 10 / 3 | `164f8cf4461f4dac` (recorded by the fork) | 78.004 % | epoch 2 (step 39,966) / 80.234 % | VAL early stopping at step 99,915 (epoch 5), finished 27 Oct 00:09:59 UTC | early-stopped on VAL (no extension); last epoch improved KL by 0.0006 nats, top-1 −0.004 pp | `6863b759…` (step 39,966) / `102a4645…` (step 99,915) |
| w4a4_kd_pilot10ep_full (D2 fork; reported W4A4 run) | W4A4 | 1 | 10 / 10 (early stopping off) | `5c17489775fe52fd` | (inherited) 78.004 % | epoch 10 (step 199,830) / 80.330 % | completed the scheduled budget (finished 28 Oct 17:32:49 UTC) | converged within the budget: last epoch improved KL by 0.0016 nats (just under the 0.002 threshold), top-1 +0.05 pp (paired SE 0.08) | `a72ab4f5…` / `1e04ce4d…` (both step 199,830; identical weights) |
| w4a4_ce_pilot10ep (optional) | W4A4 | — | 10 / 3 | — | NOT RUN (optional stage 10.4 not executed; no C9/E7 rows) | | | | |

## 8. Evaluation (stages 11–13, actual simulator) — EXECUTED (VAL 28–29 Oct, TEST 29 Oct, extension D3 29–30 Oct); final audits 145 PASS / 3 WARN / 0 FAIL (29 Oct), 212 / 6 / 0 (1 Nov) and 214 / 6 / 0 (17 Nov, v2.8.1, §8.6)

**VAL status, 28 Oct ≈23:30 UTC (29 Oct ≈05:00 IST; reported by user; full numbers in `eval/val/REPORT_val_W8A8.md` /
`REPORT_val_W4A4.md` on the remote):**
- D2 applied with `select-d2` (v2.6) before any TEST read: `W4NEW` = the run printed by select-d2
  (pilot best 80.23 %, fork best 80.33 %, surrogate VAL); audit `L9.d2.rule` and `L9.d2.before_test` PASS.
- W4A4 E6 (12 workers) and E6s (16 workers, 0.97 s/img): **logits bit-identical** (best and final are
  the same weights, confirmed tensor by tensor); C7b 0 discordant images. A determinism check of the
  simulator across worker counts on 50,000 images.
- Step E diagnostic (VAL, simulator, 16 workers, 0.98 s/img), declared 28 Oct before it ran:

  | W4A4, simulator, VAL 50,000 | pilot best (epoch 2) | fork best (epoch 10, D2 choice) |
  |---|---|---|
  | top-1 | 80.16 % | 80.40 % |
  | NLL | 0.8467 | 0.8428 |
  | KL(FP32‖run) | 0.0279 | 0.0222 |
  | agreement with FP32 | 94.92 % | 95.63 % |

  Both checkpoints and the D2 choice were selected on VAL, so this VAL difference favours the fork.
  Paired test C10: VAL +0.24 pp (+0.07, +0.40), McNemar p 0.005; TEST +0.27 pp (−0.15, +0.66), p 0.21 (§8.3).
- Reported: W4A4 C6 (new QAT vs PTQ) +2.37 pp; W8A8 C6 −0.03 pp with the CI including 0.
- Pre-TEST audit: 120 PASS, 3 WARN (S4.info: train∩val 797, within-VAL 29, within-TRAIN 5,909
  identical-content images; informational), 0 FAIL; E1.val 17 runs on identical inputs; one TEST read
  logged (v2-integrity, allowed purpose).
- ADC-utilisation median / p99.9 columns: recomputed with `qat_evaluate.py restat` (v2.7) before the
  final reports (§9, 29 Oct); the tables below are the recomputed ones.
- **Train/VAL overlap sensitivity (EXECUTED 29 Oct 07:21 UTC, read-only):** 799 VAL images (1.6 %; 797 distinct contents, two of them repeated inside VAL) have an identical-content copy in the TRAIN split. Measured, not assumed (`tools_qat/val_overlap_sensitivity.py`, 29 Oct): every model scores only 35–39 % top-1 on them against 72–81 % overall, so including them LOWERS VAL top-1 by 0.59–0.70 pp for every row rather than inflating it, and every paired comparison changes by at most 0.07 pp (C10 +0.24 → +0.26 pp, CI +0.10 to +0.42). The overlap is therefore not a source of VAL optimism; the optimism of the QAT rows on VAL comes from selecting on VAL. ImageNetV2 (TEST) has no content overlap with TRAIN.

Separate tables per split. VAL QAT rows are selection-biased; TEST is ImageNetV2 (a different
distribution); never compare absolute numbers across the two tables.

**TEST status, 29 Oct ≈06:15 UTC (reported by user):**
- W8A8 TEST matrix (v2.6, 16 workers): E1–E6s finished, no errors, ≈0.99 s/img on analog rows.
- W4A4 TEST matrix: E1–E5b finished. E6/E6s first attempt **refused** before any read: `W4NEW` was set
  but not exported into the `bash -c` that launched the matrix, so the model was `run::best` /
  `run::final`; the TEST guard stopped both rows ("no summary.json; the run is not finished").
  In `qat_evaluate.py sim` the model is resolved before the TEST manifest is read and before the
  output folder is created, so the refused rows read no TEST manifest or image, wrote no access-log
  line and created no folder (from the code; confirmed on the remote 29 Oct: no access-log entry from that launch, no `_best`/`_final` folder). Relaunched with `W4NEW=w4a4_kd_pilot10ep_full`
  exported, dry run checked, E6 and E6s only (log `eval_test_W4A4_E6.log`): finished, 0.95 /
  0.96 s/img, no errors.
- Pilot-best TEST row (§11.3b) run 29 Oct 06:18 UTC. From the v2.7 folder: `restat` VAL (12 runs) and
  TEST (12 runs), the four reports (no † rows), final audit **145 PASS, 3 WARN (S4.info), 0 FAIL**
  (`AUDIT_20261029T065916+0000.md`): L9.d2.rule / before_test (D2 18:29:22Z < first TEST read
  23:44:41Z) / test PASS; L6 for every TEST row incl. order; L4 19 TEST reads (final_eval,
  v2_integrity); E1.test and E1.val 17 runs on identical inputs. Access-log check: no entry from the
  empty-W4NEW launch, no `_best`/`_final` folder. After restat E6 and E6s agree (VAL util median
  12.766/12.117/2.425 % both; p99.9 80.14/81.24 vs 80.14/80.97, 64.63 vs 64.54 %).

### 8.1 VAL (ImageNet-1k val, 50,000) — EXECUTED 28–29 Oct (from REPORT_val_*.json, 29 Oct 06:55–06:57 UTC)

Cells: top-1 % / NLL / KL(FP32‖run). Clip: % of conversions per op (exact counts).

| id | W8A8 | W8A8 clip % qkv/qk/av | W4A4 | W4A4 clip % qkv/qk/av |
|---|---|---|---|---|
| E1 FP32 pretrained | 80.73 / 0.8266 / 0.0000 | — | 80.73 / 0.8266 / 0.0000 | — |
| E2 ADC-off (operand quantization only) | 80.72 / 0.8268 / 0.0004 | — | 78.10 / 1.0375 / 0.1413 | — |
| E3 PTQ, G = 1 | 76.85 / 1.0771 / 0.1849 | 0.0000/0.0000/0.0000 | 72.00 / 1.4294 / 0.4701 | 0.0000/0.0000/0.0000 |
| E4 PTQ, calibrated gains | 80.68 / 0.8288 / 0.0024 | 0.0099/0.0102/0.0095 | 78.03 / 1.0447 / 0.1439 | 0.0100/0.0097/0.0083 |
| E4b PTQ, historical gains | 80.71 / 0.8277 / 0.0022 | 0.0152/0.0058/0.0140 | — | — |
| E5 old QAT, new hardware | 79.56 / 0.7846 / 0.4209 | 0.0108/0.0077/0.0092 | 78.98 / 0.8007 / 0.4298 | 0.0113/0.0073/0.0085 |
| E5b old QAT, legacy ADC (its training point) | 79.76 / 0.7658 / 0.4222 | — | 79.01 / 0.7917 / 0.4477 | — |
| E6 new QAT best (primary) | 80.64 / 0.8284 / 0.0022 | 0.0100/0.0072/0.0094 | 80.40 / 0.8428 / 0.0222 | 0.0107/0.0074/0.0094 |
| E6s new QAT final | 80.66 / 0.8280 / 0.0018 | 0.0099/0.0070/0.0095 | 80.40 / 0.8428 / 0.0222 | 0.0107/0.0074/0.0094 |
| D2 diagnostic: W4A4 pilot best (epoch 2) | — | — | 80.16 / 0.8467 / 0.0279 | 0.0103/0.0090/0.0089 |
| E7 CE pilot | NOT RUN | — | NOT RUN | — |

### 8.2 TEST (ImageNetV2 matched-frequency, 10,000) — EXECUTED 29 Oct (from REPORT_test_*.json, 06:57–06:58 UTC)

Cells: top-1 % / NLL / KL(FP32‖run). Clip: % of conversions per op (exact counts).

| id | W8A8 | W8A8 clip % qkv/qk/av | W4A4 | W4A4 clip % qkv/qk/av |
|---|---|---|---|---|
| E1 FP32 pretrained | 69.69 / 1.3368 / 0.0000 | — | 69.69 / 1.3368 / 0.0000 | — |
| E2 ADC-off (operand quantization only) | 69.69 / 1.3373 / 0.0006 | — | 66.95 / 1.5821 / 0.1786 | — |
| E3 PTQ, G = 1 | 65.71 / 1.6323 / 0.2370 | 0.0000/0.0000/0.0000 | 60.38 / 2.0248 / 0.5776 | 0.0000/0.0000/0.0000 |
| E4 PTQ, calibrated gains | 69.59 / 1.3402 / 0.0034 | 0.0099/0.0096/0.0096 | 66.70 / 1.5869 / 0.1842 | 0.0100/0.0093/0.0088 |
| E4b PTQ, historical gains | 69.75 / 1.3387 / 0.0032 | 0.0151/0.0053/0.0140 | — | — |
| E5 old QAT, new hardware | 68.22 / 1.3630 / 0.4733 | 0.0107/0.0073/0.0100 | 67.38 / 1.3884 / 0.4918 | 0.0112/0.0070/0.0092 |
| E5b old QAT, legacy ADC (its training point) | 68.84 / 1.3348 / 0.4728 | — | 67.57 / 1.3725 / 0.5103 | — |
| E6 new QAT best (primary) | 69.70 / 1.3401 / 0.0030 | 0.0100/0.0067/0.0095 | 69.26 / 1.3575 / 0.0314 | 0.0106/0.0070/0.0095 |
| E6s new QAT final | 69.64 / 1.3387 / 0.0026 | 0.0099/0.0065/0.0096 | 69.26 / 1.3575 / 0.0314 | 0.0106/0.0070/0.0095 |
| D2 diagnostic: W4A4 pilot best (epoch 2) | — | — | 68.99 / 1.3644 / 0.0389 | 0.0103/0.0084/0.0091 |
| E7 CE pilot | NOT RUN | — | NOT RUN | — |

### 8.3 Paired comparisons (b vs a, same images): Δtop-1 pp (95% bootstrap CI), exact McNemar p

| id | isolates | W8A8 VAL | W8A8 TEST | W4A4 VAL | W4A4 TEST |
|---|---|---|---|---|---|
| C1 | operand quantization (exact integer MAC, no ADC) | -0.01 (-0.06, +0.04), p 0.76 | +0.00 (-0.13, +0.13), p 1 | -2.64 (-2.85, -2.41), p 2.6e-120 | -2.74 (-3.26, -2.22), p 5.7e-24 |
| C2 | ADC + analog chain at unity gain | -3.87 (-4.11, -3.62), p 9.6e-213 | -3.98 (-4.57, -3.35), p 5.2e-39 | -6.10 (-6.40, -5.82), p 0 | -6.57 (-7.26, -5.81), p 1.3e-74 |
| C3 | ADC + analog chain at calibrated gains | -0.05 (-0.13, +0.03), p 0.28 | -0.10 (-0.30, +0.10), p 0.38 | -0.06 (-0.23, +0.10), p 0.47 | -0.25 (-0.65, +0.15), p 0.24 |
| C4 | gain calibration (unity -> calibrated) | +3.82 (+3.57, +4.07), p 1.4e-207 | +3.88 (+3.28, +4.47), p 2e-37 | +6.03 (+5.74, +6.33), p 0 | +6.32 (+5.61, +7.02), p 1.1e-68 |
| C4b | calibrated vs historical synthetic-input gains | -0.04 (-0.11, +0.04), p 0.4 | -0.16 (-0.36, +0.03), p 0.14 | — | — |
| C5 | old QAT vs PTQ (same hardware, same gains) | -1.11 (-1.34, -0.90), p 3.4e-24 | -1.37 (-1.90, -0.82), p 6.5e-07 | +0.94 (+0.69, +1.20), p 4.7e-13 | +0.68 (+0.02, +1.32), p 0.04 |
| C8 | old QAT checkpoint: its training point (legacy ADC) -> new hardware | -0.20 (-0.40, +0.01), p 0.059 | -0.62 (-1.11, -0.13), p 0.014 | -0.03 (-0.23, +0.18), p 0.8 | -0.19 (-0.71, +0.33), p 0.49 |
| C6 | new QAT (VAL-selected) vs PTQ | -0.03 (-0.12, +0.06), p 0.5 | +0.11 (-0.09, +0.33), p 0.36 | +2.37 (+2.13, +2.60), p 1e-92 | +2.56 (+1.99, +3.09), p 1.4e-20 |
| C7 | new QAT vs old QAT (same hardware, same gains) | +1.08 (+0.87, +1.31), p 9.1e-23 | +1.48 (+0.92, +2.03), p 1.1e-07 | +1.42 (+1.19, +1.65), p 1.6e-34 | +1.88 (+1.30, +2.47), p 8.3e-11 |
| C7b | new QAT: last vs VAL-selected checkpoint | +0.02 (-0.05, +0.09), p 0.63 | -0.06 (-0.25, +0.14), p 0.62 | +0.00 (+0.00, +0.00), p 1 | +0.00 (+0.00, +0.00), p 1 |
| C10 | decision D2: fork (10 epochs, no early stop) vs early-stopped pilot, both VAL-selected | — | — | +0.24 (+0.07, +0.40), p 0.0051 | +0.27 (-0.15, +0.66), p 0.21 |

C9 not applicable (CE pilot not run). C10 declared 29 Oct before any TEST evaluation of a trained model; on VAL it favours the fork (VAL selected both checkpoints and the D2 winner). The reports are the authoritative record; this table is copied from their JSON.

### 8.4 Extension D3: ImageNetV2 threshold-0.7 and top-images — EXECUTED 29–30 Oct (package v2.8); audit 1 Nov 212 PASS / 6 WARN / 0 FAIL

Requested by the professor on 29 Oct 2026, after the freeze (`FINAL_SHA256SUMS_20261029.txt`, 673 files)
and after the 10k TEST reports were seen. Reason given: the W8A8 new-QAT vs FP32 difference
(+0.01 pp on the 10k TEST, −0.09 pp on VAL) is within noise; a larger test set narrows the intervals.
Declared before any image of the two new variants was read (QAT_DESIGN.md §0.4 D3):
- each variant evaluated on its own in a new folder, with the frozen models, gains and simulator, and the
  same row set as the 10k TEST; nothing frozen re-run or changed;
- exclusion rule: images equal in content to a CAL, CALCHECK, VAL or TRAIN image are removed;
- pooled report: each distinct image once (first occurrence in the order matched-frequency, threshold-0.7,
  top-images); duplicate outputs must be identical;
- the matched-frequency 10k stays the primary pre-declared result; the extension is descriptive; P1
  (new QAT vs FP32, pooled and per variant) is post hoc.

| step | status (reported by user; records in `ext_results_20261101.zip`, SHA-256 `60b967ac…`) |
|---|---|
| install | v2.8 zip `da380f05…` in `code/CIM-SOC-QAT-v2.8`; old freeze verified intact before any extension step |
| archives (no published SHA-256; our record) | threshold-0.7 `32142bfe998f9cfa4141be691f6a80eeb1e27b24bd532a668eb08c26d98897a2`; top-images `85e4ace30f3f0d660152062254b8759dda66267b27f7ec0fd8d5768f1073a04b` |
| registration (29 Oct 18:07 / 18:08 UTC) | each 10,000 files = 10,000 evaluated, **0 excluded** (content overlap with CAL / CALCHECK / VAL / TRAIN = 0); 1000 classes × 10; label = int(folder) and wnid check ok; no within-variant duplicates; no shared file paths. Manifests `baf8d5d4…` (th-0.7), `da14691b…` (top) |
| overlap between variants (content) | threshold-0.7 ∩ matched-frequency 3,186; top-images ∩ matched-frequency 3,070; top-images ∩ threshold-0.7 4,996. Pooled: 10,000 + 6,814 + 3,869 = **20,683 distinct images** (9,317 duplicates; 0 label conflicts; 0 duplicates with different outputs, for every row) |
| v2-integrity (29 Oct 18:20 UTC) | PASS for both: modal pretrained prediction == int(folder) for 960/1000 classes (lexicographic alternative 12 / 14) |
| pre-extension audit | `AUDIT_20261029T182450+0000` |
| evaluation (`tools_qat/run_ext_d3.sh`, nohup, 16 workers) | 29 Oct 18:30:09 → 23:51:51 UTC (5 h 22 min), 0 errors; 17 rows per variant (the 10k row set + the W4A4 pilot-best row); 0.94–1.01 s/img on analog rows |
| reports | `REPORT_test_th07_*`, `REPORT_test_top_*`, `REPORT_test_pooled_*` (30 Oct 17:15–17:18 UTC) |
| final audit (1 Nov 15:17:55 UTC) | **212 PASS, 6 WARN, 0 FAIL**: X0 all 673 frozen files unchanged; X1–X3 counts, labels, no leakage; X5 19 reads per variant, allowed purposes only, none in the frozen log; X6 44 checks (every row `--final`, full set, frozen checkpoint). WARNs: the 3 old S4.info notices, X4 overlap (both variants), X7 evaluated after the freeze |
| new freeze | `FINAL_SHA256SUMS_20261030_ext.txt`, 316 files (written after the final audit; contains it). The old freeze file is untouched |

Results (from the report JSONs):

Top-1 % per variant (each variant complete, 10,000 images) and pooled (20,683 distinct images). Matched-frequency = the frozen primary result (unchanged; copied from REPORT_test_*).

| row | W8A8 mf | W8A8 th-0.7 | W8A8 top | **W8A8 pooled** | W4A4 mf | W4A4 th-0.7 | W4A4 top | **W4A4 pooled** |
|---|---|---|---|---|---|---|---|---|
| E1 FP32 pretrained | 69.69 | 78.40 | 82.58 | **75.49** | 69.69 | 78.40 | 82.58 | **75.49** |
| E2 ADC-off | 69.69 | 78.29 | 82.60 | **75.49** | 66.95 | 75.19 | 79.98 | **72.67** |
| E3 PTQ, G = 1 | 65.71 | 73.89 | 78.65 | **71.47** | 60.38 | 68.46 | 74.40 | **66.38** |
| E4 PTQ, calibrated gains | 69.59 | 78.18 | 82.47 | **75.38** | 66.70 | 75.55 | 80.06 | **72.55** |
| E4b PTQ, historical gains | 69.75 | 78.23 | 82.59 | **75.51** | — | — | — | **—** |
| E5 old QAT, new hardware | 68.22 | 76.10 | 80.69 | **73.72** | 67.38 | 75.50 | 80.42 | **73.07** |
| E5b old QAT, legacy ADC | 68.84 | 76.67 | 81.25 | **74.23** | 67.57 | 76.07 | 80.83 | **73.45** |
| E6 new QAT best (primary) | 69.70 | 78.11 | 82.37 | **75.35** | 69.26 | 77.56 | 82.05 | **74.90** |
| E6s new QAT final | 69.64 | 78.21 | 82.60 | **75.42** | 69.26 | 77.56 | 82.05 | **74.90** |
| D2 diagnostic: W4A4 pilot best | — | — | — | **—** | 68.99 | 77.43 | 81.96 | **74.75** |

Paired comparisons, Δtop-1 pp (95% CI), b − a on the same images:

| id | what | W8A8 mf (primary) | W8A8 pooled | W4A4 mf (primary) | W4A4 pooled |
|---|---|---|---|---|---|
| C1 | operand quantization | +0.00 (-0.13, +0.13) | +0.00 (-0.09, +0.09) | -2.74 (-3.26, -2.22) | -2.82 (-3.17, -2.47) |
| C2 | ADC at unity gain | -3.98 (-4.57, -3.35) | -4.02 (-4.41, -3.62) | -6.57 (-7.26, -5.81) | -6.29 (-6.78, -5.83) |
| C3 | ADC at calibrated gains | -0.10 (-0.30, +0.10) | -0.11 (-0.24, +0.02) | -0.25 (-0.65, +0.15) | -0.13 (-0.38, +0.15) |
| C4 | gain calibration | +3.88 (+3.28, +4.47) | +3.91 (+3.51, +4.31) | +6.32 (+5.61, +7.02) | +6.16 (+5.66, +6.64) |
| C4b | calibrated vs historical gains | -0.16 (-0.36, +0.03) | -0.13 (-0.26, +0.00) | — | — |
| C5 | old QAT vs PTQ | -1.37 (-1.90, -0.82) | -1.66 (-2.01, -1.29) | +0.68 (+0.02, +1.32) | +0.53 (+0.11, +0.97) |
| C8 | old QAT: legacy ADC → this hardware | -0.62 (-1.11, -0.13) | -0.51 (-0.86, -0.20) | -0.19 (-0.71, +0.33) | -0.37 (-0.70, -0.04) |
| C6 | new QAT vs PTQ | +0.11 (-0.09, +0.33) | -0.03 (-0.17, +0.10) | +2.56 (+1.99, +3.09) | +2.35 (+2.01, +2.73) |
| C7 | new QAT vs old QAT | +1.48 (+0.92, +2.03) | +1.63 (+1.26, +1.98) | +1.88 (+1.30, +2.47) | +1.82 (+1.44, +2.20) |
| C7b | new QAT final vs best | -0.06 (-0.25, +0.14) | +0.08 (-0.05, +0.20) | +0.00 (+0.00, +0.00) | +0.00 (+0.00, +0.00) |
| C10 | D2: fork vs pilot | — | — | +0.27 (-0.15, +0.66) | +0.15 (-0.14, +0.44) |

**P1 (POST HOC, chosen after the 10k results were seen): new QAT on the CIM simulator vs FP32 pretrained**, Δtop-1 pp (95% CI), exact McNemar p:

| images | W8A8 | W4A4 |
|---|---|---|
| **pooled (20,683)** | **-0.15 (-0.28, -0.00), p 0.049** | **-0.59 (-0.85, -0.34), p 1.6e-06** |
| matched-frequency (10,000) | +0.01 (-0.20, +0.23), p 1 | -0.43 (-0.77, -0.07), p 0.021 |
| threshold-0.7 (10,000) | -0.29 (-0.48, -0.10), p 0.0051 | -0.84 (-1.18, -0.47), p 3.8e-06 |
| top-images (10,000) | -0.21 (-0.40, -0.03), p 0.033 | -0.53 (-0.84, -0.22), p 0.0013 |

Reading (plain): the pooled set has about twice the images of the 10k, so intervals are about 1.4× narrower.
- **W8A8:** every pre-declared finding of the 10k holds on the pooled set (C1 0.00, C3 −0.11 n.s., C4 +3.91,
  C6 −0.03 n.s., C7 +1.63). P1 (post hoc): the new W8A8 QAT on the simulator is 0.15 pp below FP32
  (95% CI −0.28 to −0.005; p = 0.049) — borderline, post hoc, and one of four P1 lines, so it is not a robust
  finding; it bounds the gap to at most about 0.3 pp. The +0.01 on the matched-frequency 10k was noise, as expected.
  QAT does not change the W8A8 gap to FP32 (C6 −0.03); the small residual is at the calibrated-ADC step (C3).
- **W4A4:** C6 +2.35 (+2.01, +2.73) and C7 +1.82 confirm the 10k. P1: new W4A4 QAT stays 0.59 pp below FP32
  (−0.85, −0.34) — a real, small residual gap. C10 (D2) +0.15 (−0.14, +0.44): not resolved.

### 8.5 Post hoc: own-FP32 decomposition (stored logits only) — EXECUTED 2 Nov 2026 by user

Chosen after the TEST results were seen: **post hoc, descriptive**. The data are not new: every simulator
row of the pre-declared evaluations already stored the same checkpoint's plain-float output
(`logits_own_fp32`, part of the evaluation job from the start); only the analysis is new.
- Script `tools_qat/own_fp32_posthoc.py` (SHA-256 `c94fdc158ca2a55e…`), read-only: stored `result.npz` files
  and the split hash files; no image, manifest or simulator run; writes only `eval/posthoc_own_fp32/`
  (refuses frozen files). Records: `posthoc_own_fp32_20261102.zip` (SHA-256 `787f9e78…`):
  `own_fp32_posthoc.md` `a5199aee…`, `.json` `1be6c097…`, `check_output.txt` `35a57534…`.
- `check`: PASS for all three test sets and both precisions: every needed row holds `logits_own_fp32`
  (10,000 × 1000) on the same images, input tensors and labels as `pretrained__fp32`; the pretrained row's
  own-FP32 equals `pretrained__fp32` exactly; the old QAT's own-FP32 is identical in its analog and
  legacy-ADC rows (determinism).
- Independent cross-checks against the existing report JSONs: A (own-FP32 top-1) equals the
  `own_fp32_top1_pct` column of all 12 per-variant rows; C equals the reports' top-1; Case 1 equals C6 (new)
  and C5 (old) including the CIs; Case 5 equals P1 including the CIs, pooled and per variant; pooled top-1
  of P, Q and C equals `REPORT_test_pooled_*`. 0 mismatches.
- Case definitions (the same as CASE1–CASE5 of the Imagenette experiment, CLAUDE.md §8): P = pretrained
  FP32, Q = pretrained on the CIM simulator (calibrated gains), A = the QAT checkpoint in plain float,
  C = the same checkpoint on the simulator. Case 4 P→Q, Case 1 Q→C, Case 2 P→A, Case 3 A→C, Case 5 P→C.
  Case 2 + Case 3 = Case 5 holds exactly for the point estimates (asserted in the script); the CIs do not add.
- Caveats: about 80 intervals in the file and no correction for multiple comparisons; the three test sets
  overlap, so they are not independent replications; one training seed.

Pooled (20,683 distinct images), top-1 % and paired differences in points (95% CI):

| precision, model | P: FP32 | Q: pretrained-CIM | A: QAT in float (own FP32) | C: QAT-CIM | Case 4 P→Q | Case 1 Q→C | Case 2 P→A | Case 3 A→C | Case 5 P→C |
|---|---|---|---|---|---|---|---|---|---|
| W8A8, new QAT | 75.49 | 75.38 | 75.39 | 75.35 | -0.11 (-0.24, +0.02) | -0.03 (-0.17, +0.10) | -0.10 (-0.21, +0.01) | -0.04 (-0.18, +0.10) | -0.15 (-0.28, -0.00) |
| W8A8, old QAT | 75.49 | 75.38 | 73.80 | 73.72 | -0.11 (-0.24, +0.02) | -1.66 (-2.01, -1.29) | -1.69 (-2.03, -1.33) | -0.08 (-0.22, +0.05) | -1.77 (-2.12, -1.40) |
| W4A4, new QAT | 75.49 | 72.55 | 74.45 | 74.90 | -2.94 (-3.30, -2.61) | +2.35 (+2.01, +2.73) | -1.04 (-1.34, -0.74) | +0.45 (+0.13, +0.77) | -0.59 (-0.85, -0.34) |
| W4A4, old QAT | 75.49 | 72.55 | 73.39 | 73.07 | -2.94 (-3.30, -2.61) | +0.53 (+0.11, +0.97) | -2.10 (-2.46, -1.73) | -0.32 (-0.63, +0.00) | -2.42 (-2.78, -2.05) |

Case 3 (A→C) for the new W4A4 QAT per test set: matched-frequency +0.41 (-0.06, +0.91); threshold-0.7 +0.58 (+0.13, +1.05); top-images +0.32 (-0.13, +0.75); pooled +0.45 (+0.13, +0.77). Agreement with the original FP32 model (pooled): float 91.4 %, CIM 94.2 %.

Reading:
- **W8A8:** the new QAT barely moves the model in float (Case 2 −0.10), and the hardware costs it nothing
  measurable (Case 3 −0.04). The old W8A8 QAT lost 1.7 points in its weights themselves (Case 2), not on
  the hardware (Case 3 −0.08).
- **W4A4, new QAT:** the trained weights in float are 1.04 points below the original (Case 2), and the same
  weights on the CIM simulator score 0.45 points **higher** than in float (Case 3; agreement with the
  original 94.2 % vs 91.4 %). This is what the training set-up asks for: the loss was computed on the
  hardware forward (the bit-exact surrogate: 4-bit numbers and the 8-bit ADC) with the original FP32 model
  as the KD teacher, never on the float forward. The float version of the QAT weights is therefore a
  reference point, not an upper limit, and the result does not mean the hardware adds accuracy: the CIM
  model is still 0.59 points below the original (Case 5). The same pattern was already visible on VAL
  during training (fork epoch 10: surrogate 80.33 % vs float 79.77 %; simulator VAL report 80.40 % vs
  79.77 %). Which part of the hardware path the weights adapted to (4-bit rounding or the ADC) was not
  measured; an ADC-off run of the QAT weights would split Case 3.

### 8.6 Audit re-run with v2.8.1, and the fourth freeze — EXECUTED 17 Nov 2026 by user

Runbook §13.8 (prepared 2 Nov, run 17 Nov). Nothing was re-evaluated; only the audit ran again.
- Package `CIM-SOC-QAT-v2.8.1.zip` SHA-256 `e21c13da…` matched; `PACKAGE_SHA256SUMS.txt` passed. Installed in
  `code/CIM-SOC-QAT-v2.8.1`; the v2.8 folder (listed in the third freeze) untouched. The installed
  `qat_audit.py` is `cce3a346…`, the same as the delivered v2.8.1 file.
- `audit/AUDIT_20261102T105922+0000.NOTE.md` written beside the failed audit, word for word the runbook text;
  the failed audit itself is unchanged.
- **`AUDIT_20261117T170653+0000`: 214 PASS, 6 WARN, 0 FAIL.**
  - X0 PASS for all three earlier freezes: 673, 316 and 6 files unchanged.
  - `X5.test_th07.log` and `X5.test_top.log` PASS ("0 entries dated after the freeze; freezes broken: []"):
    the two FAILs of 2 Nov were false, as diagnosed.
  - Compared check by check with the last clean audit (`AUDIT_20261101T151755+0000`, v2.8, 212/6/0): same
    checks in the same order and the same 6 WARNs (S4.info ×3, X4 overlap ×2, X7.after_freeze). Differences,
    all expected:
    - two new X0 lines, for the freezes written after 1 Nov;
    - the reworded X5.*.log messages (v2.8.1);
    - `L4.log` 163 → 165 entries: one `all`/`audit` line from each of the 2 Nov and 17 Nov audits, both in
      `access_log_ext2.jsonl`.
    TEST and each extension variant still show 19 reads, the same as on 1 Nov: since then the only new
    log lines are the audits' own `all`/`audit` lines.
- Environment: the audit ran in the conda `base` environment (Python 3.12.2), not `cim_soc_qat`; reported by
  the user. It cannot change a result: the checks are file hashes, comparisons of stored JSON, CSV and
  log files, and exact `numpy` array equality on stored outputs. `qat_audit.py` imports modules that load
  torch only for constants and hash helpers; it runs no model and no simulator. The report does not record
  the interpreter.
- **Fourth freeze `FINAL_SHA256SUMS_20261102_audit.txt`** (7 files): both audits of 2 and 17 Nov (`.json`, `.md`),
  the NOTE, `access_log_ext2.jsonl` and `code/CIM-SOC-QAT-v2.8.1/qat_audit.py`. The file name carries the
  date from the runbook (2 Nov); it was written on 17 Nov. Checked in the development sandbox: the new audit
  (`.json`, `.md`) and the NOTE match their lines in the freeze; the 2 Nov audit files and
  `access_log_ext2.jsonl` were not sent, so their lines were not checked there. Any later audit writes to
  `access_log_ext3.jsonl`, by design.
