# W8A8 — ImageNetV2, all three variants pooled (extension) (n = 20,683 distinct images)

Status: EXECUTED (2026-10-30T17:17:21+00:00). Combined at report time from finished per-variant results; nothing was re-run. KL and agreement are vs the pretrained FP32 model on the same images.

> All three ImageNetV2 variants pooled, each distinct image counted once (EXTENSION requested after the results were frozen (29 Oct 2026): descriptive only. The pre-declared primary TEST result is ImageNetV2 matched-frequency (10,000 images).) The variants differ in difficulty; compare models WITHIN this table.

## Composition

| variant | images evaluated | counted in the pool | duplicates of an earlier variant |
|---|---|---|---|
| matched-frequency | 10,000 | 10,000 | 0 |
| threshold-0.7 | 10,000 | 6,814 | 3,186 |
| top-images | 10,000 | 3,869 | 6,131 |

Duplicate rule: each distinct file content (SHA-256) once; first occurrence in the order test, test_th07, test_top. 9,317 duplicate images; label conflicts 0; duplicates with different outputs (should be 0: same image, same model): 0.

## Results (pooled)

| run | top-1 % (±SE) | top-5 % | NLL | KL(FP32‖run) | agree FP32 % | top-1 % matched-frequency | top-1 % threshold-0.7 | top-1 % top-images |
|---|---|---|---|---|---|---|---|---|
| old_int8__analog_W8A8__cal | 73.72 (±0.31) | 91.85 | 1.0537 | 0.4599 | 88.27 | 68.22 | 76.10 | 80.69 |
| old_int8__legacyadc_W8A8 | 74.23 (±0.30) | 92.19 | 1.0323 | 0.4604 | 89.08 | 68.84 | 76.67 | 81.25 |
| pretrained__adcoff_W8A8 | 75.49 (±0.30) | 92.77 | 1.0707 | 0.0005 | 99.16 | 69.69 | 78.29 | 82.60 |
| pretrained__analog_W8A8__cal | 75.38 (±0.30) | 92.70 | 1.0735 | 0.0029 | 98.00 | 69.59 | 78.18 | 82.47 |
| pretrained__analog_W8A8__hist | 75.51 (±0.30) | 92.71 | 1.0719 | 0.0027 | 98.16 | 69.75 | 78.23 | 82.59 |
| pretrained__analog_W8A8__unity | 71.47 (±0.31) | 90.02 | 1.3353 | 0.2042 | 85.07 | 65.71 | 73.89 | 78.65 |
| pretrained__fp32 | 75.49 (±0.30) | 92.79 | 1.0704 | 0.0000 | 100.00 | 69.69 | 78.40 | 82.58 |
| w8a8_kd_pilot5ep_best__analog_W8A8__cal | 75.35 (±0.30) | 92.76 | 1.0732 | 0.0025 | 98.04 | 69.70 | 78.11 | 82.37 |
| w8a8_kd_pilot5ep_final__analog_W8A8__cal | 75.42 (±0.30) | 92.77 | 1.0725 | 0.0022 | 98.24 | 69.64 | 78.21 | 82.60 |

## Paired comparisons on the pooled images (b vs a; C1–C9 pre-declared)

| id | isolates | a | b | Δtop-1 pp (95% CI) | McNemar p (a-only/b-only) | ΔNLL (95% CI) |
|---|---|---|---|---|---|---|
| C1 | operand quantization (exact integer MAC, no ADC) | pretrained__fp32 | pretrained__adcoff_W8A8 | +0.00 (-0.09, +0.09) | 1 (43/43) | +0.0004 (-0.0000, +0.0008) |
| C2 | ADC + analog chain at unity gain | pretrained__adcoff_W8A8 | pretrained__analog_W8A8__unity | -4.02 (-4.41, -3.62) | 4.34e-89 (1312/480) | +0.2646 (+0.2555, +0.2738) |
| C3 | ADC + analog chain at calibrated gains | pretrained__adcoff_W8A8 | pretrained__analog_W8A8__cal | -0.11 (-0.24, +0.02) | 0.113 (108/85) | +0.0027 (+0.0018, +0.0037) |
| C4 | gain calibration (unity -> calibrated) | pretrained__analog_W8A8__unity | pretrained__analog_W8A8__cal | +3.91 (+3.51, +4.31) | 3.41e-84 (491/1300) | -0.2618 (-0.2710, -0.2528) |
| C4b | calibrated vs historical synthetic-input gains | pretrained__analog_W8A8__hist | pretrained__analog_W8A8__cal | -0.13 (-0.26, +0.00) | 0.0596 (109/82) | +0.0015 (+0.0006, +0.0025) |
| C5 | old QAT vs PTQ (same hardware, same gains) | pretrained__analog_W8A8__cal | old_int8__analog_W8A8__cal | -1.66 (-2.01, -1.29) | 8.41e-20 (888/544) | -0.0198 (-0.0287, -0.0111) |
| C8 | old QAT checkpoint: its training point (legacy ADC) -> new hardware | old_int8__legacyadc_W8A8 | old_int8__analog_W8A8__cal | -0.51 (-0.86, -0.20) | 0.00204 (633/527) | +0.0214 (+0.0146, +0.0283) |
| C6 | new QAT (VAL-selected) vs PTQ | pretrained__analog_W8A8__cal | w8a8_kd_pilot5ep_best__analog_W8A8__cal | -0.03 (-0.17, +0.10) | 0.687 (114/107) | -0.0002 (-0.0014, +0.0008) |
| C7 | new QAT vs old QAT (same hardware, same gains) | old_int8__analog_W8A8__cal | w8a8_kd_pilot5ep_best__analog_W8A8__cal | +1.63 (+1.26, +1.98) | 9.42e-19 (560/897) | +0.0195 (+0.0108, +0.0286) |
| C7b | new QAT: last vs VAL-selected checkpoint | w8a8_kd_pilot5ep_best__analog_W8A8__cal | w8a8_kd_pilot5ep_final__analog_W8A8__cal | +0.08 (-0.05, +0.20) | 0.261 (81/97) | -0.0007 (-0.0016, +0.0002) |

## P1 — POST HOC: new QAT (VAL-selected) on the CIM simulator vs the FP32 pretrained model

Chosen after the TEST results were seen: descriptive, not a pre-declared test.

| images | Δtop-1 pp, b − a (95% CI) | McNemar p (a-only/b-only) | ΔNLL (95% CI) |
|---|---|---|---|
| pooled (20,683) | -0.15 (-0.28, -0.00) | 0.0493 (124/94) | +0.0029 (+0.0019, +0.0038) |
| matched-frequency (10,000) | +0.01 (-0.20, +0.23) | 1 (59/60) | +0.0033 (+0.0017, +0.0050) |
| threshold-0.7 (10,000) | -0.29 (-0.48, -0.10) | 0.00508 (65/36) | +0.0031 (+0.0018, +0.0043) |
| top-images (10,000) | -0.21 (-0.40, -0.03) | 0.0334 (55/34) | +0.0025 (+0.0014, +0.0035) |
