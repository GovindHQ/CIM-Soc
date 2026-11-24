# W4A4 — ImageNetV2, all three variants pooled (extension) (n = 20,683 distinct images)

Status: EXECUTED (2026-10-30T17:18:03+00:00). Combined at report time from finished per-variant results; nothing was re-run. KL and agreement are vs the pretrained FP32 model on the same images.

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
| old_int4__analog_W4A4__cal | 73.07 (±0.31) | 91.61 | 1.0801 | 0.4724 | 86.85 | 67.38 | 75.50 | 80.42 |
| old_int4__legacyadc_W4A4 | 73.45 (±0.31) | 91.74 | 1.0628 | 0.4931 | 86.68 | 67.57 | 76.07 | 80.83 |
| pretrained__adcoff_W4A4 | 72.67 (±0.31) | 90.72 | 1.2961 | 0.1576 | 87.61 | 66.95 | 75.19 | 79.98 |
| pretrained__analog_W4A4__cal | 72.55 (±0.31) | 90.70 | 1.3004 | 0.1615 | 87.35 | 66.70 | 75.55 | 80.06 |
| pretrained__analog_W4A4__unity | 66.38 (±0.33) | 86.38 | 1.7057 | 0.5145 | 76.00 | 60.38 | 68.46 | 74.40 |
| pretrained__fp32 | 75.49 (±0.30) | 92.79 | 1.0704 | 0.0000 | 100.00 | 69.69 | 78.40 | 82.58 |
| w4a4_kd_pilot10ep_best__analog_W4A4__cal | 74.75 (±0.30) | 92.59 | 1.0935 | 0.0331 | 93.56 | 68.99 | 77.43 | 81.96 |
| w4a4_kd_pilot10ep_full_best__analog_W4A4__cal | 74.90 (±0.30) | 92.41 | 1.0884 | 0.0266 | 94.20 | 69.26 | 77.56 | 82.05 |
| w4a4_kd_pilot10ep_full_final__analog_W4A4__cal | 74.90 (±0.30) | 92.41 | 1.0884 | 0.0266 | 94.20 | 69.26 | 77.56 | 82.05 |

## Paired comparisons on the pooled images (b vs a; C1–C9 pre-declared, C10 declared 29 Oct before TEST)

| id | isolates | a | b | Δtop-1 pp (95% CI) | McNemar p (a-only/b-only) | ΔNLL (95% CI) |
|---|---|---|---|---|---|---|
| C1 | operand quantization (exact integer MAC, no ADC) | pretrained__fp32 | pretrained__adcoff_W4A4 | -2.82 (-3.17, -2.47) | 4.43e-54 (1018/435) | +0.2257 (+0.2185, +0.2331) |
| C2 | ADC + analog chain at unity gain | pretrained__adcoff_W4A4 | pretrained__analog_W4A4__unity | -6.29 (-6.78, -5.83) | 2.22e-149 (1955/654) | +0.4096 (+0.3982, +0.4215) |
| C3 | ADC + analog chain at calibrated gains | pretrained__adcoff_W4A4 | pretrained__analog_W4A4__cal | -0.13 (-0.38, +0.15) | 0.376 (411/385) | +0.0043 (+0.0007, +0.0079) |
| C4 | gain calibration (unity -> calibrated) | pretrained__analog_W4A4__unity | pretrained__analog_W4A4__cal | +6.16 (+5.66, +6.64) | 1.46e-142 (673/1948) | -0.4053 (-0.4171, -0.3934) |
| C5 | old QAT vs PTQ (same hardware, same gains) | pretrained__analog_W4A4__cal | old_int4__analog_W4A4__cal | +0.53 (+0.11, +0.97) | 0.0172 (973/1082) | -0.2203 (-0.2319, -0.2084) |
| C8 | old QAT checkpoint: its training point (legacy ADC) -> new hardware | old_int4__legacyadc_W4A4 | old_int4__analog_W4A4__cal | -0.37 (-0.70, -0.04) | 0.0358 (694/617) | +0.0173 (+0.0095, +0.0248) |
| C6 | new QAT (VAL-selected) vs PTQ | pretrained__analog_W4A4__cal | w4a4_kd_pilot10ep_full_best__analog_W4A4__cal | +2.35 (+2.01, +2.73) | 2.1e-36 (511/997) | -0.2120 (-0.2193, -0.2046) |
| C7 | new QAT vs old QAT (same hardware, same gains) | old_int4__analog_W4A4__cal | w4a4_kd_pilot10ep_full_best__analog_W4A4__cal | +1.82 (+1.44, +2.20) | 4.4e-21 (615/992) | +0.0083 (-0.0012, +0.0180) |
| C7b | new QAT: last vs VAL-selected checkpoint | w4a4_kd_pilot10ep_full_best__analog_W4A4__cal | w4a4_kd_pilot10ep_full_final__analog_W4A4__cal | +0.00 (+0.00, +0.00) | 1 (0/0) | +0.0000 (+0.0000, +0.0000) |
| C10 | decision D2: fork vs early-stopped pilot, both VAL-selected | w4a4_kd_pilot10ep_best__analog_W4A4__cal | w4a4_kd_pilot10ep_full_best__analog_W4A4__cal | +0.15 (-0.14, +0.44) | 0.3 (404/435) | -0.0051 (-0.0089, -0.0012) |

## P1 — POST HOC: new QAT (VAL-selected) on the CIM simulator vs the FP32 pretrained model

Chosen after the TEST results were seen: descriptive, not a pre-declared test.

| images | Δtop-1 pp, b − a (95% CI) | McNemar p (a-only/b-only) | ΔNLL (95% CI) |
|---|---|---|---|
| pooled (20,683) | -0.59 (-0.85, -0.34) | 1.63e-06 (387/264) | +0.0181 (+0.0149, +0.0212) |
| matched-frequency (10,000) | -0.43 (-0.77, -0.07) | 0.0212 (188/145) | +0.0207 (+0.0154, +0.0257) |
| threshold-0.7 (10,000) | -0.84 (-1.18, -0.47) | 3.82e-06 (205/121) | +0.0180 (+0.0137, +0.0221) |
| top-images (10,000) | -0.53 (-0.84, -0.22) | 0.0013 (158/105) | +0.0142 (+0.0104, +0.0176) |
