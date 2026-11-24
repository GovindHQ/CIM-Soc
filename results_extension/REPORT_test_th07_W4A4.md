# W4A4 — ImageNetV2 threshold-0.7 (extension) (n = 10000)

Status: EXECUTED (2026-10-30T17:15:26+00:00). All CIM rows from the actual cim_sim simulator (batch 1, CPU). KL and agreement are vs the pretrained FP32 model on the same images.
Identical inputs verified: manifest `baf8d5d4feab0a63`, per-image input-tensor SHA-256 digest `2a9c34c92754061c`, preprocessing `cim_exp.imagenette.load_uint8+to_model_input|interp=bicubic|resize_short=256|crop=224|mean=[0.485, 0.456, 0.406]|std=[0.229, 0.224, 0.225]`.

> ImageNetV2 threshold-0.7 (EXTENSION requested after the results were frozen (29 Oct 2026): descriptive only. The pre-declared primary TEST result is ImageNetV2 matched-frequency (10,000 images).) Compare models WITHIN this table.

| run | top-1 % (±SE) | top-5 % | NLL | ECE % | KL(FP32‖run) | agree FP32 % | own-FP32 top-1 % | KL(own FP32‖run) |
|---|---|---|---|---|---|---|---|---|
| old_int4__analog_W4A4__cal | 75.50 (±0.43) | 93.81 | 0.9204 | 4.05 | 0.4638 | 87.94 | 75.86 | 0.0928 |
| old_int4__legacyadc_W4A4 | 76.07 (±0.43) | 94.12 | 0.8946 | 3.53 | 0.4823 | 87.75 | 75.86 | 0.1519 |
| pretrained__adcoff_W4A4 | 75.19 (±0.43) | 92.99 | 1.1545 | 14.02 | 0.1534 | 88.44 | 78.40 | 0.1534 |
| pretrained__analog_W4A4__cal | 75.55 (±0.43) | 93.05 | 1.1581 | 14.67 | 0.1560 | 88.35 | 78.40 | 0.1560 |
| pretrained__analog_W4A4__unity | 68.46 (±0.46) | 88.66 | 1.5782 | 18.54 | 0.5087 | 77.16 | 78.40 | 0.5087 |
| pretrained__fp32 | 78.40 (±0.41) | 94.97 | 0.9280 | 10.12 | 0.0000 | 100.00 | — | — |
| w4a4_kd_pilot10ep_best__analog_W4A4__cal | 77.43 (±0.42) | 94.62 | 0.9484 | 9.77 | 0.0310 | 94.04 | 76.72 | 0.0880 |
| w4a4_kd_pilot10ep_full_best__analog_W4A4__cal | 77.56 (±0.42) | 94.64 | 0.9460 | 9.77 | 0.0251 | 94.69 | 76.98 | 0.0745 |
| w4a4_kd_pilot10ep_full_final__analog_W4A4__cal | 77.56 (±0.42) | 94.64 | 0.9460 | 9.77 | 0.0251 | 94.69 | 76.98 | 0.0745 |

## Hardware diagnostics per operation (analog runs)

| run | gains qkv/qk/av | clip % qkv/qk/av | rail % qkv/qk/av | ADC util median % qkv/qk/av | util p99.9 % qkv/qk/av | ADC saturation % (all ops) | images with any saturation % |
|---|---|---|---|---|---|---|---|
| old_int4__analog_W4A4__cal | 29.05/11.28/3.82 | 0.0112/0.0071/0.0094 | 0.012/0.009/0.010 | 12.1/11.8/2.2 | 80/81/63 | 0.0101 | 100.0 |
| old_int4__legacyadc_W4A4 | — | — (legacy ADC: per-op stats not recorded) | — (legacy ADC: per-op stats not recorded) | — (legacy ADC: per-op stats not recorded) | — (legacy ADC: per-op stats not recorded) | 0.0001 | 1.6 |
| pretrained__analog_W4A4__cal | 29.05/11.28/3.82 | 0.0100/0.0095/0.0088 | 0.010/0.012/0.009 | 12.8/12.4/2.2 | 79/83/62 | 0.0097 | 100.0 |
| pretrained__analog_W4A4__unity | 1.00/1.00/1.00 | 0.0000/0.0000/0.0000 | 0.000/0.000/0.000 | 0.4/1.0/0.5 | 3/7/12 | 0.0000 | 0.0 |
| w4a4_kd_pilot10ep_best__analog_W4A4__cal | 29.05/11.28/3.82 | 0.0103/0.0087/0.0092 | 0.011/0.011/0.010 | 12.8/12.1/2.3 | 79/82/63 | 0.0098 | 100.0 |
| w4a4_kd_pilot10ep_full_best__analog_W4A4__cal | 29.05/11.28/3.82 | 0.0107/0.0072/0.0095 | 0.011/0.009/0.010 | 12.8/12.1/2.4 | 80/82/64 | 0.0098 | 100.0 |
| w4a4_kd_pilot10ep_full_final__analog_W4A4__cal | 29.05/11.28/3.82 | 0.0107/0.0072/0.0095 | 0.011/0.009/0.010 | 12.8/12.1/2.4 | 80/82/64 | 0.0098 | 100.0 |

ADC-utilisation percentiles: weighted uniform subsample (~2 M conversions per operation, cim_qat.analog_merge); clip, rail and saturation columns are exact counts.

## Paired comparisons (b vs a, same images; C1–C9 pre-declared, C10 declared 29 Oct before TEST)

| id | isolates | a | b | Δtop-1 pp (95% CI) | McNemar p (a-only/b-only) | ΔNLL (95% CI) | ΔKL (95% CI) |
|---|---|---|---|---|---|---|---|
| C1 | operand quantization (exact integer MAC, no ADC) | pretrained__fp32 | pretrained__adcoff_W4A4 | -3.21 (-3.75, -2.70) | 1.18e-33 (523/202) | +0.2264 (+0.2156, +0.2371) | +0.1534 (+0.1491, +0.1576) |
| C2 | ADC + analog chain at unity gain | pretrained__adcoff_W4A4 | pretrained__analog_W4A4__unity | -6.73 (-7.46, -6.06) | 3.52e-83 (973/300) | +0.4237 (+0.4073, +0.4402) | +0.3552 (+0.3457, +0.3652) |
| C3 | ADC + analog chain at calibrated gains | pretrained__adcoff_W4A4 | pretrained__analog_W4A4__cal | +0.36 (-0.01, +0.74) | 0.077 (178/214) | +0.0036 (-0.0015, +0.0088) | +0.0026 (+0.0007, +0.0044) |
| C4 | gain calibration (unity -> calibrated) | pretrained__analog_W4A4__unity | pretrained__analog_W4A4__cal | +7.09 (+6.41, +7.80) | 2.87e-91 (291/1000) | -0.4201 (-0.4366, -0.4034) | -0.3527 (-0.3628, -0.3432) |
| C5 | old QAT vs PTQ (same hardware, same gains) | pretrained__analog_W4A4__cal | old_int4__analog_W4A4__cal | -0.05 (-0.69, +0.56) | 0.899 (495/490) | -0.2377 (-0.2538, -0.2212) | +0.3078 (+0.3010, +0.3150) |
| C8 | old QAT checkpoint: its training point (legacy ADC) -> new hardware | old_int4__legacyadc_W4A4 | old_int4__analog_W4A4__cal | -0.57 (-1.08, -0.06) | 0.0293 (359/302) | +0.0258 (+0.0150, +0.0366) | -0.0186 (-0.0226, -0.0143) |
| C6 | new QAT (VAL-selected) vs PTQ | pretrained__analog_W4A4__cal | w4a4_kd_pilot10ep_full_best__analog_W4A4__cal | +2.01 (+1.49, +2.54) | 7.1e-14 (261/462) | -0.2121 (-0.2227, -0.2009) | -0.1309 (-0.1349, -0.1268) |
| C7 | new QAT vs old QAT (same hardware, same gains) | old_int4__analog_W4A4__cal | w4a4_kd_pilot10ep_full_best__analog_W4A4__cal | +2.06 (+1.53, +2.64) | 1.67e-13 (288/494) | +0.0257 (+0.0130, +0.0389) | -0.4387 (-0.4452, -0.4326) |
| C7b | new QAT: last vs VAL-selected checkpoint | w4a4_kd_pilot10ep_full_best__analog_W4A4__cal | w4a4_kd_pilot10ep_full_final__analog_W4A4__cal | +0.00 (+0.00, +0.00) | 1 (0/0) | +0.0000 (+0.0000, +0.0000) | +0.0000 (+0.0000, +0.0000) |
| C10 | decision D2: fork (10 epochs, no early stop) vs early-stopped pilot, both VAL-selected | w4a4_kd_pilot10ep_best__analog_W4A4__cal | w4a4_kd_pilot10ep_full_best__analog_W4A4__cal | +0.13 (-0.25, +0.52) | 0.557 (202/215) | -0.0024 (-0.0075, +0.0025) | -0.0058 (-0.0066, -0.0050) |