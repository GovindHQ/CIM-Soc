# W4A4 — ImageNetV2 top-images (extension) (n = 10000)

Status: EXECUTED (2026-10-30T17:16:07+00:00). All CIM rows from the actual cim_sim simulator (batch 1, CPU). KL and agreement are vs the pretrained FP32 model on the same images.
Identical inputs verified: manifest `da14691bdf4a3f6b`, per-image input-tensor SHA-256 digest `b7355b0e9dba1814`, preprocessing `cim_exp.imagenette.load_uint8+to_model_input|interp=bicubic|resize_short=256|crop=224|mean=[0.485, 0.456, 0.406]|std=[0.229, 0.224, 0.225]`.

> ImageNetV2 top-images (EXTENSION requested after the results were frozen (29 Oct 2026): descriptive only. The pre-declared primary TEST result is ImageNetV2 matched-frequency (10,000 images).) Compare models WITHIN this table.

| run | top-1 % (±SE) | top-5 % | NLL | ECE % | KL(FP32‖run) | agree FP32 % | own-FP32 top-1 % | KL(own FP32‖run) |
|---|---|---|---|---|---|---|---|---|
| old_int4__analog_W4A4__cal | 80.42 (±0.40) | 96.26 | 0.6927 | 2.15 | 0.4379 | 90.47 | 80.45 | 0.0737 |
| old_int4__legacyadc_W4A4 | 80.83 (±0.39) | 96.18 | 0.6793 | 2.24 | 0.4586 | 90.32 | 80.45 | 0.1223 |
| pretrained__adcoff_W4A4 | 79.98 (±0.40) | 95.50 | 0.9357 | 15.15 | 0.1320 | 91.11 | 82.58 | 0.1320 |
| pretrained__analog_W4A4__cal | 80.06 (±0.40) | 95.61 | 0.9400 | 15.60 | 0.1351 | 90.89 | 82.58 | 0.1351 |
| pretrained__analog_W4A4__unity | 74.40 (±0.44) | 91.85 | 1.3141 | 20.38 | 0.4458 | 81.34 | 82.58 | 0.4458 |
| pretrained__fp32 | 82.58 (±0.38) | 96.93 | 0.7379 | 11.00 | 0.0000 | 100.00 | — | — |
| w4a4_kd_pilot10ep_best__analog_W4A4__cal | 81.96 (±0.38) | 96.87 | 0.7518 | 10.91 | 0.0254 | 95.26 | 81.44 | 0.0740 |
| w4a4_kd_pilot10ep_full_best__analog_W4A4__cal | 82.05 (±0.38) | 96.75 | 0.7521 | 10.86 | 0.0203 | 95.97 | 81.73 | 0.0616 |
| w4a4_kd_pilot10ep_full_final__analog_W4A4__cal | 82.05 (±0.38) | 96.75 | 0.7521 | 10.86 | 0.0203 | 95.97 | 81.73 | 0.0616 |

## Hardware diagnostics per operation (analog runs)

| run | gains qkv/qk/av | clip % qkv/qk/av | rail % qkv/qk/av | ADC util median % qkv/qk/av | util p99.9 % qkv/qk/av | ADC saturation % (all ops) | images with any saturation % |
|---|---|---|---|---|---|---|---|
| old_int4__analog_W4A4__cal | 29.05/11.28/3.82 | 0.0112/0.0071/0.0093 | 0.012/0.009/0.010 | 12.1/11.8/2.2 | 80/81/62 | 0.0101 | 100.0 |
| old_int4__legacyadc_W4A4 | — | — (legacy ADC: per-op stats not recorded) | — (legacy ADC: per-op stats not recorded) | — (legacy ADC: per-op stats not recorded) | — (legacy ADC: per-op stats not recorded) | 0.0001 | 1.5 |
| pretrained__analog_W4A4__cal | 29.05/11.28/3.82 | 0.0100/0.0095/0.0087 | 0.010/0.012/0.009 | 12.8/12.1/2.2 | 79/83/62 | 0.0097 | 100.0 |
| pretrained__analog_W4A4__unity | 1.00/1.00/1.00 | 0.0000/0.0000/0.0000 | 0.000/0.000/0.000 | 0.4/1.0/0.5 | 3/7/12 | 0.0000 | 0.0 |
| w4a4_kd_pilot10ep_best__analog_W4A4__cal | 29.05/11.28/3.82 | 0.0103/0.0086/0.0091 | 0.011/0.011/0.010 | 12.8/12.1/2.3 | 79/82/63 | 0.0098 | 100.0 |
| w4a4_kd_pilot10ep_full_best__analog_W4A4__cal | 29.05/11.28/3.82 | 0.0107/0.0072/0.0095 | 0.011/0.009/0.010 | 12.8/12.1/2.4 | 80/82/64 | 0.0098 | 100.0 |
| w4a4_kd_pilot10ep_full_final__analog_W4A4__cal | 29.05/11.28/3.82 | 0.0107/0.0072/0.0095 | 0.011/0.009/0.010 | 12.8/12.1/2.4 | 80/82/64 | 0.0098 | 100.0 |

ADC-utilisation percentiles: weighted uniform subsample (~2 M conversions per operation, cim_qat.analog_merge); clip, rail and saturation columns are exact counts.

## Paired comparisons (b vs a, same images; C1–C9 pre-declared, C10 declared 29 Oct before TEST)

| id | isolates | a | b | Δtop-1 pp (95% CI) | McNemar p (a-only/b-only) | ΔNLL (95% CI) | ΔKL (95% CI) |
|---|---|---|---|---|---|---|---|
| C1 | operand quantization (exact integer MAC, no ADC) | pretrained__fp32 | pretrained__adcoff_W4A4 | -2.60 (-3.10, -2.12) | 2.04e-26 (436/176) | +0.1978 (+0.1881, +0.2075) | +0.1320 (+0.1281, +0.1361) |
| C2 | ADC + analog chain at unity gain | pretrained__adcoff_W4A4 | pretrained__analog_W4A4__unity | -5.58 (-6.24, -4.94) | 4.15e-64 (846/288) | +0.3784 (+0.3618, +0.3934) | +0.3139 (+0.3044, +0.3231) |
| C3 | ADC + analog chain at calibrated gains | pretrained__adcoff_W4A4 | pretrained__analog_W4A4__cal | +0.08 (-0.29, +0.44) | 0.711 (174/182) | +0.0043 (-0.0004, +0.0088) | +0.0031 (+0.0013, +0.0050) |
| C4 | gain calibration (unity -> calibrated) | pretrained__analog_W4A4__unity | pretrained__analog_W4A4__cal | +5.66 (+5.01, +6.32) | 7.12e-66 (285/851) | -0.3741 (-0.3889, -0.3580) | -0.3107 (-0.3199, -0.3014) |
| C5 | old QAT vs PTQ (same hardware, same gains) | pretrained__analog_W4A4__cal | old_int4__analog_W4A4__cal | +0.36 (-0.19, +0.92) | 0.233 (412/448) | -0.2472 (-0.2613, -0.2333) | +0.3028 (+0.2963, +0.3098) |
| C8 | old QAT checkpoint: its training point (legacy ADC) -> new hardware | old_int4__legacyadc_W4A4 | old_int4__analog_W4A4__cal | -0.41 (-0.88, +0.05) | 0.09 (299/258) | +0.0134 (+0.0039, +0.0222) | -0.0208 (-0.0244, -0.0171) |
| C6 | new QAT (VAL-selected) vs PTQ | pretrained__analog_W4A4__cal | w4a4_kd_pilot10ep_full_best__analog_W4A4__cal | +1.99 (+1.53, +2.51) | 2.86e-15 (220/419) | -0.1879 (-0.1976, -0.1783) | -0.1148 (-0.1187, -0.1111) |
| C7 | new QAT vs old QAT (same hardware, same gains) | old_int4__analog_W4A4__cal | w4a4_kd_pilot10ep_full_best__analog_W4A4__cal | +1.63 (+1.13, +2.14) | 2.26e-10 (248/411) | +0.0594 (+0.0486, +0.0709) | -0.4176 (-0.4241, -0.4120) |
| C7b | new QAT: last vs VAL-selected checkpoint | w4a4_kd_pilot10ep_full_best__analog_W4A4__cal | w4a4_kd_pilot10ep_full_final__analog_W4A4__cal | +0.00 (+0.00, +0.00) | 1 (0/0) | +0.0000 (+0.0000, +0.0000) | +0.0000 (+0.0000, +0.0000) |
| C10 | decision D2: fork (10 epochs, no early stop) vs early-stopped pilot, both VAL-selected | w4a4_kd_pilot10ep_best__analog_W4A4__cal | w4a4_kd_pilot10ep_full_best__analog_W4A4__cal | +0.09 (-0.26, +0.46) | 0.66 (161/170) | +0.0003 (-0.0043, +0.0047) | -0.0051 (-0.0058, -0.0044) |