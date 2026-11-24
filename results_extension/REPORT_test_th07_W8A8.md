# W8A8 — ImageNetV2 threshold-0.7 (extension) (n = 10000)

Status: EXECUTED (2026-10-30T17:15:05+00:00). All CIM rows from the actual cim_sim simulator (batch 1, CPU). KL and agreement are vs the pretrained FP32 model on the same images.
Identical inputs verified: manifest `baf8d5d4feab0a63`, per-image input-tensor SHA-256 digest `2a9c34c92754061c`, preprocessing `cim_exp.imagenette.load_uint8+to_model_input|interp=bicubic|resize_short=256|crop=224|mean=[0.485, 0.456, 0.406]|std=[0.229, 0.224, 0.225]`.

> ImageNetV2 threshold-0.7 (EXTENSION requested after the results were frozen (29 Oct 2026): descriptive only. The pre-declared primary TEST result is ImageNetV2 matched-frequency (10,000 images).) Compare models WITHIN this table.

| run | top-1 % (±SE) | top-5 % | NLL | ECE % | KL(FP32‖run) | agree FP32 % | own-FP32 top-1 % | KL(own FP32‖run) |
|---|---|---|---|---|---|---|---|---|
| old_int8__analog_W8A8__cal | 76.10 (±0.43) | 94.06 | 0.8899 | 4.04 | 0.4540 | 89.43 | 76.23 | 0.0025 |
| old_int8__legacyadc_W8A8 | 76.67 (±0.42) | 94.38 | 0.8694 | 3.91 | 0.4530 | 89.97 | 76.23 | 0.0887 |
| pretrained__adcoff_W8A8 | 78.29 (±0.41) | 94.97 | 0.9284 | 10.02 | 0.0005 | 99.31 | 78.40 | 0.0005 |
| pretrained__analog_W8A8__cal | 78.18 (±0.41) | 94.95 | 0.9315 | 9.90 | 0.0027 | 98.29 | 78.40 | 0.0027 |
| pretrained__analog_W8A8__hist | 78.23 (±0.41) | 94.88 | 0.9305 | 9.90 | 0.0025 | 98.45 | 78.40 | 0.0025 |
| pretrained__analog_W8A8__unity | 73.89 (±0.44) | 92.35 | 1.1933 | 13.47 | 0.1970 | 86.02 | 78.40 | 0.1970 |
| pretrained__fp32 | 78.40 (±0.41) | 94.97 | 0.9280 | 10.12 | 0.0000 | 100.00 | — | — |
| w8a8_kd_pilot5ep_best__analog_W8A8__cal | 78.11 (±0.41) | 95.00 | 0.9311 | 9.92 | 0.0024 | 98.28 | 78.30 | 0.0024 |
| w8a8_kd_pilot5ep_final__analog_W8A8__cal | 78.21 (±0.41) | 94.94 | 0.9302 | 9.95 | 0.0021 | 98.56 | 78.26 | 0.0024 |

## Hardware diagnostics per operation (analog runs)

| run | gains qkv/qk/av | clip % qkv/qk/av | rail % qkv/qk/av | ADC util median % qkv/qk/av | util p99.9 % qkv/qk/av | ADC saturation % (all ops) | images with any saturation % |
|---|---|---|---|---|---|---|---|
| old_int8__analog_W8A8__cal | 22.92/8.75/2.72 | 0.0107/0.0075/0.0103 | 0.012/0.009/0.011 | 12.4/11.9/1.9 | 80/81/59 | 0.0100 | 100.0 |
| old_int8__legacyadc_W8A8 | — | — (legacy ADC: per-op stats not recorded) | — (legacy ADC: per-op stats not recorded) | — (legacy ADC: per-op stats not recorded) | — (legacy ADC: per-op stats not recorded) | 0.0002 | 10.7 |
| pretrained__analog_W8A8__cal | 22.92/8.75/2.72 | 0.0099/0.0099/0.0099 | 0.011/0.012/0.010 | 12.5/12.3/2.1 | 80/83/61 | 0.0099 | 100.0 |
| pretrained__analog_W8A8__hist | 23.77/8.41/2.92 | 0.0152/0.0055/0.0143 | 0.017/0.007/0.015 | 12.9/11.8/2.2 | 82/80/66 | 0.0132 | 100.0 |
| pretrained__analog_W8A8__unity | 1.00/1.00/1.00 | 0.0000/0.0000/0.0000 | 0.000/0.000/0.000 | 0.5/1.4/0.7 | 3/9/17 | 0.0000 | 0.0 |
| w8a8_kd_pilot5ep_best__analog_W8A8__cal | 22.92/8.75/2.72 | 0.0100/0.0069/0.0096 | 0.011/0.008/0.010 | 12.5/12.2/2.1 | 79/81/61 | 0.0094 | 100.0 |
| w8a8_kd_pilot5ep_final__analog_W8A8__cal | 22.92/8.75/2.72 | 0.0099/0.0068/0.0097 | 0.011/0.008/0.010 | 12.5/12.2/2.1 | 80/81/61 | 0.0093 | 100.0 |

ADC-utilisation percentiles: weighted uniform subsample (~2 M conversions per operation, cim_qat.analog_merge); clip, rail and saturation columns are exact counts.

## Paired comparisons (b vs a, same images; C1–C9 pre-declared)

| id | isolates | a | b | Δtop-1 pp (95% CI) | McNemar p (a-only/b-only) | ΔNLL (95% CI) | ΔKL (95% CI) |
|---|---|---|---|---|---|---|---|
| C1 | operand quantization (exact integer MAC, no ADC) | pretrained__fp32 | pretrained__adcoff_W8A8 | -0.11 (-0.23, +0.01) | 0.0989 (24/13) | +0.0004 (-0.0002, +0.0009) | +0.0005 (+0.0004, +0.0005) |
| C2 | ADC + analog chain at unity gain | pretrained__adcoff_W8A8 | pretrained__analog_W8A8__unity | -4.40 (-4.98, -3.84) | 2.22e-51 (663/223) | +0.2649 (+0.2524, +0.2775) | +0.1965 (+0.1910, +0.2021) |
| C3 | ADC + analog chain at calibrated gains | pretrained__adcoff_W8A8 | pretrained__analog_W8A8__cal | -0.11 (-0.30, +0.08) | 0.305 (53/42) | +0.0031 (+0.0017, +0.0043) | +0.0022 (+0.0021, +0.0023) |
| C4 | gain calibration (unity -> calibrated) | pretrained__analog_W8A8__unity | pretrained__analog_W8A8__cal | +4.29 (+3.75, +4.87) | 4.18e-49 (226/655) | -0.2618 (-0.2746, -0.2493) | -0.1943 (-0.1998, -0.1888) |
| C4b | calibrated vs historical synthetic-input gains | pretrained__analog_W8A8__hist | pretrained__analog_W8A8__cal | -0.05 (-0.24, +0.13) | 0.672 (47/42) | +0.0010 (-0.0003, +0.0023) | +0.0002 (+0.0001, +0.0003) |
| C5 | old QAT vs PTQ (same hardware, same gains) | pretrained__analog_W8A8__cal | old_int8__analog_W8A8__cal | -2.08 (-2.62, -1.56) | 9.19e-16 (441/233) | -0.0416 (-0.0534, -0.0298) | +0.4513 (+0.4454, +0.4574) |
| C8 | old QAT checkpoint: its training point (legacy ADC) -> new hardware | old_int8__legacyadc_W8A8 | old_int8__analog_W8A8__cal | -0.57 (-1.04, -0.09) | 0.0172 (305/248) | +0.0205 (+0.0114, +0.0300) | +0.0010 (-0.0025, +0.0045) |
| C6 | new QAT (VAL-selected) vs PTQ | pretrained__analog_W8A8__cal | w8a8_kd_pilot5ep_best__analog_W8A8__cal | -0.07 (-0.27, +0.13) | 0.558 (56/49) | -0.0004 (-0.0019, +0.0011) | -0.0003 (-0.0004, -0.0002) |
| C7 | new QAT vs old QAT (same hardware, same gains) | old_int8__analog_W8A8__cal | w8a8_kd_pilot5ep_best__analog_W8A8__cal | +2.01 (+1.49, +2.53) | 1.4e-14 (242/443) | +0.0412 (+0.0294, +0.0530) | -0.4516 (-0.4577, -0.4457) |
| C7b | new QAT: last vs VAL-selected checkpoint | w8a8_kd_pilot5ep_best__analog_W8A8__cal | w8a8_kd_pilot5ep_final__analog_W8A8__cal | +0.10 (-0.06, +0.27) | 0.302 (33/43) | -0.0009 (-0.0021, +0.0003) | -0.0003 (-0.0004, -0.0003) |