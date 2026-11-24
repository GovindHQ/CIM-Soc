# W8A8 — ImageNetV2 top-images (extension) (n = 10000)

Status: EXECUTED (2026-10-30T17:15:46+00:00). All CIM rows from the actual cim_sim simulator (batch 1, CPU). KL and agreement are vs the pretrained FP32 model on the same images.
Identical inputs verified: manifest `da14691bdf4a3f6b`, per-image input-tensor SHA-256 digest `b7355b0e9dba1814`, preprocessing `cim_exp.imagenette.load_uint8+to_model_input|interp=bicubic|resize_short=256|crop=224|mean=[0.485, 0.456, 0.406]|std=[0.229, 0.224, 0.225]`.

> ImageNetV2 top-images (EXTENSION requested after the results were frozen (29 Oct 2026): descriptive only. The pre-declared primary TEST result is ImageNetV2 matched-frequency (10,000 images).) Compare models WITHIN this table.

| run | top-1 % (±SE) | top-5 % | NLL | ECE % | KL(FP32‖run) | agree FP32 % | own-FP32 top-1 % | KL(own FP32‖run) |
|---|---|---|---|---|---|---|---|---|
| old_int8__analog_W8A8__cal | 80.69 (±0.39) | 96.27 | 0.6751 | 2.40 | 0.4332 | 91.27 | 80.87 | 0.0019 |
| old_int8__legacyadc_W8A8 | 81.25 (±0.39) | 96.43 | 0.6573 | 2.31 | 0.4336 | 91.96 | 80.87 | 0.0714 |
| pretrained__adcoff_W8A8 | 82.60 (±0.38) | 96.91 | 0.7382 | 11.01 | 0.0004 | 99.37 | 82.58 | 0.0004 |
| pretrained__analog_W8A8__cal | 82.47 (±0.38) | 96.85 | 0.7402 | 10.87 | 0.0022 | 98.65 | 82.58 | 0.0022 |
| pretrained__analog_W8A8__hist | 82.59 (±0.38) | 96.88 | 0.7384 | 10.94 | 0.0020 | 98.85 | 82.58 | 0.0020 |
| pretrained__analog_W8A8__unity | 78.65 (±0.41) | 94.86 | 0.9711 | 14.38 | 0.1690 | 88.84 | 82.58 | 0.1690 |
| pretrained__fp32 | 82.58 (±0.38) | 96.93 | 0.7379 | 11.00 | 0.0000 | 100.00 | — | — |
| w8a8_kd_pilot5ep_best__analog_W8A8__cal | 82.37 (±0.38) | 96.95 | 0.7403 | 10.85 | 0.0019 | 98.68 | 82.46 | 0.0019 |
| w8a8_kd_pilot5ep_final__analog_W8A8__cal | 82.60 (±0.38) | 96.90 | 0.7396 | 11.04 | 0.0017 | 98.88 | 82.45 | 0.0020 |

## Hardware diagnostics per operation (analog runs)

| run | gains qkv/qk/av | clip % qkv/qk/av | rail % qkv/qk/av | ADC util median % qkv/qk/av | util p99.9 % qkv/qk/av | ADC saturation % (all ops) | images with any saturation % |
|---|---|---|---|---|---|---|---|
| old_int8__analog_W8A8__cal | 22.92/8.75/2.72 | 0.0107/0.0075/0.0102 | 0.012/0.009/0.011 | 12.4/11.9/1.9 | 80/81/59 | 0.0100 | 100.0 |
| old_int8__legacyadc_W8A8 | — | — (legacy ADC: per-op stats not recorded) | — (legacy ADC: per-op stats not recorded) | — (legacy ADC: per-op stats not recorded) | — (legacy ADC: per-op stats not recorded) | 0.0002 | 10.5 |
| pretrained__analog_W8A8__cal | 22.92/8.75/2.72 | 0.0099/0.0099/0.0098 | 0.011/0.012/0.010 | 12.5/12.3/2.1 | 80/83/61 | 0.0099 | 100.0 |
| pretrained__analog_W8A8__hist | 23.77/8.41/2.92 | 0.0152/0.0055/0.0141 | 0.017/0.007/0.015 | 13.0/11.8/2.2 | 83/80/65 | 0.0133 | 100.0 |
| pretrained__analog_W8A8__unity | 1.00/1.00/1.00 | 0.0000/0.0000/0.0000 | 0.000/0.000/0.000 | 0.5/1.4/0.7 | 3/9/17 | 0.0000 | 0.0 |
| w8a8_kd_pilot5ep_best__analog_W8A8__cal | 22.92/8.75/2.72 | 0.0100/0.0070/0.0095 | 0.011/0.008/0.010 | 12.5/12.2/2.1 | 80/81/61 | 0.0094 | 100.0 |
| w8a8_kd_pilot5ep_final__analog_W8A8__cal | 22.92/8.75/2.72 | 0.0100/0.0068/0.0096 | 0.011/0.008/0.010 | 12.5/12.2/2.1 | 80/81/61 | 0.0093 | 100.0 |

ADC-utilisation percentiles: weighted uniform subsample (~2 M conversions per operation, cim_qat.analog_merge); clip, rail and saturation columns are exact counts.

## Paired comparisons (b vs a, same images; C1–C9 pre-declared)

| id | isolates | a | b | Δtop-1 pp (95% CI) | McNemar p (a-only/b-only) | ΔNLL (95% CI) | ΔKL (95% CI) |
|---|---|---|---|---|---|---|---|
| C1 | operand quantization (exact integer MAC, no ADC) | pretrained__fp32 | pretrained__adcoff_W8A8 | +0.02 (-0.10, +0.14) | 0.871 (18/20) | +0.0004 (-0.0001, +0.0009) | +0.0004 (+0.0004, +0.0004) |
| C2 | ADC + analog chain at unity gain | pretrained__adcoff_W8A8 | pretrained__analog_W8A8__unity | -3.95 (-4.48, -3.43) | 3.11e-48 (580/185) | +0.2329 (+0.2221, +0.2437) | +0.1686 (+0.1635, +0.1737) |
| C3 | ADC + analog chain at calibrated gains | pretrained__adcoff_W8A8 | pretrained__analog_W8A8__cal | -0.13 (-0.30, +0.04) | 0.171 (45/32) | +0.0020 (+0.0009, +0.0032) | +0.0018 (+0.0017, +0.0019) |
| C4 | gain calibration (unity -> calibrated) | pretrained__analog_W8A8__unity | pretrained__analog_W8A8__cal | +3.82 (+3.30, +4.37) | 2.95e-45 (190/572) | -0.2309 (-0.2415, -0.2201) | -0.1668 (-0.1719, -0.1616) |
| C4b | calibrated vs historical synthetic-input gains | pretrained__analog_W8A8__hist | pretrained__analog_W8A8__cal | -0.12 (-0.30, +0.05) | 0.213 (45/33) | +0.0018 (+0.0007, +0.0029) | +0.0001 (+0.0001, +0.0002) |
| C5 | old QAT vs PTQ (same hardware, same gains) | pretrained__analog_W8A8__cal | old_int8__analog_W8A8__cal | -1.78 (-2.27, -1.29) | 8.23e-13 (399/221) | -0.0651 (-0.0748, -0.0554) | +0.4310 (+0.4254, +0.4371) |
| C8 | old QAT checkpoint: its training point (legacy ADC) -> new hardware | old_int8__legacyadc_W8A8 | old_int8__analog_W8A8__cal | -0.56 (-1.01, -0.13) | 0.0133 (275/219) | +0.0178 (+0.0104, +0.0260) | -0.0004 (-0.0037, +0.0028) |
| C6 | new QAT (VAL-selected) vs PTQ | pretrained__analog_W8A8__cal | w8a8_kd_pilot5ep_best__analog_W8A8__cal | -0.10 (-0.28, +0.08) | 0.332 (48/38) | +0.0001 (-0.0012, +0.0014) | -0.0003 (-0.0003, -0.0002) |
| C7 | new QAT vs old QAT (same hardware, same gains) | old_int8__analog_W8A8__cal | w8a8_kd_pilot5ep_best__analog_W8A8__cal | +1.68 (+1.19, +2.17) | 2.36e-11 (232/400) | +0.0652 (+0.0555, +0.0750) | -0.4313 (-0.4374, -0.4257) |
| C7b | new QAT: last vs VAL-selected checkpoint | w8a8_kd_pilot5ep_best__analog_W8A8__cal | w8a8_kd_pilot5ep_final__analog_W8A8__cal | +0.23 (+0.06, +0.40) | 0.0117 (27/50) | -0.0007 (-0.0017, +0.0003) | -0.0002 (-0.0003, -0.0002) |