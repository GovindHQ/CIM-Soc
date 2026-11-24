# POST HOC: own-FP32 and the case decomposition (stored logits only; descriptive)

All differences are second minus first, top-1 points, paired on the same images, 95% paired-bootstrap CI (cim_qat.metrics, seed 0, 2000 resamples). Pooled = each distinct image once (first occurrence in the order matched-frequency, threshold-0.7, top-images). Chosen after the TEST results were seen.


## W8A8  (pooled n = 20,683 distinct images)


### W8A8, NEW QAT (primary): top-1 % and agreement with pretrained FP32

| test set (images) | P: pretrained FP32 | Q: pretrained-CIM | A: QAT FP32 (own) | C: QAT-CIM | agree A vs P % | agree C vs P % |
|---|---|---|---|---|---|---|
| matched-frequency 10k (10,000) | 69.69 | 69.59 | 69.65 | 69.70 | 98.6 | 97.6 |
| threshold-0.7 10k (10,000) | 78.40 | 78.18 | 78.30 | 78.11 | 99.0 | 98.3 |
| top-images 10k (10,000) | 82.58 | 82.47 | 82.46 | 82.37 | 99.1 | 98.7 |
| POOLED (20,683) | 75.49 | 75.38 | 75.39 | 75.35 | 98.8 | 98.0 |

### W8A8, NEW QAT (primary): paired differences, top-1 points (95% CI)

| test set | Case 4: P→Q | Case 1: Q→C | Case 2: P→A | Case 3: A→C | Case 5: P→C |
|---|---|---|---|---|---|
| matched-frequency 10k | -0.10 (-0.31, +0.11) | +0.11 (-0.09, +0.33) | -0.04 (-0.21, +0.13) | +0.05 (-0.17, +0.27) | +0.01 (-0.20, +0.23) |
| threshold-0.7 10k | -0.22 (-0.41, -0.03) | -0.07 (-0.27, +0.13) | -0.10 (-0.26, +0.05) | -0.19 (-0.38, +0.00) | -0.29 (-0.48, -0.10) |
| top-images 10k | -0.11 (-0.29, +0.07) | -0.10 (-0.28, +0.08) | -0.12 (-0.27, +0.03) | -0.09 (-0.27, +0.09) | -0.21 (-0.40, -0.03) |
| POOLED | -0.11 (-0.24, +0.02) | -0.03 (-0.17, +0.10) | -0.10 (-0.21, +0.01) | -0.04 (-0.18, +0.10) | -0.15 (-0.28, -0.00) |

### W8A8, OLD QAT (supporting): top-1 % and agreement with pretrained FP32

| test set (images) | P: pretrained FP32 | Q: pretrained-CIM | A: QAT FP32 (own) | C: QAT-CIM | agree A vs P % | agree C vs P % |
|---|---|---|---|---|---|---|
| matched-frequency 10k (10,000) | 69.69 | 69.59 | 68.26 | 68.22 | 86.1 | 85.7 |
| threshold-0.7 10k (10,000) | 78.40 | 78.18 | 76.23 | 76.10 | 89.5 | 89.4 |
| top-images 10k (10,000) | 82.58 | 82.47 | 80.87 | 80.69 | 91.5 | 91.3 |
| POOLED (20,683) | 75.49 | 75.38 | 73.80 | 73.72 | 88.5 | 88.3 |

### W8A8, OLD QAT (supporting): paired differences, top-1 points (95% CI)

| test set | Case 4: P→Q | Case 1: Q→C | Case 2: P→A | Case 3: A→C | Case 5: P→C |
|---|---|---|---|---|---|
| matched-frequency 10k | -0.10 (-0.31, +0.11) | -1.37 (-1.90, -0.82) | -1.43 (-1.97, -0.88) | -0.04 (-0.24, +0.16) | -1.47 (-2.01, -0.91) |
| threshold-0.7 10k | -0.22 (-0.41, -0.03) | -2.08 (-2.62, -1.56) | -2.17 (-2.68, -1.69) | -0.13 (-0.33, +0.05) | -2.30 (-2.82, -1.81) |
| top-images 10k | -0.11 (-0.29, +0.07) | -1.78 (-2.27, -1.29) | -1.71 (-2.20, -1.24) | -0.18 (-0.35, -0.01) | -1.89 (-2.36, -1.40) |
| POOLED | -0.11 (-0.24, +0.02) | -1.66 (-2.01, -1.29) | -1.69 (-2.03, -1.33) | -0.08 (-0.22, +0.05) | -1.77 (-2.12, -1.40) |

### W8A8 self-check: my pooled top-1 vs REPORT_test_pooled_W8A8.json

| row | this script | existing report | match |
|---|---|---|---|
| pretrained__fp32 | 75.4919 | 75.4919 | yes |
| pretrained__analog_W8A8__cal | 75.3807 | 75.3807 | yes |
| w8a8_kd_pilot5ep_best__analog_W8A8__cal | 75.3469 | 75.3469 | yes |
| old_int8__analog_W8A8__cal | 73.7175 | 73.7175 | yes |

## W4A4  (pooled n = 20,683 distinct images)


### W4A4, NEW QAT (primary): top-1 % and agreement with pretrained FP32

| test set (images) | P: pretrained FP32 | Q: pretrained-CIM | A: QAT FP32 (own) | C: QAT-CIM | agree A vs P % | agree C vs P % |
|---|---|---|---|---|---|---|
| matched-frequency 10k (10,000) | 69.69 | 66.70 | 68.85 | 69.26 | 89.5 | 92.9 |
| threshold-0.7 10k (10,000) | 78.40 | 75.55 | 76.98 | 77.56 | 92.2 | 94.7 |
| top-images 10k (10,000) | 82.58 | 80.06 | 81.73 | 82.05 | 94.0 | 96.0 |
| POOLED (20,683) | 75.49 | 72.55 | 74.45 | 74.90 | 91.4 | 94.2 |

### W4A4, NEW QAT (primary): paired differences, top-1 points (95% CI)

| test set | Case 4: P→Q | Case 1: Q→C | Case 2: P→A | Case 3: A→C | Case 5: P→C |
|---|---|---|---|---|---|
| matched-frequency 10k | -2.99 (-3.51, -2.43) | +2.56 (+1.99, +3.09) | -0.84 (-1.28, -0.39) | +0.41 (-0.06, +0.91) | -0.43 (-0.77, -0.07) |
| threshold-0.7 10k | -2.85 (-3.37, -2.31) | +2.01 (+1.49, +2.54) | -1.42 (-1.83, -0.96) | +0.58 (+0.13, +1.05) | -0.84 (-1.18, -0.47) |
| top-images 10k | -2.52 (-3.01, -2.05) | +1.99 (+1.53, +2.51) | -0.85 (-1.25, -0.44) | +0.32 (-0.13, +0.75) | -0.53 (-0.84, -0.22) |
| POOLED | -2.94 (-3.30, -2.61) | +2.35 (+2.01, +2.73) | -1.04 (-1.34, -0.74) | +0.45 (+0.13, +0.77) | -0.59 (-0.85, -0.34) |

### W4A4, OLD QAT (supporting): top-1 % and agreement with pretrained FP32

| test set (images) | P: pretrained FP32 | Q: pretrained-CIM | A: QAT FP32 (own) | C: QAT-CIM | agree A vs P % | agree C vs P % |
|---|---|---|---|---|---|---|
| matched-frequency 10k (10,000) | 69.69 | 66.70 | 67.93 | 67.38 | 83.9 | 84.2 |
| threshold-0.7 10k (10,000) | 78.40 | 75.55 | 75.86 | 75.50 | 88.0 | 87.9 |
| top-images 10k (10,000) | 82.58 | 80.06 | 80.45 | 80.42 | 90.3 | 90.5 |
| POOLED (20,683) | 75.49 | 72.55 | 73.39 | 73.07 | 86.7 | 86.9 |

### W4A4, OLD QAT (supporting): paired differences, top-1 points (95% CI)

| test set | Case 4: P→Q | Case 1: Q→C | Case 2: P→A | Case 3: A→C | Case 5: P→C |
|---|---|---|---|---|---|
| matched-frequency 10k | -2.99 (-3.51, -2.43) | +0.68 (+0.02, +1.32) | -1.76 (-2.33, -1.17) | -0.55 (-1.08, -0.03) | -2.31 (-2.92, -1.72) |
| threshold-0.7 10k | -2.85 (-3.37, -2.31) | -0.05 (-0.69, +0.56) | -2.54 (-3.07, -2.00) | -0.36 (-0.82, +0.10) | -2.90 (-3.45, -2.36) |
| top-images 10k | -2.52 (-3.01, -2.05) | +0.36 (-0.19, +0.92) | -2.13 (-2.63, -1.66) | -0.03 (-0.47, +0.41) | -2.16 (-2.66, -1.65) |
| POOLED | -2.94 (-3.30, -2.61) | +0.53 (+0.11, +0.97) | -2.10 (-2.46, -1.73) | -0.32 (-0.63, +0.00) | -2.42 (-2.78, -2.05) |

### W4A4 self-check: my pooled top-1 vs REPORT_test_pooled_W4A4.json

| row | this script | existing report | match |
|---|---|---|---|
| pretrained__fp32 | 75.4919 | 75.4919 | yes |
| pretrained__analog_W4A4__cal | 72.5475 | 72.5475 | yes |
| w4a4_kd_pilot10ep_full_best__analog_W4A4__cal | 74.8973 | 74.8973 | yes |
| old_int4__analog_W4A4__cal | 73.0745 | 73.0745 | yes |
