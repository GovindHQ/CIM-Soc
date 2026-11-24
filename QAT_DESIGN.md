# QAT_DESIGN.md — hardware-aware QAT of TinyViT-5M against the cim_sim analog model

**Version 2.1, 25 Oct 2026** (v2.1 adds the held-out eligibility rule, §8.1, after the first real
`make` found 3 CAL images whose content equals VAL images). Supersedes v1 (22 Oct 2026). v2 adopts the revised data protocol:
- VAL is the full 50,000-image ImageNet-1k val set, used for checkpoint selection and early stopping;
- TEST is ImageNetV2 matched-frequency.

v2 also re-assesses the v1 design (§0). **Nothing has been run on the GPU machine or on real
images yet.** Only the CPU tests in §13 have run, in the development sandbox on fake images.

Every claim is labelled:
- **[HW]** hardware-specified.
- **[DECIDED]** fixed by your decision.
- **[ASSUMPTION]** a provisional simulator modelling assumption.
- **[VERIFIED-DEV]** verified by execution in the development sandbox: CPU only, Python 3.13,
  torch 2.14.1, fake images, so data-dependent numbers mean nothing.
- **[VERIFIED-SRC]** read from source code or logs that you supplied.
- **[PUBLIC]** from a public source (see "Sources").
- **[REMOTE]** observed by you on the GPU machine.
- **[PENDING]** must still be run.

---

## 0. Re-assessment of v1 and what changed in v2

### 0.1 Kept unchanged (re-checked, still correct)

| area | decision | why it stands |
|---|---|---|
| hardware model | capacitive DAC 8 b / 0.6 V; C2C 32×32; per-op pre-ADC gain; SAR ADC **8 b**; ideal components; all `cim_sim` code byte-identical | your instruction; no correctness issue found in the simulator (one counter-only edge, §4.3, left as is) |
| quantization | W8A8 first, then W4A4; per-token / per-column symmetric; AV unsigned; A4 MSB-aligned | [DECIDED], identical to the simulator |
| surrogate | the forward is a bit-exact re-implementation of the simulator's integer/ADC arithmetic; STE backward as in §5 | verified exact per matmul, per module, and for the full model on CPU [VERIFIED-DEV] |
| training vs verification | the **surrogate** trains (GPU, differentiable); the **actual simulator** verifies (spot checks, calibration check, every reported number) | §3 |
| loss | pure KD, forward KL, T = 1 (optional controlled W4A4 CE pilot) | [DECIDED] |
| optimizer / budget | AdamW lr 1e-5, wd 1e-4, batch 64, warm-up 0.1 epoch + cosine; pilots W8A8 5 epochs, W4A4 10 epochs | [DECIDED] |
| BN / teacher | BN running statistics frozen, all parameters trainable, teacher frozen | [DECIDED] |
| calibration | exact probe on CAL; P99.99 primary; P99.9 / P99.999 sensitivity; shared gain per op; CALCHECK simulator check; explicit freeze | [DECIDED] |
| preprocessing | `cim_exp` bicubic eval for every model; bicubic RandomResizedCrop for training | [DECIDED] |
| numerics | fp32, TF32 off, deterministic, bit-exact resume | [VERIFIED-DEV] |

### 0.2 Changed in v2, with the reason for each

| # | v1 | v2 | reason |
|---|---|---|---|
| 1 | VAL = 10k and TEST = 40k, both drawn from ImageNet val | **VAL = all 50,000 ImageNet val; TEST = ImageNetV2 matched-frequency (10,000)** | your requirement A. VAL is now the same set the old QAT selected on; TEST is a separately collected set |
| 2 | primary = final checkpoint; best-VAL secondary | **primary = best-VAL checkpoint** (hardware top-1 under the surrogate; ties → lower NLL); final checkpoint = pre-declared secondary | requirement A: "use VAL for checkpoint selection … matching the old QAT protocol". VAL-selected numbers on VAL are optimistic; the TEST numbers are not (§8.4) |
| 3 | no early stopping (plateau rule only) | **early stopping on VAL** with a pre-declared patience per run: **W8A8 5 epochs** (= its budget, so it cannot stop early), **W4A4 3 epochs** (CE pilot 3) | requirement A. Old QAT used patience 8 within a 40-epoch budget [VERIFIED-SRC]; 8 would never fire in 5- or 10-epoch pilots. v2 proposed 2 as a flagged default. **Your decision, 26 Oct 2026, after the stage 10.1 smoke runs and before any training run; it rests on budget and throughput only, not on any QAT accuracy** (deviation D1, §0.4) |
| 4 | VAL evaluated every 0.25 epoch | VAL evaluated at epoch 0 (PTQ) and at **every epoch end** | the old protocol evaluated per epoch. Each candidate checkpoint adds max-selection optimism, so 4× more candidates would be 4× more chances to pick a lucky one. A full-VAL surrogate pass is a GPU job of minutes, measured in stage 7 |
| 5 | (no TEST loader needed) | **ImageNetV2 loader: label = `int(folder name)`**; structural check that the folders are exactly `0`…`999` | **critical data issue:** the ImageNetV2 README suggests `torchvision.datasets.ImageFolder`, which sorts folder names as strings (`0, 1, 10, 100, …`). That mislabels **988 of the 1000 classes** [VERIFIED-DEV, pinned by test D7]. The TFDS builder uses `int(class_id)` [PUBLIC] |
| 6 | TRAIN content-dedup optional; reported only | `make --hash-train` hashes all of TRAIN (resumable). **TRAIN files whose content equals a CAL/CALCHECK file are removed from TRAIN and listed.** TRAIN∩VAL and TRAIN∩TEST duplicates are counted and reported, not removed (§8.3) | requirement C/E: TRAIN, CAL and CALCHECK must be disjoint by content too, not only by path |
| 7 | count accounting implicit | **exact accounting from disk:** TRAIN = files_on_disk − CAL − CALCHECK − excluded_duplicates, with the identity asserted and re-counted by the audit | requirement C (the expected 1,281,167 is never assumed) |
| 8 | path overlap on manifest keys | also on **resolved real paths** (symlinks followed) | the existing `data/imagenet/val` copy consists of symlinks [REMOTE]; two names can point to one file |
| 9 | TEST guarded by a `--final` flag only | **TEST guard in code:** `Splits` refuses TEST unless opened for the final evaluation (or the label-integrity check). **Every split access by every process is logged** to `access_log.jsonl`. `qat_audit.py` inspects the configs, run logs, calibration records, eval records and that log | requirement E: "must inspect the actual experiment configs and run logs" |
| 10 | (no per-image provenance) | every simulator evaluation stores a **SHA-256 of every model-input tensor** plus a preprocessing id. The report and the audit **refuse** to compare runs whose images, labels, input tensors or preprocessing differ | requirement E: identical TEST images and preprocessing across models |
| 11 | `all50k` (VAL∪TEST) split for old-QAT transfer | removed: VAL is the full 50k | — |
| 12 | old QAT only at the new hardware | adds an optional diagnostic row: old checkpoints at **their own training point** (legacy functional ADC 8 b, an existing simulator path) | separates "old checkpoint on new hardware" from "old pipeline's own target" (C8, §10) |
| 13 | INT4 history "22 completed epochs" | **corrected from the delivered logs:** INT4 has **28** completed epochs (`metrics.csv` rows 0–27), best 79.354 % at epoch 22 (row 21), stopped manually during epoch 29 | §12; your text said 22, the logs say the *best* epoch is 22 |

### 0.3 Critical issues still open (not fixable in code)

1. **ImageNetV2 is not on the machine yet** (not found in your listings). It must be downloaded
   on the remote (runbook stage 5.2). The source used is the TFDS builder's URL on Hugging Face
   [PUBLIC]. **No checksum is published**, so you record the SHA-256 of your copy and the file
   counts are verified (10,000 images, 1,000 folders, 10 per class).
   - If the download is impossible, TEST is **blocked**.
   - Fallback A: download on another computer and copy it over.
   - Fallback B: report VAL results only, with their selection optimism stated. Nothing
     substitutes for TEST.
2. **VAL selection optimism.** Old and new QAT are both selected on VAL, so their VAL numbers are
   biased upward. The PTQ/FP32 rows are not. VAL is still reported for comparability with the old
   80.11 %, but **conclusions about QAT come from TEST**.
   *Train/VAL content overlap (measured 29 Oct):* 799 VAL images (1.6 %; 797 distinct contents, two of them repeated inside VAL) have an identical-content copy in the TRAIN split. Measured, not assumed (`tools_qat/val_overlap_sensitivity.py`, 29 Oct): every model scores only 35–39 % top-1 on them against 72–81 % overall, so including them LOWERS VAL top-1 by 0.59–0.70 pp for every row rather than inflating it, and every paired comparison changes by at most 0.07 pp (C10 +0.24 → +0.26 pp, CI +0.10 to +0.42). The overlap is therefore not a source of VAL optimism; the optimism of the QAT rows on VAL comes from selecting on VAL. ImageNetV2 (TEST) has no content overlap with TRAIN.
3. **Old QAT VAL accuracy is not a simulator number.** The 80.11 % (INT8) and 79.354 % (INT4)
   were measured with the old training forward: legacy ADC fake-quant, fp16 AMP, bilinear
   preprocessing [VERIFIED-SRC]. They are historical context and are not comparable to any
   simulator row.
4. **Disk:** `/` is 99 % full [REMOTE]. Everything (code, work, tmp, ImageNetV2) goes on
   `/media/user/DATA 1`, which has 5.2 TB free but is a rotational HDD. Data loading and
   `--hash-train` are I/O-bound, and their speed is measured, not assumed (stages 6–7).

### 0.4 Documented deviations from the v2 proposal

| id | item | v2 proposal | decided | when / basis | consequence |
|---|---|---|---|---|---|
| D1 | early-stopping patience | 2 epochs (flagged default) | **W8A8 5, W4A4 3, W4A4 CE pilot 3** | 26 Oct 2026, after the smoke runs (throughput only), before any training run | **W8A8:** with patience = budget, the stopping rule can never fire (at most 3 non-improving epoch ends precede the last one), so the run always completes its 5 epochs; selection is still best-VAL; the plateau rule decides whether an extension is recommended. **W4A4:** stops if 3 consecutive epoch-end VAL evaluations do not improve on the best (3/10 vs the old 8/40). Old and new patience/budget ratios differ, so stopping behaviour is not identical to the old QAT. Recorded in each run's `config_resolved.json`; `qat_audit.py` FAILs a run whose patience differs from these values |
| D2 | W4A4 early stopping vs the cosine schedule | — (not foreseen) | **if `w4a4_kd_pilot10ep` early-stops, fork it as `w4a4_kd_pilot10ep_full` and continue the same 10-epoch schedule with early stopping off; report the W4A4 run with the higher best-VAL hardware top-1 (ties → lower VAL NLL → the pilot)** | Requested by you on 27 Oct 2026 at 02:31 IST (= 26 Oct 21:01 UTC), **about 3 h before** the pilot's epoch-5 evaluation and early stop (logged 27 Oct 00:09:58 UTC); best at the time of the decision: epoch 2, 80.23 %; epoch 3 80.13 %; VAL evidence only, TEST untouched. Package v2.5 implemented it; the fork was started 27 Oct 10:31:28 UTC | Patience 3 can stop the run at epoch 5 with the learning rate still about 5e-6, before the low-learning-rate tail of the cosine schedule (in which W8A8's KL kept falling). The fork restores the stopped run's exact state (weights, optimizer, scheduler, every RNG) and continues; this is **bit-identical to a run that never had early stopping** (`tests_qat/test_fork.py` F5) and saves re-training epochs 1–5. Only `run_name` and `early_stopping_patience` (10 ≥ 10 epochs) may differ (`cim_qat.trainer.prepare_fork`). The fork inherits the pilot's history and best checkpoint and replaces the best only if a later epoch is strictly better, so its best-VAL top-1 is always ≥ the pilot's; D2 therefore reports the fork's best (the identical file if no later epoch improves). Recorded by `qat_evaluate.py select-d2` in `eval/D2_selection.json` **before any TEST read**. `qat_audit.py` checks: the fork is the declared one; the source early-stopped and is unchanged; only the allowed settings changed; the fork's history equals the source's; the D2 record matches the rule and precedes the first TEST read; and the selected run is the one evaluated on TEST. If the pilot does not early-stop, no fork is made and D2 selects the pilot |
| D3 | TEST size | ImageNetV2 matched-frequency only (10,000) | **Extension, descriptive only: also evaluate ImageNetV2 threshold-0.7 and top-images (10,000 each) with the frozen models, gains and code path; report each variant on its own and the three pooled (each distinct image once). The matched-frequency 10k stays the pre-declared primary result** | Requested by the professor on 29 Oct 2026, **after** the results were frozen (`FINAL_SHA256SUMS_20261029.txt`) and after the 10k TEST reports were seen. Declared before any image of the two new variants was read. Package v2.8 | Nothing frozen is re-run or changed: no retraining, no re-selection (D2 stands), same gains, same simulator. The new variants are registered as guarded TEST splits (`test_th07`, `test_top`) in `splits/extra_tests.json`; `splits.json` is not rewritten. **Exclusion rule (declared before reading them):** an image whose file content equals a CAL, CALCHECK, VAL or TRAIN image is removed from the variant (listed in `excluded_<split>.csv`). Duplicates of matched-frequency images, and between variants, stay in each variant and are counted once in the pooled report (first occurrence in the order matched-frequency, threshold-0.7, top-images); their outputs must be identical (checked). New results go to new folders (`eval/test_th07`, `eval/test_top`, `eval/test_pooled`); v2.8 refuses to overwrite any file listed in a checksum freeze, skips finished rows without touching them, and logs new reads in `access_log_ext.jsonl` (the frozen `access_log.jsonl` is never appended to). The comparison new QAT vs FP32 on the pooled set (**P1**) was chosen after the 10k results were seen: **post hoc**, labelled so. The variants differ in difficulty, so pooled absolute numbers mix three distributions; paired differences (same images for both models) remain valid. The audit section X checks all of this; the WARNs it raises (exclusions, overlaps between variants, evaluation after the freeze) are disclosures, not failures **Outcome (29 Oct–1 Nov):** 0 images excluded; the variants overlap (9,317 duplicates), so the pool is 20,683 distinct images; every pre-declared finding holds; P1 post hoc: W8A8 −0.15 pp (95% CI −0.28 to −0.005), W4A4 −0.59 pp (−0.85 to −0.34) vs FP32; audit 212 PASS / 6 WARN / 0 FAIL; second freeze `FINAL_SHA256SUMS_20261030_ext.txt` (QAT_EXPERIMENT_LOG.md §8.4). A post hoc decomposition against each checkpoint's own float output, from the stored logits only, followed on 2 Nov (§8.5; `tools_qat/own_fp32_posthoc.py`) |

---

## 1. Forward path of the simulator as used by TinyViT

Only the three attention matmuls of the 10 `Attention` modules (stages 1–3) run on the CIM model
[VERIFIED-DEV]: `cim_sim.attention.CIMAttention`, `cim_out_proj=False`. The following run in
float32:
- patch-embed convs and stage-0 MBConvs;
- LayerNorm, the qkv-bias add, the attention-bias add and softmax;
- the output projection and the MLP;
- the depthwise `local_conv` (Conv2d + BN);
- residuals and the head.

| op | X (DAC side) | W (array side) | X scale | W scale | X signed |
|---|---|---|---|---|---|
| `qkv` | LayerNorm(x) `[B·N, C]` | `W_qkv^T` `[C, 3C]`, static | per token | per output column | yes |
| `qk` | Q `[N, d]` per (image·window, head) | `K^T` `[d, N]`, dynamic | per query token | per key token | yes |
| `av` | A = softmax(QKᵀ·s + bias) `[N, N]` | V `[N, d]`, dynamic | per query row | per channel over all tokens | **no** |

```
1 operand quantization   q_a = 2^(b_a-1)-1 (signed) | 2^b_a - 1 (AV);  q_w = 2^(b_w-1)-1
                         s_x[n] = max(max_k |X[n,k]|, 1e-12)/q_a      X_q = clip(rint(X/s_x))
                         s_w[m] = max(max_k |W[k,m]|, 1e-12)/q_w      W_q = clip(rint(W/s_w))
2 input DAC [HW]         V = 0.6·c/256,  c = X_q·2^(8-b_a) + 128 (signed) | X_q·2^(8-b_a) (AV)
                         V_IA = V - V_cm (0.3 V signed | 0 AV)  = (0.6/256)·2^(8-b_a)·X_q
3 C2C macro [HW]         V_OA = (1/32) Σ_j V_IA,j · W_q[j]/2^b_w = K_MAC · P_b,
                         P_b = Σ_{j in 32-row block b} X_q[j]·W_q[j]   (exact integer)
                         K_MAC = (0.6/256)·2^(8-b_a)/(32·2^b_w):  2.861e-7 (W8A8), 7.324e-5 (W4A4) V/unit
4 amplifier [ASSUMPTION] V_ADC_IN = G_op · V_OA   (ideal; one G per op shared by all 10 layers)
5 SAR ADC [HW: SAR, 0.6 V; DECIDED: 8 b]
                         d = clip(floor(V_ADC_IN/LSB_V + 1/2), -128, 127),  LSB_V = 0.6/256
                         ties resolve upward (guard 1e-9·V_REF); "clipped" = |V_ADC_IN| > 0.3 V
6 composition            d_b = clip(floor(P_b/LSB_P + 1/2), -128, 127)
                         LSB_P = LSB_V/(G·K_MAC) = 8192/G (W8A8) | 32/G (W4A4);  FS_op = 128·LSB_P
7 digital accumulation   D = Σ_b d_b (int);  Y = ((D·LSB_P)·s_x)·s_w  → float32; qkv adds its bias
```

`qat_calibrate.py scale-check` verifies `K_MAC`, `LSB_P`, `FS`, the DAC and the ADC against these
closed forms [VERIFIED-DEV: PASS for W8A8 and W4A4].

## 2. Facts, decisions and assumptions

**[HW]**
- capacitive input DAC, 8 b, V_REF 0.6 V;
- 32×32 C2C charge-domain macro with a charge-sharing column node, one SAR ADC per column,
  input range 0–0.6 V.

**[DECIDED]**
- ADC 8 b;
- W8A8, then W4A4;
- capacitive DAC only;
- one gain per op type;
- attention sites qkv / qk / av.

**[ASSUMPTION]** (unchanged; every result is conditional on these):
1. Offset-binary signed inputs with ideal common-mode removal.
2. A4 MSB-aligned.
3. Sign-magnitude weights (behavioural only).
4. Ideal linear amplifier (saturation not modelled; whether gains of about 10–30× are feasible is
   open).
5. `m = 32` also for zero-padded rows.
6. Digital accumulation across depth blocks.
7. All non-idealities off (an ideal upper bound).
8. Run-time max-scaling of dynamic operands.

In the ideal chain only `G·K_MAC` matters, so calibrated results are invariant to assumptions 1–2;
G = 1 results are not.

## 3. Components and roles

```
reference (forward truth)   cim_sim/ (unchanged) — calibration, CAL-check, training spot checks,
                            surrogate validation T1–T4, and EVERY reported number (qat_evaluate.py)
surrogate (gradient path)   cim_qat/surrogate.py + attention.py — training forward/backward and the
                            per-epoch VAL selection metric only
link                        test_surrogate_vs_simulator.py (T1–T4) + the per-epoch spot check
```

| file | role |
|---|---|
| `cim_qat/hw_config.py` | HW configuration. LSB_P and the tie guard are read from the simulator's `AnalogChain`/`SARADC`. Frozen-gains loader with a precision guard |
| `cim_qat/surrogate.py`, `attention.py` | the differentiable surrogate (custom autograd function) and the attention module mirroring `CIMAttention.forward` |
| `cim_qat/model.py` | builds through `run_tinyvit.build_model`; student, teacher, BN freezing; SHA-checked registry (pretrained, old INT8 / INT4) |
| `cim_qat/data.py` | listing of train (class folders), val (flat + `LOC_val_solution.csv`, or class folders) and ImageNetV2 (int folders); splits; exact accounting; content and real-path disjointness; V2 mapping report; **TEST guard + access log**; datasets |
| `cim_qat/calibration.py` | probe observation, candidate gains, diagnostics, CAL-check, flags, freeze |
| `cim_qat/sim_eval.py` | the actual simulator over images (batch 1, 1 thread, multiprocess, resumable); kinds `fp32`, `adc_off`, `analog`, `legacy_adc`; per-image input hashes and ADC counts |
| `cim_qat/trainer.py` | training core: full-VAL selection, early stopping, spot checks, resume |
| `cim_qat/metrics.py` | per-image metrics, ECE, McNemar, paired bootstrap |
| `qat_env_check.py` | environment and GPU numerics checks |
| `qat_prepare_data.py` | `inspect`, `make`, `add-test`, `show` |
| `qat_calibrate.py` | `scale-check`, `run`, `freeze` |
| `qat_train.py` | `--smoke-steps`, `--resume` |
| `qat_evaluate.py` | `fp32-check`, `v2-integrity`, `sim`, `report` |
| `qat_eval_matrix.py` | the §10 matrix per precision and split |
| `qat_audit.py` | integrity and leakage audit |
| tests | `test_surrogate_vs_simulator.py`, `test_training_mechanics.py`, `tests_qat/test_cim_qat_data.py`, `tests_qat/test_selection.py`, `tests_qat/test_device_placement.py`, `tests_qat/run_selftest.py` (end-to-end on fake data) |

**Training forward = surrogate only** (approved). Training through the NumPy simulator would cost
about 1.3–2.5 s per image per CPU core, i.e. hundreds of core-hours per epoch, for no gain: the
surrogate computes the same function.

## 4. Surrogate forward and its validation (unchanged from v1)

### 4.1 Implementation
- Operand quantization runs in float64 (`torch.round` = `np.rint`).
- `P_b` comes from a float32 matmul of integer-valued tensors, which is exact: |partial sums| ≤
  1,036,320 < 2^24 in any order, also under TF32.
- ADC rounding, floor and clip run in float64.
- The rescale is done in NumPy's order.
- Autocast is disabled inside.
- The histogram is deterministic and sort-based.

### 4.2 Tolerances
- Per matmul and per module: **0 differing elements**.
- Full model on CPU (batch 1, 1 thread, `cim_exp` input layout): 0 differing logits.
- GPU vs CPU: judged distributionally (§4.4).

### 4.3 Clip counter at exactly full scale (counter only)
At `|P_b| = FS` the simulator's volt path computes 0.30000000000000004 V and counts the conversion
as clipped.
- The output code is identical, so outputs are unaffected.
- The simulator is **not** changed.
- The validation accepts a counter difference only when it equals the counted exact-FS hits.
- A one-line guard in `SARADC.convert` is possible; whether to apply it is your decision.

### 4.4 Discretization avalanche
A 1-ulp input change moves the simulator's own output by KL ≈ 1.5–2e-3 on fake images
[VERIFIED-DEV].
- Bit-exactness therefore needs batch 1, one thread and the same input layout.
- GPU agreement is judged against that 1-ulp floor F: T4 passes if KL(sim ‖ GPU surrogate) ≤ 2F.
- Every final number uses one fixed implementation: `sim_eval`, CPU, batch 1.

### 4.5 Tiers
- T1: a 216-case grid.
- T1b: constructed boundary cases (exact ties, rounding ties, the FS edge, ragged K, unsigned AV,
  G = 300).
- T2: the 10 modules.
- T3: the full model.
- T4: GPU vs simulator on real VAL images.

## 5. Backward rules (STE) — unchanged

| # | forward | gradient |
|---|---|---|
| 1 | operand rounding | identity (STE) |
| 2 | operand clip | never active (max scaling) |
| 3 | scales | detached (hardware rule, not parameters) |
| 4 | ADC rounding | identity inside the code range |
| 5 | ADC saturation | zero where `floor(·)` is outside [−128, 127]; a conversion that rounds correctly onto a rail code passes the gradient (explicit, since torch's `clamp` would block it) |
| 6 | accumulation and rescale | exact |
| 7 | off-CIM ops | exact |

In range: `∂L/∂P_b = ∂L/∂Y · s_x·s_w ⊙ mask_b`. The custom backward equals an independent
autograd reference to 2.9e-7 (relative) [VERIFIED-DEV].

## 6. Preprocessing

- **Evaluation, every model and every split** (VAL, TEST, CALCHECK, spot checks): `cim_exp`
  `load_uint8` + `to_model_input`, imported and not re-implemented.
  - Short side resized to 256 with PIL **bicubic**, rounded centre crop 224, ImageNet mean/std.
  - Id: `cim_exp.imagenette.load_uint8+to_model_input|interp=bicubic|resize_short=256|crop=224|…`,
    recorded per evaluation.
- **Training:** `RandomResizedCrop(224, BICUBIC)` + flip, seeded per (seed, epoch, index).
- **Old QAT (for reference):** bilinear `RandomResizedCrop` for training; `Resize(256)` (bilinear)
  + `CenterCrop(224)` via `torchvision.datasets.ImageFolder` for validation [VERIFIED-SRC].
- The old checkpoints are evaluated **as they are**, under the common bicubic protocol. Their
  bilinear sensitivity is measured on VAL in FP32 by `fp32-check`. TEST is never used for that.

## 7. Training

| item | setting |
|---|---|
| start | `tinyvit5m.pth` (SHA `e4894bab…`, checked) |
| trainable | all parameters; BN statistics frozen; teacher frozen (KD) |
| loss | KD, forward KL, T = 1 (CE only in the pilot config) |
| optimizer / schedule | AdamW lr 1e-5, wd 1e-4; per-step linear warm-up 0.1 epoch, then cosine to 0 over the budget |
| batch / numerics | 64; fp32; TF32 off; deterministic; seed 0 |
| budget | W8A8 5 epochs, W4A4 10 epochs (pilots) |
| VAL evaluation | **full 50k VAL**, surrogate forward on the GPU, at step 0 (epoch-0 PTQ) and every epoch end. Logged: top-1/5, NLL, ECE, KL and agreement vs the teacher, per-op clip % and code use, student-FP32 drift; per-image arrays saved |
| simulator spot check | at the same points: 128 fixed VAL images. (a) CPU surrogate, batch 1 == simulator bit for bit (**the run stops if not**); (b) GPU surrogate vs simulator KL against the simulator's 1-ulp floor; (c) simulator clip % |
| **checkpoint selection** | **primary `best_model.pth`** = max VAL hardware top-1 (surrogate), ties → lower VAL NLL, among trained epochs (epoch 0 = PTQ is not a candidate, as in the old code). **Secondary `final_model.pth`** = the last checkpoint |
| **early stopping** | stop when `early_stopping_patience` consecutive epoch-end evaluations do not improve on the best (same improvement rule as selection): **W8A8 5** (never fires within its 5-epoch budget), **W4A4 3**, CE pilot 3 (§0.4 D1). The old code used strict `val_acc > best` with patience 8 [VERIFIED-SRC] |
| plateau / extension | reported at the end: converged iff the objective improved < 0.002 nats over the last epoch **and** \|Δtop-1\| ≤ 2 paired SE. A run that is not converged and not early-stopped → *recommend* a fresh run at 2× budget (your decision; never a continuation) |
| resume | `last.pth` at every evaluation; refuses a changed config, changed gains or a finished run; bit-exact [VERIFIED-DEV] |
| TEST | never opened (the `Splits` object has no TEST access; checked by the audit) |

Throughput, memory and h/epoch [REMOTE, smoke runs 26 Oct, one run at a time]:
- 0.669 s/step and 96 img/s for both precisions; data wait 0.000 s; peak GPU memory 8.12 GB;
- estimated 4.56 h/epoch: W8A8 ≈ 22.8 h for 5 epochs, W4A4 ≈ 45.5 h for 10.
- The per-epoch figure contains two full-VAL passes (hardware and student FP32) estimated at about
  25 min each. That estimate is extrapolated from 256 images, including DataLoader start-up, so it
  is probably high. The real figure is `eval_seconds` in `evals.jsonl` after the first evaluation.
For reference, the old QAT logged 105.7 h for 34 epochs ≈ 3.1 h/epoch [VERIFIED-SRC], with AMP and
a different forward.

## 8. Data protocol and leakage controls

### 8.1 Splits

| split | source | size | used for | never used for |
|---|---|---|---|---|
| CAL | ImageNet train, 2/class (seeded) | 2,000 | gain calibration (probe histograms) | training, selection |
| CALCHECK | ImageNet train, 1 in each of 200 seeded classes, disjoint from CAL | 200 | checking candidate gains on the simulator | training, selection |
| TRAIN | ImageNet train − CAL − CALCHECK − content duplicates of CAL/CALCHECK | on-disk count − 2,200 − k | gradient updates | — |
| VAL | **all** ImageNet-1k val, labels from `LOC_val_solution.csv` | 50,000 (verified on disk) | per-epoch monitoring, **checkpoint selection, early stopping**, CE-vs-KD decision, FP32 preprocessing check | calibration |
| TEST | **ImageNetV2 matched-frequency**, label = int(folder) | 10,000 (verified on disk) | final evaluation only | everything else |

- The seed is 20261022, and the file lists are sorted before any random draw.
- **Held-out eligibility rule (v2.1).** CAL/CALCHECK are drawn by walking each class's seeded
  permutation. A candidate is skipped (and stays in TRAIN) if its SHA-256 content equals:
  - any VAL image;
  - any TEST image;
  - an already selected CAL/CALCHECK image.

  The next eligible file in the permutation is taken instead.
  - The random draws themselves do not depend on the rule, so every class without a collision
    keeps exactly the files the plain draw gives.
  - Only file contents are compared: no labels, no model outputs.
  - Every skipped candidate is listed in `splits.json["heldout_skipped"]`.
  - Reason: ImageNet train contains exact copies of some val images (794 found on your disk).
    The v2 draw hit 3 of them for CAL, which the critical check `cal|val = 0` rejected
    [REMOTE, 25 Oct].
- `make` refuses to overwrite. `make --replace-failed` archives the outputs of a **failed** `make`
  to `splits/failed_<time>/` and draws again, reusing the hash caches. It refuses if:
  - the stored splits passed; or
  - the access log shows any use of them other than prepare/audit.
- The hash caches assume the dataset files never change. Each `make` re-hashes a seeded sample of
  200 cached entries and stops on any mismatch.
- Every consumer refuses a manifest whose SHA changed.

### 8.2 Verified dataset facts [REMOTE, from your listings]
- train: 1,000 class folders, 1,281,167 files at `.../Data/CLS-LOC/train/`;
- val: flat, 50,000 files;
- `LOC_val_solution.csv` present;
- `.../data/imagenet/val` contains symlinks (that copy is **not** used).

`qat_prepare_data.py inspect` re-counts the files at stage 5, and the splits use those counts.

### 8.3 Duplicate and overlap policy (requirement E)
- **Path level:** manifest keys and resolved real paths must be unique within and between all
  splits (FAIL otherwise).
- **Content level** (SHA-256). These must be zero (FAIL otherwise):
  - CAL∩CALCHECK, CAL∩VAL, CAL∩TEST, CALCHECK∩VAL, CALCHECK∩TEST, VAL∩TEST;
  - repeats inside CAL and inside CALCHECK;
  - TRAIN∩CAL and TRAIN∩CALCHECK, which hold by construction with `--hash-train`: duplicates are
    excluded from TRAIN and listed in `train_excluded.csv`.
- **Reported, not removed:** TRAIN∩VAL and TRAIN∩TEST exact duplicates.
  - Rationale: the pretrained model and the old QAT checkpoints were trained on the same full
    ImageNet train, so removing them from new QAT alone would make new and old differ in training
    data without making the teacher clean.
  - Any non-zero count is printed and must be stated with the results.
  - ImageNetV2's authors de-duplicated against ImageNet [PUBLIC]; near-duplicates cannot be
    detected by hashing.

### 8.4 TEST isolation (requirement E)
1. **Code guard.**
   - `cim_qat.data.Splits` raises `TestAccessError` for TEST unless it was created with
     `allow_test=True` and a TEST purpose.
   - Only `qat_evaluate.py sim/report --split test --final`, `v2-integrity` (mapping statistics,
     no accuracy), `prepare` and `audit` do that.
   - Training (`purpose=train`), calibration (`calibration`) and the checks cannot.
2. **Finished runs only.** A new-QAT checkpoint is evaluated on TEST only if its run has a
   `summary.json`, i.e. it finished and was selected on VAL.
   - `--limit` is refused on TEST.
   - The frozen gains must be the ones the run trained with.
3. **Access log.** Every split read by every process is appended to `access_log.jsonl`, with
   time, purpose, argv, pid and run name.
4. **Audit** (`qat_audit.py`). It inspects:
   - every training config and run config;
   - each run's `evals.jsonl` (every selection or early-stopping evaluation must carry
     `split = val` and the VAL manifest SHA);
   - `summary.json` (selection split, `test_split_used = false`);
   - the spot-check jobs (their item hash must equal the fixed VAL spot images);
   - `train.log`;
   - calibration reports, frozen gains and CAL-check jobs (CAL/CALCHECK manifests and items only);
   - the access log (TEST read only for final_eval / v2_integrity / prepare / audit, never by a
     training-like process);
   - TEST eval records (`--final`, no limit, run finished before its TEST evaluation).

   Activity after the first TEST read is listed as WARN for disclosure: a training run started,
   or gains frozen.
5. **Identical inputs.** Each simulator evaluation stores per-image input-tensor SHA-256, labels,
   indices, preprocessing id and manifest SHA. `report` and `audit` FAIL if any run in a split
   differs.

### 8.5 ImageNetV2 facts [PUBLIC] and checks
- Archive: `imagenetv2-matched-frequency.tar.gz` (about 1.2 GB) from the Hugging Face dataset
  `vaishaal/ImageNetV2`, the URL used by the TFDS builder.
- It extracts to `imagenetv2-matched-frequency-format-val/<0..999>/<hash>.jpeg`, 10 images per
  class.
- The label is the ImageNet-2012 class index = `int(folder)`, the same index order as our sorted
  synsets and TinyViT's head.
- No checksum is published: record your SHA-256.

Automated checks:
- **structure:** exactly the folders `0`…`999`, 10 images each, 10,000 total;
- **mapping:** label == int(folder); wnid == synsets[label]; the same 1000 classes as VAL; the
  synset list is sorted and unique;
- **trap test:** ImageFolder would mislabel 988 classes (D7);
- **empirical** (`v2-integrity`, pretrained FP32): the number of classes whose modal prediction
  equals the label (pass ≥ 700/1000), against the lexicographic alternative (must be ≤ 50). It
  prints these counts only, not accuracy.

## 9. Calibration (unchanged method)

1. scale check;
2. exact-probe |P_b| histograms on CAL, per op and per layer, in the actual simulator network;
3. P99.9 / 99.99 / 99.999 / 100 gains (shared per op, rounded to 0.01), with:
   - predicted clip % per op and per layer;
   - code use and SQNR;
   - the MSE-optimal FS;
4. CAL-check on CALCHECK with the analog simulator at the three levels + ADC-off + FP32;
5. pre-declared flags:
   - measured clip > 2× target;
   - another level lowers KL ≥ 10 % with a CI excluding 0;
   - MSE-optimal FS off by > 2×;
   - a layer's clip > 10× target;
   - median |code| ≤ 2;
6. `freeze` (refuses to overwrite).

Gains are frozen per precision, used unchanged by training and by every analog evaluation (old
and new checkpoints alike), and derived from CAL only.

## 10. Experiments, comparisons and attribution

### 10.1 Rows

All rows are evaluated **on VAL and on TEST** with the actual simulator (CPU, batch 1). P =
W8A8 or W4A4; OLD = old_int8 for W8A8, old_int4 for W4A4. Status of every row: **PENDING**.

| id | checkpoint (as delivered / as trained) | precision | hardware | gains (calibration source) | notes |
|---|---|---|---|---|---|
| E1 | pretrained `tinyvit5m.pth` | FP32 | no CIM | — | reference for KL/agreement |
| E2 | pretrained | P | operand quantization + tiling, **exact integer MAC, no ADC** (`adc_off`) | — | quantization-only floor |
| E3 | pretrained | P | analog chain, ADC 8 b | **G = 1** | PTQ, uncalibrated |
| E4 | pretrained | P | analog chain, ADC 8 b | **frozen calibrated** (CAL, P99.99 unless a flag led to another documented choice) | **PTQ baseline** |
| E4b | pretrained | W8A8 | analog chain | historical 23.77 / 8.41 / 2.92 (synthetic-input P99.99, CLAUDE.md §6) | W8A8 only |
| E5 | OLD (SHA-checked, unmodified) | P | analog chain | same frozen calibrated gains as E4 | old QAT on the new hardware |
| E5b | OLD | P | **legacy functional ADC 8 b** (`legacy_adc`) | — | old checkpoint at its own training point (diagnostic) |
| E6 | new QAT **best-VAL** (primary) | P | analog chain | same frozen gains (used in training) | `run:<name>:best` |
| E6s | new QAT final (secondary) | P | analog chain | same | `run:<name>:final` |
| E7 | W4A4 CE pilot best/final (optional) | W4A4 | analog chain | same | only if run |

Every analog row also reports each model's own FP32 output (`logits_own_fp32`), so the hardware
effect on *that* checkpoint, KL(own FP32 ‖ CIM), is available.

**Metrics per row:**
- top-1 (±SE), top-5, NLL, ECE;
- KL(FP32 ‖ row) and agreement vs the pretrained FP32 model;
- own-FP32 top-1 and KL(own FP32 ‖ row);
- per op (qkv / qk / av): conversions, clip % (hi / lo), rail %, ADC utilisation median and
  p99.9 (% of ±0.3 V), max;
- % of images with any saturation.

### 10.2 Pre-declared paired comparisons (same images; Δ = b − a)

| id | a → b | isolates | can be attributed to | cannot be attributed to |
|---|---|---|---|---|
| C1 | E1 → E2 | operand quantization | per-token/per-column quantization at P | ADC |
| C2 | E2 → E3 | ADC + analog chain at G = 1 | the 8-b ADC with unscaled swing | — |
| C3 | E2 → E4 | ADC + analog chain at calibrated gains | the 8-b ADC at the chosen gains | gain choice alone |
| C4 | E3 → E4 | gain calibration | the calibrated gains (CAL-derived) | — |
| C4b | E4b → E4 | real-image calibration vs historical synthetic gains | calibration data | — |
| C5 | E4 → E5 | old QAT vs PTQ | old QAT fine-tuning *as transferred to this hardware* | a training effect specific to this hardware (old QAT targeted the legacy ADC) |
| C6 | E4 → E6 | **new QAT vs PTQ** | hardware-aware QAT with this surrogate (same gains, same images) | — |
| C7 | E5 → E6 | new vs old pipeline | the **bundle** of differences: ADC model trained against, loss (KD vs CE), BN, numerics, preprocessing, budget, selection rule | any single one of them |
| C7b | E6 → E6s | selection | last vs VAL-selected checkpoint | — |
| C8 | E5b → E5 | old checkpoint: training point → new hardware | the hardware change for a fixed model | — |
| C9 | E6 → E7 | CE vs KD (W4A4, identical otherwise) | the loss | — (VAL decides; TEST only reported) |

**Generalisation:**
- Every comparison is computed on VAL and on TEST **separately**.
- TEST comparisons are the clean ones.
- On VAL, rows E5/E6/E7 are selection-biased.
- VAL and TEST absolute accuracies are **never** compared with each other: they are different
  distributions, and ImageNetV2 is known to be harder [PUBLIC].
- What *is* meaningful: whether a paired Δ keeps its sign and size from VAL to TEST.

CE-vs-KD decision (VAL only, pre-declared): keep KD unless CE is better on VAL top-1 with
McNemar p < 0.05 and ≥ 0.3 pp. If CE is better, report back before changing anything.

**Cost:** the simulator takes about 1.3–2.5 s per image per core on the sandbox. Measure your
s/image at stage 7. Per row: VAL ≈ 50k × t / workers; TEST ≈ 10k × t / workers.

## 11. Differences from the old QAT (summary)

| aspect | old [VERIFIED-SRC] | new |
|---|---|---|
| ADC target | legacy functional ADC (worst-case FS, mid-rise, LSB 2FS/(2^N−1)) | analog chain: per-op calibrated gain, mid-tread, LSB V_REF/2^N |
| loss | CE | KD (T = 1) |
| BN | running statistics updated | frozen |
| numerics | fp16 AMP | fp32, TF32 off, deterministic |
| preprocessing | bilinear | bicubic (`cim_exp`) |
| selection | best per-epoch val_acc (strict >), full 50k val, training forward | best per-epoch VAL hardware top-1 (surrogate; ties → NLL), full 50k VAL |
| early stopping | patience 8, budget 40 epochs, cosine T_max = 40 per epoch | patience 5 (W8A8, = budget) / 3 (W4A4), pilot budgets 5 / 10, per-step warm-up + cosine |
| independent test | none | ImageNetV2 |
| verification | — | 4 validation tiers + per-epoch simulator spot check; all reported numbers from the simulator |

## 12. Historical context [VERIFIED-SRC: delivered logs, configs and checkpoints]

| | INT8 (W8A8) | INT4 (W4A4) |
|---|---|---|
| config | 40-epoch budget, patience 8, batch 64, lr 1e-5, wd 1e-4, AMP, legacy ADC 8 b, seed 0 | same, W4A4 |
| completed epochs | 34 (`metrics.csv` rows 0–33); stopped by early stopping | **28** (rows 0–27); stopped manually during epoch 29 (log ends mid-epoch) |
| best val_acc (old forward) | **80.11 %** at epoch 26 (row 25) | **79.354 %** at epoch 22 (row 21) |
| wall time | 105.7 h (≈ 3.1 h/epoch), RTX 4000 Ada | — |
| delivered checkpoint | `best_model_int8_qat.pth`, SHA `48f0e484…` | `best_model_int4_qat.pth`, SHA `a903663b…` |

These are **historical** results of a different forward and are not reproduced or re-claimed by
this pipeline.

## 13. Evidence actually run (development sandbox, CPU, fake images) [VERIFIED-DEV]

All of the following were run on 25 Oct 2026 **from the extracted delivery zip**, in a folder
whose path contains a space (`.../DATA 1/...`). Fake images carry no class information, so no
accuracy, gain or timing figure from them means anything for the thesis.

| test | result |
|---|---|
| `python -m compileall` (all code) | OK |
| simulator suites (unchanged simulator) | verify_array: all checks passed; test_analog 98/98; test_qat_contract 17/17; test_multi_array 42/42; test_experiment_pipeline 71/71 |
| pre-existing files vs your tree | 46 compared, 0 differ (your Windows `.venv` and `results/` were not packaged) |
| `qat_calibrate.py scale-check` W8A8 / W4A4 | PASS / PASS (K_MAC 2.861023e-07 / 7.324219e-05; LSB_P 8192 / 32) |
| `test_surrogate_vs_simulator.py` CPU (T1, T1b, T2, T3 synthetic) | 39/39, 0 exact-tier failures |
| `tests_qat/test_cim_qat_data.py` | 39/39 (incl. D14 eligibility rule, D15 replace-failed and stale-cache check) |
| `tests_qat/run_selftest.py` (end-to-end, as subprocesses) | **101/101** in 23.3 min (v2.8 build, 29 Oct; adds the freeze guards, the two extension variants with planted duplicates, the pooled report and audit section X incl. a planted frozen-file change; v2.7: 75/75 in 19.0 min, adds `tests_qat/test_stats_merge.py` 12/12, the D2 rows with C10, the † flag and `restat`; v2.6: 66/66 with `tests_qat/test_threads.py` 14/14 and the per-worker BLAS-thread check; v2.5: 64/64; v2.2–v2.4: 58/58; v2.0: 57/57). Covers: the D2 fork (`tests_qat/test_fork.py` 7/7: fork bit-identical to a never-early-stopped run, refusals, history and source checks) and the D2 selection; device placement; inspect; make --hash-train; exact accounting; refusal to overwrite; mini splits; calibrate + freeze; surrogate incl. the T4 code path; mechanics; selection rules 6/6; smoke; training A; training B killed and resumed (final weights bit-identical to A); refusals of a finished run and of three TEST misuse attempts (no TEST read logged); the 9-row matrix on VAL and TEST; reports with the identical-input assertion (and its deliberate failure); v2-integrity (fails on fake labels, as it must); audit PASS; four planted leaks each caught as FAIL |

Not run here: anything on CUDA, anything on real ImageNet / ImageNetV2.

**v2.2 fix (26 Oct), found by stage 9 on the GPU [REMOTE]: teacher device placement.**
- Upstream TinyViT's `Attention` caches the attention-bias table as a plain attribute `ab` when it
  is put in eval mode. `model.to(cuda)` moves parameters and buffers only, so the teacher (built and
  put in eval mode on the CPU, then moved) kept a CPU `ab`.
- The first teacher forward on the GPU failed ("found at least two devices", `tiny_vit.py:278`).
- It could not appear in the CPU-only sandbox.
- It would also have stopped `qat_train.py --smoke-steps` (first training step) and any resumed KD
  run. A fresh run escaped by luck: its step-0 evaluation calls `.eval()`, which rebuilds `ab`.
- Fix: `cim_qat.model.to_device` re-derives `ab` on the target device, identically to
  `Attention.train(False)`, and `assert_on_device` checks every parameter, buffer and plain tensor
  attribute. The trainer and the env-check benchmark use it; `tiny_vit.py` is unchanged.
- Pinned by `tests_qat/test_device_placement.py` (4/4; uses the `meta` device to reproduce the bug
  without a GPU) and by the new mechanics check C0.

**v2.3 fix (26 Oct), found by the stage 9 re-run on the GPU [REMOTE]: resume RNG restore.**
- `torch.load(checkpoint, map_location=cuda)` also moves the saved RNG states (CPU `uint8`
  tensors) to the GPU.
- `torch.set_rng_state` then raises "RNG state must be a torch.ByteTensor". This failed mechanics
  C8 and would have failed every `qat_train.py --resume` on the GPU.
- The CUDA generator states had the same exposure: `torch.cuda.set_rng_state` also needs a CPU
  `uint8` tensor.
- Fix:
  - `trainer.restore_rng` converts every saved state back to a contiguous CPU `uint8` tensor
    before restoring the python, numpy, torch-CPU and torch-CUDA generators;
  - `qat_train.py --resume` (and C8) load checkpoints with `map_location="cpu"`. This is correct
    for everything else in the checkpoint: `load_state_dict` copies weights onto the model's
    device, and the optimizer moves its state to each parameter's device.
- Pinned by `test_device_placement.py` P5 (it reproduces the TypeError on CPU via a non-uint8
  state) and by mechanics C8b (resume with `map_location=<GPU>` must still be bit-exact).


**v2.6 (28 Oct), found during the stage 11 VAL matrix [REMOTE]: BLAS threads in the simulator workers.**
- The simulator's float64 `v_ia @ effective` GEMM ran on NumPy's multi-threaded OpenBLAS
  (scipy-openblas 0.3.31, up to 64 threads). `configure_threads(1)` set the thread variables only
  after NumPy had loaded, which has no effect, so each of the 16 workers ran ~32 BLAS threads
  (load average ≈ 321 reported).
- A/B on the remote: 64/64 logits bit-identical with 1 thread; 1.05 s/img/worker. **Speed only;
  no result changes**, so E1–E3 computed before the fix stand.
- Fix (no simulator file touched; `cim_sim/` fingerprint unchanged):
  - the evaluation / calibration entry points set the thread variables before NumPy loads;
  - every simulator and calibration worker also limits the running BLAS pool to 1 thread
    (threadpoolctl or the OpenBLAS/MKL C API) and records the count (`blas_threads_per_worker`
    in each evaluation `summary.json`); the caller's environment is restored after the pool;
  - workers request `PR_SET_PDEATHSIG`, so they end when their parent dies;
  - `qat_stop.py` stops evaluation jobs including spawn workers that `pkill -f` misses, and never
    touches training processes.
- Pinned by `tests_qat/test_threads.py` (H1–H6, 14/14 from the extracted v2.6 zip; the H2 and H4 checks start from 2 BLAS threads, so they fail if the limit does not act) and a self-test check
  that every VAL simulator run recorded `[1]`.

**v2.7 (29 Oct), found on the VAL reports [REMOTE]: percentile statistics of chunked simulator runs.**
- Symptom: W4A4 E6 and E6s have bit-identical logits, but their ADC-utilisation median / p99.9
  columns differed (13.5/17.9/1.4 vs 5.7/13.5/4.8 %).
- Cause: cim_sim `AnalogStats.merge()` appends a chunk's subsample (kept at the chunk's own rate)
  to the running one and halves everything whenever the cap is exceeded. After ~1,000 chunks nearly
  all weight sits on a few surviving samples; expected weights are right, realised ones are not
  (synthetic: realised Σw/n = 0.06, one sample holds 32 % of the weight).
- Scope: only the sampled percentiles (util median / p99.9, V_OA / V_ADC_IN percentiles). Logits,
  accuracy metrics, clip / rail / saturation counts, util max and the frozen gains (exact histograms)
  are unaffected. `cim_exp` (Imagenette Experiments A/B) uses the same merge.
- Fix without touching cim_sim: `cim_qat.analog_merge` brings every chunk's subsample to one common
  keep fraction (independent random streams per chunk and op), then summarises with the simulator's
  own `AnalogStats.summary()`. New runs use it in `sim_eval.merge`; finished runs are recomputed from
  their chunk files by `qat_evaluate.py restat`; reports mark rows still on the old merge with †.
- Pinned by `tests_qat/test_stats_merge.py` (M1–M4): uniform re-merge within ±1.2 % (median) and
  ±4.1 % (p99.9) of the exact values for three worker layouts while the old merge is off by up to 58 %;
  identical exact scalars; real 2- vs 3-worker runs identical in logits and counts.
- Also v2.7: comparison **C10** (fork best vs pilot best, D2) in the report when both rows exist,
  declared 29 Oct before any TEST evaluation of a trained model; descriptive only.

**v2.8 (29 Oct): the ImageNetV2 extension (D3) and write protection of the frozen results.**
- `cim_qat/data.py`: two more guarded TEST splits (`test_th07`, `test_top`) registered by
  `qat_prepare_data.py add-test-variant` in `splits/extra_tests.json` with their manifests, content
  hashes, the leakage check against CAL/CALCHECK/VAL/TRAIN (exclusion rule D3), the overlap with the
  matched-frequency TEST and with each other, and the label check (`label = int(folder)`).
- Freeze protection: any file listed in a `FINAL_SHA256SUMS_*.txt` is never overwritten
  (`assert_not_frozen`: reports, restat, v2-integrity, evaluation folders); a finished evaluation row is
  skipped without writing or logging anything; once `access_log.jsonl` is frozen, new entries go to
  `access_log_ext.jsonl` (the audit and `select-d2` read all logs).
- `qat_evaluate.py`: `--split test_th07|test_top` for `sim`, `report`, `restat`, `v2-integrity`
  (own output file); new `report-pooled` (the three variants combined at report time, distinct images
  once, duplicate outputs checked, P1 post hoc).
- `qat_audit.py` section X: X0 every frozen file unchanged; X1 manifests and counts; X2 labels;
  X3 no leakage (+ exclusions disclosed); X4 path and content overlaps; X5 access purposes and logs;
  X6 every extension row is a `--final` full-set evaluation of the registered manifest with a frozen
  checkpoint; X7 evaluation after the freeze disclosed.
- Pinned by the self-test (fake data): freeze guards, registration with planted duplicates, evaluation
  in new folders, the pooled report (distinct images, identical duplicate outputs, P1), audit PASS, and
  an audit FAIL when a frozen file is changed.

**v2.8.1 (2 Nov): audit check X5.<split>.log corrected (false FAIL); no result or frozen file changed.**
- Found by the audit after the third freeze (`AUDIT_20261102T105922+0000`: 212 PASS, 6 WARN, **2 FAIL**,
  `X5.test_th07.log` and `X5.test_top.log`, 18 each).
- Cause, a defect in the check, not in the data: it flagged every extension read stored in a log that
  is frozen *now* and dated after the variant's registration. The extension freeze (runbook 13.6) lists
  `access_log_ext.jsonl` itself, so all legitimate reads written **before** that freeze counted as
  "appended to a frozen log". The v2.8 self-test missed it because it never froze the extension log.
- Fix: a read is a violation only if it is dated after the freeze of the log that holds it, or that
  freeze no longer matches (X0). Pinned by two new self-test checks: freezing the extension log after
  its reads gives PASS, and a read appended afterwards gives FAIL (X0 and X5). Self-test 105/105 from the
  extracted v2.8.1 zip (all other suites unchanged and passing).
- `qat_audit.py` is the only code change. The add-on `tools_qat/own_fp32_posthoc.py` is included
  byte-identical.
- **Outcome (17 Nov, run by the user):** with v2.8.1 the audit gives 214 PASS, 6 WARN, 0 FAIL
  (`AUDIT_20261117T170653+0000`). `X5.test_th07.log` and `X5.test_top.log` PASS, so the 2 Nov FAILs were
  false; X0 PASS for all three freezes; same 6 WARNs as on 1 Nov. Fourth freeze
  `FINAL_SHA256SUMS_20261102_audit.txt`. Details: QAT_EXPERIMENT_LOG.md §8.6.

## 14. Risks and open items
1. ImageNetV2 download (§0.3.1).
2. HDD I/O: training data loading and VAL/TEST evaluation speed are unmeasured (stage 7).
3. GPU throughput of the float64 per-block ops is unmeasured (stage 7).
4. W8A8 may show little measurable headroom; W4A4 is where QAT should matter.
5. Early-stopping patience was decided per run (§0.4 D1); W8A8 cannot stop early by design.
6. All results are conditional on the §2 assumptions.
7. The optional §4.3 counter guard awaits your decision.

## Sources
- [ImageNetV2 GitHub (modestyachts/ImageNetV2)](https://github.com/modestyachts/ImageNetV2)
- [TFDS catalog: imagenet_v2](https://www.tensorflow.org/datasets/catalog/imagenet_v2)
- [TFDS builder source (imagenet_v2)](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/image_classification/imagenet_v2.py)
- [Hugging Face dataset vaishaal/ImageNetV2](https://huggingface.co/datasets/vaishaal/ImageNetV2)
