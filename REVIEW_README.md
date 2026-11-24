# Hardware-aware QAT of TinyViT-5M for an analog CIM accelerator — review bundle (17 Nov 2026)

This bundle has three parts:

1. **`CIM-SOC-QAT-v2.8.1-review.zip`**: the code (v2.8.1) and documentation, with copies of the
   extension reports and audit in `results_extension/`, of the post hoc analysis in `results_posthoc/`,
   and of the latest audit with its freeze in `results_audit/`.
2. **`qat_review_results_20261029.tar.gz`**: the records of the main runs, copied from the GPU machine
   (`/media/user/DATA 1/cim_qat_work`).
3. **`qat_review_results_ext_20261101.tar.gz`**: the records of the extension (two more ImageNetV2
   variants), copied from the same machine.

## Where to start

| read | what it gives you |
|---|---|
| `RESULTS_SUMMARY.md` | Final VAL and ImageNetV2 TEST tables, the extension to all three ImageNetV2 variants, and all paired comparisons, with caveats. |
| `QAT_DESIGN.md` | The design and protocol: what is trained, against which hardware model, and how the data are split. §0 lists every deviation decided during the work (D1 patience, D2 fork, D3 the TEST extension). §13 lists every code change (v2.1–v2.8) with its cause and its test. |
| `QAT_EXPERIMENT_LOG.md` | A factual record of everything that ran, in order, with dates and hashes, failures included. |
| `QAT_RUNBOOK.md` | The exact commands for stages 1–13, as they were run. |
| results `audit/AUDIT_20261029T065916+0000.md` | The final audit of the main results: 145 PASS, 3 WARN (informational duplicate-image notices), 0 FAIL. |
| `results_posthoc/own_fp32_posthoc.md` | Post hoc (2 Nov): every trained model against its own float version, from the stored outputs only. |
| `results_extension/AUDIT_20261101T151755+0000.md` | The audit after the extension: 212 PASS, 6 WARN (the 3 notices, the overlap between the variants, and "evaluated after the freeze"), 0 FAIL. It also confirms all 673 frozen files are unchanged. |
| `results_audit/AUDIT_20261117T170653+0000.md` | **The latest audit (17 Nov, v2.8.1): 214 PASS, 6 WARN (the same six), 0 FAIL.** All three earlier freezes (673, 316 and 6 files) are unchanged. An audit run on 2 Nov with v2.8 had reported 2 FAILs; they were a defect in that check, not in the data, explained in `results_audit/AUDIT_20261102T105922+0000.NOTE.md` and in `QAT_DESIGN.md` §13 (v2.8.1). |

## Protocol in one paragraph

- **Model and hardware.** TinyViT-5M was fine-tuned with knowledge distillation against a
  differentiable surrogate. The surrogate is bit-exact to the `cim_sim` analog simulator: 8-bit
  capacitive DAC, 32×32 C2C macro, per-operation pre-ADC gain, 8-bit SAR ADC. Precisions are W8A8
  and W4A4.
- **Data splits.**
  - CAL (2,000) and CALCHECK (200), drawn from ImageNet train: gain calibration, frozen at P99.99.
  - TRAIN: the rest of ImageNet train.
  - VAL: the full 50,000-image ImageNet val, used for checkpoint selection and early stopping.
  - TEST: ImageNetV2 matched-frequency (10,000 images), read only for the final evaluation.
    `access_log.jsonl` logs every read of every split, and `qat_audit.py` checks the log.
  - Extension (requested after the results were frozen, descriptive only): ImageNetV2 threshold-0.7
    and top-images, 10,000 each, evaluated with the same frozen models and gains in their own folders,
    and pooled with the original 10k at report time (20,683 distinct images, because the three
    variants share images). The original 10k stays the primary result.
- **Evaluation.** Every row was evaluated with the actual simulator (batch 1, CPU) on identical
  inputs. Rows: pretrained FP32, PTQ, the old INT8/INT4 QAT checkpoints (unchanged), and the new
  QAT runs.

## Which code version produced what

Each result folder's `provenance.json` records the exact code fingerprint. The simulator
fingerprint (`cim_sim/` + `tiny_vit.py`) is `cea8950c…` in every version from v2 to v2.8; no
hardware model was changed. v2.1–v2.8 changed tooling, not the hardware model (QAT_DESIGN.md §13).

| stage | version |
|---|---|
| splits, calibration (25 Oct) | v2.1 |
| surrogate validation and training mechanics (stage 9) | v2.1–v2.3 (mechanics completed with v2.3) |
| training: `w8a8_kd_pilot5ep`, `w4a4_kd_pilot10ep` | v2.4 |
| D2 fork `w4a4_kd_pilot10ep_full` | v2.5 |
| VAL matrix: W8A8 rows and W4A4 E1–E5b | v2.5 (BLAS threads limited by exported environment variables) |
| `select-d2`; W4A4 E6/E6s and the pilot diagnostic on VAL; all TEST rows | v2.6 |
| `restat`, the four reports, final audit of the main results | v2.7 |
| `tools_qat/` read-only checks (overlap sensitivity, pair comparison) | v2.7 add-on, included in v2.8 |
| extension D3: registration, evaluation, reports, audit, second freeze | v2.8 |
| post hoc own-FP32 decomposition (stored logits only), third freeze | `tools_qat/own_fp32_posthoc.py`, a read-only add-on to v2.8 |
| audit check corrected; audit re-run (17 Nov), fourth freeze | v2.8.1 (this zip; only `qat_audit.py` changed, and the add-on is included unchanged) |

The code here is byte-identical to the delivered v2.8.1. `QAT_DESIGN.md`, `QAT_EXPERIMENT_LOG.md` and
`QAT_RUNBOOK.md` were updated afterwards with the 17 Nov audit result. This README, `RESULTS_SUMMARY.md`,
`results_extension/`, `results_posthoc/` and `results_audit/` were also added.

## Honest caveats (also in the docs)

- **One training seed per configuration.** Confidence intervals are over images, not over
  training runs.
- **Selection on VAL.** VAL selected the checkpoints and the D2 winner, so the conclusions come
  from TEST.
- **Different distribution.** ImageNetV2 differs from ImageNet val: compare within a table,
  never across.
- **Train/VAL overlap.** 799 VAL images also occur in TRAIN. Measured: this lowers VAL top-1 by
  about 0.7 pp for every row and moves comparisons by ≤ 0.07 pp
  (`eval/val/SENSITIVITY_train_overlap.md`).
- **Simulator assumptions.** The simulator is ideal (no nonidealities). The handling of signed
  inputs (offset binary) is a provisional assumption, not a hardware fact.
- **Extension after the freeze.** The two extra ImageNetV2 variants were added on request after the
  results were frozen and seen. They are descriptive. The comparison "new QAT vs FP32" (P1) was chosen
  after seeing the results, so it is labelled post hoc. So is the own-FP32 decomposition (`results_posthoc/`).
- **Short budgets.** The budgets were 5 and 10 epochs. The W4A4 pilot early-stopped at epoch 5;
  under decision D2, declared beforehand, it was continued to 10 epochs as a fork, and the fork
  is the reported W4A4 run.
- **ADC-utilisation percentiles** were recomputed after a defect in the simulator's statistics
  merge (QAT_DESIGN.md §13, v2.7). Accuracy, logits and clip counts were never affected.

## Verify

- **The records:** `sha256sum -c --ignore-missing FINAL_SHA256SUMS_20261029.txt` inside the main results
  folder, and `sha256sum -c --ignore-missing FINAL_SHA256SUMS_20261030_ext.txt` inside the extension
  results folder. The post hoc files have their checksums in `results_posthoc/SHA256SUMS.txt`. The latest
  audit and the note are listed in `results_audit/FINAL_SHA256SUMS_20261102_audit.txt` (written on 17 Nov;
  the name keeps the date planned in the runbook). Paths in that file are relative to the work folder on
  the GPU machine, so check it there, or check the two copies here by hash.
- **The code tests:** run the stage-4 commands in QAT_RUNBOOK.md, or run
  `python tests_qat/run_selftest.py` (CPU, fake data, about 23 min, expects 105/105).
- **Re-derive a paired comparison:**
  `python tools_qat/pair_compare.py --paths configs/paths.json --split test --a <run> --b <run>`.
  This needs `result.npz` files, which are not in the results bundle because of their size.
