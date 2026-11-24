# QAT_RUNBOOK.md — complete ordered execution guide (protocol v2, package v2.8, 29 Oct 2026)

Start from zero: nothing below has been executed on your machine. Run the stages **in order**.
Do not start a stage before the previous one has passed. If a check fails, stop, keep the output,
and report it before going on.

Conventions:
- Every command is meant to be pasted into a terminal on the GPU machine
  (`confluence-System-Product-Name`).
- Paths contain spaces (`DATA 1`, `Deva QAT`) and are always quoted.
- `$QAT_WORK` and `$QAT_CODE` are set in stage 2 by `qat_env.sh`.
- Long jobs run inside `tmux` (stage 2.4) so that closing VS Code does not kill them.

| stage | what | GPU? | typical duration |
|---|---|---|---|
| 1 | inspect machine, GPU, disk, dataset | — | minutes |
| 2 | environment | — | minutes |
| 3 | obtain and inspect the package | — | minutes |
| 4 | CPU tests, static checks, GPU numerics check | brief | 15–40 min |
| 5 | dataset verification (+ ImageNetV2 download) | — | 10–30 min + download |
| 6 | split manifests (+ hashing), mapping check, audit | brief | 0.5–3 h (HDD) |
| 7 | real-image smoke tests (FP32 on VAL, simulator speed) | yes | ~1 h |
| 8 | gain calibration + CALCHECK, per precision | — | measured |
| 9 | simulator-vs-surrogate + mechanics + selection rules | yes | ~1 h per precision |
| 10 | GPU QAT (smoke → W8A8 → W4A4 [→ CE pilot]) | yes | measured in 10.1 |
| 11 | evaluation on VAL, then on ImageNetV2 (TEST) | — | measured in 7.2 |
| 12 | reports and final audit | — | minutes |

---

## Stage 1 — inspect the machine, GPU, disk and dataset (read-only)

```bash
hostname; whoami; date
nvidia-smi
nvidia-smi --query-gpu=index,name,pci.bus_id,memory.used,memory.total,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
nproc; free -g; uptime
df -h / "/media/user/DATA 1"
lsblk -o NAME,SIZE,ROTA,MOUNTPOINT
ls -la "/media/user/DATA 1/Deva QAT/data/raw/"
ls "/media/user/DATA 1/Deva QAT/data/raw/ILSVRC/Data/CLS-LOC/"
```

Counts. `-L` follows symlinks, so symlinked files are counted. These `find` runs take a few
minutes on the HDD.

```bash
R="/media/user/DATA 1/Deva QAT/data/raw"
find -L "$R/ILSVRC/Data/CLS-LOC/train" -mindepth 1 -maxdepth 1 -type d | wc -l
find -L "$R/ILSVRC/Data/CLS-LOC/train" -mindepth 2 -maxdepth 2 -type f | wc -l
find -L "$R/ILSVRC/Data/CLS-LOC/val" -maxdepth 1 -type f | wc -l
find "$R/ILSVRC/Data/CLS-LOC/val" -maxdepth 1 -type l | wc -l      # how many are symlinks
wc -l "$R/LOC_val_solution.csv"
head -3 "$R/LOC_val_solution.csv"
```

Work root: does it exist or can it be created? Is it writable, with enough space?

```bash
mkdir -p "/media/user/DATA 1/cim_qat_work" && touch "/media/user/DATA 1/cim_qat_work/.w" \
  && rm "/media/user/DATA 1/cim_qat_work/.w" && echo "work root writable"
df -h "/media/user/DATA 1/cim_qat_work"
```

**Expected:**
- two RTX 4000 Ada GPUs; pick the idle one (GPU 1 in your last diagnostics);
- 1000 train class folders;
- train file count ≈ 1,281,167 (the pipeline uses whatever is actually on disk);
- 50000 val files;
- `wc -l` of the CSV = 50001 (header + 50,000 rows), header `ImageId,PredictionString`;
- "work root writable" and several TB free.

**If it fails:**
- different counts: write them down. Stage 5 counts again, and the splits use the on-disk counts;
- val count 0 with plain `find -type f`: you forgot `-L` (symlinks);
- not writable: check `ls -ld "/media/user/DATA 1"` and the mount options (`mount | grep "DATA 1"`).

To watch the GPUs later: `watch -n 2 nvidia-smi`, or
`nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv -l 5`.

---

## Stage 2 — environment

### 2.1 Activate the existing environment and check versions

```bash
conda activate cim_soc_qat
python -c "import sys, torch, torchvision, numpy, PIL; print(sys.version.split()[0], torch.__version__, torchvision.__version__, numpy.__version__, PIL.__version__, torch.cuda.is_available())"
```

**Expected:** `3.11.16 2.14.1+cu130 0.29.1 2.4.6 12.3.0 True`. Nothing needs installing.

**If it fails:**
- `conda: command not found`: run `source ~/miniconda3/etc/profile.d/conda.sh` or
  `source ~/anaconda3/etc/profile.d/conda.sh` first. `conda info --envs` lists the environments.
- versions differ: report them; do not upgrade the PhD scholar's environment.
- If a package is missing, use `pip install -r requirements_qat.txt` **only after asking**.

### 2.2 (after stage 3) `qat_env.sh`

The package contains `qat_env.sh`. It sets:
- `QAT_WORK` and `QAT_CODE`;
- `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=1` (GPU 1);
- `TMPDIR` on the data disk (`/` is 99 % full);
- `CUBLAS_WORKSPACE_CONFIG`.

It then `cd`s into the code. You run `source qat_env.sh` in **every new terminal and tmux
window**.

To use GPU 0 instead: `export CUDA_VISIBLE_DEVICES=0` **before** sourcing.

### 2.3 Check tmux

```bash
which tmux || echo "no tmux: use the nohup form given in stage 10"
```

### 2.4 tmux basics

| action | command |
|---|---|
| new session | `tmux new -s qat` |
| detach | `Ctrl-b` then `d` |
| re-attach | `tmux attach -t qat` |
| list sessions | `tmux ls` |

---

## Stage 3 — obtain and inspect the package

1. Download `CIM-SOC-QAT-v2.5.zip` (or later) from this chat to your laptop.
2. Copy it to the machine. Either:
   - drag it into VS Code's Explorer pane, onto the folder `/media/user/DATA 1/cim_qat_work`; or
   - run, from your laptop:

     ```bash
     scp CIM-SOC-QAT-v2.5.zip user@10.1.19.79:"/media/user/DATA 1/cim_qat_work/"
     ```

     (replace `user` if your login differs).
3. Unpack and verify it on the machine:

```bash
cd "/media/user/DATA 1/cim_qat_work"
sha256sum CIM-SOC-QAT-v2.5.zip            # must equal the SHA-256 given with the package
mkdir -p code && cd code
unzip -q ../CIM-SOC-QAT-v2.5.zip        # or: python -m zipfile -e ../CIM-SOC-QAT-v2.5.zip .
cd CIM-SOC-QAT
sha256sum -c --quiet PACKAGE_SHA256SUMS.txt && echo "ALL FILES OK"
source qat_env.sh
git status | head -5                    # branch qat-analog-surrogate; nothing committed
git log --oneline -1                    # your original HEAD commit (unchanged)
ls
```

**Expected:**
- the zip SHA matches;
- `ALL FILES OK`;
- `git status` shows branch `qat-analog-surrogate`, with your earlier working-copy
  modifications and the new QAT files as uncommitted changes (nothing was committed or pushed);
- the prompt is in `.../code/CIM-SOC-QAT`.

**If it fails:**
- SHA mismatch: re-download;
- `unzip` missing: use the `python -m zipfile` form;
- any `FAILED` line: re-extract into an empty folder.

Do **not** run `git commit` or `git push`.

---

## Stage 4 — CPU tests, static checks, GPU numerics

```bash
source qat_env.sh
python -m compileall -q cim_sim cim_exp cim_qat qat_*.py test_*.py tests_qat && echo "COMPILE OK"
python verify_array.py | tail -3
python test_analog.py | tail -2
python test_qat_contract.py | tail -2
python test_multi_array.py | tail -2
python test_experiment_pipeline.py | tail -2
python qat_calibrate.py scale-check --precision W8A8
python qat_calibrate.py scale-check --precision W4A4
python tests_qat/test_cim_qat_data.py | tail -3
python tests_qat/test_device_placement.py | tail -1
python tests_qat/run_selftest.py --workers 8 2>&1 | tee "$QAT_WORK/selftest.log" | grep -E "^\[|SELFTEST"
python qat_env_check.py --device cuda --out "$QAT_WORK/checks/env_check.json"
```

**Expected:**
- `COMPILE OK`;
- `verify_array` PASS;
- `test_analog` 98/98, `test_qat_contract` 17/17, `test_multi_array` 42/42,
  `test_experiment_pipeline` 71/71;
- both scale checks print `[PASS]` for qkv, qk and av:
  - W8A8: K_MAC 2.8610e-07, LSB_P 8192;
  - W4A4: K_MAC 7.3242e-05, LSB_P 32;
- data tests 39/39;
- the self-test ends `SELFTEST: N/N passed` (10–30 min; fake data, so accuracies in its logs are
  meaningless);
- `qat_env_check`:
  - `CUDA float32 integer matmul exact` for TF32 on and off;
  - deterministic conv runs;
  - one GPU visible;
  - its UUID matches the nvidia-smi line of GPU 1;
  - ends `... 0 problem(s)`.

**If it fails:**
- simulator suites: the tree is not the delivered one (re-extract);
- the self-test: the failing line is printed with its last 40 output lines. Keep
  `"$QAT_WORK/selftest.log"`;
- `env_check` "exactly one GPU visible": `CUDA_VISIBLE_DEVICES` was not exported (re-source
  `qat_env.sh`);
- integer matmul not exact: **stop**, because the surrogate's exactness premise fails on this
  GPU.

---

## Stage 5 — dataset verification

### 5.1 Download ImageNetV2 matched-frequency (TEST)

It is not on the machine. This only downloads and unpacks it; nothing reads labels or images for
any decision.

```bash
mkdir -p "$QAT_WORK/data/imagenetv2" && cd "$QAT_WORK/data/imagenetv2"
wget -c "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz" \
  || curl -L -C - -O "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz"
ls -l imagenetv2-matched-frequency.tar.gz
sha256sum imagenetv2-matched-frequency.tar.gz | tee imagenetv2-matched-frequency.tar.gz.sha256
tar -tf imagenetv2-matched-frequency.tar.gz | head -3     # -f alone: GNU tar detects whether it is gzipped
tar -xf imagenetv2-matched-frequency.tar.gz              # (the file you downloaded on 25 Oct is an uncompressed tar)
ls imagenetv2-matched-frequency-format-val | wc -l
find imagenetv2-matched-frequency-format-val -type f | wc -l
ls imagenetv2-matched-frequency-format-val/0 | head -3
cd "$QAT_CODE"
```

**Expected:**
- an archive of about 1.2 GB;
- record its SHA-256 (none is published);
- entries look like `imagenetv2-matched-frequency-format-val/0/<hash>.jpeg`;
- 1000 folders and 10000 files.

**If it fails:**
- download blocked: download the same URL on another computer and copy the `.tar.gz` here
  (`scp`/VS Code);
- if it cannot be obtained at all, TEST is **blocked**. Continue stages 6–11 on VAL only
  (`make` works without it, and `add-test` adds it later), and report it. Never substitute
  another set.

### 5.2 Write `configs/paths.json`

```bash
cat > configs/paths.json <<'EOF'
{
 "imagenet_train_roots": ["/media/user/DATA 1/Deva QAT/data/raw/ILSVRC/Data/CLS-LOC/train"],
 "imagenet_val_root": "/media/user/DATA 1/Deva QAT/data/raw/ILSVRC/Data/CLS-LOC/val",
 "imagenet_val_labels_csv": "/media/user/DATA 1/Deva QAT/data/raw/LOC_val_solution.csv",
 "imagenetv2_root": "/media/user/DATA 1/cim_qat_work/data/imagenetv2/imagenetv2-matched-frequency-format-val",
 "work_root": "/media/user/DATA 1/cim_qat_work"
}
EOF
cat configs/paths.json
```

If ImageNetV2 is not available yet, set `"imagenetv2_root": null`.

### 5.3 Inspect (read-only)

```bash
python qat_prepare_data.py inspect --paths configs/paths.json
python qat_env_check.py --device cuda --paths configs/paths.json --out "$QAT_WORK/checks/env_check_paths.json" | tail -6
```

**Expected:**
- `[PASS] work_root ... writable=True, free ... GB`;
- `[PASS] ImageNet train: 1,281,167 image files in 1000 class folders` (or your exact count, with
  a NOTE if it differs);
- `[PASS] ImageNet val (flat+csv): 50,000 images, CSV rows 50,000, every image labelled exactly
  once, per class 50..50`;
- `[PASS] ImageNetV2 matched-frequency: 10,000 images in folders 0..999 (per class 10..10) ...
  label = int(folder name)`;
- `No files were written.`

**If it fails:** the message names the exact problem:
- missing class folders;
- an image without a CSV label;
- a CSV row without an image;
- an unexpected V2 folder (e.g. zero-padded names);
- a missing `-format-val` level: set `imagenetv2_root` to the folder that directly contains
  `0` … `999`.

---

## Stage 6 — prepare the manifests (fixed once)

### 6.1 Make the splits with full content hashing

This runs in tmux. It hashes all ~1.28 M train files on the HDD, which is resumable: if it is
interrupted, re-run the same command.

```bash
tmux new -s prep            # then inside:
source "/media/user/DATA 1/cim_qat_work/code/CIM-SOC-QAT/qat_env.sh"
python qat_prepare_data.py make --paths configs/paths.json --workers 16 --hash-train 2>&1 | tee "$QAT_WORK/prepare.log"
```

**Expected:** at the end, a JSON summary followed by:

```
OK: TRAIN = 1,281,167 - 2,000 - 200 - k = 1,278,967 - k; VAL = 50,000; TEST = 10,000
```

Here k = TRAIN files whose content duplicates a CAL/CALCHECK image, usually 0. The exact numbers
come from your disk.

Also expected:
- `size_check` ok for cal/calcheck/val/test;
- `path_overlap.ok = true`;
- `content_between` 0 for every critical pair;
- `v2_mapping.ok = true`, `imagefolder_trap_classes_that_would_be_mislabelled: 988`.
- NOTE lines with non-zero `train|val` or `train|test` counts are possible (ImageNet has a few
  exact train/val duplicates). Record them; they are reported, not removed (QAT_DESIGN.md §8.3).

Also printed: `held-out eligibility rule: ...; N candidate(s) skipped`, one line per skipped
candidate. These are train images whose content equals a VAL/TEST image; they stay in TRAIN, and
the next file of the same class is used.

**If it fails:**
- `FAIL: a critical check failed`: do **not** continue. Send `"$QAT_WORK/splits/splits.json"`
  (keys `disjointness`, `accounting`, `v2_mapping`).
- If the failure was produced by **v2.0** of the package (no eligibility rule) and nothing has used
  the splits, install v2.1 and run:

  ```bash
  python qat_prepare_data.py make --paths configs/paths.json --workers 16 --hash-train --replace-failed
  ```

  This archives the failed outputs to `splits/failed_<time>/`, reuses all hash caches (minutes,
  not hours) and draws again with the rule.
- `splits.json exists`: the splits are already fixed. Never use `--force` once anything has used
  them.

Duration: hashing is I/O-bound (about 140 GB). Note the wall time in the experiment log.

### 6.2 ImageNetV2 label-mapping check (pretrained FP32; mapping statistics only)

```bash
python qat_evaluate.py v2-integrity --paths configs/paths.json --device cuda --num-workers 12
```

**Expected:**
`[PASS] ImageNetV2 label mapping: modal pretrained prediction == label for ≥700/1000 classes with
label=int(folder); ≤50/1000 with the ImageFolder lexicographic order`.

No accuracy is printed. It cannot influence any decision, because the pretrained model is not
trained.

**If it fails:** the labels or mapping are wrong. **Stop** and send `"$QAT_WORK/eval/v2_integrity.json"`.

### 6.3 First audit

```bash
python qat_audit.py --paths configs/paths.json 2>&1 | tail -25
```

**Expected:** `... PASS, k WARN, 0 FAIL`. WARNs are possible only for reported TRAIN∩VAL /
TRAIN∩TEST duplicates. No runs or calibration exist yet, so no `L2`/`L3` lines appear.

**If it fails:** each FAIL line names the check (`S*` splits, `M*` mapping); stop and report it.

---

## Stage 7 — real-image smoke tests

### 7.1 FP32 preprocessing check on VAL (GPU, no CIM)

```bash
python qat_evaluate.py fp32-check --paths configs/paths.json --device cuda --num-workers 12
```

Three models × bicubic/bilinear on all 50k VAL images (I/O-bound).

**Expected:**
- the pretrained model under **bicubic** is close to the published **80.7 %** top-1 / **95.6 %**
  top-5;
- the old INT8/INT4 FP32 rows are new measurements; record them;
- bilinear vs bicubic differences are typically a few tenths of a pp.

**If it fails:** if pretrained bicubic is far below (e.g. < 78 %), labels or preprocessing are
wrong. **Stop.**

### 7.2 Simulator speed on real images (VAL only, separate `val_limit48` folder, never reported)

```bash
python qat_evaluate.py sim --paths configs/paths.json --split val --limit 48 --model pretrained \
  --kind analog --precision W8A8 --gain-set hist --workers 24
```

**Expected:** `done: pretrained__analog_W8A8__hist n=48 cpu X.XX s/img`. Write X down.

Full-run cost per row: VAL ≈ 50,000 × X / 24 s; TEST ≈ 10,000 × X / 24 s.

**If it fails:** worker errors are printed. Check that the `cim_sim` tests in stage 4 passed.

---

## Stage 8 — gain calibration and CALCHECK (per precision)

```bash
python qat_calibrate.py run --paths configs/paths.json --precision W8A8 --workers 24 2>&1 | tee "$QAT_WORK/calib_W8A8.log"
less "$QAT_WORK/calibration/W8A8/calibration_report.md"
```

**Expected:** the report contains:
- candidate gains for P99.9 / 99.99 / 99.999 / 100 and the MSE-optimal diagnostic;
- predicted clip % per op and per layer;
- the CAL-check table (FP32, ADC-off, three levels) with KL, agreement, NLL, top-1, measured
  clip %, and ADC utilisation per op;
- a **Flags** section.

**Decision (pre-declared):**
- If the status is "no material trade-off flagged", freeze P99.99:

  ```bash
  python qat_calibrate.py freeze --paths configs/paths.json --precision W8A8 --level p99.99
  ```

- If any flag is raised, **do not freeze**. Send the report: choosing another level is a recorded
  deviation that you decide.

Repeat for W4A4:

```bash
python qat_calibrate.py run --paths configs/paths.json --precision W4A4 --workers 24 2>&1 | tee "$QAT_WORK/calib_W4A4.log"
less "$QAT_WORK/calibration/W4A4/calibration_report.md"
python qat_calibrate.py freeze --paths configs/paths.json --precision W4A4 --level p99.99   # only if no flag
python qat_audit.py --paths configs/paths.json 2>&1 | grep -E "L3|FAIL|PASS,"
```

**Expected:**
- `FROZEN {...} (level p99.99) -> .../frozen_gains.json` for each precision;
- audit `L3.*` lines PASS: CAL/CALCHECK manifests only, CAL-check jobs ran exactly on CALCHECK.

**If it fails:**
- `scale check failed`: stop (simulator mismatch);
- `frozen_gains.json exists`: the gains are already frozen.

---

## Stage 9 — simulator vs surrogate, mechanics, selection rules (per precision)

```bash
G8="$QAT_WORK/calibration/W8A8/frozen_gains.json"
python test_surrogate_vs_simulator.py --device cuda --precision W8A8 --frozen-gains "$G8" \
  --paths configs/paths.json --n-real 500 --workers 24 --out "$QAT_WORK/checks/surrogate_W8A8" 2>&1 | tail -15
python test_training_mechanics.py --device cuda --precision W8A8 --paths configs/paths.json \
  --frozen-gains "$G8" --out "$QAT_WORK/checks/mechanics_W8A8" 2>&1 | tail -20
python tests_qat/test_selection.py --paths configs/paths.json --frozen-gains "$G8"
```

Then the same three commands with `W4A4` and
`G4="$QAT_WORK/calibration/W4A4/frozen_gains.json"`.

**Expected:**
- surrogate: `... checks passed (0 exact-tier failures)`, exit 0;
- T4 lines:
  - CPU surrogate batch 1 bit-exact on 500/500 real VAL images;
  - `KL(sim‖GPU surrogate)` against the 1-ulp floor, with verdict **PASS** (≤ 2× floor);
- mechanics 19/19:
  - C0 checks that every model tensor, including TinyViT's cached attention-bias tensor, is on
    the GPU;
  - C8 / C8b check resume both with the checkpoint loaded on the CPU and loaded straight onto the
    GPU;
- selection 8/8 (the precision is read from the frozen-gains file, so the W4A4 command works too).

**If it fails:**
- any exact-tier failure or CPU-exact < 500/500: **stop**. The surrogate no longer reproduces
  the simulator.
- T4 verdict REVIEW: stop and send `surrogate_validation_cuda.json`. The GPU path deviates
  beyond the hardware model's own float sensitivity.

---

## Stage 10 — GPU QAT

### 10.1 Smoke run with measured throughput (separate `*_smoke` folder, never used for results)

```bash
python qat_train.py --config configs/qat_w8a8_kd.json --device cuda --smoke-steps 60
cat "$QAT_WORK/runs/w8a8_kd_pilot5ep_smoke/throughput.json"
python qat_train.py --config configs/qat_w4a4_kd.json --device cuda --smoke-steps 60
```

**Expected:** a `THROUGHPUT:` line with:
- s/step and img/s;
- data wait;
- peak GPU memory (< 20 GB);
- the full-VAL eval time;
- `est. H h/epoch` and the full-budget hours.

The spot-check line must show `CPU surrogate batch-1 bit-exact on 16/16`.

**If it fails:**
- CUDA OOM: report it (the batch size is a decided setting; do not change it silently);
- `data_bound=true` (data wait > 10 % of the step): the HDD is the bottleneck. Report it before
  the long run. Raising `num_workers` (default 12) in the config is allowed; record it.

### 10.2 + 10.3 The two runs (W8A8 on GPU 1, W4A4 on GPU 0, at the same time)

The patience values were decided on 26 Oct, after the smoke runs and before any training (QAT_DESIGN.md §0.4 D1):

| run | budget | patience |
|---|---|---|
| `w8a8_kd_pilot5ep` | 5 epochs | 5 (it cannot stop early) |
| `w4a4_kd_pilot10ep` | 10 epochs | 3 |

The two runs are independent, deterministic, and write to separate run folders. Running them at the same time changes no number in either run. It only changes the wall time.

**Before you start:**
```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
free -g; uptime
```
- **GPU 0:** it must be free for about 2 days. Check that nobody else in the lab is using it. Each run needs about 8.1 GB of GPU memory at its peak.
- **CPU:** the load average should be low. Each run uses 1 main process, 12 data-loader workers, and 16 simulator workers during each spot check.

**Terminal A (W8A8 on GPU 1):**
```bash
export CUDA_VISIBLE_DEVICES=1
source "/media/user/DATA 1/cim_qat_work/code/CIM-SOC-QAT/qat_env.sh"
nohup python qat_train.py --config configs/qat_w8a8_kd.json --device cuda > "$QAT_WORK/runs/w8a8_console.log" 2>&1 &
disown
```

**Terminal B (W4A4 on GPU 0).** Start it after A has printed its first `step 50` line (about 2 minutes), so that the two start-ups don't compete:
```bash
export CUDA_VISIBLE_DEVICES=0
source "/media/user/DATA 1/cim_qat_work/code/CIM-SOC-QAT/qat_env.sh"
nohup python qat_train.py --config configs/qat_w4a4_kd.json --device cuda > "$QAT_WORK/runs/w4a4_console.log" 2>&1 &
disown
```

**Check that each run is on its own GPU:**
- the `device:` line at the top of each `train.log` shows a different UUID;
- `nvidia-smi` shows one python process on each GPU.

**Monitor** (from any terminal):
```bash
grep -E "device|\[eval|EARLY|DONE|Error|Traceback" "$QAT_WORK/runs/w8a8_kd_pilot5ep/train.log" "$QAT_WORK/runs/w4a4_kd_pilot10ep/train.log"
tail -n 3 "$QAT_WORK/runs/w8a8_kd_pilot5ep/train.log" "$QAT_WORK/runs/w4a4_kd_pilot10ep/train.log"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
uptime; free -g
ps -ef | grep "qat_train" | grep -v grep
```

**Watch these three things:**

1. **The data wait.** Each step line ends with `step S s data D s`. When one run was alone, `data` was 0.000 s. If both runs regularly show `data` above about 0.07 s (10 % of the 0.67 s step), the shared HDD or CPU is limiting them. This is most likely when the page cache is cold. In that case:
   - keep both runs going (it is only slower, never wrong); or
   - stop B with `kill <pid>` and later resume it alone with the same command plus `--resume`.

   The loader worker count is exempt from the config hash, so you may also lower `num_workers` in a config before resuming.
2. **Evaluation and spot-check time.** One full-VAL pass was estimated at about 25 min, but that is likely an overestimate. The real figure is `eval_seconds` in `evals.jsonl` after epoch 0. Two runs evaluating at the same time share the CPU for JPEG decoding, and the 16 + 16 simulator workers of two simultaneous spot checks oversubscribe the 32 CPU threads for a few minutes. This is harmless; only time is lost.
3. **Memory.** `free -g` should keep tens of GB available, and swap was already almost full on this machine. If `available` falls below about 10 GB, stop B and resume it later.

**Don't run stage 11 simulator evaluations while both runs train.** They compete for the same CPUs and disk. Run them after W8A8 finishes, on the cores its end frees up: use `--workers 12` while W4A4 is still running, and 24 afterwards.

**Resume after an interruption.** Re-run the same `nohup` command for that run, adding `--resume` before the `>`, with the same `CUDA_VISIBLE_DEVICES`. It refuses if the config (other than the worker counts) or the gains changed.

**Expected in each `train.log`:**
- `[eval step0000000]` first: the epoch-0 PTQ, surrogate, full VAL;
- then one `[eval …]` line per epoch, showing:
  - VAL hardware top-1, NLL, KL(T‖S), agreement;
  - clipping % per operation;
  - the spot check, CPU-exact 128/128, KL against the floor;
  - student FP32 top-1;
  - `NEW BEST` or `k epoch(s) since best`;
- at the end, `DONE (completed the scheduled budget | VAL early stopping …)`.

W8A8 always completes 5 epochs. `summary.json` records the primary (`best_model.pth`) and the secondary (`final_model.pth`) checkpoints with their SHA-256, the stop reason and the plateau recommendation.

**If it fails:**
- `spot check … Stopping`: the surrogate diverged from the simulator. Keep the folder and report it.
- `non-finite loss`, CUDA OOM, or a traceback: report it with the last 40 lines of `train.log`.
- `NOT converged: recommend a fresh run at 2x`: this is a recommendation only. Running it is your decision.

### 10.5 Decision D2: fork W4A4 if it early-stops

This step applies **only if** `w4a4_kd_pilot10ep` stops early. Its `train.log` then shows `EARLY STOP at step …` and `DONE (VAL early stopping …)`, and its process exits. If the run reaches epoch 10, skip this section.

Run the fork from the v2.5 code folder. It may use either GPU, so check which one is free first. On GPU 1 (free after W8A8 finished):
```bash
export CUDA_VISIBLE_DEVICES=1
source "/media/user/DATA 1/cim_qat_work/code/CIM-SOC-QAT-v2.5/qat_env.sh"
python -c "import json,os; s=json.load(open(os.environ['QAT_WORK']+'/runs/w4a4_kd_pilot10ep/summary.json')); print(s['stopped_reason']); print(s['steps_done'], s['best']['tag'])"
nohup python qat_train.py --config configs/qat_w4a4_kd_full.json --device cuda --fork-from w4a4_kd_pilot10ep > "$QAT_WORK/runs/w4a4_full_console.log" 2>&1 &
disown
grep -E "FORKED|device|Error|refused" "$QAT_WORK/runs/w4a4_kd_pilot10ep_full/train.log" "$QAT_WORK/runs/w4a4_full_console.log"
```

**Expected:**
- `train.log` contains the pilot's whole log, then a `device:` line, then `FORKED (decision D2) from w4a4_kd_pilot10ep at global step <the stop step>/199830: early stopping OFF (patience 10)`.
- Training then resumes at the next step, with the next `[eval …]` at the following epoch end.
- It never stops early and finishes at epoch 10 with `DONE (completed the scheduled budget)`.
- The pilot's own folder is not modified.
- Time: the remaining epochs at about 4.5 h each (≈ 0.67 s/step alone on a GPU) plus about 30 min of evaluation per epoch. Forking at epoch 5 leaves 5 epochs, about 25 h.

**If it fails:**
- `fork refused: … completed its scheduled budget`: the pilot did not early-stop, so there is nothing to fork.
- `… may change only …`: the config differs from the pilot's in another setting. Use `configs/qat_w4a4_kd_full.json` unchanged.
- `… exists`: the fork was already made. Resume it with `--resume` (not `--fork-from`).

**If the fork is interrupted**, resume it like any run:
```bash
nohup python qat_train.py --config configs/qat_w4a4_kd_full.json --device cuda --resume > "$QAT_WORK/runs/w4a4_full_console.log" 2>&1 &
```

### 10.4 (optional) W4A4 CE pilot

Only if you decide the GPU time is available. It is identical to 10.3 except the loss:
`configs/qat_w4a4_ce_pilot.json` → run folder `w4a4_ce_pilot10ep`.

The CPU work of stage 11 on VAL (rows E1–E5b) can run while the GPU trains. Give it fewer workers
(`--workers 12`) and watch the `data` time in `train.log`.

---

## Stage 11 — evaluation (actual simulator): VAL first, then TEST

### 11.0 Threads, workers, timing and stopping (package v2.6)

**One BLAS thread per simulator process.** Parallelism is by processes (`--workers`). Since v2.6
the code enforces this itself:
- `qat_evaluate.py`, `qat_calibrate.py` and `qat_eval_matrix.py` set `OMP/OPENBLAS/MKL/NUMEXPR/
  VECLIB_*_NUM_THREADS=1` before numpy is loaded (an explicit value already in your environment
  is respected);
- every simulator and calibration worker also lowers the running BLAS pool to 1 thread
  (`threadpoolctl` if installed, otherwise the OpenBLAS/MKL C call), so it holds even for
  workers of the training process's spot checks;
- each chunk records the thread count it ran with, and `summary.json` of every evaluation has
  `blas_threads_per_worker` (expected `[1]`; `null` for chunks computed by v2.5).

Manual `export OMP_NUM_THREADS=1 …` lines are no longer needed with v2.6 (keep them for jobs
still running from the v2.5 folder). The thread count changes speed only: the 28 Oct A/B gave
64/64 bit-identical logits with and without the limit.

**Timing.** Measured on this machine: X = 1.05 s per image per worker (W8A8 and W4A4 analog
rows; FP32 rows are much faster). One 50,000-image VAL row ≈ 50,000 × 1.05 / workers seconds:
0.91 h at 16 workers, 0.61 h at 24, 0.52 h at 28. A 10,000-image TEST row is one fifth of that.
Leave CPU cores for a running training job (its data loader).

**Stopping an evaluation, including its workers.** The workers have a generic command line
(`python -c from multiprocessing.spawn import spawn_main …`), so `pkill -f qat_evaluate.py`
does not reach them. Use:

```bash
python qat_stop.py list               # every cim_qat process: train / eval / calib, with thread counts
python qat_stop.py eval --dry-run     # exactly what would be stopped
python qat_stop.py eval               # SIGTERM, wait 15 s, SIGKILL; exit 0 = nothing left
```

It selects processes tagged `CIM_QAT_JOB=eval:…` (v2.6), Python processes running
`qat_evaluate.py` / `qat_eval_matrix.py` (also v2.5 jobs) and **all their descendants**, and
never a `qat_train.py` process, a `train`-tagged process or anything below them. It works on
v2.5 jobs too, so you can use the v2.6 copy to stop a matrix running from the v2.5 folder.
Since v2.6 every worker also asks the kernel to end it when its parent dies
(`PR_SET_PDEATHSIG`), so a killed evaluation can no longer leave orphaned workers. A stopped
row resumes from its finished chunks when the same command is run again.

### 11.1 VAL matrix per precision (resumable; dry run first)

```bash
python qat_eval_matrix.py --paths configs/paths.json --precision W8A8 --split val --new w8a8_kd_pilot5ep --dry-run
python qat_eval_matrix.py --paths configs/paths.json --precision W8A8 --split val --new w8a8_kd_pilot5ep --workers 24
# W4A4: first apply decision D2 (both W4A4 runs finished; BEFORE any TEST read)
python qat_evaluate.py select-d2 --paths configs/paths.json
#   -> prints both candidates and "D2 -> report <RUN>"; use that RUN below as W4NEW
W4NEW=<the run printed by select-d2>
python qat_eval_matrix.py --paths configs/paths.json --precision W4A4 --split val --new "$W4NEW" --workers 24
#   add  --new-ce w4a4_ce_pilot10ep  to the W4A4 line if 10.4 was run
python qat_evaluate.py report --paths configs/paths.json --split val --precision W8A8 --new w8a8_kd_pilot5ep
python qat_evaluate.py report --paths configs/paths.json --split val --precision W4A4 --new "$W4NEW"
# (diagnostic, VAL only, if a fork was made) the pilot's checkpoint at its early stop:
#   python qat_evaluate.py sim --paths configs/paths.json --split val --model run:w4a4_kd_pilot10ep:final --kind analog --precision W4A4 --gain-set cal --workers 24
```

**Expected:**
- one `done: <tag> n=50000` line per row;
- `REPORT_val_W8A8.md` / `REPORT_val_W4A4.md` in `"$QAT_WORK/eval/val/"`, containing:
  - the identical-input statement;
  - the rows table;
  - per-op hardware diagnostics;
  - paired comparisons C1–C8 (C9 if the CE pilot ran).
- Missing rows show `PENDING`.

**If it fails:**
- a row fails: the matrix continues and lists it. Re-run the same command (finished chunks are
  kept);
- `runs are not on identical inputs`: **stop** and report it.

### 11.2 Pre-TEST audit

All decisions must be final before any TEST read.

```bash
python qat_audit.py --paths configs/paths.json 2>&1 | tail -30
```

**Expected:** 0 FAIL. From here on, no training, calibration or configuration change for these
experiments. Anything done later is listed by the audit (WARN L5) and must be disclosed.

### 11.3 TEST (ImageNetV2) matrix — the final evaluation

```bash
python qat_eval_matrix.py --paths configs/paths.json --precision W8A8 --split test --final --new w8a8_kd_pilot5ep --workers 24
python qat_eval_matrix.py --paths configs/paths.json --precision W4A4 --split test --final --new "$W4NEW" --workers 24
```

**Expected:**
- one `done: ... n=10000` line per row;
- the pretrained FP32 row is shared by both precisions (computed once);
- this covers requirement D: pretrained FP32 (E1), PTQ (E4), old INT8 (E5, W8A8), old INT4 (E5,
  W4A4), new W8A8 (E6), new W4A4 (E6), plus the diagnostic rows, all on the same 10,000 images
  with the same preprocessing.

**If it fails:**
- `--split test reads ImageNetV2 ... pass --final`: the flag is missing;
- `not finished`: a training run has no `summary.json`.

---

### 11.3b (declared 29 Oct, optional) the pilot's best checkpoint on TEST — diagnostic for D2

On VAL the fork-vs-pilot comparison (C10) favours the fork, because VAL chose both checkpoints
and the D2 winner. TEST is the only unbiased measurement of what D2 changed. The reported W4A4
model stays the D2 choice whatever this row shows. One 10,000-image row (≈ 10 min at 16 workers):

```bash
python qat_evaluate.py sim --paths configs/paths.json --split test --final --model run:w4a4_kd_pilot10ep:best --kind analog --precision W4A4 --gain-set cal --workers 16
```

The TEST W4A4 report then lists it as a row and adds C10.

---

## Stage 12 — reports and final audit

**First (v2.7): recompute the analog statistics summaries** of every analog run from its chunk
files. The ADC-utilisation percentiles written by v2.5/v2.6 came from a defective merge (QAT_DESIGN.md
§13, v2.7); exact counts, clipping and all accuracy numbers are unaffected. Reads only the
simulator's chunk statistics; writes `analog_summary_uniform.json` next to each run. Do **not**
delete any `chunks/` folder before this has run.

```bash
python qat_evaluate.py restat --paths configs/paths.json --split val
python qat_evaluate.py restat --paths configs/paths.json --split test --final
python qat_evaluate.py report --paths configs/paths.json --split val --precision W8A8 --new w8a8_kd_pilot5ep
python qat_evaluate.py report --paths configs/paths.json --split val --precision W4A4 --new "$W4NEW"
```
Each `restat` line prints old → new util median / p99.9 per run. The reports then carry no `†`
and the VAL W4A4 report contains C10.

Then:

```bash
python qat_evaluate.py report --paths configs/paths.json --split test --final --precision W8A8 --new w8a8_kd_pilot5ep
python qat_evaluate.py report --paths configs/paths.json --split test --final --precision W4A4 --new "$W4NEW"
python qat_audit.py --paths configs/paths.json 2>&1 | tail -40
ls "$QAT_WORK/eval/val" "$QAT_WORK/eval/test" "$QAT_WORK/audit"
```

**Expected:**
- `REPORT_test_W8A8.md` and `REPORT_test_W4A4.md`;
- the audit shows 0 FAIL and passes `E1.test` (identical inputs for every TEST run) and the
  `L6.*` lines (every TEST run `--final`, after its training run finished).

How to read the reports: QAT_DESIGN.md §10.2.
- TEST comparisons are the clean ones.
- VAL QAT rows are selection-biased.
- Never compare VAL and TEST absolute numbers with each other.

**Optional read-only tools (`tools_qat/`, added 29 Oct after the final audit; they write only
`SENSITIVITY_train_overlap.*`):**
- `python tools_qat/val_overlap_sensitivity.py --paths configs/paths.json` — VAL top-1 and C3/C5/C6/
  C7/C10 with and without the VAL images whose content also occurs in TRAIN (audit S4.info.train|val).
- `python tools_qat/pair_compare.py --paths configs/paths.json --split test --a <run> --b <run>` —
  one paired comparison with the reports' statistics. Anything chosen after the TEST reports were
  seen is post hoc and is labelled so.
- `python tools_qat/imagenette_percentile_check.py <Imagenette results folder>` — which older
  Imagenette jobs have percentile columns affected by the AnalogStats.merge() defect.

Send back the following (small files):
- `"$QAT_WORK/splits/splits.json"`;
- `calibration/W*/calibration_report.md` and `frozen_gains.json`;
- `checks/*/surrogate_validation_cuda.json` and `mechanics.json`;
- `runs/*/summary.json` and `throughput.json`;
- `eval/val/REPORT_*.md` and `eval/test/REPORT_*.md`;
- the latest `audit/AUDIT_*.md`.

Fill in `QAT_EXPERIMENT_LOG.md` (sections marked PENDING) from them.

---

## Stage 13 — Extension D3: ImageNetV2 threshold-0.7 and top-images (package v2.8)

Requested on 29 Oct 2026, after the results were frozen (QAT_DESIGN.md §0.4 D3). Descriptive only.
The matched-frequency 10k stays the pre-declared primary result. Nothing frozen is re-run or
changed. Same models, gains and simulator. Everything new goes into new files and folders.

What could overwrite frozen results, and how v2.8 prevents it:
- re-running `report`, `restat` or `v2-integrity` on `--split test`, or any `sim` row that is
  already finished. v2.8 refuses to overwrite any file listed in a `FINAL_SHA256SUMS_*.txt` file,
  and skips a finished row without writing or logging anything;
- appending to `access_log.jsonl`. Once it is frozen, v2.8 writes new entries to
  `access_log_ext.jsonl`;
- running anything from the **old** v2.5–v2.7 folders. They have none of these protections.
  Use only the v2.8 folder from now on;
- deleting `chunks/` folders. Nothing in v2.8 deletes them.

### 13.0 Install v2.8 in a new folder; check the disk

```bash
cd "/media/user/DATA 1/cim_qat_work"
sha256sum CIM-SOC-QAT-v2.8.zip                     # == the SHA-256 given with the package
mkdir -p code/_v28 && cd code/_v28 && unzip -q ../../CIM-SOC-QAT-v2.8.zip && mv CIM-SOC-QAT ../CIM-SOC-QAT-v2.8 && cd .. && rmdir _v28
cp CIM-SOC-QAT-v2.7/configs/paths.json CIM-SOC-QAT-v2.8/configs/
cd CIM-SOC-QAT-v2.8 && sha256sum -c --quiet PACKAGE_SHA256SUMS.txt && source qat_env.sh
cd "$QAT_WORK" && sha256sum -c --quiet FINAL_SHA256SUMS_20261029.txt && echo "FROZEN FILES INTACT"; cd "$QAT_CODE"
df -h "/media/user/DATA 1"                          # need about 70 GB free (two archives + about 60 GB of chunks)
```

### 13.1 Download, record, unpack, count

```bash
mkdir -p "$QAT_WORK/data/imagenetv2_ext" && cd "$QAT_WORK/data/imagenetv2_ext"
for V in threshold0.7 top-images; do
  wget -c "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-$V.tar.gz"
  sha256sum "imagenetv2-$V.tar.gz" | tee "imagenetv2-$V.tar.gz.sha256"
  mkdir -p "$V" && tar -xf "imagenetv2-$V.tar.gz" -C "$V"
  R=$(dirname "$(find "$V" -maxdepth 3 -type d -name 999 | head -1)"); echo "$V root: $R"
  ls "$R" | wc -l                                   # 1000 class folders "0".."999"
  find "$R" -type f | wc -l                         # 10000 images
  for d in "$R"/*; do ls "$d" | wc -l; done | sort | uniq -c   # "1000 10": 10 per class
done
cd "$QAT_CODE"
```
No SHA-256 is published for these archives, so record your own (the `.sha256` files); the
registration copies it into `extra_tests.json`.

Then add the two roots (the folders printed as `root:`) to `configs/paths.json`:
```json
 "imagenetv2_extra_roots": {
  "test_th07": "/media/user/DATA 1/cim_qat_work/data/imagenetv2_ext/threshold0.7/<the root printed>",
  "test_top":  "/media/user/DATA 1/cim_qat_work/data/imagenetv2_ext/top-images/<the root printed>"
 }
```

### 13.2 Register each variant (once): labels, leakage, overlap

```bash
python qat_prepare_data.py add-test-variant --paths configs/paths.json --variant test_th07 \
  --archive imagenetv2-threshold0.7.tar.gz --archive-sha256 "$(cut -d' ' -f1 "$QAT_WORK/data/imagenetv2_ext/imagenetv2-threshold0.7.tar.gz.sha256")" --workers 16
python qat_prepare_data.py add-test-variant --paths configs/paths.json --variant test_top \
  --archive imagenetv2-top-images.tar.gz --archive-sha256 "$(cut -d' ' -f1 "$QAT_WORK/data/imagenetv2_ext/imagenetv2-top-images.tar.gz.sha256")" --workers 16
```
**Expected**, for each variant:
- a last line `OK: test_… = N images (10,000 on disk - E excluded …)`;
- label = int(folder) True, 1000 classes, 10 per class.

The line also reports:
- **content equal to CAL / CALCHECK / VAL / TRAIN:** these images are excluded by the declared
  rule (`splits/excluded_<split>.csv`). Expected 0, or a handful if ImageNetV2 shares a file with
  ImageNet;
- **shared content with the matched-frequency TEST and with the other variant:** this is the
  overlap between the variants. The images stay in each variant and are counted once in the
  pooled report.

The step writes new files only: `splits/{manifest,hashes,excluded,extcache}_<split>.csv` and
`splits/extra_tests.json`. `splits.json` is not touched. Registration resolves about 1.3 M
TRAIN paths and hashes the 10,000 images, which takes a few minutes per variant.

### 13.3 Label-mapping check on each variant (GPU, pretrained FP32, mapping statistics only)

```bash
python qat_evaluate.py v2-integrity --paths configs/paths.json --split test_th07 --device cuda
python qat_evaluate.py v2-integrity --paths configs/paths.json --split test_top --device cuda
```
**Expected:** `[PASS] … label mapping: … >= 700 of 1000 classes …` for each variant. The check
writes `eval/v2_integrity_<split>.json`. The frozen `eval/v2_integrity.json` is not touched.

### 13.4 Audit before evaluating

```bash
python qat_audit.py --paths configs/paths.json 2>&1 | grep -E "X[0-9]|PASS,|FAIL" | tail -40
```
**Expected:**
- 0 FAIL;
- `X0.FINAL_SHA256SUMS_20261029.txt` PASS (every frozen file unchanged);
- `X3.<split>.leakage` PASS.

The WARNs to expect and disclose:
- `X3.<split>.excluded`, if any image was excluded;
- `X4.<split>.overlap`, the overlap between variants;
- the three old `S4.info` lines.

### 13.5 Evaluate overnight (nohup, 16 workers)

```bash
cd "$QAT_CODE"
nohup bash tools_qat/run_ext_d3.sh > "$QAT_WORK/eval_ext_d3.log" 2>&1 &
echo $! > "$QAT_WORK/eval_ext_d3.pid"
tail -f "$QAT_WORK/eval_ext_d3.log"                # Ctrl-C stops only the viewing, not the run
```
The script exports `W8NEW=w8a8_kd_pilot5ep`, `W4NEW=w4a4_kd_pilot10ep_full` (checked against
`eval/D2_selection.json`) and `WORKERS=16`. The code sets the thread variables itself.

Rows, in order:
1. **Phase A (priority), per variant:** pretrained FP32; for W8A8 and W4A4 each, calibrated PTQ
   (E4), old QAT (E5) and new QAT best (E6).
2. **Phase B:** the rest of the 10k row set (E2, E3, E4b, E5b, E6s), plus the W4A4 pilot-best row
   for C10, so that every row and comparison exists for each variant and for the pooled set.

Run time, at the measured 0.97–0.99 s per image per worker with 16 workers:
- one 10,000-image analog row: about 10 min;
- phase A: about 1 h per variant, about 2.1 h in total;
- phase B: about 2.6 h;
- **total about 5 h.**

`PHASE=A nohup bash …` runs phase A only. To stop: `python qat_stop.py eval`. Start it again
later to continue: finished rows are skipped untouched.

### 13.6 Reports, audit, new freeze (all new files)

```bash
W4NEW=w4a4_kd_pilot10ep_full
for S in test_th07 test_top; do
  python qat_evaluate.py report --paths configs/paths.json --split $S --final --precision W8A8 --new w8a8_kd_pilot5ep
  python qat_evaluate.py report --paths configs/paths.json --split $S --final --precision W4A4 --new $W4NEW
done
python qat_evaluate.py report-pooled --paths configs/paths.json --precision W8A8 --new w8a8_kd_pilot5ep --final
python qat_evaluate.py report-pooled --paths configs/paths.json --precision W4A4 --new $W4NEW --final
python qat_audit.py --paths configs/paths.json 2>&1 | tail -70
cd "$QAT_WORK"
find eval/test_th07 eval/test_top eval/test_pooled eval/v2_integrity_test_th07.json eval/v2_integrity_test_top.json \
     splits/extra_tests.json splits/*_test_th07.csv splits/*_test_top.csv audit access_log_ext.jsonl \
     -type f ! -path '*/chunks/*' | sort | xargs sha256sum > FINAL_SHA256SUMS_20261030_ext.txt
wc -l FINAL_SHA256SUMS_20261030_ext.txt; cd "$QAT_CODE"
```
**Expected:**
- `REPORT_test_th07_*`, `REPORT_test_top_*` (one variant each, C1–C10);
- `eval/test_pooled/REPORT_test_pooled_*`: composition, distinct images, duplicates with identical
  outputs, all comparisons, and **P1 (post hoc) new QAT vs FP32**, pooled and per variant;
- audit 0 FAIL, with `X0` PASS for the old freeze, `X6.*` PASS for every new row, and
  `X7.after_freeze` WARN (disclosure).

The new freeze file sits next to the old one, which is never replaced. A later audit also checks
the new file.

### 13.7 (post hoc, 2 Nov) own-FP32 decomposition from the stored logits

Read-only on every existing result. It writes only `eval/posthoc_own_fp32/`. It is post hoc and
descriptive (QAT_EXPERIMENT_LOG.md §8.5).

```bash
python tools_qat/own_fp32_posthoc.py check   --paths configs/paths.json   # must end "OVERALL: PASS"
python tools_qat/own_fp32_posthoc.py compute --paths configs/paths.json
cd "$QAT_WORK"
ls eval/posthoc_own_fp32                                   # the new files (+ check_output.txt, if saved there)
sha256sum eval/posthoc_own_fp32/* code/CIM-SOC-QAT-v2.8/tools_qat/own_fp32_posthoc.py > FINAL_SHA256SUMS_20261102_posthoc.txt
cd "$QAT_CODE" && python qat_audit.py --paths configs/paths.json 2>&1 | grep -E "X0|PASS,"
```

**Expected:**
- The audit has three `X0.FINAL_SHA256SUMS_*` lines, all PASS: every file of all three freezes is
  unchanged, and the script is the one that produced the results.
- The audit does not otherwise look into `eval/posthoc_own_fp32/`, which holds no evaluation runs.
- The audit's own access entry goes to `access_log_ext2.jsonl`, because both earlier logs are
  frozen.
- The result is 0 FAIL with the same 6 WARNs as before.

### 13.8 (prepared 2 Nov, EXECUTED 17 Nov) v2.8.1: audit fix, then the audit again

The audit run after 13.7 with v2.8 gave 2 false FAILs (`X5.test_th07.log`, `X5.test_top.log`): the
check did not allow for the extension log itself being frozen later (QAT_DESIGN.md §13, v2.8.1).
Install v2.8.1 in a new folder. Leave the v2.8 folder unchanged, because the third freeze lists a
file in it.

```bash
cd "/media/user/DATA 1/cim_qat_work"
sha256sum CIM-SOC-QAT-v2.8.1.zip                   # == the SHA-256 given with the package
mkdir -p code/_v281 && cd code/_v281 && unzip -q ../../CIM-SOC-QAT-v2.8.1.zip && mv CIM-SOC-QAT ../CIM-SOC-QAT-v2.8.1 && cd .. && rmdir _v281
cp CIM-SOC-QAT-v2.8/configs/paths.json CIM-SOC-QAT-v2.8.1/configs/
cd CIM-SOC-QAT-v2.8.1 && sha256sum -c --quiet PACKAGE_SHA256SUMS.txt && source qat_env.sh
cat > "$QAT_WORK/audit/AUDIT_20261102T105922+0000.NOTE.md" <<'NOTE'
# Note on AUDIT_20261102T105922+0000 (kept unchanged)
Its 2 FAILs (X5.test_th07.log, X5.test_top.log) are false: they are a defect of the v2.8 check. The check
counted the extension reads stored in access_log_ext.jsonl as "appended to a frozen log", because
that log was frozen later, on 1 Nov 15:23 UTC, after its last entry (15:16 UTC). X0 in the same
audit shows nothing was appended since. The check was corrected in v2.8.1 (QAT_DESIGN.md §13), and
the audit was re-run with it (see the next AUDIT_* file).
NOTE
python qat_audit.py --paths configs/paths.json 2>&1 | grep -E "X0|X5|PASS,"
cd "$QAT_WORK"
sha256sum audit/AUDIT_20261102T105922+0000.* audit/AUDIT_<new stamp>.* access_log_ext2.jsonl \
          code/CIM-SOC-QAT-v2.8.1/qat_audit.py > FINAL_SHA256SUMS_20261102_audit.txt
```

**Expected:**
- three `X0` lines PASS;
- `X5.test_th07.log` and `X5.test_top.log` PASS;
- **0 FAIL** and the same 6 WARNs.

The fourth freeze then records the failed audit, its note, the passing audit and the v2.8.1 audit
script. Any later audit logs to `access_log_ext3.jsonl`, by design.

**Result (17 Nov):** `AUDIT_20261117T170653+0000`, 214 PASS, 6 WARN, 0 FAIL, as expected; fourth freeze
written with 7 files (QAT_EXPERIMENT_LOG.md §8.6). Run in the `cim_soc_qat` environment next time
(`conda activate cim_soc_qat` before `source qat_env.sh`); the audit result does not depend on it.

