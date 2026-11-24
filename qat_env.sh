# qat_env.sh — source this in EVERY new terminal / tmux window:   source qat_env.sh
# (QAT_RUNBOOK.md stage 2.2). Sets paths and GPU selection, then cd's into the code.
_qat_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export QAT_CODE="$_qat_here"
export QAT_WORK="${QAT_WORK:-/media/user/DATA 1/cim_qat_work}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID                       # CUDA index == nvidia-smi index
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"  # GPU 1 unless you exported another one first
export TMPDIR="$QAT_WORK/tmp"                             # '/' is 99% full: temp files on DATA 1
export CUBLAS_WORKSPACE_CONFIG=":4096:8"                  # deterministic cuBLAS
mkdir -p "$TMPDIR" "$QAT_WORK/checks" "$QAT_WORK/runs"
cd "$QAT_CODE" || return
echo "QAT_CODE=$QAT_CODE"
echo "QAT_WORK=$QAT_WORK"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER)  TMPDIR=$TMPDIR"
python -c "import sys; print('python', sys.version.split()[0], sys.executable)" 2>/dev/null || echo "python not found: conda activate cim_soc_qat"
