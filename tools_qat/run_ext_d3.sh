#!/usr/bin/env bash
# Extension D3 (package v2.8): evaluate ImageNetV2 threshold-0.7 and top-images with the FROZEN
# models, gains and simulator, in their own folders (QAT_RUNBOOK.md stage 13).
#
#   cd "<the v2.8 code folder>"
#   nohup bash tools_qat/run_ext_d3.sh > "/media/user/DATA 1/cim_qat_work/eval_ext_d3.log" 2>&1 &
#
# PHASE=A runs only the priority rows; default AB runs the priority rows first, then the rest.
# A finished row is skipped without being touched, so the script can be started again after
# an interruption (stop it with: python qat_stop.py eval).
set -u
cd "$(dirname "$0")/.."
source qat_env.sh
export W8NEW=w8a8_kd_pilot5ep
export W4NEW=w4a4_kd_pilot10ep_full          # the D2 selection (eval/D2_selection.json)
export WORKERS="${WORKERS:-16}"
export PHASE="${PHASE:-AB}"
P=configs/paths.json
sel=$(python -c "import json,os; print(json.load(open(os.environ['QAT_WORK']+'/eval/D2_selection.json'))['selected'])")
if [ "$sel" != "$W4NEW" ]; then echo "D2_selection.json says $sel, expected $W4NEW: stopping"; exit 1; fi
echo "start $(date -u +%FT%TZ)  code $(pwd)  workers $WORKERS  phase $PHASE  W8NEW=$W8NEW  W4NEW=$W4NEW"
for SPLIT in test_th07 test_top; do
  echo "=== phase A (priority rows) $SPLIT $(date -u +%FT%TZ)"
  python qat_eval_matrix.py --paths $P --precision W8A8 --split $SPLIT --final --new $W8NEW \
         --skip E2,E3,E4b,E5b,E6s --workers $WORKERS
  python qat_eval_matrix.py --paths $P --precision W4A4 --split $SPLIT --final --new $W4NEW \
         --skip E2,E3,E5b,E6s --workers $WORKERS
done
echo "=== phase A done $(date -u +%FT%TZ)"
if [ "$PHASE" = "AB" ]; then
  for SPLIT in test_th07 test_top; do
    echo "=== phase B (remaining rows) $SPLIT $(date -u +%FT%TZ)"
    python qat_eval_matrix.py --paths $P --precision W8A8 --split $SPLIT --final --new $W8NEW --workers $WORKERS
    python qat_eval_matrix.py --paths $P --precision W4A4 --split $SPLIT --final --new $W4NEW --workers $WORKERS
    python qat_evaluate.py sim --paths $P --split $SPLIT --final --model run:w4a4_kd_pilot10ep:best \
           --kind analog --precision W4A4 --gain-set cal --workers $WORKERS
  done
  echo "=== phase B done $(date -u +%FT%TZ)"
fi
echo "end $(date -u +%FT%TZ)"
