#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# run_iverilog.sh -- simulate the control path with Icarus Verilog.
#
#   ./run_iverilog.sh         NUM_ARRAYS=1, NUM_TOKENS=4
#   ./run_iverilog.sh 2       NUM_ARRAYS=2  (testbench then uses array 1)
#   ./run_iverilog.sh 1 1     one token: the original single-MVM demo
# -----------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"
NARR="${1:-1}"
NTOK="${2:-4}"
TAG="a${NARR}_t${NTOK}"
mkdir -p build

iverilog -g2005 -Ptb_control.NUM_ARRAYS="$NARR" -Ptb_control.NUM_TOKENS="$NTOK" \
    -s tb_control -o "build/sim_$TAG.vvp" \
    token_buffer.v weight_memory.v result_buffer.v cim_array.v \
    array_controller.v cim_top.v tb_control.v
vvp -n "build/sim_$TAG.vvp" | tee "build/sim_$TAG.log"
grep -q "RESULT: PASS" "build/sim_$TAG.log"
