# control/ — simplified CIM control path (v0.2)

A deliberately small, sequential first control architecture:

> **N tokens → one weight matrix (loaded once) → one 32×32 CIM array → N MVMs → N results → DONE**

`cfg_num_tokens = 1` is the original one-token, one-MVM demonstration.

Verilog-2005, no IP, no external libraries. Runs on Vivado XSim (verified),
and is written to also run on Icarus Verilog and Verilator.

The previous controller (`../cim_ctrl_rtl_v0.1/`) is left untouched and is used
only as a reference (see §12).

---

## 1. Architecture

```
            CPU / Testbench
   (writes token + weights, sets cfg_*, pulses START, waits DONE, reads result)
                   |
                   v
          +------------------+   addresses + strobes only
          | array_controller |---------------------------+
          |    (one FSM)     |                           |
          +--+------------+--+                           |
             |            |                              |
   tok_rd_addr|  w_rd_addr|                   res_we, res_wr_addr
             v            v                              |
     +-------------+ +---------------+                   |
     | token_buffer| | weight_memory |                   |
     | (act. RAM)  | |  (row words)  |                   |
     +------+------+ +-------+-------+                   |
            | token          | weight row                |
            v                v                           |
     +--------------------------------+                  |
     | cim_array  x NUM_ARRAYS        |                  |
     |  32x32, x register, weight regs|                  |
     +---------------+----------------+                  |
                     | y (32 column sums)                |
                     v                                   v
              +---------------------------------------------+
              |               result_buffer                 |
              +---------------------------------------------+
```

| file | what it is |
|---|---|
| `array_controller.v` | **the** controller: one FSM, config registers, address generation, latency counter |
| `token_buffer.v` | activation memory; one word = one whole token (ROWS × ACT_BITS) |
| `weight_memory.v` | weight memory; one word = one weight **row** (COLS × W_BITS) |
| `cim_array.v` | behavioural 32×32 array: weight registers, input register, ideal MAC |
| `result_buffer.v` | result memory; one word = one MVM output (COLS × ACC_W) |
| `cim_top.v` | wires the above together; instantiates `NUM_ARRAYS` arrays |
| `tb_control.v` | testbench: `NUM_TOKENS` tokens, one matrix, self-checking |
| `run_xsim.bat` / `run_iverilog.sh` | one-command simulation |

**Key rule: data never goes through the controller.** The controller only
produces addresses and strobes. Token, weight and result data flow straight
between memories and the array. This is what keeps the FSM small.

## 2. The FSM

```
          start
 IDLE ----------> LOAD_WEIGHT --(ROWS cycles)--> LOAD_TOKEN --> START_MVM
  ^                                                  ^              |
  |  start                       more tokens:        |              v
 DONE <--last token-- STORE_RESULT ---token_idx++----+         WAIT_MVM
                           ^                                        |
                           +-----------(MVM_LATENCY cycles)---------+
```

| state | cycles | what the controller drives |
|---|---|---|
| IDLE | — | waits for `start`; latches all `cfg_*` registers, `token_idx = 0` |
| LOAD_WEIGHT | ROWS (32) | `w_rd_addr = weight_base + w_row`, `arr_w_we`, `arr_w_row = w_row` |
| LOAD_TOKEN | 1 | `tok_rd_addr = token_base + token_idx`, `arr_x_load` |
| START_MVM | 1 | `arr_mvm_start` (the "analog start") |
| WAIT_MVM | MVM_LATENCY | counts `lat_cnt` |
| STORE_RESULT | 1 | `res_we`, `res_wr_addr = result_base + token_idx`; then `token_idx + 1 >= num_tokens` → DONE, else `token_idx++` → LOAD_TOKEN |
| DONE | until next `start` | `done = 1` (level). `start` here launches a new run |

`busy` is high in every state except IDLE and DONE.

Weights are loaded **before** the tokens on purpose: the array is
weight-stationary, so the token loop goes back to `LOAD_TOKEN`, never to
`LOAD_WEIGHT`. The matrix is written once per run, however many tokens follow.

Total latency from `start` to `done`:

```
ROWS + NUM_TOKENS · (MVM_LATENCY + 3)   cycles
         per token: LOAD_TOKEN 1 + START_MVM 1 + WAIT_MVM L + STORE_RESULT 1
```

= 45 for 1 token, 84 for 4, 201 for 13 at the defaults. The testbench checks
this number. `cfg_num_tokens = 0` behaves as 1.

## 3. Token flow

```
TB --tok_wr_*--> token_buffer[token_base + t] = token t     (before start)
controller, for token_idx = 0 .. num_tokens-1:
    tok_rd_addr = token_base + token_idx                     (LOAD_TOKEN)
    token_buffer.rd_data --> cim_array.x_data, latched by arr_x_load
    ... MVM ...
    res_wr_addr = result_base + token_idx                    (STORE_RESULT)
```

One memory word holds a whole token (32 × 8 bit = 256 bit), so
`TOKEN_SIZE = 1` word and loading a token is one cycle. Tokens must therefore be
stored at **consecutive** addresses, and results come out at consecutive
addresses in the same order. The buffers are 16 words deep (`TOK_DEPTH`,
`RES_DEPTH` in `cim_top.v`); nothing checks that `base + num_tokens` fits.

## 4. Weight flow

```
TB --w_wr_*--> weight_memory[weight_base + r] = row r of W   (before start)
controller: for w_row = 0..ROWS-1:                            (LOAD_WEIGHT)
    w_rd_addr = weight_base + w_row
    cim_array.wrow[w_row] <= weight_memory[w_rd_addr]
```

One weight row per cycle, like a real macro written row by row (same idea as
`MC_LOAD` in v0.1). Row `r` of the weight memory is row `r` of the array, i.e.
the `K` (contraction) index; the column index inside the word is the output.

## 5. MVM execution

`cim_array` computes, on `mvm_start`:

```
y[c] = Σ_{r=0..ROWS-1}  x[r] · W[r][c]          (signed, ACC_W-bit result)
```

i.e. `y = x · W` with `x[1×32]`, `W[32×32]` — the same orientation as
`cim_matmul(X[N,K], W[K,M])` in the Python simulator. This is an **ideal
behavioural MAC**: no DAC, no charge sharing, no ADC quantisation. The analog
chain (`cim_sim/analog/`) is not modelled here; this RTL only proves the
control sequence.

## 6. Fixed latency

The analog datapath is deterministic, so there is no ready/valid handshake:

```
START_MVM: arr_mvm_start = 1, lat_cnt <= 0
WAIT_MVM : lat_cnt <= lat_cnt + 1;  leave when lat_cnt == MVM_LATENCY-1
```

`MVM_LATENCY` is a single parameter (default 10, must be ≥ 1), set in
`cim_top.v` and overridable from the testbench / command line. It is a
**placeholder**: no analog timing has been supplied. Nothing else in the RTL
contains a delay.

To switch to a real completion signal later, add an input `arr_mvm_done` and
replace the one exit condition in `WAIT_MVM` (`lat_cnt == MVM_LATENCY-1`) with
`arr_mvm_done`. Nothing else changes.

## 7. Configuring NUM_ARRAYS

1. **Where:** `parameter NUM_ARRAYS = 1` in `cim_top.v` (also in `tb_control.v`,
   which passes it down).
2. **How it creates arrays:** `cim_top.v` has a `generate for (g = 0; g < NUM_ARRAYS; ...)`
   loop that instantiates `cim_array` once per `g` (`g_arr[g].u_arr`). All
   arrays see the same weight-row and token data buses.
3. **How the controller selects one:** `cfg_array_id` is latched on `start`
   and driven out as `arr_sel`. In `cim_top.v` each array's `w_we`, `x_load`
   and `mvm_start` are ANDed with `(arr_sel == g)`, and the result is muxed from
   `y_all[arr_sel]`. So exactly one array does the work. `cfg_array_id` must be
   `< NUM_ARRAYS` (not checked in hardware).
4. **What changes later** to spread work over arrays: see §10.

The testbench uses `ARRAY_ID = NUM_ARRAYS-1`, so with `NUM_ARRAYS=2` it really
exercises array 1 while array 0 stays idle.

## 8. Running the simulation

From `control/`:

Arguments are `[NUM_ARRAYS] [NUM_TOKENS]`, defaults 1 and 4:

```bat
run_xsim.bat            :: Vivado XSim, 1 array, 4 tokens
run_xsim.bat 1 1        :: one token: the original single-MVM demo
run_xsim.bat 2 4        :: 2 arrays (testbench uses array 1), 4 tokens
run_xsim.bat 1 13       :: 13 tokens (maximum for the testbench's buffer layout)
```

Icarus Verilog (if installed):

```bash
./run_iverilog.sh 1 4
```

Verilator 5 (if installed):

```bash
verilator --binary --timing -Wno-fatal -GNUM_ARRAYS=1 -GNUM_TOKENS=4 --top-module tb_control *.v && ./obj_dir/Vtb_control
```

The run prints the FSM trace, the full result of token 0, a one-line check
per token, and ends with `RESULT: PASS` or `RESULT: FAIL`. Build products go
to `control/build/` (log `sim_a<arrays>_t<tokens>.log`).

**Verified (Vivado 2022.2 XSim):**

| NUM_ARRAYS | NUM_TOKENS | start→done | result |
|---|---|---|---|
| 1 | 1 | 45 | PASS |
| 1 | 4 | 84 | PASS |
| 2 | 4 | 84 | PASS |
| 1 | 13 | 201 | PASS |
| 4 | 13 | 201 | PASS |

Single-token runs also PASS at `MVM_LATENCY = 1, 3` (36 / 38 cycles).
Two deliberately broken controllers FAIL as they should:
- weight address ignoring `weight_base`: every column reads 528 (Σx over the decoy tile);
- loop exit off by one (`token_idx >= num_tokens`): 97 instead of 84 cycles,
  and a result is written past the end of the range.

Icarus and Verilator are not installed on this machine, so those two flows are
written but not yet run.

Trace of the 4-token run (abridged):

```
---- START
  cycle   83          IDLE ->  LOAD_WEIGHT  token_idx=0
  cycle  115   LOAD_WEIGHT ->   LOAD_TOKEN  token_idx=0      <- weights loaded once
  cycle  116    LOAD_TOKEN ->    START_MVM  token_idx=0
  cycle  117     START_MVM ->     WAIT_MVM  token_idx=0
  cycle  127      WAIT_MVM -> STORE_RESULT  token_idx=0
  cycle  128  STORE_RESULT ->   LOAD_TOKEN  token_idx=1      <- next token, same weights
  ...
  cycle  167  STORE_RESULT ->         DONE  token_idx=3
---- DONE after 84 cycles (expected ROWS + NUM_TOKENS*(MVM_LATENCY+3) = 84)
   token  0 @ result addr  2: 32/32 columns match  (y[0]=-30, y[31]=63)
   token  1 @ result addr  3: 32/32 columns match  (y[0]=-38, y[31]=55)
   token  2 @ result addr  4: 32/32 columns match  (y[0]=-46, y[31]=47)
   token  3 @ result addr  5: 32/32 columns match  (y[0]=-54, y[31]=39)
RESULT: PASS
```

## 9. Changing the input / weights

Everything is in `tb_control.v`, in the two functions at the top:

```verilog
function integer tok_val(input integer t, input integer r);  // x_t[r], token t
function integer w_val  (input integer r, input integer c);  // W[r][c]
```

The testbench computes the expected result with its own plain loop from these
same functions, so any change is checked automatically. Values must fit in
`ACT_BITS` / `W_BITS` signed. The default test is chosen to be checkable by hand:

```
x_t[r]  = (r + 1) − 8t                   token 0: 1..32, token 1: −7..24, ...
W[r][c] = +2 if r == c
          -1 if r == 31 - c
           0 otherwise
y_t[c]  = 2·x_t[c] − x_t[31−c] = 3c − 30 − 8t
          token 0: −30, −27, ..., 63      token 1: −38, ..., 55      ...
```

The diagonal catches row/column ordering, the anti-diagonal catches a reversed
index, the −1 checks signed arithmetic, and the −8t shift makes every token's
result different (so a repeated or skipped token fails). Later tokens also have
negative activations. The testbench uses non-zero `token_base=1`,
`weight_base=32`, `result_base=2`, puts **decoy** tokens just before and just
after the token range, puts a decoy weight tile at 0, and checks that the
result words just before and just after the result range were never written.

## 10. How this extends

### N tokens → N MVMs (implemented)

Set `cfg_num_tokens = N` before `start`. The whole change in
`array_controller.v` is one latched register and one branch in `STORE_RESULT`:

```verilog
if (token_idx + 1'b1 >= num_tokens)  state <= S_DONE;          // last token
else begin token_idx <= token_idx + 1'b1;  state <= S_LOAD_TOKEN; end
```

Because one memory word is one token, `TOKEN_SIZE = 1` and
`token_addr = token_base + token_idx`. If the activation memory becomes narrower
(e.g. a 32-bit bus), `token_addr = token_base + token_idx * TOKEN_SIZE` and
`LOAD_TOKEN` becomes a small beat loop exactly like `LOAD_WEIGHT`.
Weights are loaded once and stay resident (weight-stationary), so one run
behaves like `CIMConfig.token_block = N` in the Python simulator: one weight
write per tile, N tokens streamed through it.

Not done yet: overlapping token loads with the MVM wait (LOAD_TOKEN for token
t+1 during WAIT_MVM of token t). That would save 1 cycle per token today and
more once token fetch is multi-cycle; it is the first pipelining step.

### Fetching weights for the next MVM

`weight_base` is already a register. For a different matrix (or a different
tile of a larger matrix), set `weight_base = tile_index * ROWS` and go through
`LOAD_WEIGHT` again. A tiled MVM (K > 32 or M > 32) adds two counters
(depth block, output block) around the same states; depth-block partial sums
must then be **added** into the result buffer instead of overwritten
(read-modify-write in `STORE_RESULT`), matching the digital accumulation in
`cim_sim/matmul.py`.

### Multiple arrays dividing the work

Recommended split = **output-tile parallelism**, the scheme the Python
simulator already uses (`cim_matmul(..., arrays=[...])`): each array holds a
different 32-column output tile; one token is **broadcast** to all of them.

What changes:
1. `LOAD_WEIGHT` loops over arrays too (array `a` gets tile
   `weight_base + a*ROWS`), using `arr_sel` as the loop counter.
2. In `cim_top.v`, `x_load` and `mvm_start` are no longer gated by `arr_sel`
   (broadcast); all arrays fire together, one `WAIT_MVM` for all.
3. `STORE_RESULT` loops over arrays, writing `y_all[a]` to
   `result_base + token_idx*NUM_ARRAYS + a`.

The alternative (token parallelism: same weights in every array, different
tokens) is also possible but costs NUM_ARRAYS × the weight writes.

### Command table (deliberately not built yet)

Today the FSM state order *is* the program. When more than one fixed sequence
is needed, each work state becomes an opcode in a small table
(`LOAD_WEIGHT`, `LOAD_TOKEN`, `MVM` = START+WAIT, `STORE_RESULT`, `END`), and
IDLE is replaced by a "fetch next command" state that jumps to the state named
by the opcode. The work states themselves stay unchanged, which is why they are
kept as independent steps now.

## 11. Assumptions (not hardware facts)

| item | value | status |
|---|---|---|
| array | 32×32, one MVM per `mvm_start` | design point (CLAUDE.md §3) |
| ACT_BITS / W_BITS | 8 / 8, signed | simulator assumption; unsigned AV activations not modelled |
| ACC_W | 32 bit | ≥ 21 bits needed for 8b×8b×32 |
| MVM_LATENCY | 10 cycles | **placeholder** |
| memory reads | combinational (register-array memories) | simplification; a real SRAM adds 1 cycle per read |
| ADC / analog | ideal, not modelled | — |

## 12. Relation to `cim_ctrl_rtl_v0.1`

Kept from v0.1: the `ROWS/COLS` naming, the idea that the macro is written one
weight row at a time (`MC_LOAD`) and fired by one command (`MC_COMPUTE` ≈
`arr_mvm_start`), and that all timing is a parameter. Deliberately left out:
operation sequencer, engine controllers, the four configuration tables, the
AGUs, accumulator credits, bit-planes, weight slots and the macro adapter.
Those remain the reference for where this design is heading.
