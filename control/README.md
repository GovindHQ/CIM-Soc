# control/ — sequential CIM array controller

The CPU generates a **command table**. One **array controller** (a single FSM)
executes it in order, driving an explicit **DAC → array → ADC → accumulate**
datapath on 32×32 tiles, reloading weights as it goes, until `DONE`.

Verilog-2005, no IP. Verified in Vivado 2022.2 XSim (see §9). The analog blocks
are behavioural models and are not meant to be synthesized.

```
                    CPU (cpu_mapper.py / tb_control.v)
                     |
                     | command table, tokens, weights
                     v
   command_table --> array_controller            addresses, enables, start pulses
                     |     |     |     |
       +-------------+     |     |     +----------------------------+
       v                   v     v                                  v
  token_buffer      weight_memory  dac_start, array_start,    result_buffer
       |                   |       adc_start, group_mask      (accumulator +
       |                   |             |                     result memory)
       v                   v             v                          ^
  +--------------------------------------------------+              |
  | one channel per array (NUM_ARRAYS)               |              |
  |   dac --v_ia--> cim_array --v_oa--> adc --codes-------------------+
  +--------------------------------------------------+
```

| file | what it is |
|---|---|
| `array_controller.v` | **the** controller: one FSM, simple counters, no data passes through it |
| `command_defs.vh` | command format and opcodes |
| `command_table.v` | the table the CPU writes |
| `token_buffer.v` | input memory; one word = one depth-block slice of one token |
| `weight_memory.v` | weights; one word = one row of one tile |
| `dac.v` | behavioural input DAC bank of one array |
| `cim_array.v` | behavioural 32×32 macro of one array |
| `adc.v` | behavioural pre-ADC gain + SAR ADC bank of one array |
| `result_buffer.v` | digital accumulator + result memory |
| `cim_top.v` | wiring; `NUM_ARRAYS` channels from a generate loop |
| `tb_control.v` | testbench, plays the CPU and checks everything |
| `cpu_mapper.py` | the CPU's mapper (X, W → table + memory images) + golden reference |
| `run_tests.py` | runs the test progression |

---

## 1. The mapping (taken from `cim_sim/matmul.py`, not invented)

`X[N,K] @ W[K,M]`, array 32×32, `n_db = ceil(K/32)`, `n_ob = ceil(M/32)`:

```
for token_block:
    for output_block ob:
        for depth_block db:
            load tile W[db, ob]                  (skipped if already resident)
            stream the token block through it
            acc[tokens, ob] += ADC codes
```

**One array holds exactly one tile, and a tile is a (depth block, output block)
pair.** So with one array every depth block needs its own weight load: reloads
happen per depth block too, not only per output tile. TinyViT stage 3 qkv is 5
depth × 15 output = **75 weight loads**, with one COMPUTE of 196 tokens after each.

Token block `T` is purely a choice in the table: `T=N` above; `T=1` gives
`token → output → depth`, with far more weight loads and bit-identical results
(tested: `t4_8` vs `t4_8_T1`, 4 vs 32 weight loads). The RTL does not change.

## 2. The command table

One 120-bit word per command; fields at fixed positions (`command_defs.vh`).
The controller knows nothing about attention, linear layers or convolution.

| opcode | does | cycles |
|---|---|---|
| `CMD_CONFIG` | set `gain` and `act_signed` for what follows (like `set_operation()`: once per operation, never per depth block) | 2 |
| `CMD_CLEAR` | zero `count` consecutive result words (like `acc = np.zeros(...)`) | 2 + count |
| `CMD_LOAD_WGT` | write one tile from `weight_addr` into `array_first` | 2 + ROWS |
| `CMD_COMPUTE` | stream `count` tokens through the resident weights | 2 + count × (see §3) |
| `CMD_END` | raise `done` | 2 |

`CMD_COMPUTE` addressing, token `t`, array `a` in the group:

```
input word  = input_addr  + t*input_stride  + (in_step  ? a : 0)
result word = output_addr + t*output_stride + (out_step ? a : 0)
```

Stage-3 qkv table on one array (first entries):

```
CONFIG  gain=24 act_signed=1
CLEAR   output_addr=3 count=2940
LOAD_WGT array=0 tile(db0,ob0)
COMPUTE count=196 input_addr=.. input_stride=5 output_addr=3 output_stride=15
LOAD_WGT array=0 tile(db1,ob0)
COMPUTE count=196 input_addr=..+1 input_stride=5 output_addr=3 output_stride=15
...
```

Stage 3 qkv + QK^T + AV together fit in **185 commands**, because one command
covers a whole token range. A command per token would be about 17 000.

## 3. The FSM

```
IDLE --start--> FETCH --> DECODE --+-- CONFIG -----------------------> FETCH
                                   +-- LOAD_WEIGHT  (ROWS cycles) ---> FETCH
                                   +-- CLEAR_OUTPUT (count cycles) --> FETCH
                                   +-- END / illegal ----------------> DONE
                                   +-- COMPUTE:

   +-> LOAD_INPUT    one cycle per array in the group
   |   DAC_START     pulse
   |   WAIT_DAC      DAC_LATENCY cycles
   |   ARRAY_START   pulse: input applied, column outputs start to settle
   |   WAIT_ARRAY    ARRAY_LATENCY cycles
   |   ADC_START     pulse: ADC samples
   |   WAIT_ADC      ADC_LATENCY cycles
   |   WAIT_ACCUM    ARRAY_ACCUM_WAIT cycles (skipped if 0)
   |   ACCUMULATE    one cycle per array in the group
   +-- more tokens                      last token --> FETCH
```

Cycles per token, `g` arrays: `g + 1 + DAC + 1 + ARRAY + 1 + ADC + ACCUM_WAIT + g`
= **20** at the defaults with `g = 1`. The testbench checks total cycles exactly.

### Every control signal

| signal | high in | meaning |
|---|---|---|
| `cmd_rd_addr` | always | `pc` |
| `weight_we`, `weight_row` | LOAD_WEIGHT | write row `row` of `active_array` |
| `weight_rd_addr` | LOAD_WEIGHT | `weight_addr + row` |
| `input_load` | LOAD_INPUT | latch input codes into `active_array`'s DAC |
| `input_rd_addr` | LOAD_INPUT | §2 formula |
| `dac_start` | DAC_START | start conversion on every array in `group_mask` |
| `array_start` | ARRAY_START | apply the input on every array in `group_mask` |
| `adc_start` | ADC_START | sample on every array in `group_mask` |
| `result_clear` | CLEAR_OUTPUT | zero `result_addr` |
| `result_accum` | ACCUMULATE | `result_addr += codes of active_array` |
| `result_addr` | CLEAR / ACCUMULATE | §2 formula |
| `active_array` | always | `array_first + arr_idx`: the array a shared memory port serves now |
| `group_mask` | set in DECODE | arrays taking part in this COMPUTE |
| `adc_gain`, `act_signed` | set by CONFIG | analog domain, held until the next CONFIG |
| `busy`, `done`, `error` | | status. `error` = illegal opcode or array range; the run stops |

Internal counters: `pc`, `token_idx`, `arr_idx`, `row`, `wait_cnt`. That is all of
the address generation: a base plus an index times a stride. There is no AGU.

## 4. DAC, array, ADC

| block | model | relation to `cim_sim/analog/` |
|---|---|---|
| `dac.v` | `v_ia[r]` = input code, read signed or unsigned per `act_signed` | `dac.py`, signed analog-equivalent, in units of `v_per_unit` (R2R / CAPACITIVE family; topology unconfirmed) |
| `cim_array.v` | `v_oa[c] = Σ_r v_ia[r]·W[r][c]` | `macro.py`, in units of `K_MAC`. `m` = ROWS always, including zero-padded rows |
| `adc.v` | `d = floor((G·v_oa + 2^12) / 2^13)`, clipped to `[-128,127]`, clips counted | `adc.py`. `2^13 = LSB/K_MAC = m·2^B·2^DAC_BITS/2^ADC_BITS`, derived from configuration only |
| `result_buffer.v` | `mem[addr] += d` per column | `acc += d` in `matmul.py` |

`array_start` is a control abstraction. The macro has no compute operation of its
own; its outputs develop once the input is applied. The pulse gives the
controller a clean sequencing point.

**The waits are enforced, not decorative.** Each analog block drives X on its
output until its own latency has elapsed. A controller that waits one cycle too
few propagates X into the results and the test fails (§9).

**Gain** is an integer per-operation configuration value, never derived from data.
The verified set for this DAC family is ≈ {24, 8, 3}; non-integer gains would need
a fixed-point field.

**Result = integer sum of ADC codes.** The single rescale to real units
(`LSB_P·S_X·S_W`) is left to the CPU after `DONE`, and the two scales stay separate.

**There is no STORE command.** With the whole token range in flight, the partial
sums for every token must stay live until the last depth block, so they live in
the result memory itself (`psum_words_live = T·32` in the simulator). The last
ACCUMULATE leaves the final result in place; a separate store would be a copy.

## 5. Timing parameters

Set on `cim_top` (in tests: `Timing` in `cpu_mapper.py`). **All are placeholders.**
`cim_sim/` holds no timing data at all.

| parameter | default | note |
|---|---|---|
| `DAC_LATENCY` | 2 | placeholder |
| `ARRAY_LATENCY` | 4 | placeholder |
| `ADC_LATENCY` | 9 | placeholder; sized as 1 sample + 8 SAR decisions |
| `ARRAY_ACCUM_WAIT` | 0 | extra wait before accumulating |
| LOAD_WEIGHT | 1 row/cycle | assumption; the simulator does not model write time |

To use a real completion signal later, replace the one exit condition of a
`WAIT_*` state. Nothing else changes.

## 6. Multiple arrays

`NUM_ARRAYS` on `cim_top` builds that many DAC + array + ADC channels. A COMPUTE
names a group (`array_first`, `array_count`). Inside a group the analog stages
fire together, one `WAIT_*` for all; only the shared memory ports (LOAD_INPUT,
ACCUMULATE, LOAD_WEIGHT) go one array at a time. The table picks the split, and
**output tiles is the default mapping** (`Program.mode = "output"`), matching the
simulator:

| | `in_step` | `out_step` | arrays hold | result |
|---|---|---|---|---|
| **output tiles** (default; what `matmul.py` does) | 0 | 1 | different output tiles of one depth block; one input goes to all | one word per array |
| depth blocks (`mode="depth"`; still tested) | 1 | 0 | different depth blocks of one output tile | codes of all arrays add into one word |

Both give bit-identical results to one array, since addition commutes. Stage 3 on
4 arrays, same data and timing placeholders:

| | timesteps | cycles |
|---|---|---|
| 1 array | 17 444 | 356 606 |
| 4 arrays, depth blocks | 7 644 | 203 038 |
| 4 arrays, output tiles | 5 684 | 161 858 |

(The 4-array runs use `ARRAY_ACCUM_WAIT=3`, the 1-array run 0.) Output tiles win
here because qkv has 15 output tiles but only 5 depth blocks. Weight loads remain
sequential, and broadcast still loads each array's input register in its own cycle.

## 7. Running

```bash
python run_tests.py              # all tests, XSim on PATH
python run_tests.py t1 t2        # only these
python run_tests.py --list
python run_tests.py --sim iverilog   # written, not run here: Icarus is not installed
```

Each test's files and `sim.log` are in `build/<test>/`; `build/summary.csv` holds
the numbers and configuration. `t1` and `t2` print the full FSM trace and every
partial sum. To try other data, add an entry in `build_tests()` in `run_tests.py`.

## 8. Tests

| test | what it proves |
|---|---|
| `t1` | 1 token, 1 depth, 1 output block. Hand-checkable: `P[c] = 764c − 7747`, `d[c] = round(P[c]/1024)` = −8, −7, −6, −5 … |
| `t2` | 3 depth blocks (ragged): col 0 goes 0 → 19 → 20 → 21 |
| `t3`, `t3_slow` | 2 depth × 3 output blocks: 6 weight reloads; other latencies |
| `t4_8`, `t4_8_T1`, `t4_8u` | 8 tokens; token block T=N vs T=1; unsigned inputs |
| `t4_196` | TinyViT stage 3 shapes, 196 tokens, qkv + QK^T + AV in one table at gains 24/8/3 |
| `t5a_*`, `t5b_*`, `t5c_*` | 2, 4, 8 arrays by depth block and by output tile, against the 1-array reference |
| `t6_clip` | gain 255: clipping counted |
| `t7_*` | illegal opcode / array out of range raise `error` and stop |

Every test checks: all result words against the golden values, exact cycle count,
weight writes, array ops, timesteps and clip count, and that nothing was written
just outside the result range.

## 9. Results

XSim 2022.2, seeded synthetic data. Latencies are placeholders.

**20/20 tests pass.** Multi-array results are identical to their 1-array references.

**Against the real simulator.** `run_tests.py` also runs `cim_matmul` through the
actual analog chain (capacitive DAC, 8b/8b/8b) and recovers its integer code sums:

- Weight writes, array ops and clip counts match its trace in every test; timesteps
  too, except depth-block mode, which the simulator has no equivalent for.
- **Stage 3: 100 of 138 768 result cells differ from `cim_sim`, each by 1 code.** All
  100 are explained by exact rounding ties (463 tie conversions in the run; 0 cells
  unexplained). The RTL always rounds an exact tie up; `cim_sim` computes the same
  voltage in floating point with Vref = 0.6, so its ties fall either way. All
  other tests have 0 differing cells.

**End-to-end error**, code sums rescaled once and compared with float `X @ W`:
qkv 1.6%, QK^T 2.0%, AV 13.6%. This is **random data, not TinyViT activations**,
and the gains were calibrated on other data, so it says nothing about accuracy.
It only shows that the rescale chain closes.

**Fault injection** (scratch copies of the RTL, one fault each): all 20 faults
below make their test fail:
DAC / array / ADC wait one cycle short (results go X), accumulate overwrites, no
clear, weight reload reads first tile, wrong row, load stops after one row, token
stride ignored on input / output, `in_step` / `out_step` ignored, second array never
started, accumulate always takes array 0, unsigned read as signed, ADC truncates,
gain ignored, clipping not counted, illegal opcode accepted, array range unchecked.

## 10. Assumptions and open points

| item | status |
|---|---|
| 32×32, one ADC per column, SAR 8b | hardware-confirmed |
| capacitive DAC transfer | candidate, the simulator's default; topology unconfirmed |
| signed weights and signed MAC | behavioural abstraction; physical realisation is an open question |
| `m = ROWS` for padded tiles | simulator assumption, open question to the hardware team |
| all latencies | placeholders |
| memory reads combinational | simplification; a real SRAM adds a cycle per read |
| ideal analog | no nonidealities modelled, unlike the hooks in `cim_sim/analog` |
| `ACC_W = 32` | generous; 7 depth blocks × 8-bit codes need about 11 bits |

**Next natural extensions** (not built): overlapping token loads with the analog
wait, parallel weight loads, per-array commands so arrays run independently, and
the DAC-to-input bit-plane question once DAC width differs from input width.
