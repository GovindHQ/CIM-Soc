"""
cpu_mapper.py -- the CPU's side of the control path.

Given X[N,K] and W[K,M] it produces everything the RTL needs: the command table,
the token buffer image and the weight memory image. The hardware only executes
the table; all knowledge of the mapping lives here.

The loop nest is the one in cim_sim/matmul.py, with one array:

    for token_block:
        for output_block:
            for depth_block:
                load tile W[db, ob]                 (skipped if already resident)
                stream the token block through it   (one CMD_COMPUTE)

Two ways of spreading work over P arrays are available:

    mode="output"  DEFAULT. Arrays hold different OUTPUT tiles of one depth block;
                   one input word is sent to all of them. This is what matmul.py does.
    mode="depth"   Arrays hold different DEPTH blocks of one output tile; each
                   gets its own input word and their codes add into one result.

golden() is an exact integer reference for the ADC-code sum. cim_sim_reference()
gets the same thing out of the real simulator, so the two can be compared.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cim_sim.array import CIMArray, CIMConfig          # noqa: E402
from cim_sim.analog import AnalogConfig                # noqa: E402
from cim_sim.matmul import TraceLog, cim_matmul        # noqa: E402
from cim_sim.quant import quantize                     # noqa: E402

R, C = 32, 32           # array rows / columns
ACT_BITS = W_BITS = ADC_BITS = 8
ADC_SHIFT = 13          # log2(m) + B + DAC_BITS - ADC_BITS, see adc.v

CMD_END, CMD_CONFIG, CMD_LOAD_WGT, CMD_CLEAR, CMD_COMPUTE = range(5)


def pack_cmd(op, array_first=0, array_count=0, weight_addr=0, input_addr=0,
             input_stride=0, output_addr=0, output_stride=0, count=0,
             in_step=0, out_step=0, act_signed=0, gain=0) -> int:
    """One command word. Field positions must match command_defs.vh."""
    return (op
            | array_first << 4
            | array_count << 12
            | weight_addr << 20
            | input_addr << 36
            | input_stride << 52
            | output_addr << 64
            | output_stride << 80
            | count << 92
            | in_step << 108
            | out_step << 109
            | act_signed << 110
            | gain << 111)


@dataclass
class Timing:
    """Clock cycles. All PLACEHOLDERS; the simulator holds no timing data."""
    dac: int = 2
    array: int = 4
    adc: int = 9
    accum_wait: int = 0


@dataclass
class Op:
    """One logical X @ W."""
    X: np.ndarray
    W: np.ndarray
    gain: int
    act_signed: bool = True
    T: int | None = None       # token block; None = all tokens


@dataclass
class Program:
    """Accumulates what the CPU loads, across every operation in one run."""
    P: int = 1
    mode: str = "output"
    timing: Timing = field(default_factory=Timing)
    cmds: list = field(default_factory=list)
    input_img: list = field(default_factory=list)
    weight_img: list = field(default_factory=list)
    expected: list = field(default_factory=list)
    output_base: int = 3       # non-zero on purpose: wrong addressing must show
    cycles: int = 0
    tile_writes: int = 0
    array_ops: int = 0
    timesteps: int = 0
    clips: int = 0
    resident: dict = field(default_factory=dict)
    bad_tail: int | None = None    # a command the controller must refuse

    def __post_init__(self):
        # decoys at the addresses no command should touch
        self.input_img = [decoy_word(R)] * 5
        self.weight_img = [decoy_word(C)] * R
        self._next_output = self.output_base

    # -- the commands, with what each one costs ----------------------------- #

    def _config(self, gain, act_signed):
        self.cmds.append(pack_cmd(CMD_CONFIG, gain=gain, act_signed=int(act_signed)))
        self.cycles += 2                                # FETCH, DECODE

    def _clear(self, addr, count):
        self.cmds.append(pack_cmd(CMD_CLEAR, output_addr=addr, count=count))
        self.cycles += 2 + count

    def _load_weight(self, array, addr, tile_key):
        if self.resident.get(array) == tile_key:
            return                                      # already resident, no write
        self.resident[array] = tile_key
        self.cmds.append(pack_cmd(CMD_LOAD_WGT, array_first=array, weight_addr=addr))
        self.cycles += 2 + R
        self.tile_writes += 1

    def _compute(self, count, array_count, **fields):
        self.cmds.append(pack_cmd(CMD_COMPUTE, array_first=0, array_count=array_count,
                                  count=count, **fields))
        t = self.timing
        per_token = (array_count                        # LOAD_INPUT
                     + 1 + t.dac + 1 + t.array + 1 + t.adc + t.accum_wait
                     + array_count)                     # ACCUMULATE
        self.cycles += 2 + count * per_token
        self.array_ops += count * array_count
        self.timesteps += count

    def _end(self):
        self.cmds.append(pack_cmd(CMD_END))
        self.cycles += 2

    # -- mapping one X @ W -------------------------------------------------- #

    def add(self, op: Op, op_id: int = 0):
        N, K = op.X.shape
        M = op.W.shape[1]
        xq, _ = quantize(op.X, bits=ACT_BITS, axis=1, signed=op.act_signed)
        wq, _ = quantize(op.W, bits=W_BITS, axis=0, signed=True)
        n_db, n_ob = (K + R - 1) // R, (M + C - 1) // C
        xp = np.zeros((N, n_db * R), dtype=np.int64)
        xp[:, :K] = xq
        wp = np.zeros((n_db * R, n_ob * C), dtype=np.int64)
        wp[:K, :M] = wq

        # memory layout, chosen here and nowhere else
        ibase = len(self.input_img)
        wbase = len(self.weight_img)
        obase = self._next_output
        self._next_output += N * n_ob
        for t in range(N):
            for db in range(n_db):
                self.input_img.append(pack_vec(xp[t, db * R:(db + 1) * R], ACT_BITS))
        for db in range(n_db):
            for ob in range(n_ob):
                for r in range(R):
                    self.weight_img.append(
                        pack_vec(wp[db * R + r, ob * C:(ob + 1) * C], W_BITS))

        in_addr = lambda t, db: ibase + t * n_db + db
        out_addr = lambda t, ob: obase + t * n_ob + ob
        tile_addr = lambda db, ob: wbase + (db * n_ob + ob) * R

        self._config(op.gain, op.act_signed)
        self._clear(obase, N * n_ob)

        T = max(1, min(op.T or N, N))
        P = self.P
        for tb in range((N + T - 1) // T):
            t0, t1 = tb * T, min((tb + 1) * T, N)
            common = dict(count=t1 - t0, input_stride=n_db, output_stride=n_ob)
            if self.mode == "output":
                for group in range((n_ob + P - 1) // P):
                    obs = range(group * P, min((group + 1) * P, n_ob))
                    for db in range(n_db):
                        for slot, ob in enumerate(obs):
                            self._load_weight(slot, tile_addr(db, ob), (op_id, db, ob))
                        self._compute(array_count=len(obs), out_step=1,
                                      input_addr=in_addr(t0, db),
                                      output_addr=out_addr(t0, obs[0]), **common)
            else:
                for ob in range(n_ob):
                    for group in range((n_db + P - 1) // P):
                        dbs = range(group * P, min((group + 1) * P, n_db))
                        for slot, db in enumerate(dbs):
                            self._load_weight(slot, tile_addr(db, ob), (op_id, db, ob))
                        self._compute(array_count=len(dbs), in_step=1,
                                      input_addr=in_addr(t0, dbs[0]),
                                      output_addr=out_addr(t0, ob), **common)

        acc, clips = golden(xp, wp, op.gain)
        self.clips += clips
        for t in range(N):
            for ob in range(n_ob):
                self.expected.append(pack_vec(acc[t, ob * C:(ob + 1) * C], 32))
        return acc[:, :M]

    # -- files for the testbench --------------------------------------------- #

    def write(self, directory: str, name: str, verbose: bool):
        if self.bad_tail is not None:
            self.cmds.append(self.bad_tail)     # FETCH, DECODE, then DONE with error
            self.cycles += 2
        else:
            self._end()
        os.makedirs(directory, exist_ok=True)
        self.input_img.append(decoy_word(R))
        self.weight_img += [decoy_word(C)] * R
        write_hex(os.path.join(directory, "cmd.hex"), self.cmds, 120)
        write_hex(os.path.join(directory, "input.hex"), self.input_img, R * ACT_BITS)
        write_hex(os.path.join(directory, "weight.hex"), self.weight_img, C * W_BITS)
        write_hex(os.path.join(directory, "expected.hex"), self.expected, C * 32)
        t = self.timing
        params = dict(NUM_ARRAYS=self.P, DAC_LATENCY=t.dac, ARRAY_LATENCY=t.array,
                      ADC_LATENCY=t.adc, ARRAY_ACCUM_WAIT=t.accum_wait,
                      N_CMD=len(self.cmds), N_INPUT=len(self.input_img),
                      N_WEIGHT=len(self.weight_img), N_EXPECTED=len(self.expected),
                      OUTPUT_BASE=self.output_base, EXP_CYCLES=self.cycles,
                      EXP_TILE_WRITES=self.tile_writes, EXP_ARRAY_OPS=self.array_ops,
                      EXP_TIMESTEPS=self.timesteps, EXP_CLIPS=self.clips,
                      EXP_ERROR=int(self.bad_tail is not None),
                      VERBOSE=int(verbose))
        with open(os.path.join(directory, "tb_params.vh"), "w") as f:
            f.write(f'localparam TEST_NAME = "{name}";\n')
            for k, v in params.items():
                f.write(f"localparam {k} = {v};\n")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def pack_vec(values, bits: int) -> int:
    """Element 0 goes in the least significant bits."""
    word, mask = 0, (1 << bits) - 1
    for i, v in enumerate(values):
        word |= (int(v) & mask) << (i * bits)
    return word


def decoy_word(n: int) -> int:
    return pack_vec([0x55] * n, 8)


def write_hex(path: str, words, bits: int) -> None:
    digits = (bits + 3) // 4
    with open(path, "w") as f:
        for w in words:
            f.write(f"{w:0{digits}x}\n")


def adc_transfer(p: np.ndarray, gain: int):
    """Exact integer form of the SAR ADC. Returns (codes, n_clipped)."""
    v = gain * p
    swing = 1 << (ADC_SHIFT + ADC_BITS - 1)
    clips = int(np.count_nonzero(np.abs(v) > swing))
    d = np.floor_divide(v + (1 << (ADC_SHIFT - 1)), 1 << ADC_SHIFT)
    return np.clip(d, -(1 << (ADC_BITS - 1)), (1 << (ADC_BITS - 1)) - 1), clips


def golden(xp: np.ndarray, wp: np.ndarray, gain: int):
    """acc[t, col] = sum over depth blocks of the ADC code of that block's column sum."""
    N, Kp = xp.shape
    acc = np.zeros((N, wp.shape[1]), dtype=np.int64)
    clips = 0
    for db in range(Kp // R):
        p = xp[:, db * R:(db + 1) * R] @ wp[db * R:(db + 1) * R, :]
        d, c = adc_transfer(p, gain)
        acc += d
        clips += c
    return acc, clips


def tie_map(op: Op) -> np.ndarray:
    """ties[t, col] = conversions landing exactly between two codes for that cell.

    The golden rule rounds an exact tie up. cim_sim computes the same voltage in
    floating point, where Vref = 0.6 is inexact, so a tie can fall either way. A
    cell can therefore differ from cim_sim by at most its tie count, and no more.
    """
    N, K = op.X.shape
    xq, _ = quantize(op.X, bits=ACT_BITS, axis=1, signed=op.act_signed)
    wq, _ = quantize(op.W, bits=W_BITS, axis=0, signed=True)
    ties = np.zeros((N, op.W.shape[1]), dtype=np.int64)
    for db in range((K + R - 1) // R):
        p = xq[:, db * R:(db + 1) * R] @ wq[db * R:(db + 1) * R, :]
        ties += (op.gain * p) % (1 << ADC_SHIFT) == 1 << (ADC_SHIFT - 1)
    return ties


def cim_sim_reference(op: Op, P: int = 1):
    """Run the real simulator's analog chain and recover its integer code sum.

    cim_matmul returns acc * LSB_P * S_X * S_W, so dividing the scales back out and
    rounding gives acc. Returns (acc, trace, y_real, y_float).
    """
    acfg = AnalogConfig(enabled=True, dac_type="CAPACITIVE", gain_default=float(op.gain),
                        adc_bits=ADC_BITS)
    cfg = CIMConfig(rows=R, cols=C, weight_bits=W_BITS, act_bits=ACT_BITS,
                    token_block=op.T, analog=acfg)
    arrays = [CIMArray(cfg) for _ in range(P)]
    log = TraceLog()
    y = cim_matmul(op.X, op.W, arrays=arrays, log=log, act_signed=op.act_signed)
    _, sx = quantize(op.X, bits=ACT_BITS, axis=1, signed=op.act_signed)
    _, sw = quantize(op.W, bits=W_BITS, axis=0, signed=True)
    acc = np.rint(y / (arrays[0].chain.lsb_p * sx * sw)).astype(np.int64)
    return acc, log.records[0], y, op.X @ op.W
