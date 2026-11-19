"""
run_tests.py -- progressive verification of the CIM array controller.

    python run_tests.py                  all tests, Vivado XSim
    python run_tests.py t1 t2            only these
    python run_tests.py --list
    python run_tests.py --sim iverilog   Icarus Verilog instead

For each test this writes the CPU's memory images into build/<test>/, checks the
exact-integer golden values against the real Python simulator, compiles, runs
tb_control there, and prints one summary line. Numbers are on seeded synthetic
data; timing values are placeholders.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time

import numpy as np

from cpu_mapper import CMD_COMPUTE, Op, Program, Timing, cim_sim_reference, pack_cmd, tie_map

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = ["command_table.v", "token_buffer.v", "weight_memory.v", "result_buffer.v",
           "dac.v", "cim_array.v", "adc.v", "array_controller.v", "cim_top.v", "tb_control.v"]


def signed_data(rng, shape):
    return rng.standard_normal(shape)


def attention_data(rng, shape):
    """Non-negative rows summing to 1, shaped like post-softmax attention."""
    e = np.exp(2.0 * rng.standard_normal(shape))
    return e / e.sum(axis=1, keepdims=True)


def hand_op():
    """Checkable by hand. Quantization leaves both operands unchanged (scale = 1).

    x[r] = 4r + 3;  W[r][c] = +127 if r == c, -64 if r == 31 - c, else 0
    => column sum  P[c] = 127*x[c] - 64*x[31-c] = 764c - 7747
    => code        d[c] = round(8 * P[c] / 8192) = round(P[c] / 1024)
    """
    X = np.array([[4.0 * r + 3 for r in range(32)]])
    W = np.zeros((32, 32))
    for c in range(32):
        W[c, c] = 127
        W[31 - c, c] = -64
    return Op(X, W, gain=8)


def stage3_ops(seed=1):
    """TinyViT-5M stage 3 shapes (N=196, dim=160, 5 heads, key_dim=32). Gains {24,8,3}.
    Seeded synthetic data, not real activations."""
    rng = np.random.default_rng(seed)
    return [Op(signed_data(rng, (196, 160)), signed_data(rng, (160, 480)), gain=24),
            Op(signed_data(rng, (196, 32)), signed_data(rng, (32, 196)), gain=8),
            Op(attention_data(rng, (196, 196)), signed_data(rng, (196, 32)), gain=3,
               act_signed=False)]


def small_op(N, K, M, gain=8, T=None, seed=1):
    rng = np.random.default_rng(seed)
    return Op(signed_data(rng, (N, K)), signed_data(rng, (K, M)), gain=gain, T=T)


def unsigned_op(N, K, M, gain=3, seed=8):
    rng = np.random.default_rng(seed)
    return Op(attention_data(rng, (N, K)), signed_data(rng, (K, M)), gain=gain, act_signed=False)


# commands the controller must refuse, placed where CMD_END would be
BAD_TAIL = {"t7_opcode": 15,
            "t7_array": pack_cmd(CMD_COMPUTE, array_first=1, array_count=2, count=1)}


# name: (description, ops, P, mode, Timing, verbose)
def build_tests():
    slow = Timing(dac=5, array=7, adc=11, accum_wait=3)
    t5a = lambda: [small_op(16, 160, 64, seed=5)]
    t5b = lambda: [small_op(16, 64, 160, seed=6)]
    return {
        "t1":        ("1 token, 1 depth block, 1 output block", [hand_op()], 1, "output", Timing(), True),
        "t2":        ("1 token, 3 depth blocks (ragged), 1 output block", [small_op(1, 80, 32, seed=2)], 1, "output", Timing(), True),
        "t3":        ("1 token, 2 depth x 3 output blocks (ragged), reloads", [small_op(1, 64, 80, seed=3)], 1, "output", Timing(), False),
        "t3_slow":   ("same as t3, different latencies", [small_op(1, 64, 80, seed=3)], 1, "output", slow, False),
        "t4_8":      ("8 tokens, T=N: tile stays resident for all tokens", [small_op(8, 64, 64, seed=4)], 1, "output", Timing(), False),
        "t4_8_T1":   ("8 tokens, T=1: tile reloaded per token", [small_op(8, 64, 64, T=1, seed=4)], 1, "output", Timing(), False),
        "t4_8u":     ("8 tokens, unsigned inputs (post-softmax style)", [unsigned_op(8, 64, 32)], 1, "output", Timing(), False),
        "t4_196":    ("TinyViT stage 3: qkv + QK^T + AV, 196 tokens, in one table", stage3_ops(), 1, "output", Timing(), False),
        "t5a_ref":   ("reference for t5a: 1 array", t5a(), 1, "depth", Timing(), False),
        "t5a_P2":    ("2 arrays on different depth blocks", t5a(), 2, "depth", Timing(accum_wait=3), False),
        "t5a_P4":    ("4 arrays on different depth blocks", t5a(), 4, "depth", Timing(accum_wait=3), False),
        "t5a_P8":    ("8 arrays instantiated, 5 depth blocks", t5a(), 8, "depth", Timing(accum_wait=3), False),
        "t5b_ref":   ("reference for t5b: 1 array", t5b(), 1, "output", Timing(), False),
        "t5b_P2":    ("2 arrays on different output tiles (simulator's mapping)", t5b(), 2, "output", Timing(accum_wait=3), False),
        "t5b_P4":    ("4 arrays on different output tiles (simulator's mapping)", t5b(), 4, "output", Timing(accum_wait=3), False),
        "t5c_depth4": ("TinyViT stage 3 on 4 arrays, depth blocks", stage3_ops(), 4, "depth", Timing(accum_wait=3), False),
        "t5c_out4":  ("TinyViT stage 3 on 4 arrays, output tiles", stage3_ops(), 4, "output", Timing(accum_wait=3), False),
        "t6_clip":   ("gain 255: ADC clipping counted, not hidden", [small_op(4, 32, 32, gain=255, seed=7)], 1, "output", Timing(), False),
        "t7_opcode": ("illegal opcode: error flag, stop", [small_op(1, 32, 32, seed=9)], 1, "output", Timing(), False),
        "t7_array":  ("COMPUTE on an array that does not exist: error flag, stop", [small_op(1, 32, 32, seed=9)], 2, "output", Timing(), False),
    }


def check_against_simulator(prog, ops, accs, P, mode):
    """Golden integer sums and mapper counters vs the real cim_sim analog chain.

    cim_sim has no depth-block scheduling, so mode="depth" is compared with its
    one-array run: same codes and same work, but its wall-clock count is not comparable.
    """
    ref_P = P if mode == "output" else 1
    differs = ties = unexplained = 0
    writes = array_ops = timesteps = clips = 0
    rel_errs = []
    for op, acc in zip(ops, accs):
        ref, trace, y_real, y_float = cim_sim_reference(op, ref_P)
        tm = tie_map(op)
        differs += int(np.count_nonzero(ref != acc))
        ties += int(tm.sum())
        unexplained += int(np.count_nonzero(np.abs(ref - acc) > tm))
        writes += trace.stats.weight_tile_writes
        array_ops += trace.stats.array_ops
        timesteps += trace.parallel_array_cycles
        clips += trace.stats.adc_saturations
        rel_errs.append(np.linalg.norm(y_real - y_float) / np.linalg.norm(y_float))
    counters_ok = (writes == prog.tile_writes and array_ops == prog.array_ops
                   and clips == prog.clips
                   and (mode == "depth" and P > 1 or timesteps == prog.timesteps))
    return dict(differs=differs, ties=ties, unexplained=unexplained, counters_ok=counters_ok,
                sim=(writes, array_ops, timesteps, clips), rel_errs=rel_errs)


def run_sim(sim, directory, src):
    sources = [os.path.join(src, s) for s in SOURCES]
    if sim == "xsim":
        steps = [["cmd", "/c", "xvlog", "-i", src, "-i", "."] + sources,
                 ["cmd", "/c", "xelab", "tb_control", "-s", "sim"],
                 ["cmd", "/c", "xsim", "sim", "-R"]]
    else:
        steps = [["iverilog", "-g2005", "-I", src, "-I", ".", "-s", "tb_control", "-o", "sim.vvp"] + sources,
                 ["vvp", "-n", "sim.vvp"]]
    output = ""
    for step in steps:
        p = subprocess.run(step, cwd=directory, capture_output=True, text=True)
        output += p.stdout + p.stderr
        if p.returncode != 0:
            break
    with open(os.path.join(directory, "sim.log"), "w") as f:
        f.write(output)
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tests", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--sim", choices=["xsim", "iverilog"], default="xsim")
    ap.add_argument("--src", default=HERE, help="directory holding the RTL (for fault injection)")
    ap.add_argument("--build", default=os.path.join(HERE, "build"))
    ap.add_argument("--no-reference", action="store_true", help="skip the cim_sim cross-check")
    args = ap.parse_args()

    tests = build_tests()
    if args.list:
        for k, v in tests.items():
            print(f"{k:10s} {v[0]}")
        return 0

    names = args.tests or list(tests)
    print("vs cim_sim = cells differing / exact ties / cells a tie cannot explain, counters\n")
    print(f"{'test':12s} {'P':>2s} {'mode':6s} {'cmds':>5s} {'cycles':>8s} {'writes':>6s} "
          f"{'ops':>6s} {'steps':>6s} {'clips':>6s}  {'vs cim_sim':>16s}  RTL")
    rows, failed = [], 0
    for name in names:
        desc, ops, P, mode, timing, verbose = tests[name]
        d = os.path.join(args.build, name)
        prog = Program(P=P, mode=mode, timing=timing, bad_tail=BAD_TAIL.get(name))
        accs = [prog.add(op, i) for i, op in enumerate(ops)]
        prog.write(d, name, verbose)

        ref = None
        if not args.no_reference:
            ref = check_against_simulator(prog, ops, accs, P, mode)
            if ref["unexplained"] or not ref["counters_ok"]:
                failed += 1
                print(f"    REFERENCE DISAGREES: {ref}")

        t0 = time.time()
        log = run_sim(args.sim, d, args.src)
        ok = "RESULT: PASS" in log
        failed += not ok
        sim_col = (f"{ref['differs']:4d}/{ref['ties']:4d}/{ref['unexplained']:d} "
                   f"{'ok' if ref['counters_ok'] else 'BAD'}") if ref else "skipped"
        print(f"{name:12s} {P:2d} {mode:6s} {len(prog.cmds):5d} {prog.cycles:8d} {prog.tile_writes:6d} "
              f"{prog.array_ops:6d} {prog.timesteps:6d} {prog.clips:6d}  "
              f"{sim_col:>16s}  {'PASS' if ok else 'FAIL'}   ({time.time()-t0:.0f}s)", flush=True)
        if not ok:
            for line in log.splitlines():
                if line.startswith(("MISMATCH", "ERROR", "RESULT")):
                    print("    " + line)
        rows.append(dict(test=name, description=desc, arrays=P, mode=mode, commands=len(prog.cmds),
                         cycles=prog.cycles, tile_writes=prog.tile_writes, array_ops=prog.array_ops,
                         timesteps=prog.timesteps, clips=prog.clips,
                         sim_diffs=ref["differs"] if ref else "", sim_ties=ref["ties"] if ref else "",
                         sim_unexplained=ref["unexplained"] if ref else "",
                         sim_counters_ok=ref["counters_ok"] if ref else "",
                         rel_err_vs_float=";".join(f"{e:.4f}" for e in ref["rel_errs"]) if ref else "",
                         dac_latency=timing.dac, array_latency=timing.array, adc_latency=timing.adc,
                         accum_wait=timing.accum_wait, result="PASS" if ok else "FAIL"))

    os.makedirs(args.build, exist_ok=True)
    with open(os.path.join(args.build, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"\n{len(names) - failed}/{len(names)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
