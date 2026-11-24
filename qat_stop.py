#!/usr/bin/env python3
"""
qat_stop.py — list or stop cim_qat jobs, including ALL their worker processes.

  python qat_stop.py list                 # every cim_qat process (eval, calib, train) and its workers
  python qat_stop.py eval --dry-run       # what `eval` would stop
  python qat_stop.py eval                 # stop every evaluation job (qat_evaluate / qat_eval_matrix)

Why: simulator workers started with the "spawn" method have a generic command line
("python -c from multiprocessing.spawn import spawn_main ...") that `pkill -f
qat_evaluate.py` does not match; on 28 Oct orphaned workers of a stopped
evaluation kept running with 32 BLAS threads each. Since v2.6 every worker also asks the kernel
to terminate it when its parent dies, so orphans cannot recur.

A process is selected for `eval` if EITHER
  * its environment carries CIM_QAT_JOB=eval:... (set by qat_evaluate.py /
    qat_eval_matrix.py since v2.6 and inherited by every worker), OR
  * it is a Python process running qat_evaluate.py or qat_eval_matrix.py (this
    also catches jobs started with v2.5), OR it descends from such a process
    (its simulator workers, whatever their command line);
and NEVER if it is, or descends from, a qat_train.py process or carries
CIM_QAT_JOB=train:... (training and its spot-check workers are left alone).
Stops with SIGTERM, waits up to 15 s, then SIGKILL. Linux only (/proc).
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time


def procs() -> dict:
    out = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        pid = int(d)
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                argv = [x.decode(errors="replace") for x in f.read().split(b"\0") if x]
            cmd = " ".join(argv)
            with open(f"/proc/{pid}/stat") as f:
                st = f.read()
            ppid = int(st.rsplit(")", 1)[1].split()[1])
            try:
                with open(f"/proc/{pid}/environ", "rb") as f:
                    env = f.read().split(b"\0")
                tag = next((e.split(b"=", 1)[1].decode() for e in env if e.startswith(b"CIM_QAT_JOB=")), "")
            except PermissionError:
                tag = ""
            with open(f"/proc/{pid}/status") as f:
                thr = next((int(line.split()[1]) for line in f if line.startswith("Threads:")), 0)
        except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError, ValueError):
            continue
        out[pid] = dict(pid=pid, ppid=ppid, argv=argv, cmd=cmd, tag=tag, threads=thr)
    return out


def descendants(P: dict, roots: set) -> set:
    kids = {}
    for p in P.values():
        kids.setdefault(p["ppid"], []).append(p["pid"])
    out, stack = set(), list(roots)
    while stack:
        q = stack.pop()
        if q in out:
            continue
        out.add(q)
        stack += kids.get(q, [])
    return out


def runs_script(p: dict, names) -> bool:
    """True if p is a Python interpreter running one of the scripts `names`
    (so an editor or `tail` with the file name on its command line is not matched)."""
    argv = p["argv"]
    if not argv or "python" not in os.path.basename(argv[0]):
        return False
    return any(os.path.basename(x) in names for x in argv[1:3])


def classify(P: dict, only: str = "") -> dict:
    train_roots = {p["pid"] for p in P.values()
                   if runs_script(p, ("qat_train.py",)) or p["tag"].startswith("train:")}
    protected = descendants(P, train_roots)

    def roots(scripts, prefix):
        r = {p["pid"] for p in P.values() if runs_script(p, scripts) or p["tag"].startswith(prefix)}
        if only:                                    # restrict to trees whose root names `only`
            r = {q for q in r if only in P[q]["cmd"] or only in P[q]["tag"]}
        return r

    me = {os.getpid(), os.getppid()}
    return dict(train=protected - me,
                eval=descendants(P, roots(("qat_evaluate.py", "qat_eval_matrix.py"), "eval:")) - protected - me,
                calib=descendants(P, roots(("qat_calibrate.py",), "calib:")) - protected - me)


def show(P, pids, title):
    print(f"{title}: {len(pids)} process(es)")
    for pid in sorted(pids):
        p = P[pid]
        print(f"  pid {pid:>8} ppid {p['ppid']:>8} threads {p['threads']:>3} tag {p['tag'] or '-':<16} "
              f"{p['cmd'][:110]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["list", "eval", "calib"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="", help="only jobs whose root command line or CIM_QAT_JOB tag "
                    "contains this text (used by the tests; normally not needed)")
    a = ap.parse_args()
    if not os.path.isdir("/proc"):
        sys.exit("Linux /proc required")
    P = procs()
    C = classify(P, a.only)
    if a.what == "list":
        for k in ("train", "eval", "calib"):
            show(P, C[k], k)
        return
    target = C[a.what]
    show(P, target, f"{a.what} (to stop)" if not a.dry_run else f"{a.what} (dry run, nothing stopped)")
    show(P, C["train"], "train (left running)")
    if a.dry_run or not target:
        return
    for pid in target:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    t0 = time.time()
    while time.time() - t0 < 15:
        alive = [pid for pid in target if os.path.exists(f"/proc/{pid}")]
        if not alive:
            break
        time.sleep(0.5)
    alive = [pid for pid in target if os.path.exists(f"/proc/{pid}")]
    for pid in alive:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(1)
    left = classify(procs(), a.only)[a.what]
    print(f"stopped {len(target)} process(es) ({len(alive)} needed SIGKILL); remaining {a.what} processes: {len(left)}")
    sys.exit(0 if not left else 1)


if __name__ == "__main__":
    main()
