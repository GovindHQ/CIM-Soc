#!/usr/bin/env python3
"""
tests_qat/test_threads.py — the v2.6 thread limit, orphan protection and stop tool.

  python tests_qat/test_threads.py

  H1  limit_blas_threads() lowers a RUNNING OpenBLAS pool from 2 threads to 1
      (threadpoolctl path, and the ctypes fallback with threadpoolctl hidden)
  H2  the evaluation / calibration entry points set the thread variables BEFORE
      numpy loads: run with no thread variables at all, numpy's BLAS ends up with
      1 thread (control: plain `import numpy` in the same environment gets > 1)
  H3  qat_eval_matrix.py exports the thread variables and the eval job tag to
      the commands it starts
  H4  simulator workers run with 1 BLAS thread even when the parent's BLAS has 2
      (fork start method), recorded per chunk and as summary
      blas_threads_per_worker == [1]; the caller's environment is restored
  H5  die_with_parent(): a worker whose parent is SIGKILLed is terminated by the
      kernel (control: without it the worker survives as an orphan)
  H6  qat_stop.py: finds a tagged evaluation tree (incl. a spawn worker), an
      untagged v2.5-style qat_eval_matrix.py tree, never a qat_train.py tree or a
      train-tagged process; `eval` stops exactly the evaluation processes.
      Every fake process carries a random marker and qat_stop runs with
      --only <marker>, so real jobs on the machine are never listed or touched.

Linux only. Uses the local pretrained checkpoint for H4.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
PY = sys.executable
RES = []
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS")


def check(name, ok, detail=""):
    RES.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    return ok


def env_without_threads(**extra):
    e = {k: v for k, v in os.environ.items() if k not in THREAD_VARS and k != "CIM_QAT_JOB"}
    e.update({k: str(v) for k, v in extra.items()})
    return e


def py(code, env, timeout=300):
    p = subprocess.run([PY, "-c", textwrap.dedent(code)], cwd=REPO, env=env, capture_output=True,
                       text=True, timeout=timeout)
    if p.returncode:
        print(p.stdout[-2000:], p.stderr[-3000:])
    return p.returncode, p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""


def alive(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def h1():
    code = """
        import sys, json
        sys.path.insert(0, ".")
        if HIDE: sys.modules["threadpoolctl"] = None
        import numpy as np
        np.ones((64, 64)) @ np.ones((64, 64))
        from cim_qat.common import blas_threads, limit_blas_threads
        b = blas_threads(); a = limit_blas_threads(1)
        print(json.dumps(dict(before=b, after=a)))
    """
    for hide in (False, True):
        rc, out = py(code.replace("HIDE", str(hide)), env_without_threads(OPENBLAS_NUM_THREADS=2))
        d = json.loads(out) if rc == 0 else {}
        b, a = d.get("before", {}), d.get("after", {})
        check(f"H1 limit_blas_threads 2 -> 1 ({'ctypes fallback' if hide else 'threadpoolctl if installed'})",
              rc == 0 and b and set(b.values()) == {2} and set(a.values()) == {1}, f"before {b} after {a}")


def h2():
    if (os.cpu_count() or 1) < 2:
        check("H2 skipped: needs > 1 CPU to tell 1 BLAS thread from the default", True)
        return
    rc, out = py("""
        import sys, json; sys.path.insert(0, ".")
        import numpy
        from cim_qat.common import blas_threads
        print(json.dumps(blas_threads()))
    """, env_without_threads())
    ctrl = json.loads(out) if rc == 0 else {}
    check("H2 control: without the entry point, numpy's BLAS starts with > 1 thread",
          ctrl and max(ctrl.values()) > 1, str(ctrl))
    for script in ("qat_evaluate.py", "qat_calibrate.py"):
        rc, out = py(f"""
            import sys, json, runpy, os
            sys.argv = ["{script}", "--help"]
            try:
                runpy.run_path("{script}", run_name="__main__")
            except SystemExit:
                pass
            import numpy        # loads here if the entry point imports it lazily (qat_calibrate)
            from cim_qat.common import blas_threads
            print(json.dumps(dict(blas=blas_threads(), env=os.environ.get("OPENBLAS_NUM_THREADS"),
                                  tag=os.environ.get("CIM_QAT_JOB", ""))))
        """, env_without_threads())
        d = json.loads(out) if rc == 0 else {}
        b = d.get("blas", {})
        want_tag = "eval:" if script == "qat_evaluate.py" else "calib:"
        check(f"H2 {script}: numpy's BLAS loads with 1 thread, job tagged {want_tag}",
              rc == 0 and b and set(b.values()) == {1} and d.get("env") == "1"
              and d.get("tag", "").startswith(want_tag), json.dumps(d))


def h3():
    rc, out = py("""
        import sys, json, runpy, os
        sys.argv = ["qat_eval_matrix.py", "--help"]
        try:
            runpy.run_path("qat_eval_matrix.py", run_name="__main__")
        except SystemExit:
            pass
        print(json.dumps({k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "CIM_QAT_JOB")}))
    """, env_without_threads())
    d = json.loads(out) if rc == 0 else {}
    check("H3 qat_eval_matrix.py exports 1 thread x 5 variables and an eval tag to its rows",
          rc == 0 and all(d.get(v) == "1" for v in THREAD_VARS) and (d.get("CIM_QAT_JOB") or "").startswith("eval:"),
          json.dumps(d))


def h4(tmp: Path):
    import numpy as np
    from PIL import Image
    imgs = []
    rng = np.random.default_rng(0)
    for k in range(4):
        f = tmp / f"img{k}.png"
        Image.fromarray(rng.integers(0, 255, (80, 90, 3), dtype=np.uint8)).save(f)
        imgs.append([k, str(f)])
    (tmp / "items.json").write_text(json.dumps(imgs))
    rc, out = py(f"""
        import sys, json, os; sys.path.insert(0, ".")
        import numpy as np
        np.ones((64, 64)) @ np.ones((64, 64))
        from cim_qat.common import blas_threads
        parent = blas_threads()
        from cim_qat import sim_eval
        from cim_qat.model import PRETRAINED
        items = [tuple(x) for x in json.load(open({str(tmp / 'items.json')!r}))]
        job = dict(kind="fp32", ckpt=str(PRETRAINED), with_own_fp32=False, interpolation="bicubic")
        r = sim_eval.run(job, items, {str(tmp / 'sim')!r}, workers=2, chunk=2, progress=False)
        summ = json.load(open({str(tmp / 'sim' / 'summary.json')!r}))
        print(json.dumps(dict(parent=parent, workers=summ.get("blas_threads_per_worker"),
                              env=os.environ.get("OPENBLAS_NUM_THREADS"))))
    """, env_without_threads(OPENBLAS_NUM_THREADS=2), timeout=900)
    d = json.loads(out) if rc == 0 else {}
    par = d.get("parent", {})
    check("H4 simulator workers: parent BLAS 2 threads, every worker 1 (summary blas_threads_per_worker == [1])",
          rc == 0 and par and set(par.values()) == {2} and d.get("workers") == [1], json.dumps(d))
    check("H4 caller's thread environment restored after the pool", rc == 0 and d.get("env") == "2",
          f"OPENBLAS_NUM_THREADS after run = {d.get('env')}")


def h5(tmp: Path):
    for use in (True, False):
        pidf = tmp / f"child_{use}.pid"
        code = f"""
            import sys, os, time; sys.path.insert(0, ".")
            from multiprocessing import get_context
            def init():
                if {use}:
                    from cim_qat.common import die_with_parent
                    assert die_with_parent()
            def sleep(_):
                open({str(pidf)!r}, "w").write(str(os.getpid())); time.sleep(120)
            if __name__ == "__main__":
                pool = get_context("fork").Pool(1, initializer=init)
                pool.map_async(sleep, [0])
                time.sleep(120)
        """
        f = tmp / f"parent_{use}.py"
        f.write_text(textwrap.dedent(code))
        p = subprocess.Popen([PY, str(f)], cwd=REPO)
        t0 = time.time()
        while not pidf.exists() and time.time() - t0 < 60:
            time.sleep(0.2)
        child = int(pidf.read_text()) if pidf.exists() else None
        p.send_signal(signal.SIGKILL)
        p.wait()
        time.sleep(2)
        survived = child is not None and alive(child)
        if use:
            check("H5 die_with_parent: worker terminated when its parent was SIGKILLed",
                  child is not None and not survived, f"worker pid {child} alive={survived}")
        else:
            check("H5 control: without it the worker survives as an orphan", survived, f"worker pid {child}")
        if survived:
            os.kill(child, signal.SIGKILL)


def h6(tmp: Path):
    mark = "selftest" + uuid.uuid4().hex[:10]
    d = tmp / mark
    d.mkdir()
    # a tagged evaluation (v2.6 style) whose worker is a "spawn" process
    (d / "fake_eval.py").write_text(textwrap.dedent("""
        import time
        from multiprocessing import get_context
        def sleep(_): time.sleep(300)
        if __name__ == "__main__":
            pool = get_context("spawn").Pool(1)
            pool.map_async(sleep, [0]); time.sleep(300)
    """))
    # an untagged v2.5-style matrix with a child, and a training process with a child
    for name in ("qat_eval_matrix.py", "qat_train.py"):
        (d / name).write_text("import subprocess, sys, time\n"
                              "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
                              "time.sleep(300)\n")
    e_tag = env_without_threads(CIM_QAT_JOB=f"eval:{mark}")
    procs = dict(eval_tagged=subprocess.Popen([PY, str(d / "fake_eval.py")], env=e_tag, cwd=d),
                 eval_v25=subprocess.Popen([PY, str(d / "qat_eval_matrix.py")], env=env_without_threads(), cwd=d),
                 train=subprocess.Popen([PY, str(d / "qat_train.py")], env=env_without_threads(), cwd=d),
                 train_tagged=subprocess.Popen([PY, "-c", f"import time; time.sleep(300)  # {mark}"],
                                               env=env_without_threads(CIM_QAT_JOB=f"train:{mark}")))
    time.sleep(6)                                         # spawn worker needs a moment

    def kids(pid):
        out = []
        for q in os.listdir("/proc"):
            if q.isdigit():
                try:
                    if int(open(f"/proc/{q}/stat").read().rsplit(")", 1)[1].split()[1]) == pid:
                        out.append(int(q))
                except (FileNotFoundError, IndexError, ValueError):
                    pass
        return out

    tree = {k: [p.pid] + kids(p.pid) for k, p in procs.items()}
    spawn_cmd = any("spawn_main" in open(f"/proc/{c}/cmdline", "rb").read().decode(errors="replace")
                    for c in tree["eval_tagged"][1:] if alive(c))
    check("H6 setup: tagged eval has a spawn worker; v2.5 matrix and training each have a child",
          spawn_cmd and len(tree["eval_v25"]) >= 2 and len(tree["train"]) >= 2, json.dumps(tree))
    want = set(tree["eval_tagged"] + tree["eval_v25"])
    never = set(tree["train"] + tree["train_tagged"])
    p = subprocess.run([PY, str(REPO / "qat_stop.py"), "eval", "--dry-run", "--only", mark],
                       capture_output=True, text=True, timeout=60)
    listed = {int(l.split()[1]) for l in p.stdout.split("train (left running)")[0].splitlines()
              if l.strip().startswith("pid ")}
    check("H6 dry run lists every evaluation process (incl. spawn worker), no training process, stops nothing",
          want <= listed and not (listed & never) and all(alive(x) for x in want),
          f"want {sorted(want)} listed {sorted(listed)}")
    p = subprocess.run([PY, str(REPO / "qat_stop.py"), "eval", "--only", mark],
                       capture_output=True, text=True, timeout=60)
    for pp in procs.values():                           # reap our direct children
        try:
            pp.wait(timeout=0.1)
        except subprocess.TimeoutExpired:
            pass
    time.sleep(1)
    check("H6 `qat_stop.py eval` stopped every evaluation process", p.returncode == 0
          and not any(alive(x) for x in want), p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr)
    check("H6 training processes left running", all(alive(x) for x in never), str(sorted(never)))
    for x in never:
        try:
            os.kill(x, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for pp in procs.values():
        try:
            pp.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def main():
    if not os.path.isdir("/proc"):
        sys.exit("Linux only")
    with tempfile.TemporaryDirectory(prefix="cim qat threads ") as t:
        tmp = Path(t)
        h1()
        h2()
        h3()
        h4(tmp)
        h5(tmp)
        h6(tmp)
    n = sum(ok for _, ok in RES)
    print(f"\nTHREADS: {n}/{len(RES)} passed")
    sys.exit(0 if n == len(RES) else 1)


if __name__ == "__main__":
    main()
