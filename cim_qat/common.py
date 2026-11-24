"""
cim_qat/common.py — paths, atomic I/O, seeds, device selection, provenance.

Kept dependency-free beyond numpy/torch so every script can import it first.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def setup_paths() -> Path:
    """Make `cim_sim`, `cim_exp`, `run_tinyvit` and the timm shim importable."""
    for p in (str(REPO_ROOT / "_shim"), str(REPO_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return REPO_ROOT


setup_paths()


# --------------------------------------------------------------------------- #
# atomic I/O
# --------------------------------------------------------------------------- #


def json_default(o):
    import numpy as np
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=json_default))
    os.replace(tmp, path)


def read_json(path) -> dict:
    return json.loads(Path(path).read_text())


def savez_atomic(path, **arrays) -> None:
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def torch_save_atomic(obj, path) -> None:
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def sha256_file(path, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=json_default)


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# seeds and determinism
# --------------------------------------------------------------------------- #


def derive_seed(*parts) -> int:
    """Stable 31-bit seed from arbitrary parts (independent of PYTHONHASHSEED)."""
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def configure_determinism(deterministic: bool, tf32: bool = False) -> None:
    """Deterministic kernels and no TF32 (call before the first CUDA op).

    CUBLAS_WORKSPACE_CONFIG must be set before cuBLAS initialises; it is set
    here if absent, which is early enough as long as no CUDA matmul has run.
    """
    import torch
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.backends.cudnn.allow_tf32 = bool(tf32)


def configure_threads(n: int) -> None:
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = str(n)
    import torch
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(n)
    except RuntimeError:
        pass


THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def single_thread_env_before_numpy() -> None:
    """Call at the very top of an EVALUATION entry point, before numpy/torch are
    imported: BLAS libraries size their thread pool when they load, so setting
    these variables later (as configure_threads does) has no effect on NumPy.
    Explicit values already in the environment are respected."""
    for v in THREAD_ENV:
        os.environ.setdefault(v, "1")


def _blas_lib_paths() -> list:
    import re
    try:
        maps = open("/proc/self/maps").read()
    except OSError:
        return []
    return sorted(set(re.findall(r"(/\S*(?:openblas|libblas|mkl_rt)\S*\.so\S*)", maps)))


_BLAS_SET = ("scipy_openblas_set_num_threads64_", "scipy_openblas_set_num_threads_64_",
             "scipy_openblas_set_num_threads", "openblas_set_num_threads64_", "openblas_set_num_threads",
             "MKL_Set_Num_Threads")
_BLAS_GET = ("scipy_openblas_get_num_threads64_", "scipy_openblas_get_num_threads_64_",
             "scipy_openblas_get_num_threads", "openblas_get_num_threads64_", "openblas_get_num_threads",
             "MKL_Get_Max_Threads")


def blas_threads() -> dict:
    """{library path: current thread count} for every BLAS loaded in this process."""
    import ctypes
    out = {}
    for path in _blas_lib_paths():
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue
        for g in _BLAS_GET:
            f = getattr(lib, g, None)
            if f is not None:
                f.restype = ctypes.c_int
                out[path] = int(f())
                break
    return out


def limit_blas_threads(n: int = 1) -> dict:
    """Set the RUNNING BLAS thread pools of this process to n threads (works after
    numpy has been imported, unlike environment variables). Uses threadpoolctl if
    installed, else calls the OpenBLAS / MKL C API directly. Returns blas_threads()."""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(n, user_api="blas")
    except ImportError:
        import ctypes
        for path in _blas_lib_paths():
            try:
                lib = ctypes.CDLL(path)
            except OSError:
                continue
            for name in _BLAS_SET:
                f = getattr(lib, name, None)
                if f is not None:
                    f(ctypes.c_int(n))
                    break
    return blas_threads()


def die_with_parent() -> bool:
    """Linux: ask the kernel to SIGTERM this process when its parent exits, so
    pool workers can never be orphaned (they kept running for hours after their
    parent was killed on 28 Oct). Returns True if set."""
    import ctypes
    import signal
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        return libc.prctl(1, int(signal.SIGTERM), 0, 0, 0) == 0   # 1 = PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        return False


def tag_job(kind: str) -> str:
    """Mark this process and everything it starts (inherited environment) as a
    cim_qat job of the given kind ("eval", "calib", "train"); qat_stop.py uses it."""
    os.environ.setdefault("CIM_QAT_JOB", f"{kind}:{os.getpid()}")
    return os.environ["CIM_QAT_JOB"]


class single_thread:
    """Temporarily run torch CPU ops with ONE thread — the setting every
    simulator worker uses. Bit-exact CPU comparisons with the simulator need
    identical thread counts: multi-threaded conv/linear kernels reduce in a
    different order, and the CIM rounding amplifies 1-ulp differences."""

    def __enter__(self):
        import torch
        self.n = torch.get_num_threads()
        torch.set_num_threads(1)

    def __exit__(self, *exc):
        import torch
        torch.set_num_threads(self.n)


# --------------------------------------------------------------------------- #
# device selection (never a silent CPU fallback)
# --------------------------------------------------------------------------- #


def select_device(requested: str):
    """`requested` is "cpu" or "cuda" / "cuda:N".

    The GPU is chosen OUTSIDE Python with CUDA_VISIBLE_DEVICES (see
    QAT_RUNBOOK.md §2); inside Python "cuda" then means the first VISIBLE GPU.
    If CUDA is requested but unavailable this raises instead of falling back.
    """
    import torch
    if requested == "cpu":
        return torch.device("cpu")
    if not requested.startswith("cuda"):
        raise ValueError(f"--device must be cpu or cuda[:N], got {requested!r}")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. Refusing to "
            "fall back to CPU silently. Check `nvidia-smi`, the conda env, and "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}.")
    dev = torch.device(requested if ":" in requested else "cuda:0")
    if dev.index is not None and dev.index >= torch.cuda.device_count():
        raise RuntimeError(
            f"{dev} does not exist: only {torch.cuda.device_count()} GPU(s) visible "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}). "
            "With CUDA_VISIBLE_DEVICES=1 the chosen GPU is cuda:0 inside Python.")
    return dev


def describe_device(dev) -> dict:
    import torch
    d = dict(device=str(dev),
             CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
             CUDA_DEVICE_ORDER=os.environ.get("CUDA_DEVICE_ORDER", "<unset>"))
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(dev)
        d.update(name=p.name, total_memory_gb=round(p.total_memory / 1e9, 2),
                 uuid=str(getattr(p, "uuid", "")),
                 pci_bus_id=str(getattr(p, "pci_bus_id", "")),
                 capability=f"{p.major}.{p.minor}")
    return d


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #

FINGERPRINT_GLOBS = ("cim_sim/**/*.py", "cim_qat/**/*.py", "cim_exp/imagenette.py",
                     "cim_exp/calibration.py", "tiny_vit.py", "run_tinyvit.py",
                     "qat_*.py", "_shim/**/*.py")


def code_fingerprint(root: Path = REPO_ROOT) -> dict:
    """SHA-256 over all output-affecting sources (CRLF-normalised).

    Returns the combined hash, a simulator-only hash (cim_sim/ + tiny_vit.py),
    and the per-file hashes so a later diff can be localised.
    """
    files = set()
    for g in FINGERPRINT_GLOBS:
        files.update(p for p in root.glob(g) if p.is_file() and "__pycache__" not in p.parts)
    files = sorted(files, key=lambda p: p.relative_to(root).as_posix())
    per, h_all, h_sim = {}, hashlib.sha256(), hashlib.sha256()
    for f in files:
        rel = f.relative_to(root).as_posix()
        b = f.read_bytes().replace(b"\r\n", b"\n")
        per[rel] = hashlib.sha256(b).hexdigest()
        for h in (h_all,) + ((h_sim,) if rel.startswith("cim_sim/") or rel == "tiny_vit.py" else ()):
            h.update(rel.encode())
            h.update(b)
    return dict(all=h_all.hexdigest(), simulator=h_sim.hexdigest(), files=per)


def git_info(root: Path = REPO_ROOT) -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=root, capture_output=True, text=True,
                                  timeout=20).stdout.strip()
        except Exception:
            return ""
    head = run("rev-parse", "HEAD")
    if not head:
        return dict(available=False)
    diff = run("diff", "HEAD")
    return dict(available=True, head=head, branch=run("rev-parse", "--abbrev-ref", "HEAD"),
                dirty=bool(run("status", "--porcelain")),
                diff_sha256=sha256_text(diff) if diff else "")


def environment_info() -> dict:
    import numpy as np
    import torch
    info = dict(python=sys.version.split()[0], platform=platform.platform(),
                hostname=socket.gethostname(), torch=torch.__version__,
                torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                numpy=np.__version__, cuda_available=torch.cuda.is_available(),
                cuda_device_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
                CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
                CUDA_DEVICE_ORDER=os.environ.get("CUDA_DEVICE_ORDER", "<unset>"),
                cpu_count=os.cpu_count())
    try:
        import torchvision
        info["torchvision"] = torchvision.__version__
    except Exception:
        info["torchvision"] = None
    try:
        import PIL
        info["pillow"] = PIL.__version__
    except Exception:
        info["pillow"] = None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.used,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout
        info["nvidia_smi"] = [l.strip() for l in out.splitlines() if l.strip()]
    except Exception:
        info["nvidia_smi"] = []
    return info


def pip_freeze() -> str:
    try:
        return subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                              text=True, timeout=120).stdout
    except Exception as e:  # pragma: no cover
        return f"<pip freeze failed: {e}>"


def write_provenance(out_dir: Path, extra: dict | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = code_fingerprint()
    prov = dict(time_utc=now_iso(), argv=sys.argv, cwd=os.getcwd(),
                environment=environment_info(), git=git_info(),
                code_fingerprint=fp["all"], simulator_fingerprint=fp["simulator"],
                **(extra or {}))
    write_json(out_dir / "provenance.json", prov)
    write_json(out_dir / "code_fingerprint_files.json", fp["files"])
    (out_dir / "pip_freeze.txt").write_text(pip_freeze())
    return prov
