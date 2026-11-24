#!/usr/bin/env python3
"""Device placement of models (CPU-runnable; uses PyTorch's 'meta' device as a
stand-in for a second device, so the CUDA-only bug found in stage 9 is reproduced
and pinned without a GPU).

  P1 plain .to(): TinyViT's cached attention-bias tensor `ab` stays behind (the bug)
  P2 cim_qat.model.to_device moves it; assert_on_device passes (teacher and student)
  P3 to_device changes no value: `ab` and the model output are bit-identical (CPU)
  P4 the trainer and the env-check benchmark place their models with to_device
  P5 RNG states arriving with the wrong dtype/device (as after torch.load(map_location=cuda))
     are restored exactly by trainer.restore_rng; qat_train --resume loads on the CPU

  python tests_qat/test_device_placement.py
"""
import copy
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402

from cim_qat.common import setup_paths  # noqa: E402,F401
from cim_qat.hw_config import HWConfig  # noqa: E402
from cim_qat.model import assert_on_device, build_student, build_teacher, to_device  # noqa: E402

R = []


def check(n, ok, **kw):
    R.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {n} " + " ".join(f"{k}={v}" for k, v in kw.items()))


meta = torch.device("meta")
t = build_teacher()
n_ab = sum(1 for m in t.modules() if "ab" in vars(m))
t_plain = copy.deepcopy(t).to(meta)
try:
    assert_on_device(t_plain, meta)
    caught = False
except RuntimeError as e:
    caught = "ab (plain attribute)" in str(e)
check("P1 plain .to() leaves TinyViT's cached `ab` on the old device (bug reproduced)", caught and n_ab == 10,
      cached_ab_modules=n_ab)
try:
    to_device(copy.deepcopy(t), meta)
    to_device(build_student(HWConfig(8, 8, (22.92, 8.75, 2.72))), meta)
    to_device(build_student(HWConfig(4, 4, (29.05, 11.28, 3.82))), meta)
    ok = True
except RuntimeError as e:
    ok, msg = False, str(e)
check("P2 to_device moves every tensor (teacher, W8A8 and W4A4 students)", ok)
t2 = to_device(copy.deepcopy(t), torch.device("cpu"))
same_ab = all(torch.equal(a.ab, b.ab) for a, b in zip(t.modules(), t2.modules()) if "ab" in vars(a))
x = torch.randn(2, 3, 224, 224, generator=torch.Generator().manual_seed(0))
with torch.no_grad():
    same_out = torch.equal(t(x), t2(x))
check("P3 to_device changes no value (ab and outputs bit-identical)", same_ab and same_out and not t2.training)
from cim_qat import trainer  # noqa: E402
import qat_env_check  # noqa: E402
src_run = inspect.getsource(trainer.Run.__init__)
src_env = inspect.getsource(qat_env_check.benchmark)
check("P4 trainer and env-check place models with to_device (no bare .to(device) on models)",
      src_run.count("to_device(") == 3 and "build_teacher().to(" not in src_run
      and "build_student(self.hw).to(" not in src_run and src_env.count("to_device(") == 2)
import random  # noqa: E402

import numpy as np  # noqa: E402
st = torch.get_rng_state()
try:
    torch.set_rng_state(st.to(torch.int64))     # what a non-ByteTensor state does (CUDA case: device)
    repro = False
except TypeError as e:
    repro = "ByteTensor" in str(e)
torch.set_rng_state(st)
want = torch.rand(5)
trainer.restore_rng(dict(python=random.getstate(), numpy=np.random.get_state(), torch=st.to(torch.int64),
                         cuda=None))
got = torch.rand(5)
src_tr = inspect.getsource(sys.modules["qat_train"]) if "qat_train" in sys.modules else \
    (Path(__file__).resolve().parent.parent / "qat_train.py").read_text()
check("P5 restore_rng accepts a non-ByteTensor state and restores the stream exactly; "
      "qat_train --resume loads on the CPU",
      repro and torch.equal(want, got) and 'map_location="cpu"' in src_tr and "map_location=dev" not in src_tr)
print(f"{sum(R)}/{len(R)} passed")
sys.exit(0 if all(R) else 1)
