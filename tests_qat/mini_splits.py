"""Plumbing-test helper: derive a SMALL split directory (marked mini=True) from a
prepared one, subsetting manifests and hash lists consistently. Used only by
tests_qat/run_selftest.py on fake data; never for real experiments.

  python tests_qat/mini_splits.py <src splits dir> <new work root> '<json sizes>' <paths.json> <out paths.json>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cim_qat.common import read_json, write_json  # noqa: E402
from cim_qat.data import SPLITS, read_hashes, read_manifest, write_manifest  # noqa: E402


def make(src: Path, dst: Path, n: dict):
    info = read_json(src / "splits.json")
    info["mini"] = True
    info["manifest_sha256"] = {}
    for s in SPLITS:
        if not (src / f"manifest_{s}.csv").exists():
            continue
        rows = read_manifest(src / f"manifest_{s}.csv")
        hs = read_hashes(src / f"hashes_{s}.csv") if (src / f"hashes_{s}.csv").exists() else None
        step = max(1, len(rows) // n.get(s, len(rows)))
        pick = list(range(0, len(rows), step))[: n.get(s, len(rows))]
        sub = [dict(rows[i], idx=k) for k, i in enumerate(pick)]
        info["manifest_sha256"][s] = write_manifest(sub, dst / f"manifest_{s}.csv")
        info["counts"][s] = len(sub)
        if hs is not None:
            with open(dst / f"hashes_{s}.csv", "w") as f:
                f.write("idx,sha256\n")
                for k, i in enumerate(pick):
                    f.write(f"{k},{hs[i]}\n")
    acc = info["accounting"]
    acc.update(train_files_on_disk=info["counts"]["cal"] + info["counts"]["calcheck"] + info["counts"]["train"],
               train_excluded_content_duplicates=0, cal=info["counts"]["cal"], calcheck=info["counts"]["calcheck"],
               train=info["counts"]["train"], val=info["counts"]["val"], test=info["counts"].get("test", 0),
               note="mini split for plumbing tests: counts are NOT the on-disk counts")
    write_json(dst / "splits.json", info)


if __name__ == "__main__":
    src, work = Path(sys.argv[1]), Path(sys.argv[2])
    n = json.loads(sys.argv[3])
    make(src, work / "splits", n)
    p = json.loads(Path(sys.argv[4]).read_text())
    p["work_root"] = str(work)
    Path(sys.argv[5]).write_text(json.dumps(p, indent=1))
    print("mini splits ->", work / "splits", n)
