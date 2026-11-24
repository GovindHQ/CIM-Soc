#!/usr/bin/env python3
"""
qat_audit.py — automated integrity and leakage audit (QAT_DESIGN.md §8).

  python qat_audit.py --paths configs/paths.json [--quick]

Inspects the ACTUAL files, not just the manifests:
  S  splits    manifest hashes; path-level overlap (manifest keys AND resolved
               real paths, symlinks followed); content-hash overlap from the
               stored SHA-256 lists; exact count accounting (re-counted on
               disk unless --quick); TRAIN/CAL/CALCHECK disjoint.
  M  mapping   ImageNetV2 label == int(folder), wnid == synsets[label], same
               1000 classes as VAL, synset list sorted/unique; the stored
               v2-integrity result if present.
  L  leakage   every training config and run (config_resolved.json,
               evals.jsonl, summary.json, spot-check jobs, train.log),
               every calibration (report, frozen gains, CAL-check jobs) and
               the data access log: ImageNetV2/TEST never used for training,
               selection, early stopping or calibration; TEST read only for
               final evaluation of FINISHED runs; activity after the first
               TEST read is listed for disclosure.
  E  eval      within each eval/<split>/ directory every run has identical
               image indices, labels, per-image input-tensor SHA-256 and
               preprocessing; TEST runs were made with --final, no --limit.
  C  ckpts     pretrained and old QAT checkpoints unchanged (SHA-256).

Exit code 1 if any check FAILs. WARN lines need disclosure, not a stop.
Writes <work_root>/audit/AUDIT_<utc>.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cim_qat.common import now_iso, read_json, sha256_file, write_json  # noqa: E402

RES = []
# pre-declared early-stopping patience per run (decided 26 Oct 2026, before any training run;
# QAT_DESIGN.md §0.2 #3). A run with another value FAILs the audit.
DECLARED_PATIENCE = {"w8a8_kd_pilot5ep": 5, "w4a4_kd_pilot10ep": 3, "w4a4_ce_pilot10ep": 3,
                     "w4a4_kd_pilot10ep_full": 10}
# decision D2 (27 Oct 2026): the only declared fork, and the runs D2 chooses between
DECLARED_FORKS = {"w4a4_kd_pilot10ep_full": "w4a4_kd_pilot10ep"}
D2_PILOT, D2_FORK = "w4a4_kd_pilot10ep", "w4a4_kd_pilot10ep_full"


def check(cid, ok, msg, level="FAIL"):
    st = "PASS" if ok else level
    RES.append(dict(id=cid, status=st, msg=msg))
    print(f"  [{st}] {cid} {msg}")


def jsonl(p):
    out = []
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    out.append(dict(_bad_line=line[:80]))
    return out


def forbidden_refs(obj, needles, path="") -> list:
    """Paths inside a JSON object whose key or string value refers to TEST/ImageNetV2."""
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in ("test", "imagenetv2_root"):
                hits.append(f"{path}/{k}")
            hits += forbidden_refs(v, needles, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits += forbidden_refs(v, needles, f"{path}[{i}]")
    elif isinstance(obj, str):
        low = obj.lower()
        if obj.strip().lower() == "test" or any(n and n.lower() in low for n in needles):
            hits.append(f"{path}={obj[:60]!r}")
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", required=True)
    ap.add_argument("--quick", action="store_true", help="skip re-counting files and resolving 1.3M real paths")
    a = ap.parse_args()
    from cim_qat import data as D
    from cim_qat.model import PRETRAINED, PRETRAINED_SHA256, REGISTRY
    dp = D.DataPaths.load(a.paths)
    wr = Path(dp.work_root)
    sd = dp.splits_dir
    info = read_json(sd / "splits.json")
    needles = [s for s in (dp.imagenetv2_root, info.get("imagenetv2_root"), "imagenetv2",
                           D.V2_DIRNAME, "manifest_test") if s]

    mini = bool(info.get("mini"))
    SZ = "WARN" if mini else "FAIL"       # size/accounting checks are meaningless on a mini (test) split
    if mini:
        check("S0", False, "this is a MINI split directory (plumbing tests only)", level="WARN")
    print("== S: splits")
    man = {}
    for s, h in info["manifest_sha256"].items():
        p = sd / f"manifest_{s}.csv"
        hh = sha256_file(p)
        check(f"S1.{s}", hh == h, f"manifest_{s}.csv sha256 {hh[:12]} == splits.json")
        man[s] = D.read_manifest(p)
    D.log_access(wr, "all", "audit", None)
    keys = {s: [(r["root"], r["rel_path"]) for r in rows] for s, rows in man.items()}
    for s in man:
        check(f"S2.within.{s}", len(keys[s]) == len(set(keys[s])), f"no duplicate manifest entries in {s}")
    names = list(man)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            n = len(set(keys[x]) & set(keys[y]))
            check(f"S2.{x}|{y}", n == 0, f"no manifest path shared by {x} and {y} ({n})")
    if not a.quick:
        real = {s: [os.path.realpath(D.resolve(info, r)) for r in rows] for s, rows in man.items()}
        for s in man:
            check(f"S3.within.{s}", len(real[s]) == len(set(real[s])),
                  f"no two {s} rows resolve to the same real file (symlinks followed)")
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                n = len(set(real[x]) & set(real[y]))
                check(f"S3.{x}|{y}", n == 0, f"no real file shared by {x} and {y} ({n})")
        del real
    else:
        check("S3", True, "resolved-path check skipped (--quick); splits.json path_overlap: "
              f"{info['disjointness']['path_overlap']['ok']}", level="WARN")
    hs = {s: D.read_hashes(sd / f"hashes_{s}.csv") for s in D.SPLITS if (sd / f"hashes_{s}.csv").exists()}
    for s, h in hs.items():
        check(f"S4.len.{s}", len(h) == len(man.get(s, [])), f"hashes_{s}.csv covers every {s} row")
    rep = D.disjointness_report(hs)
    for k in rep["critical_pairs"]:
        check(f"S4.{k}", rep["between"][k] == 0, f"no identical file content between {k} ({rep['between'][k]})")
    for s in rep["critical_within"]:
        check(f"S4.within.{s}", rep["within"][s] == 0, f"no identical content inside {s}")
    for k in rep["not_checked"]:
        check(f"S4.{k}", False, f"content overlap {k} NOT checked (TRAIN not hashed: use make --hash-train)",
              level="WARN")
    for k in ("cal|train", "calcheck|train", "train|val", "train|test", "val|test"):
        if k in rep["between"] and k not in rep["critical_pairs"]:
            check(f"S4.info.{k}", rep["between"][k] == 0, f"identical content {k}: {rep['between'][k]}", level="WARN")
    for s in ("val", "test", "train"):
        if s in rep["within"] and rep["within"][s]:
            check(f"S4.info.within.{s}", False, f"{rep['within'][s]} identical-content repeats inside {s}",
                  level="WARN")
    check("S4.errors", all(v == 0 for v in rep["errors"].values()), f"no unreadable files {rep['errors']}")
    acc = info["accounting"]
    excl = acc["train_excluded_content_duplicates"]
    check("S5.identity", acc["train_files_on_disk"] == len(man["cal"]) + len(man["calcheck"]) + len(man["train"]) + excl,
          f"TRAIN {len(man['train']):,} = on disk {acc['train_files_on_disk']:,} - CAL {len(man['cal']):,} - "
          f"CALCHECK {len(man['calcheck']):,} - excluded duplicates {excl:,}", level=SZ)
    check("S5.sizes", len(man["cal"]) == 2000 and len(man["calcheck"]) == 200 and len(man["val"]) == 50000,
          f"CAL {len(man['cal'])} / CALCHECK {len(man['calcheck'])} / VAL {len(man['val'])} == 2000 / 200 / 50000", level=SZ)
    if not a.quick:
        _, td = D.list_train(info["train_roots"])
        check("S5.recount.train", td["files"] == acc["train_files_on_disk"],
              f"train files on disk now {td['files']:,} == at prepare {acc['train_files_on_disk']:,}", level=SZ)
        _, vd = D.list_val(info["val_root"], info.get("val_labels_csv"))
        check("S5.recount.val", vd["files"] == acc["val_files_on_disk"] == len(man["val"]),
              f"val files on disk now {vd['files']:,} == manifest {len(man['val']):,}", level=SZ)
    cc = {r["label"] for r in man["calcheck"]}
    check("S6.calcheck", len(cc) == 200, "CALCHECK: 200 distinct classes", level=SZ)
    check("S6.cal", all(sum(1 for r in man["cal"] if r["label"] == c) == 2 for c in range(0, 1000, 97)),
          "CAL: 2 per class (spot-checked every 97th class)", level=SZ)

    print("== M: ImageNetV2 class mapping")
    if "test" in man:
        m = D.v2_mapping_report(man["test"], man["val"])
        check("M1", m["label_is_int_folder"], "TEST label == int(folder name) for every image")
        check("M2", m["wnid_matches_synset_index"], "TEST wnid == synsets[label] (ImageNet-2012 index order)")
        check("M3", m["val_wnid_matches_synset_index"], "VAL wnid == synsets[label]")
        check("M4", m["same_1000_classes_as_val"], "TEST and VAL cover the same 1000 class indices", level=SZ)
        check("M5", len(man["test"]) == 10000, f"TEST has {len(man['test'])} images (expected 10,000)", level=SZ)
        vi = wr / "eval" / "v2_integrity.json"
        if vi.exists():
            v = read_json(vi)
            check("M6", v["ok"], f"empirical mapping check: {v['classes_modal_match_int_folder']}/1000 classes "
                  f"(lexicographic alternative {v['classes_modal_match_imagefolder_lexicographic']})", level=SZ)
        else:
            check("M6", False, "empirical mapping check not run yet (qat_evaluate.py v2-integrity)", level="WARN")
    else:
        check("M0", False, "TEST (ImageNetV2) not prepared yet", level="WARN")

    print("== L: leakage (configs, run logs, calibration, access log)")
    for cfgp in sorted(Path("configs").glob("qat_*.json")):
        hits = forbidden_refs(read_json(cfgp), needles)
        check(f"L1.{cfgp.name}", not hits, f"training config refers to no TEST/ImageNetV2 data {hits[:3]}")
    val_sha = info["manifest_sha256"]["val"]
    runs = sorted(p for p in (wr / "runs").glob("*") if (p / "config_resolved.json").exists()) \
        if (wr / "runs").exists() else []
    run_finished = {}
    run_started = {}
    for rd in runs:
        cr = read_json(rd / "config_resolved.json")
        hits = forbidden_refs(cr["config"], needles)
        check(f"L2.{rd.name}.config", not hits, f"run config refers to no TEST/ImageNetV2 data {hits[:3]}")
        evs = [e for e in jsonl(rd / "evals.jsonl") if "tag" in e]
        bad = [e["tag"] for e in evs if e.get("split") != "val" or e.get("val_manifest_sha256") != val_sha]
        if not evs and (rd / "throughput.json").exists():
            check(f"L2.{rd.name}.evals", True, "smoke/throughput run: no selection evaluations")
        else:
            check(f"L2.{rd.name}.evals", bool(evs) and not bad,
                  f"all {len(evs)} selection/early-stopping evaluations used VAL (manifest {val_sha[:12]}) {bad[:3]}")
        pat = cr["config"].get("early_stopping_patience")
        if rd.name in DECLARED_PATIENCE:
            check(f"L2.{rd.name}.patience", pat == DECLARED_PATIENCE[rd.name],
                  f"early-stopping patience {pat} == pre-declared {DECLARED_PATIENCE[rd.name]} "
                  "(QAT_DESIGN.md §0.2 #3)")
        if (rd / "summary.json").exists():
            s = read_json(rd / "summary.json")
            check(f"L2.{rd.name}.summary", s.get("selection", {}).get("split") == "val"
                  and s.get("test_split_used") is False,
                  f"checkpoint selection split = {s.get('selection', {}).get('split')}, test_split_used = "
                  f"{s.get('test_split_used')}")
            run_finished[rd.name] = s.get("finished_utc")
        log = (rd / "train.log").read_text(errors="replace") if (rd / "train.log").exists() else ""
        hits = [n for n in needles if n and n in log]
        check(f"L2.{rd.name}.log", not hits, f"train.log mentions no ImageNetV2 path {hits[:2]}")
        run_started[rd.name] = _first_time(rd)
        fk = cr.get("fork")
        if fk or rd.name in DECLARED_FORKS:
            _audit_fork(rd, cr, fk)
            if fk:
                run_started[rd.name] = fk["time_utc"]           # the fork starts training at fork time
        if (rd / "spot").exists():
            spot_items = _spot_items_ok(rd, man["val"], info)
            check(f"L2.{rd.name}.spot", spot_items[0], spot_items[1])
    cal_times = {}
    for cd in sorted((wr / "calibration").glob("W*A*")) if (wr / "calibration").exists() else []:
        rp = cd / "calibration_report.json"
        if rp.exists():
            r = read_json(rp)
            check(f"L3.{cd.name}.report", r["cal_manifest_sha256"] == info["manifest_sha256"]["cal"]
                  and r["calcheck_manifest_sha256"] == info["manifest_sha256"]["calcheck"],
                  "calibration used the CAL and CALCHECK manifests only")
        fg = cd / "frozen_gains.json"
        if fg.exists():
            g = read_json(fg)
            check(f"L3.{cd.name}.frozen", g["cal_manifest_sha256"] == info["manifest_sha256"]["cal"],
                  f"frozen gains {g['gains']} derived from CAL (level {g['level']})")
            cal_times[cd.name] = g["frozen_utc"]
            for rd in runs:
                cr = read_json(rd / "config_resolved.json")
                if cr["config"]["precision"].upper() == cd.name:
                    check(f"L3.{rd.name}.gains", read_json(fg) == cr["frozen_gains_record"],
                          f"run {rd.name} trained with the current frozen {cd.name} gains (file unchanged)")
        if (cd / "calcheck").exists():
            from cim_qat.sim_eval import _items_sha
            want = _items_sha([(r["idx"], str(D.resolve(info, r))) for r in man["calcheck"]])
            jobs = [read_json(j) for j in (cd / "calcheck").glob("*/job.json")]
            check(f"L3.{cd.name}.calcheck", jobs and all(j["items_sha"] == want for j in jobs),
                  f"{len(jobs)} CAL-check simulator jobs ran on exactly the CALCHECK images")
    # once access_log.jsonl is frozen, new entries go to access_log_ext.jsonl (v2.8): read both
    log = D.read_access_logs(wr)
    check("L4.log", bool(log), f"data access log has {len(log)} entries")
    test_acc = [e for e in log if e.get("split") == "test"]
    bad = [e for e in test_acc if e.get("purpose") not in ("final_eval", "v2_integrity", "prepare", "audit")]
    check("L4.test_purposes", not bad,
          f"TEST read {len(test_acc)} times, only for final_eval/v2_integrity/prepare/audit "
          f"{[(e.get('purpose'), e.get('time_utc')) for e in bad][:3]}")
    trainlike = [e for e in log if e.get("purpose") in ("train", "smoke", "calibration", "unit_test",
                                                        "surrogate_check")]
    check("L4.trainlike", all(e.get("split") != "test" for e in trainlike),
          "no training/calibration/check process read TEST")
    finals = sorted(e["time_utc"] for e in test_acc if e.get("purpose") == "final_eval")
    if finals:
        first = finals[0]
        later = [f"run {k} started {v}" for k, v in run_started.items() if v and v > first] + \
                [f"gains {k} frozen {v}" for k, v in cal_times.items() if v > first]
        check("L5.after_first_test", not later,
              f"no training run started / gains frozen after the first TEST read ({first}) {later[:4]}",
              level="WARN")
    tdir = wr / "eval" / "test"
    for ed in sorted(tdir.glob("*")) if tdir.exists() else []:
        rp = ed / "eval_record.json"
        if not rp.exists():
            continue
        r = read_json(rp)
        check(f"L6.{ed.name}", r.get("final") is True and r.get("limit") is None and r.get("purpose") == "final_eval",
              "TEST evaluation was a --final, full-set evaluation")
        mdl = r["model"]
        if mdl.get("source") == "new_qat_run":
            fin = run_finished.get(mdl["run"])
            check(f"L6.{ed.name}.order", fin is not None and fin < r["time_utc"],
                  f"run {mdl['run']} finished ({fin}) before its TEST evaluation ({r['time_utc']})")

    print("== D2: W4A4 run selection")
    _audit_d2(wr, log)

    print("== X: freeze and the ImageNetV2 extension variants (v2.8)")
    _audit_ext(wr, sd, dp, log, run_finished, SZ)

    print("== E: identical evaluation inputs")
    import numpy as np
    for sdir in sorted((wr / "eval").glob("*")) if (wr / "eval").exists() else []:
        if not sdir.is_dir():
            continue
        res = sorted(p for p in sdir.glob("*") if (p / "result.npz").exists())
        if not res:
            continue
        ref = None
        bad = []
        for p in res:
            with np.load(p / "result.npz") as z:
                cur = (z["idx"], z["input_sha"], np.load(p / "labels.npy"),
                       read_json(p / "summary.json")["job"]["preprocess_id"],
                       read_json(p / "eval_record.json")["split_manifest_sha256"])
            if ref is None:
                ref = (p.name, cur)
                continue
            same = (np.array_equal(cur[0], ref[1][0]) and np.array_equal(cur[1], ref[1][1])
                    and np.array_equal(cur[2], ref[1][2]) and cur[3] == ref[1][3] and cur[4] == ref[1][4])
            if not same:
                bad.append(p.name)
        check(f"E1.{sdir.name}", not bad, f"{len(res)} runs on identical images, labels, input tensors and "
              f"preprocessing (reference {ref[0]}) {bad[:3]}")

    print("== C: checkpoints")
    check("C1.pretrained", sha256_file(PRETRAINED) == PRETRAINED_SHA256, "pretrained checkpoint unchanged")
    for k, (p, h) in REGISTRY.items():
        if k != "pretrained":
            check(f"C1.{k}", Path(p).exists() and sha256_file(p) == h, f"{k} unchanged ({h[:12]})")

    n_fail = sum(r["status"] == "FAIL" for r in RES)
    n_warn = sum(r["status"] == "WARN" for r in RES)
    out = wr / "audit"
    out.mkdir(parents=True, exist_ok=True)
    stamp = now_iso().replace(":", "").replace("-", "")
    write_json(out / f"AUDIT_{stamp}.json", dict(time_utc=now_iso(), results=RES, n_fail=n_fail, n_warn=n_warn))
    md = [f"# Audit {now_iso()}\n", f"FAIL {n_fail} · WARN {n_warn} · PASS {len(RES) - n_fail - n_warn}\n",
          "| id | status | check |", "|---|---|---|"] + [f"| {r['id']} | {r['status']} | {r['msg']} |" for r in RES]
    (out / f"AUDIT_{stamp}.md").write_text("\n".join(md))
    print(f"\n{len(RES) - n_fail - n_warn} PASS, {n_warn} WARN, {n_fail} FAIL -> {out}/AUDIT_{stamp}.md")
    sys.exit(1 if n_fail else 0)


def _audit_fork(rd: Path, cr: dict, fk):
    """A D2 fork: declared, from an early-stopped source, only the allowed changes,
    and its history is exactly the source's history up to the fork point."""
    from cim_qat.trainer import FORK_ALLOWED
    n = rd.name
    if not fk:
        check(f"L8.{n}.fork", False, "this run name is reserved for the declared D2 fork, but the run is not a fork")
        return
    check(f"L8.{n}.declared", DECLARED_FORKS.get(n) == fk["source_run"],
          f"fork {n} <- {fk['source_run']} is the declared D2 fork ({DECLARED_FORKS.get(n)})")
    src = rd.parent / fk["source_run"]
    ok_src = (src / "summary.json").exists() and bool(read_json(src / "summary.json").get("stopped_early"))
    check(f"L8.{n}.source", ok_src and (src / "checkpoints" / "last.pth").exists()
          and sha256_file(src / "checkpoints" / "last.pth") == fk["source_last_sha256"],
          f"source {fk['source_run']} early-stopped and its state file is unchanged since the fork")
    ch = set(fk["changed"])
    pat = cr["config"]["early_stopping_patience"]
    check(f"L8.{n}.changes", ch <= set(FORK_ALLOWED) and pat >= cr["config"]["epochs"],
          f"only {sorted(ch)} changed; early stopping off (patience {pat} >= {cr['config']['epochs']} epochs)")
    if (src / "evals.jsonl").exists():
        a = [l for l in (src / "evals.jsonl").read_text().splitlines() if '"tag"' in l]
        b = [l for l in (rd / "evals.jsonl").read_text().splitlines() if '"tag"' in l]
        check(f"L8.{n}.history", b[:len(a)] == a and len(b) >= len(a),
              f"the fork's first {len(a)} evaluations are byte-identical to the source's")


def _audit_d2(wr: Path, log: list):
    p = wr / "eval" / "D2_selection.json"
    test_w4 = [d for d in (wr / "eval" / "test").glob("*")
               if d.name.startswith((D2_PILOT + "_", D2_FORK + "_"))] if (wr / "eval" / "test").exists() else []
    if not p.exists():
        check("L9.d2", not test_w4, "no D2 selection yet" + (f", but new-W4A4 TEST evaluations exist: "
                                                              f"{[d.name for d in test_w4][:3]}" if test_w4 else ""),
              level="FAIL" if test_w4 else "WARN")
        return
    d = read_json(p)
    c = sorted(d["candidates"], key=lambda x: (-x["top1_pct"], x["nll"], x["run"] != D2_PILOT))
    check("L9.d2.rule", c[0]["run"] == d["selected"], f"D2 selected {d['selected']} by the declared rule")
    finals = sorted(e["time_utc"] for e in log if e.get("split") == "test" and e.get("purpose") == "final_eval")
    check("L9.d2.before_test", not finals or d["time_utc"] < finals[0],
          f"D2 recorded {d['time_utc']} before the first TEST read ({finals[0] if finals else 'none yet'})")
    if test_w4:
        sel = [x for x in test_w4 if x.name.startswith(d["selected"] + "_best__")]
        check("L9.d2.test", bool(sel), f"the D2-selected run {d['selected']} is evaluated on TEST")


def _audit_ext(wr: Path, sd: Path, dp, log: list, run_finished: dict, SZ: str):
    from cim_qat import data as D
    # X0: every file of every checksum freeze is still byte-identical
    import datetime as _dt
    freezes = {}                        # freeze file -> (listed files, X0 ok, file time UTC)
    for fz in sorted(wr.glob(D.FREEZE_GLOB)):
        want = {}
        for line in fz.read_text().splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2:
                rel = parts[1].lstrip("*")
                want[rel[2:] if rel.startswith("./") else rel] = parts[0]
        bad = [rel for rel, h in want.items() if not (wr / rel).exists() or sha256_file(wr / rel) != h]
        check(f"X0.{fz.name}", not bad, f"all {len(want)} frozen files unchanged {bad[:4]}")
        freezes[fz.name] = (set(want), not bad,
                            _dt.datetime.fromtimestamp(fz.stat().st_mtime, _dt.timezone.utc).isoformat())
    reg_p = sd / D.EXT_REGISTRY
    if not reg_p.exists():
        check("X1", True, "no extension variants registered")
        return
    reg = read_json(reg_p)
    hs = {s: set(D.read_hashes(sd / f"hashes_{s}.csv")) for s in ("cal", "calcheck", "val", "train", "test")
          if (sd / f"hashes_{s}.csv").exists()}
    exth = {s: D.read_hashes(sd / f"hashes_{s}.csv") for s in reg}
    syn = D.synsets()
    for s, e in sorted(reg.items()):
        mp = sd / f"manifest_{s}.csv"
        h = sha256_file(mp)
        check(f"X1.{s}.manifest", h == e["manifest_sha256"] and sha256_file(sd / f"hashes_{s}.csv") == e["hashes_sha256"],
              f"manifest_{s}.csv {h[:12]} and hashes_{s}.csv == {D.EXT_REGISTRY}")
        rows = D.read_manifest(mp)
        check(f"X1.{s}.count", e["n"] + e["excluded"] == e["files_on_disk"] == 10000 and len(rows) == e["n"],
              f"{e['variant']}: {e['files_on_disk']:,} files on disk = {e['n']:,} evaluated + {e['excluded']} excluded "
              "(expected 10,000 on disk)", level=SZ)
        per = [0] * 1000
        for r in rows:
            per[r["label"]] += 1
        ok_map = all(r["label"] == int(Path(r["rel_path"]).parts[0]) and r["wnid"] == syn[r["label"]] for r in rows)
        check(f"X2.{s}.mapping", ok_map, "label == int(folder name) and wnid == synsets[label] for every image")
        check(f"X2.{s}.classes", sum(c > 0 for c in per) == 1000, f"{sum(c > 0 for c in per)} classes covered, "
              f"{min(per)}-{max(per)} images per class (10 per class before exclusions)", level=SZ)
        mine = exth[s]
        leak = {k: sum(x in hs[k] for x in mine) for k in ("cal", "calcheck", "val", "train") if k in hs}
        check(f"X3.{s}.leakage", not any(leak.values()), f"no evaluated image equals a CAL/CALCHECK/VAL/TRAIN "
              f"image by content {leak}")
        check(f"X3.{s}.excluded", e["excluded"] == 0, f"{e['excluded']} image(s) removed by the declared rule "
              f"(content overlaps found before exclusion: {e['content_overlap_before_exclusion']}; excluded_{s}.csv)", level="WARN")
        check(f"X4.{s}.path", e["path_shared_with_registered_splits"] == 0,
              "no file path shared with any other split")
        ov = {"test": sum(x in hs.get("test", set()) for x in mine)}
        ov.update({o: len(set(mine) & set(exth[o])) for o in reg if o != s})
        check(f"X4.{s}.overlap", not any(ov.values()), f"images also in other ImageNetV2 variants {ov} "
              "(kept in each variant; counted once in the pooled report)", level="WARN")
        acc = [x for x in log if x.get("split") == s]
        bad = [x.get("purpose") for x in acc if x.get("purpose") not in ("final_eval", "v2_integrity", "prepare", "audit")]
        check(f"X5.{s}.purposes", not bad, f"{s} read {len(acc)} times, only for final_eval/v2_integrity/prepare {bad[:3]}")
        # v2.8.1: a log that holds these reads may itself have been frozen LATER (the extension freeze lists
        # access_log_ext.jsonl). That is fine as long as nothing was appended after its freeze: the freeze
        # still matches (X0) and no entry is dated after the freeze file was written.
        late, broken = [], []
        for lg in sorted({x.get("_log") for x in acc}):
            fl = [(n, ok, t) for n, (files, ok, t) in freezes.items() if lg in files]
            if not fl:
                continue                                  # log not frozen: nothing to check
            broken += [n for n, ok, _ in fl if not ok]
            t_first = min(t for _, _, t in fl)
            late += [x for x in acc if x.get("_log") == lg and x.get("time_utc", "") > t_first]
        check(f"X5.{s}.log", not late and not broken,
              f"no {s} read was appended to an access log after that log was frozen "
              f"({len(late)} entries dated after the freeze; freezes broken: {broken})")
        vi = wr / "eval" / f"v2_integrity_{s}.json"
        if vi.exists():
            v = read_json(vi)
            check(f"X2.{s}.empirical", v["ok"], f"empirical mapping check: {v['classes_modal_match_int_folder']}/1000 "
                  f"classes (lexicographic alternative {v['classes_modal_match_imagefolder_lexicographic']})", level=SZ)
        ed = wr / "eval" / s
        for d in sorted(ed.glob("*")) if ed.exists() else []:
            rp = d / "eval_record.json"
            if not rp.exists():
                continue
            r = read_json(rp)
            check(f"X6.{s}.{d.name}", r.get("final") is True and r.get("limit") is None
                  and r.get("purpose") == "final_eval" and r.get("split_manifest_sha256") == e["manifest_sha256"]
                  and r.get("n") == e["n"], "--final, full-set evaluation of the registered manifest")
            mdl = r["model"]
            if mdl.get("source") == "new_qat_run":
                ck = read_json(wr / "runs" / mdl["run"] / "checkpoints" / "checkpoints.json")
                fname = "best_model.pth" if mdl.get("which") == "best" else "final_model.pth"
                fin = run_finished.get(mdl["run"])
                check(f"X6.{s}.{d.name}.model", ck.get(fname, {}).get("sha256") == mdl["sha256"]
                      and fin is not None and fin < r["time_utc"],
                      f"the frozen {mdl['run']}/{fname} ({mdl['sha256'][:12]}), finished {fin} before this evaluation")
    fz = sorted(wr.glob(D.FREEZE_GLOB))
    reads = sorted(x["time_utc"] for x in log if x.get("split") in reg and x.get("purpose") == "final_eval")
    if fz and reads:
        import datetime as _dt
        t_fz = _dt.datetime.fromtimestamp(min(f.stat().st_mtime for f in fz), _dt.timezone.utc).isoformat()
        check("X7.after_freeze", reads[0] < t_fz,
              f"extension evaluated after the results were frozen ({t_fz[:19]}; first read {reads[0][:19]}): "
              "an extension requested after the freeze, descriptive only (QAT_DESIGN.md §0.4 D3)", level="WARN")


def _first_time(rd: Path):
    """Start time of a run: the first timestamp in train.log."""
    p = rd / "train.log"
    if p.exists():
        for line in p.read_text(errors="replace").splitlines():
            if line.startswith("[") and "]" in line:
                return line[1:line.index("]")]
    return None


def _spot_items_ok(rd: Path, val_rows, info):
    """The spot-check simulator jobs store a hash of their (idx, path) items.
    Recompute the hash for the VAL rows the trainer would pick and compare."""
    from cim_qat import data as D
    from cim_qat.sim_eval import _items_sha
    cr = read_json(rd / "config_resolved.json")["config"]
    n_val = len(val_rows)
    lim = cr.get("val_limit")
    import numpy as np
    ind = list(range(n_val)) if lim is None else list(np.linspace(0, n_val - 1, lim).astype(int))
    n = min(cr.get("spot_check_n", 128), len(ind))
    sp = [ind[i] for i in np.linspace(0, len(ind) - 1, n).astype(int)]
    want = _items_sha([(val_rows[i]["idx"], str(D.resolve(info, val_rows[i]))) for i in sp])
    jobs = [read_json(j) for j in (rd / "spot").glob("*_sim/job.json")] + \
           [read_json(j) for j in (rd / "spot").glob("*_sim_ulp/job.json")]
    bad = [j for j in jobs if j["items_sha"] != want]
    return (not bad and bool(jobs),
            f"{len(jobs)} simulator spot-check jobs used the fixed VAL spot images ({len(bad)} differ)")


if __name__ == "__main__":
    main()
