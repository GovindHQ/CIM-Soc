"""
cim_qat/analog_merge.py — correct merge of per-chunk analog statistics.

Why (found 29 Oct 2026 on the VAL matrix): cim_sim's AnalogStats.merge()
concatenates a new chunk's subsample with the running one and then halves
EVERYTHING whenever the cap is exceeded. A chunk's subsample arrives at the
chunk's own inclusion rate (keep fraction kf ~ 1e-3), while the running
subsample has already been thinned to a far lower rate; after 1,000 chunks
the samples of the early chunks have been halved ~1,000 times. Expected
weights stay correct, but nearly all of the run's weight ends up on a handful
of surviving samples with astronomically large weights. Percentiles of the
merged subsample (ADC utilisation median / p99.9, V_OA / V_ADC_IN
percentiles) are then dominated by a few random samples: biased in practice
and different between runs that differ only in the worker count (E6 vs E6s).
Exact running scalars (counts, clipping, rail hits, min/max, mean/std,
util_max) are NOT affected.

The fix here does not touch the simulator: it re-merges the per-chunk
subsamples that every simulator run keeps on disk (chunks/chunk_*.npz/json).
Each chunk's subsample is uniform within the chunk at the chunk's final keep
fraction kf_c. All chunks are brought to ONE common rate kf* =
min(min_c kf_c, budget / total conversions) by keeping each sample with
probability kf*/kf_c (weight divided by that probability), with a random
stream seeded by (seed, label, chunk index) — independent of worker count and
order. The concatenation is then a uniform subsample of the whole run with
~budget samples per operation, and the summary is computed by the
simulator's own AnalogStats.summary(), so every formula is unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

METHOD = "uniform re-merge at a common keep fraction (cim_qat.analog_merge, v2.7)"


def _label_key(label) -> str:
    return str(label)


def uniform_merge_states(states, budget: int = 2_000_000, seed: int = 0):
    """states(arrays: bool) -> fresh iterator of (scalars, arrays) per chunk, in chunk order.

    It is called twice: first with arrays=False for the totals (the arrays may then
    be empty), then with arrays=True for the samples, so chunks stream from disk.
    Returns an AnalogStats holding exact scalars and a uniform subsample.
    """
    from cim_sim.analog import AnalogStats
    # pass 1: totals per label (exact scalars only)
    tot_n, min_kf = {}, {}
    for scal, _ in states(False):
        for key, sv in scal.items():
            if sv["n"] <= 0:
                continue
            tot_n[key] = tot_n.get(key, 0) + int(sv["n"])
            min_kf[key] = min(min_kf.get(key, np.inf), float(sv["kf"]))
    target = {k: min(min_kf[k], budget / tot_n[k]) for k in tot_n}
    labels = sorted(tot_n)
    out = AnalogStats(cap=2 ** 62, seed=seed)
    for ci, (scal, arrs) in enumerate(states(True)):
        st = AnalogStats.from_state(scal, arrs, cap=2 ** 62)
        for key in list(st.b):
            s = st.b[key]
            k = _label_key(key)
            if k not in target or not s["w"]:
                continue
            p = target[k] / float(s["kf"])
            rng = np.random.default_rng([seed, labels.index(k), ci])
            w = np.concatenate(s["w"])
            keep = rng.random(w.size) < p
            for name in ("voa", "vadc", "code"):
                s[name] = [np.concatenate(s[name])[keep]]
            s["w"] = [w[keep] / p]
            s["n_kept"] = int(keep.sum())
            s["kf"] = target[k]
        out.merge(st)
    return out


def chunk_states(out_dir: Path):
    """Factory over a sim_eval run folder's chunk statistics (chunk order)."""
    cdir = Path(out_dir) / "chunks"
    metas = sorted(cdir.glob("chunk_*.json"))

    def gen(arrays: bool = True):
        for mp in metas:
            meta = json.loads(mp.read_text())
            if "analog_scalars" not in meta:
                raise RuntimeError(f"{mp}: no analog statistics (not an analog run)")
            if not arrays:
                yield meta["analog_scalars"], {}
                continue
            with np.load(mp.with_suffix(".npz")) as z:
                arrs = {k[4:]: z[k] for k in z.files if k.startswith("as__")}
            yield meta["analog_scalars"], arrs
    return gen, len(metas)


def summary_from_chunks(out_dir: Path, adc_vref: float = 0.6, adc_bits: int = 8,
                        budget: int = 2_000_000, seed: int = 0) -> dict:
    gen, n = chunk_states(out_dir)
    if not n:
        raise RuntimeError(f"{out_dir}/chunks: no chunk files (were they deleted?)")
    st = uniform_merge_states(gen, budget=budget, seed=seed)
    summ = st.summary(adc_vref=adc_vref, adc_bits=adc_bits)
    for v in summ.values():
        v["subsample_method"] = METHOD
    return summ
