"""
adc_probe.py — side-channel recorder for the pre-ADC column-sum distribution.

WHY THIS IS NOT PART OF CIMStats
--------------------------------
CIMStats is a per-call DELTA-TRACKED counter struct. matmul.cim_matmul()
snapshots it before the loop nest, subtracts afterwards, and merges the deltas
per tag into TraceLog. That machinery assumes every field is a scalar that can
be subtracted or maxed. Putting a growing sample list in there breaks it three
separate ways:

  1. `None`-initialised fields cannot be subtracted   -> TypeError
  2. the snapshot must copy the whole list every call -> quadratic slowdown
  3. merge() concatenates, so per-tag totals double-count

So the distribution lives here instead, in an object that nothing snapshots,
nothing deltas and nothing merges. `CIMStats` goes back to being what it was:
scalar activity counters.

WHAT IT RECORDS
---------------
Exact integer histograms of the column sums leaving the array, keyed by
operation, recorded BEFORE the ADC — and recorded whether or not the ADC is
enabled. The ADC-off case is the important one: that is the undistorted signal
that should size the ADC in the first place.

WHY A HISTOGRAM AND NOT SAMPLES
-------------------------------
A column sum is an integer bounded by the array geometry and operand widths:

    FS_theoretical = rows * 2^(weight_bits-1) * 2^(act_bits-1)

so a histogram over [-FS, +FS] is EXACT. Every percentile computed from it is
the true percentile, not an estimate from a subsample. Memory is O(FS) and is
independent of how many samples pass through: 4b/4b gives FS = 2048, i.e. 4097
bins = 33 kB per operation, no matter whether you run 2 images or 2000.

PADDED COLUMNS ARE EXCLUDED
---------------------------
Only the first `useful_cols` columns are recorded. Zero-padded output columns
hold zero weights and therefore produce column sums of exactly 0. They are not
signal, and including them would drag every percentile toward zero and make the
under-utilisation look worse than it is.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# The recorder
# --------------------------------------------------------------------------- #


class ADCProbe:
    """Per-operation exact histograms of pre-ADC column sums.

    Args:
        full_scale: half-width of the histogram, in column-sum units. Must be
            the THEORETICAL worst-case bound, not whatever full scale the ADC
            happens to be configured with, so that values which would overflow
            a reduced ADC range are still recorded rather than lost.
        granularity: "op" buckets by the trailing component of the trace tag
            ("...attn.qkv" -> "qkv"), giving three histograms for the whole
            model. "full" keeps one histogram per attention module, which is
            finer but costs one histogram per (module, op) pair.
        flush_every: how many samples to buffer before folding them into the
            histogram with np.bincount. Bounds peak buffer memory.
    """

    def __init__(self,
                 full_scale: int,
                 granularity: str = "op",
                 flush_every: int = 1_000_000):
        assert granularity in ("op", "full"), granularity
        self.full_scale = int(full_scale)
        self.n_bins = 2 * self.full_scale + 1
        self.granularity = granularity
        self.flush_every = int(flush_every)

        self.hist: dict[str, np.ndarray] = {}
        self._buf: dict[str, list[np.ndarray]] = {}
        self._buf_n: dict[str, int] = {}

        # Per-operation theoretical bound. Not the same for every op: the AV
        # path quantizes activations unsigned and so has a bound almost twice
        # the signed one. Percentages of "full scale" are meaningless unless
        # each op is measured against its own bound.
        self.op_bound: dict[str, int] = {}

    # ---- key ------------------------------------------------------------ #

    def key(self, tag: str) -> str:
        if self.granularity == "full":
            return tag
        return tag.rsplit(".", 1)[-1] if "." in tag else tag

    # ---- recording ------------------------------------------------------ #

    def record(self, tag: str, y: np.ndarray, bound: int | None = None) -> None:
        """Record one MVM's worth of exact (pre-ADC) column sums.

        `bound` is the theoretical worst-case |column sum| for THIS operation,
        which depends on activation signedness. It is stored per operation so
        the report can express percentages against the right denominator.
        """
        if y.size == 0:
            return
        k = self.key(tag)
        if bound is not None:
            self.op_bound[k] = max(self.op_bound.get(k, 0), int(bound))
        idx = y.ravel().astype(np.int64) + self.full_scale

        # A value outside the theoretical bound means the bound is wrong —
        # loud failure is better than a silently truncated histogram.
        if idx.min() < 0 or idx.max() >= self.n_bins:
            raise ValueError(
                f"column sum outside theoretical full scale +/-{self.full_scale} "
                f"for tag {tag!r}: observed range "
                f"[{int(idx.min()) - self.full_scale}, "
                f"{int(idx.max()) - self.full_scale}]"
            )

        self._buf.setdefault(k, []).append(idx)
        self._buf_n[k] = self._buf_n.get(k, 0) + idx.size
        if self._buf_n[k] >= self.flush_every:
            self._flush_key(k)

    def _flush_key(self, k: str) -> None:
        bufs = self._buf.get(k)
        if not bufs:
            return
        idx = np.concatenate(bufs)
        h = np.bincount(idx, minlength=self.n_bins)
        if k in self.hist:
            self.hist[k] += h
        else:
            self.hist[k] = h.astype(np.int64)
        self._buf[k] = []
        self._buf_n[k] = 0

    def flush(self) -> None:
        for k in list(self._buf):
            self._flush_key(k)

    # ---- readout -------------------------------------------------------- #

    def tags(self) -> list[str]:
        self.flush()
        return sorted(self.hist)

    def counts(self, tag: str) -> np.ndarray:
        self.flush()
        return self.hist[tag]

    def values(self) -> np.ndarray:
        """The integer column-sum value each histogram bin corresponds to."""
        return np.arange(-self.full_scale, self.full_scale + 1, dtype=np.int64)

    def save(self, path: str, meta: dict | None = None) -> None:
        """Write every histogram to an .npz for offline analysis."""
        self.flush()
        payload = {f"hist::{k}": v for k, v in self.hist.items()}
        payload["full_scale"] = np.int64(self.full_scale)
        for k, b in self.op_bound.items():
            payload[f"bound::{k}"] = np.int64(b)
        for mk, mv in (meta or {}).items():
            payload[f"meta::{mk}"] = np.asarray(mv)
        np.savez_compressed(path, **payload)


# --------------------------------------------------------------------------- #
# Process-wide handle
# --------------------------------------------------------------------------- #
# CIMArray.mvm() is called from deep inside the scheduler and has no reference
# to the runner, so the probe is reached through a module-level handle. It is
# None unless a run explicitly turns it on, which keeps the normal path free of
# any diagnostic cost.

_PROBE: ADCProbe | None = None


def enable(full_scale: int, granularity: str = "op") -> ADCProbe:
    global _PROBE
    _PROBE = ADCProbe(full_scale, granularity=granularity)
    return _PROBE


def disable() -> None:
    global _PROBE
    _PROBE = None


def get() -> ADCProbe | None:
    return _PROBE


def record(tag: str, y: np.ndarray, bound: int | None = None) -> None:
    if _PROBE is not None:
        _PROBE.record(tag, y, bound=bound)


# --------------------------------------------------------------------------- #
# Histogram statistics
# --------------------------------------------------------------------------- #


def percentiles_from_hist(counts: np.ndarray,
                          values: np.ndarray,
                          qs) -> list[float]:
    """Exact percentiles of the signed distribution."""
    total = counts.sum()
    if total == 0:
        return [float("nan")] * len(qs)
    cum = np.cumsum(counts)
    out = []
    for q in qs:
        i = int(np.searchsorted(cum, q / 100.0 * total, side="left"))
        out.append(float(values[min(i, len(values) - 1)]))
    return out


def abs_percentiles_from_hist(counts: np.ndarray,
                              values: np.ndarray,
                              qs) -> list[float]:
    """Exact percentiles of |column sum| — this is what sizes a symmetric ADC."""
    fs = int((len(values) - 1) // 2)
    a = counts[fs:].astype(np.int64).copy()          # |v| = 0 .. fs
    a[1:] += counts[:fs][::-1]                       # fold the negative half
    total = a.sum()
    if total == 0:
        return [float("nan")] * len(qs)
    cum = np.cumsum(a)
    out = []
    for q in qs:
        i = int(np.searchsorted(cum, q / 100.0 * total, side="left"))
        out.append(float(min(i, fs)))
    return out


def moments_from_hist(counts: np.ndarray,
                      values: np.ndarray) -> tuple[float, float, int, int]:
    """(mean, std, min, max) of the signed distribution."""
    total = counts.sum()
    if total == 0:
        return float("nan"), float("nan"), 0, 0
    v = values.astype(np.float64)
    mean = float((counts * v).sum() / total)
    var = float((counts * (v - mean) ** 2).sum() / total)
    nz = np.nonzero(counts)[0]
    return mean, float(np.sqrt(max(var, 0.0))), int(values[nz[0]]), int(values[nz[-1]])


def adc_error_from_hist(counts: np.ndarray,
                        values: np.ndarray,
                        adc_full_scale: int,
                        bits: int) -> dict:
    """Exactly reproduce array._adc_convert offline, for one (FS, bits) pair.

    Because the histogram is exact, so is every number returned here. This is
    what makes the design sweep free: one instrumented run gives the
    distribution, and every candidate ADC configuration is then evaluated on
    that distribution without touching the model again.
    """
    total = int(counts.sum())
    if total == 0:
        return {}

    fs = int(adc_full_scale)
    qmax = 2 ** bits - 1
    v = values.astype(np.float64)

    v_clipped = np.clip(v, -fs, fs)
    n_clipped = int(counts[v != v_clipped].sum())

    v_norm = (v_clipped + fs) / (2.0 * fs)
    code = np.clip(np.rint(v_norm * qmax), 0, qmax)
    v_hat = np.rint(code / qmax * (2.0 * fs) - fs)

    err = v_hat - v                                   # includes clipping error
    mae = float((counts * np.abs(err)).sum() / total)
    mse = float((counts * err ** 2).sum() / total)
    rmse = float(np.sqrt(mse))
    max_err = float(np.abs(err[counts > 0]).max()) if (counts > 0).any() else 0.0

    # How many of the ADC's codes the workload ever actually produces.
    codes_used = int(np.unique(code[counts > 0]).size)

    sig_rms = float(np.sqrt((counts * v ** 2).sum() / total))
    if rmse > 0 and sig_rms > 0:
        snr_db = 20.0 * np.log10(sig_rms / rmse)
        enob = (snr_db - 1.76) / 6.02
    else:
        snr_db, enob = float("inf"), float(bits)

    return {
        "fs": fs,
        "bits": bits,
        "clip_pct": 100.0 * n_clipped / total,
        "mae": mae,
        "rmse": rmse,
        "max_err": max_err,
        "codes_used": codes_used,
        "codes_total": qmax + 1,
        "signal_rms": sig_rms,
        "snr_db": snr_db,
        "enob": enob,
    }


# --------------------------------------------------------------------------- #
# Human-readable distribution summary
# --------------------------------------------------------------------------- #


def distribution_report(probe: ADCProbe,
                        adc_full_scale: int,
                        adc_bits: int | None = None) -> str:
    """One block per operation describing the pre-ADC signal."""
    probe.flush()
    values = probe.values()
    fs_th = probe.full_scale

    lines = []
    lines.append("=" * 78)
    lines.append("PRE-ADC COLUMN-SUM DISTRIBUTION  (exact, from integer histogram)")
    lines.append("=" * 78)
    lines.append(f"  histogram span         : +/-{fs_th}")
    lines.append(f"  ADC full scale in use  : +/-{adc_full_scale}"
                 f"   (per-op overrides may apply)")
    if adc_bits is not None:
        lines.append(f"  ADC resolution         : {adc_bits} bits "
                     f"({2 ** adc_bits} codes)")
    lines.append("")

    for tag in probe.tags():
        c = probe.counts(tag)
        total = int(c.sum())
        mean, std, vmin, vmax = moments_from_hist(c, values)
        p50, p90, p99, p999, p9999 = abs_percentiles_from_hist(
            c, values, [50, 90, 99, 99.9, 99.99])

        lines.append("-" * 78)
        lines.append(f"{tag}")
        lines.append("-" * 78)
        lines.append(f"  samples                : {total:,}")
        lines.append(f"  signed range           : [{vmin}, {vmax}]")
        lines.append(f"  mean / std             : {mean:.2f} / {std:.2f}")
        bound = probe.op_bound.get(tag, fs_th)
        lines.append(f"  theoretical bound      : +/-{bound}")
        lines.append("")
        lines.append("  |column sum| percentiles      value    as % of theoretical FS")
        for label, p in [("P50", p50), ("P90", p90), ("P99", p99),
                         ("P99.9", p999), ("P99.99", p9999),
                         ("max", float(max(abs(vmin), abs(vmax))))]:
            lines.append(f"    {label:<8}                 {p:>10.0f}"
                         f"        {100.0 * p / bound:>8.2f}%")
        lines.append("")
        if adc_bits is not None:
            r = adc_error_from_hist(c, values, adc_full_scale, adc_bits)
            lines.append(f"  at the configured ADC ({adc_bits}b, FS=+/-{adc_full_scale}):")
            lines.append(f"    clipped              : {r['clip_pct']:.4f}%")
            lines.append(f"    quantisation RMSE    : {r['rmse']:.2f} column-sum units")
            lines.append(f"    codes actually used  : {r['codes_used']} of {r['codes_total']}")
            lines.append(f"    SNR / ENOB           : {r['snr_db']:.1f} dB / "
                         f"{r['enob']:.2f} effective bits")
            lines.append("")

    lines.append("=" * 78)
    return "\n".join(lines)
