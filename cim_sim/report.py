"""
report.py — turn a TraceLog into tables you can read (or paste into a slide).
"""

from __future__ import annotations

from collections import OrderedDict

from .matmul import TraceLog


def _fmt(n: float) -> str:
    """Human-readable large numbers."""
    for unit, div in (("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return f"{n:.0f}"


def tile_map(N: int, K: int, M: int, rows: int = 32, cols: int = 32) -> str:
    """One-line description of how a matmul decomposes onto the array."""
    n_db = (K + rows - 1) // rows
    n_ob = (M + cols - 1) // cols
    last_r = K - (n_db - 1) * rows
    last_c = M - (n_ob - 1) * cols
    return (f"[{N}x{K}] @ [{K}x{M}] -> {n_db} depth x {n_ob} output = "
            f"{n_db * n_ob} tiles; last depth block {last_r}/{rows} rows, "
            f"last output block {last_c}/{cols} cols; "
            f"array utilization {K * M / (n_db * rows * n_ob * cols):.1%}")


def report(log: TraceLog, group: str = "shape") -> str:
    """Render a trace log as a table.

    group='shape' collapses identical (tag-kind, N, K, M) shapes together, which
    is what you want for a repeated transformer stage.
    group='tag'   groups by full module path.
    """
    lines: list[str] = []

    buckets: "OrderedDict[tuple, dict]" = OrderedDict()
    for r in log.records:
        parts = r.tag.split(".")
        kind = parts[-1]
        # tags look like "layers.2.blocks.0.attn.qk_t"; keep the stage so that
        # identically-shaped matmuls in different stages do not merge.
        stage = ".".join(parts[:2]) if len(parts) > 2 else ""
        if group == "shape":
            key = (f"{stage}/{kind}" if stage else kind, r.N, r.K, r.M)
        else:
            key = (r.tag, r.N, r.K, r.M)
        b = buckets.setdefault(key, {"calls": 0, "rec": r,
                                     "writes": 0, "ops": 0,
                                     "issued": 0, "useful": 0})
        b["calls"] += 1
        b["writes"] += r.stats.weight_tile_writes
        b["ops"] += r.stats.array_ops
        b["issued"] += r.stats.macs_issued
        b["useful"] += r.stats.macs_useful

    hdr = (f"{'op':<20}{'N':>6}{'K':>6}{'M':>6}{'calls':>7}"
           f"{'tiles':>7}{'writes':>9}{'cycles':>10}{'util':>8}")
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for (kind, N, K, M), b in buckets.items():
        r = b["rec"]
        util = b["useful"] / b["issued"] if b["issued"] else 0.0
        lines.append(
            f"{kind:<20}{N:>6}{K:>6}{M:>6}{b['calls']:>7}"
            f"{r.tiles:>7}{_fmt(b['writes']):>9}{_fmt(b['ops']):>10}{util:>7.1%}"
        )

    tot = log.total()
    total_parallel_cycles = sum(
        r.parallel_array_cycles
        for r in log.records
    )

    total_active_arrays = sum(
        r.active_array_invocations
        for r in log.records
    )

    total_array_slots = sum(
        r.array_slots
        for r in log.records
    )

    parallel_util = (
        total_active_arrays / total_array_slots
        if total_array_slots else 0.0
    )
    lines.append("-" * len(hdr))
    lines.append(
        f"{'TOTAL':<20}{'':>6}{'':>6}{'':>6}{len(log.records):>7}{'':>7}"
        f"{_fmt(tot.weight_tile_writes):>9}{_fmt(tot.array_ops):>10}"
        f"{tot.utilization:>7.1%}"
    )
    lines.append("")
    lines.append(f"weight cell writes   : {_fmt(tot.weight_cell_writes)}")
    lines.append(f"column-sum readouts  : {_fmt(tot.column_sums_read)}  "
                 f"(= ADC conversions if 1 conversion per column sum)")
    lines.append(f"ADC conversions       : {_fmt(tot.adc_conversions)}")
    lines.append(f"ADC saturations       : {_fmt(tot.adc_saturations)}")
    if log.records:
        num_arrays = max(r.num_arrays for r in log.records)

        lines.append("")
        lines.append("parallel-array metrics")
        lines.append(f"physical CIM arrays   : {num_arrays}")
        lines.append(f"parallel array cycles : {_fmt(total_parallel_cycles)}")
        lines.append(
            f"array parallel util    : {parallel_util:.1%} "
            f"({total_active_arrays}/{total_array_slots} active slots)"
        )
    lines.append(f"MACs issued / useful : {_fmt(tot.macs_issued)} / "
                 f"{_fmt(tot.macs_useful)}   ({tot.utilization:.1%})")
    lines.append(f"peak psum words live : {tot.psum_words_live}")
    lines.append(f"max single-tile colsum: {tot.max_abs_colsum} "
                 f"-> {tot.colsum_bits}b signed")
    lines.append(f"max accumulated value : {tot.max_abs_accum} "
                 f"-> {tot.accum_bits}b signed accumulator")
    return "\n".join(lines)