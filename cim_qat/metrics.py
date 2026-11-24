"""
cim_qat/metrics.py — evaluation metrics and paired statistics (float64).

Per-image quantities are kept so that two models evaluated on the SAME images
can be compared pairwise (McNemar, paired bootstrap), which is far more
sensitive than comparing two independent accuracies.
"""

from __future__ import annotations

import math

import numpy as np


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    m = z.max(-1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(-1, keepdims=True))


def per_image(logits: np.ndarray, labels: np.ndarray, ref_logits: np.ndarray | None = None) -> dict:
    """Per-image correctness / NLL, and vs a reference model: agreement, KL(ref||model)."""
    lp = log_softmax(logits)
    labels = np.asarray(labels)
    top5 = np.argsort(-lp, axis=-1)[:, :5]
    out = dict(pred=top5[:, 0], correct1=(top5[:, 0] == labels),
               correct5=(top5 == labels[:, None]).any(1),
               nll=-lp[np.arange(len(labels)), labels],
               conf=np.exp(lp.max(-1)), logit_l2=np.linalg.norm(np.asarray(logits, np.float64), axis=1))
    if ref_logits is not None:
        lr = log_softmax(ref_logits)
        out["agree"] = lr.argmax(-1) == top5[:, 0]
        out["kl_ref"] = (np.exp(lr) * (lr - lp)).sum(-1)
        d = np.asarray(logits, np.float64) - np.asarray(ref_logits, np.float64)
        out["sq_err"] = (d ** 2).sum(-1)
        out["sq_ref"] = (np.asarray(ref_logits, np.float64) ** 2).sum(-1)
    return out


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def summarize(pi: dict) -> dict:
    n = len(pi["nll"])
    p = pi["correct1"].mean()
    s = dict(n=n, top1_pct=100 * p, top1_se_pct=100 * math.sqrt(p * (1 - p) / max(n, 1)),
             top5_pct=100 * pi["correct5"].mean(), nll=float(pi["nll"].mean()),
             ece_pct=100 * ece(pi["conf"], pi["correct1"]),
             mean_logit_l2=float(pi["logit_l2"].mean()))
    if "agree" in pi:
        s.update(agree_ref_pct=100 * pi["agree"].mean(), kl_ref=float(pi["kl_ref"].mean()),
                 rel_logit_rms_pct=100 * math.sqrt(pi["sq_err"].sum() / pi["sq_ref"].sum()))
    return s


def mcnemar(c_a: np.ndarray, c_b: np.ndarray) -> dict:
    """Exact two-sided McNemar on paired correctness vectors."""
    b = int((c_a & ~c_b).sum())       # a right, b wrong
    c = int((~c_a & c_b).sum())       # a wrong, b right
    n = b + c
    if n == 0:
        return dict(a_only=b, b_only=c, p=1.0)
    k = min(b, c)
    # P(X <= k) for Binomial(n, 1/2), doubled
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return dict(a_only=b, b_only=c, p=min(1.0, 2 * tail))


def paired_diff(x_a: np.ndarray, x_b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """mean(x_b - x_a) with paired SE and a percentile bootstrap 95% CI."""
    d = np.asarray(x_b, np.float64) - np.asarray(x_a, np.float64)
    n = len(d)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    bs = d[idx].mean(1)
    return dict(mean=float(d.mean()), se=float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
                ci95=(float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))), n=n)


def compare(pi_a: dict, pi_b: dict, seed: int = 0) -> dict:
    """b relative to a, on identical images."""
    out = dict(top1_diff_pct=paired_diff(pi_a["correct1"], pi_b["correct1"], seed=seed),
               nll_diff=paired_diff(pi_a["nll"], pi_b["nll"], seed=seed),
               mcnemar=mcnemar(pi_a["correct1"], pi_b["correct1"]))
    out["top1_diff_pct"] = {k: (100 * v if k in ("mean", "se") else
                                tuple(100 * x for x in v) if k == "ci95" else v)
                            for k, v in out["top1_diff_pct"].items()}
    if "kl_ref" in pi_a and "kl_ref" in pi_b:
        out["kl_ref_diff"] = paired_diff(pi_a["kl_ref"], pi_b["kl_ref"], seed=seed)
    return out
