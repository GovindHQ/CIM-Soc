"""
metrics.py — the metric definitions used in every table. One place, tested.

All functions take numpy arrays: logits [n, 1000] (same image order and class
order on both sides of any comparison), integer labels [n].

  top1_pct(logits, y)            100 * mean(argmax == y)              (vs TRUE labels)
  cross_entropy(logits, y)       mean_i  -log softmax(logits_i)[y_i]  natural log,
                                 computed ONCE from raw logits with a float64
                                 log-sum-exp (softmax is never applied twice)
  agreement_pct(a, b)            100 * mean(argmax a == argmax b)     (vs a NAMED reference)
  rel_rms_pct(t, r)              100 * ||t - r||_2 / ||r||_2  over the whole [n, C] tensor
  nonfinite_images(logits)       images with any NaN/Inf logit

10-way Imagenette variants restrict the logits to the 10 mapped ImageNet
classes before argmax / softmax. Agreement is NOT accuracy.
"""

from __future__ import annotations

import numpy as np


def _check(logits: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    logits = np.asarray(logits)
    if logits.ndim != 2:
        raise ValueError(f"logits must be [n, C], got {logits.shape}")
    if y is not None and len(y) != logits.shape[0]:
        raise ValueError("labels and logits disagree on n")
    return logits


def top1_pct(logits: np.ndarray, y: np.ndarray) -> float:
    logits = _check(logits, y)
    return 100.0 * float(np.mean(np.argmax(logits, axis=1) == np.asarray(y)))


def cross_entropy(logits: np.ndarray, y: np.ndarray) -> float:
    logits = _check(logits, y).astype(np.float64)
    m = logits.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(logits - m).sum(axis=1))
    return float(np.mean(lse - logits[np.arange(len(y)), np.asarray(y)]))


def agreement_pct(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _check(a), _check(b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    return 100.0 * float(np.mean(np.argmax(a, axis=1) == np.argmax(b, axis=1)))


def rel_rms_pct(tested: np.ndarray, reference: np.ndarray) -> float:
    t, r = _check(tested).astype(np.float64), _check(reference).astype(np.float64)
    if t.shape != r.shape:
        raise ValueError(f"shape mismatch {t.shape} vs {r.shape}")
    return 100.0 * float(np.linalg.norm(t - r) / np.linalg.norm(r))


def nonfinite_images(logits: np.ndarray) -> int:
    return int(np.count_nonzero(~np.isfinite(_check(logits)).all(axis=1)))


def restrict(logits: np.ndarray, class_indices: np.ndarray) -> np.ndarray:
    return _check(logits)[:, np.asarray(class_indices)]


def model_metrics(logits: np.ndarray, y1000: np.ndarray, y10: np.ndarray,
                  idx10: np.ndarray) -> dict:
    """Accuracy / CE against true labels, 1000-way (primary) and 10-way."""
    l10 = restrict(logits, idx10)
    return dict(n_images=int(logits.shape[0]),
                n_nonfinite_images=nonfinite_images(logits),
                top1_1000way_pct=top1_pct(logits, y1000),
                top1_10way_pct=top1_pct(l10, y10),
                ce_1000way_nats=cross_entropy(logits, y1000),
                ce_10way_nats=cross_entropy(l10, y10))
