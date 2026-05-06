"""Cluster-quality metrics: ARI + pairwise F1, precision, recall.

`score_clusters(predicted, true) -> dict`. Predicted and true labels are
sequence-likes of equal length. Treated as ID-permutation-invariant:
the actual numeric labels don't matter, only the partition they induce.
"""

from __future__ import annotations

from collections import Counter
from math import comb
from typing import Iterable

import pyarrow as pa


def _to_list(xs) -> list:
    if isinstance(xs, pa.Array):
        return xs.to_pylist()
    if isinstance(xs, pa.ChunkedArray):
        return xs.to_pylist()
    return list(xs)


def score_clusters(predicted: Iterable, true: Iterable) -> dict:
    """Return a dict of cluster-quality metrics.

    Metrics:
      - adjusted_rand: Adjusted Rand Index (Hubert & Arabie 1985). 1.0 = perfect,
        0.0 = chance, can be negative for worse-than-chance.
      - f1_pairs:    pairwise F1 — does each pair end up co-clustered correctly?
      - precision:   pairwise precision (predicted-positives that are correct).
      - recall:      pairwise recall   (true-positives the prediction caught).
      - n_predicted_clusters
      - n_true_clusters
      - n_items

    Both inputs must have the same length. Null labels (None / pa null) are
    treated as singletons (each null label = its own cluster).
    """
    p = _to_list(predicted)
    t = _to_list(true)
    if len(p) != len(t):
        raise ValueError(f"length mismatch: predicted={len(p)}, true={len(t)}")
    n = len(p)
    if n == 0:
        return {
            "adjusted_rand": 1.0,
            "f1_pairs": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "n_predicted_clusters": 0,
            "n_true_clusters": 0,
            "n_items": 0,
        }

    # Each null becomes its own singleton cluster. Tuple key prevents collision
    # with any user-supplied label (a string label can never equal a 3-tuple).
    p_norm = [_singletonize(v, i, 0) for i, v in enumerate(p)]
    t_norm = [_singletonize(v, i, 1) for i, v in enumerate(t)]

    pair = _pairwise_metrics(p_norm, t_norm)
    ari = _adjusted_rand_index(p_norm, t_norm)
    return {
        "adjusted_rand": ari,
        "f1_pairs": pair["f1"],
        "precision": pair["precision"],
        "recall": pair["recall"],
        "n_predicted_clusters": len(set(p_norm)),
        "n_true_clusters": len(set(t_norm)),
        "n_items": n,
    }


def _singletonize(v, idx: int, side: int):
    if v is None:
        return ("__null__", side, idx)
    return v


def _pairwise_metrics(predicted: list, true: list) -> dict:
    """Pairwise precision / recall / F1.

    For each pair of items (i, j):
      - same in predicted AND same in true   -> TP
      - same in predicted but NOT in true    -> FP
      - NOT same in predicted but same true  -> FN
      - neither same                          -> TN (unused)

    Compute via cluster sizes (avoids O(n^2) iteration).
    """
    p_sizes = Counter(predicted)
    t_sizes = Counter(true)

    # Pairs co-clustered in predicted
    pp = sum(comb(c, 2) for c in p_sizes.values())
    # Pairs co-clustered in true
    tt = sum(comb(c, 2) for c in t_sizes.values())
    # Pairs co-clustered in BOTH (need joint distribution)
    joint = Counter(zip(predicted, true))
    pt = sum(comb(c, 2) for c in joint.values())

    tp = pt

    precision = tp / pp if pp > 0 else 1.0
    recall = tp / tt if tt > 0 else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def _adjusted_rand_index(predicted: list, true: list) -> float:
    """Adjusted Rand Index per Hubert & Arabie (1985)."""
    n = len(predicted)
    if n < 2:
        return 1.0

    contingency = Counter(zip(predicted, true))
    a = Counter(predicted)  # row sums
    b = Counter(true)  # col sums

    sum_comb_c = sum(comb(c, 2) for c in contingency.values())
    sum_comb_a = sum(comb(c, 2) for c in a.values())
    sum_comb_b = sum(comb(c, 2) for c in b.values())
    total = comb(n, 2)

    if total == 0:
        return 1.0

    expected = sum_comb_a * sum_comb_b / total
    max_index = (sum_comb_a + sum_comb_b) / 2

    if max_index == expected:
        # Both perfectly equal → ARI undefined; return 1 if they truly match
        return 1.0 if sum_comb_c == expected else 0.0
    return (sum_comb_c - expected) / (max_index - expected)
