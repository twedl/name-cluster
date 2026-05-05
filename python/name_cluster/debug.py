"""Debug surfaces — `candidates()` and `explain()`.

`candidates(data, name_col, ...)` returns scored candidate pairs from the
LSH+rerank stages without clustering. Useful for picking a `threshold`
before committing to a `cluster()` run.

`explain(result, cluster_id)` returns per-cluster diagnostics: members,
canonical, internal edges with scores, hub eccentricity, size. Edges are
recomputed on demand from the cluster's members (cheap O(k²) for small
clusters; lib does not retain edge state post-clustering).
"""
from __future__ import annotations

from typing import Sequence

import narwhals as nw

from ._lowlevel import (
    candidate_pairs_lists as _candidate_pairs_lists,
    pairwise_cosines_lists as _pairwise_cosines_lists,
)


@nw.narwhalify
def candidates(
    data,
    name_col: str = "name",
    min_score: float = 0.0,
    seed: int = 0,
    ngram_size: int = 3,
    lsh_bands: int = 32,
    lsh_rows: int = 4,
    max_name_length: int = 256,
    aliases: dict[str, list[str]] | None = None,
):
    """Return LSH candidate pairs with their TF-IDF cosine scores.

    Parameters
    ----------
    data, name_col, seed, ngram_size, lsh_bands, lsh_rows, max_name_length, aliases
        Same as `cluster()`.
    min_score : float in [0.0, 1.0]
        Drop pairs scoring below this. Default 0.0 = return everything.
        Use to inspect what's near a candidate threshold.

    Returns
    -------
    Same df type as input, with five columns:
        - name_a, name_b : str   — raw names from the input
        - normalized_a, normalized_b : str — what the lib actually compared
        - score : float          — cosine similarity of the n-gram vectors

    Pairs are over UNIQUE normalized names; if the input has duplicate
    rows that normalize identically, they appear once in the output (with
    the first-occurrence row's name).
    """
    raw_names = data[name_col].to_list()
    coerced = [n if isinstance(n, str) else None for n in raw_names]
    idx_a, idx_b, scores, unique_norm, original_to_unique = _candidate_pairs_lists(
        coerced,
        min_score=min_score,
        seed=seed,
        ngram_size=ngram_size,
        lsh_bands=lsh_bands,
        lsh_rows=lsh_rows,
        max_name_length=max_name_length,
        aliases=aliases,
    )

    unique_to_first_row: list[int | None] = [None] * len(unique_norm)
    for row_idx, u in enumerate(original_to_unique):
        if u is not None and unique_to_first_row[u] is None:
            unique_to_first_row[u] = row_idx

    name_a = [raw_names[unique_to_first_row[a]] for a in idx_a]
    name_b = [raw_names[unique_to_first_row[b]] for b in idx_b]
    normalized_a = [unique_norm[a] for a in idx_a]
    normalized_b = [unique_norm[b] for b in idx_b]

    backend = data.implementation
    return nw.from_dict(
        {
            "name_a": name_a,
            "name_b": name_b,
            "normalized_a": normalized_a,
            "normalized_b": normalized_b,
            "score": scores,
        },
        backend=backend,
    )


def explain(result, cluster_id: int, name_col: str = "name", ngram_size: int = 3) -> dict:
    """Return diagnostic info for a single cluster from a `cluster()` result.

    Parameters
    ----------
    result : DataFrame
        The output of `nc.cluster(...)` with `cluster_id` and
        `canonical_name` columns appended.
    cluster_id : int
        The cluster to introspect.
    name_col : str
        The original input name column on `result` (same value passed to
        `cluster()`). Required because `cluster()` doesn't carry the
        column name as metadata.
    ngram_size : int
        Same as `cluster()`. Defaults to the lib default.

    Returns
    -------
    dict with keys:
        - canonical : str         — the cluster's canonical (hub) name
        - members : list[str]     — RAW names belonging to this cluster,
                                    deduplicated by their normalized form
        - edges : list[(str, str, float)]
                                  — every pair of unique members, scored by
                                    n-gram-count cosine (no IDF — IDF on a
                                    tiny corpus is degenerate)
        - hub_radius : int        — BFS eccentricity from the hub treating
                                    every pair as an edge (= 0 for singleton,
                                    1 for fully connected)
        - size : int              — number of unique members
    """
    nw_result = nw.from_native(result, eager_only=True)
    df = nw_result.filter(nw.col("cluster_id") == cluster_id)
    if df.shape[0] == 0:
        raise ValueError(f"no rows with cluster_id={cluster_id}")
    canonical = df["canonical_name"].to_list()[0]

    if name_col not in nw_result.columns:
        raise ValueError(
            f"name_col={name_col!r} not in result columns; pass the same "
            f"column name you passed to cluster()"
        )
    raw_names: list[str] = [n for n in df[name_col].to_list() if isinstance(n, str)]
    if not raw_names:
        return {
            "canonical": canonical,
            "members": [],
            "edges": [],
            "hub_radius": 0,
            "size": 0,
        }

    from ._lowlevel import normalize as _normalize
    normalized = [_normalize(n) for n in raw_names]
    seen: dict[str, str] = {}
    for raw, norm in zip(raw_names, normalized):
        if norm and norm not in seen:
            seen[norm] = raw
    unique_raw = list(seen.values())
    unique_norm = list(seen.keys())

    if len(unique_norm) < 2:
        return {
            "canonical": canonical,
            "members": unique_raw,
            "edges": [],
            "hub_radius": 0,
            "size": len(unique_raw),
        }

    idx_a, idx_b, scores = _pairwise_cosines_lists(unique_norm, ngram_size=ngram_size)
    edges = [
        (unique_raw[a], unique_raw[b], score) for a, b, score in zip(idx_a, idx_b, scores)
    ]

    hub_radius = _bfs_eccentricity_from_canonical(unique_norm, canonical, idx_a, idx_b, scores)

    return {
        "canonical": canonical,
        "members": unique_raw,
        "edges": edges,
        "hub_radius": hub_radius,
        "size": len(unique_raw),
    }


def _bfs_eccentricity_from_canonical(
    unique_norm: Sequence[str],
    canonical: str,
    idx_a: Sequence[int],
    idx_b: Sequence[int],
    scores: Sequence[float],
) -> int:
    """BFS eccentricity from the hub (=canonical) over edges with score>=0,
    i.e. every pair. With a complete graph this is always ≤ 1 (every other
    node is one hop). Useful indicator nonetheless: for cluster sizes >= 2
    with no edges this returns 0 (singleton).
    """
    try:
        hub = unique_norm.index(canonical)
    except ValueError:
        return 0
    n = len(unique_norm)
    adj: list[list[int]] = [[] for _ in range(n)]
    for a, b, _ in zip(idx_a, idx_b, scores):
        adj[a].append(b)
        adj[b].append(a)
    dist: list[int] = [-1] * n
    dist[hub] = 0
    queue = [hub]
    head = 0
    max_d = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                queue.append(v)
                if dist[v] > max_d:
                    max_d = dist[v]
    return max_d
