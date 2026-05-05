"""Public cluster API.

`cluster(data, name_col, **opts)`  — narwhalified entry over polars/pandas/pyarrow
`cluster_names(names, **opts)`     — list[str] convenience for REPL/notebooks
`lsh_calibrate(jaccard, recall)`   — recommend (bands, rows) for a target

The rust core lives in `_lowlevel.cluster_lists`; this layer marshals
narwhals/list inputs to the rust list API and attaches results back.
"""
from __future__ import annotations

from typing import NamedTuple, Sequence

import narwhals as nw

from ._lowlevel import cluster_lists as _cluster_lists


def _run(
    names: Sequence[str | None],
    *,
    return_canonical: bool,
    threshold: float,
    seed: int,
    ngram_size: int,
    lsh_bands: int,
    lsh_rows: int,
    hub_radius_max: int,
    diameter_check_min_size: int,
    max_name_length: int,
) -> tuple[list[int | None], list[str | None], list[int]]:
    # Coerce np.nan / non-str to None — pandas null repr differs from polars/pyarrow.
    coerced = [n if isinstance(n, str) else None for n in names]
    return _cluster_lists(
        coerced,
        threshold=threshold,
        seed=seed,
        ngram_size=ngram_size,
        lsh_bands=lsh_bands,
        lsh_rows=lsh_rows,
        hub_radius_max=hub_radius_max,
        diameter_check_min_size=diameter_check_min_size,
        max_name_length=max_name_length,
        return_canonical=return_canonical,
    )


@nw.narwhalify
def cluster(
    data,
    name_col: str = "name",
    threshold: float = 0.85,
    seed: int = 0,
    ngram_size: int = 3,
    lsh_bands: int = 32,
    lsh_rows: int = 4,
    hub_radius_max: int = 2,
    diameter_check_min_size: int = 5,
    max_name_length: int = 256,
):
    """Cluster business names into entity groups.

    Parameters
    ----------
    data : DataFrame
        Any narwhals-supported df (polars, pandas, pyarrow Table, modin).
    name_col : str
        Column holding the name strings (str or null).
    threshold : float in [0.0, 1.0]
        Cosine threshold for the TF-IDF rerank step. Default 0.85 = high precision.
    seed : int
        Deterministic RNG seed for MinHash + LSH bucket hashing.
    ngram_size, lsh_bands, lsh_rows : int
        Internal pipeline knobs (defaults match ARCHITECTURE.md).
    hub_radius_max, diameter_check_min_size : int
        Per-cluster diameter check thresholds (flag-only in v1).
    max_name_length : int
        Truncate raw input names beyond this many bytes (UTF-8 safe).

    Returns
    -------
    Same df type as input, with two columns appended:
        - cluster_id (Int64, nullable)  — 0..N-1, sorted by canonical asc
        - canonical_name (String, nullable) — the cluster's hub name
    Rows whose name is null or normalizes to empty get null in both.
    """
    cluster_ids, canonical, _flagged = _run(
        data[name_col].to_list(),
        return_canonical=True,
        threshold=threshold,
        seed=seed,
        ngram_size=ngram_size,
        lsh_bands=lsh_bands,
        lsh_rows=lsh_rows,
        hub_radius_max=hub_radius_max,
        diameter_check_min_size=diameter_check_min_size,
        max_name_length=max_name_length,
    )
    backend = data.implementation
    return data.with_columns(
        cluster_id=nw.new_series("cluster_id", cluster_ids, backend=backend),
        canonical_name=nw.new_series("canonical_name", canonical, backend=backend),
    )


def cluster_names(
    names: Sequence[str | None],
    threshold: float = 0.85,
    seed: int = 0,
    ngram_size: int = 3,
    lsh_bands: int = 32,
    lsh_rows: int = 4,
    hub_radius_max: int = 2,
    diameter_check_min_size: int = 5,
    max_name_length: int = 256,
) -> list[int | None]:
    """REPL convenience: cluster a flat list, return cluster ids only.

    >>> cluster_names(["IBM Corp", "IBM Inc", "Apple Inc"])
    [0, 0, 1]
    """
    cluster_ids, _canon, _flagged = _run(
        names,
        return_canonical=False,
        threshold=threshold,
        seed=seed,
        ngram_size=ngram_size,
        lsh_bands=lsh_bands,
        lsh_rows=lsh_rows,
        hub_radius_max=hub_radius_max,
        diameter_check_min_size=diameter_check_min_size,
        max_name_length=max_name_length,
    )
    return cluster_ids


# Try num_perm budgets ascending so we return the smallest config meeting the
# target. Common factor pairs (rows × bands = num_perm).
_CANDIDATE_NUM_PERM = [16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]


class _LshConfig(NamedTuple):
    bands: int
    rows: int
    p_target: float
    p_fp: float


def lsh_calibrate(
    target_jaccard: float,
    target_recall: float = 0.95,
    fp_jaccard: float | None = None,
) -> dict:
    """Pick (bands, rows, num_perm) achieving `target_recall` at `target_jaccard`
    while keeping the candidate-collision probability LOW at `fp_jaccard`.

    Uses the classic LSH formula `P(candidate | s) = 1 - (1 - s^r)^b`. Without
    the false-positive constraint, the optimization collapses to `rows=1`
    (which inflates collisions at every Jaccard). Adding `fp_jaccard` (default:
    half of `target_jaccard`) selects the configuration with the sharpest
    transition centered around the target.

    Parameters
    ----------
    target_jaccard : float in (0, 1)
        The Jaccard similarity above which we want pairs to land as candidates.
    target_recall : float in (0, 1), default 0.95
        Probability that a pair at `target_jaccard` becomes a candidate.
    fp_jaccard : float in (0, target_jaccard), default target_jaccard / 2
        Reference low-similarity at which collision probability should be
        minimised — controls how steep the LSH transition is.

    Returns
    -------
    dict with keys: bands, rows, num_perm, p_at_target, p_at_fp.
    """
    if not 0.0 < target_jaccard < 1.0:
        raise ValueError(f"target_jaccard must be in (0, 1), got {target_jaccard}")
    if not 0.0 < target_recall < 1.0:
        raise ValueError(f"target_recall must be in (0, 1), got {target_recall}")
    if fp_jaccard is None:
        fp_jaccard = target_jaccard / 2.0
    if not 0.0 < fp_jaccard < target_jaccard:
        raise ValueError(
            f"fp_jaccard must be in (0, target_jaccard={target_jaccard}), got {fp_jaccard}"
        )

    for num_perm in _CANDIDATE_NUM_PERM:
        best: _LshConfig | None = None
        for r in range(1, num_perm + 1):
            if num_perm % r != 0:
                continue
            b = num_perm // r
            p_target = 1.0 - (1.0 - target_jaccard ** r) ** b
            if p_target < target_recall:
                continue
            p_fp = 1.0 - (1.0 - fp_jaccard ** r) ** b
            if best is None or p_fp < best.p_fp:
                best = _LshConfig(b, r, p_target, p_fp)
        if best is not None:
            return {
                "bands": best.bands,
                "rows": best.rows,
                "num_perm": num_perm,
                "p_at_target": best.p_target,
                "p_at_fp": best.p_fp,
            }

    raise ValueError(
        f"no LSH config with target_jaccard={target_jaccard}, "
        f"target_recall={target_recall} fits within num_perm <= {_CANDIDATE_NUM_PERM[-1]}; "
        f"raise num_perm or lower target_recall"
    )
