#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "name_cluster",
#     "polars>=1.20",
# ]
# ///
"""End-to-end demo of the name_cluster public API.

Five sections, each runnable in isolation:

  1. Quickstart        — cluster a 6-row toy DataFrame
  2. cluster_names     — REPL-friendly list-only entry
  3. Generator + score — synthetic round-trip with quality metrics
  4. Threshold sweep   — same data at 5 thresholds, observe cluster count
  5. lsh_calibrate     — pick (bands, rows) for a target jaccard/recall
  6. Real-data demo    — run on cached OFAC names if downloaded; skipped otherwise

Run:
    python examples/quickstart.py
or with uv (auto-installs deps in an ephemeral env):
    uv run examples/quickstart.py
"""
from __future__ import annotations

import os
from pathlib import Path

import polars as pl

import name_cluster as nc


def _section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


# ---------------------------------------------------------------------------
# 1. Quickstart — cluster a small DataFrame
# ---------------------------------------------------------------------------

def quickstart() -> None:
    _section("1. cluster() on a polars DataFrame")
    df = pl.DataFrame({
        "name": [
            "Acme Corporation",
            "ACME Corp",
            "Acme Corporation Inc",
            "Apple Computer Co.",
            "Apple Inc",
            "Microsoft Corp",
            "Microsoft Corporation, USA",
            None,
        ],
    })
    result = nc.cluster(df, name_col="name")
    print(result)


# ---------------------------------------------------------------------------
# 2. cluster_names — REPL convenience
# ---------------------------------------------------------------------------

def repl_convenience() -> None:
    _section("2. cluster_names() — list[str] in, list[int|None] out")
    names = ["IBM Corp", "I.B.M. Inc", "Apple Inc", "Apple", None, ""]
    ids = nc.cluster_names(names)
    for n, i in zip(names, ids):
        print(f"  {repr(n):20} -> cluster {i}")


# ---------------------------------------------------------------------------
# 3. Generator + score_clusters — quality on synthetic data
# ---------------------------------------------------------------------------

def synthetic_round_trip() -> None:
    _section("3. generate -> cluster -> score (round-trip on synthetic)")
    for difficulty in ("easy", "medium", "hard"):
        ds = nc.generate_examples(
            n_entities=40,           # cap <= toy-canonical pool size
            difficulty=difficulty,
            seed=42,
        )
        result = nc.cluster(ds, name_col="variant_name")
        m = nc.score_clusters(
            result["cluster_id"].to_pylist(),  # pyarrow result → ChunkedArray
            ds["true_entity_id"].to_pylist(),
        )
        print(
            f"  difficulty={difficulty:6}  "
            f"variants={ds.num_rows:4}  "
            f"clusters_predicted={m['n_predicted_clusters']:3}  "
            f"true={m['n_true_clusters']}  "
            f"ARI={m['adjusted_rand']:.3f}  "
            f"F1={m['f1_pairs']:.3f}  "
            f"P={m['precision']:.3f}  R={m['recall']:.3f}"
        )


# ---------------------------------------------------------------------------
# 4. Threshold sweep — see how cosine cutoff shifts the partition
# ---------------------------------------------------------------------------

def threshold_sweep() -> None:
    _section("4. threshold sweep (medium-difficulty synthetic)")
    ds = nc.generate_examples(n_entities=30, difficulty="medium", seed=0)
    print(f"  variants: {ds.num_rows}, true entities: 30")
    print(f"  {'threshold':>9}  {'n_clusters':>10}  {'ARI':>5}  {'F1':>5}")
    for t in (0.50, 0.70, 0.85, 0.95, 0.99):
        result = nc.cluster(ds, name_col="variant_name", threshold=t)
        n_clusters = len(set(result["cluster_id"].to_pylist()))
        m = nc.score_clusters(
            result["cluster_id"].to_pylist(),
            ds["true_entity_id"].to_pylist(),
        )
        print(f"  {t:>9.2f}  {n_clusters:>10}  {m['adjusted_rand']:.2f}   {m['f1_pairs']:.2f}")


# ---------------------------------------------------------------------------
# 5. lsh_calibrate — pick (bands, rows) for a target jaccard / recall
# ---------------------------------------------------------------------------

def calibrate() -> None:
    _section("5. lsh_calibrate() recommendations")
    cases = [
        (0.6, 0.95),
        (0.7, 0.95),
        (0.8, 0.99),
        (0.85, 0.95),
    ]
    for j, r in cases:
        cfg = nc.lsh_calibrate(target_jaccard=j, target_recall=r)
        print(
            f"  target jaccard={j} recall={r}  -> "
            f"bands={cfg['bands']:>3} rows={cfg['rows']} "
            f"num_perm={cfg['num_perm']:>4} "
            f"P_target={cfg['p_at_target']:.3f} "
            f"P_fp={cfg['p_at_fp']:.3f}"
        )


# ---------------------------------------------------------------------------
# 6. Real-data demo — runs only if the dev caches exist
# ---------------------------------------------------------------------------

def real_data_demo() -> None:
    _section("6. Real-data demo (cached OFAC names if downloaded)")
    cache_root = Path(
        os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    ) / "name_cluster"
    ofac = sorted((cache_root / "ofac" / "parquet").glob("sdn-aliases-*.parquet"))
    if not ofac:
        print("  no OFAC cache found; skip")
        print("  to populate: uv run scripts/download_corpora.py ofac")
        return

    df = pl.read_parquet(ofac[-1]).filter(pl.col("entity_type") == "Entity")
    print(f"  loaded {df.height:,} OFAC entity-name rows from {ofac[-1].name}")

    # Sample down so the demo runs in a few seconds
    sample = df.sample(n=min(3000, df.height), seed=0)
    result = nc.cluster(sample, name_col="name")

    n_clusters = len(set(result["cluster_id"].to_list())) - (
        1 if None in result["cluster_id"].to_list() else 0
    )
    n_nulls = sum(1 for c in result["cluster_id"].to_list() if c is None)
    print(f"  clustered {sample.height:,} names -> {n_clusters} clusters "
          f"(plus {n_nulls} skipped as null/empty post-norm)")

    # Show a few biggest clusters by size for inspection
    print()
    print("  Top 5 clusters by size:")
    sizes = (
        result.drop_nulls("cluster_id")
              .group_by("cluster_id", "canonical_name")
              .len()
              .sort("len", descending=True)
              .head(5)
    )
    for row in sizes.iter_rows(named=True):
        members = (
            result.filter(pl.col("cluster_id") == row["cluster_id"])
                  .select("name")
                  .head(3)
                  .to_series()
                  .to_list()
        )
        print(f"    [{row['cluster_id']:>4}] {row['canonical_name']!r}  "
              f"({row['len']} members; e.g. {members})")


# ---------------------------------------------------------------------------

def main() -> None:
    print("name_cluster quickstart — exercises the v1 public API")
    print(f"version: {nc.__version__}")
    quickstart()
    repl_convenience()
    synthetic_round_trip()
    threshold_sweep()
    calibrate()
    real_data_demo()
    print()


if __name__ == "__main__":
    main()
