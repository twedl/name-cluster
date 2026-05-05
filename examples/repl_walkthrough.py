# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "name_cluster",
#     "polars>=1.20",
# ]
# ///
"""Step-by-step walkthrough of the name_cluster pipeline.

Designed for copy-paste into a REPL — every block is a top-level statement,
no main() to dive through. Run from `python -i examples/repl_walkthrough.py`
to land at the prompt with everything loaded.

Each block prints what it just did. Comments explain why. Edit thresholds
and inputs in place.
"""
from __future__ import annotations

import os
from pathlib import Path

import name_cluster as nc
import polars as pl

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
# Prefer the GLEIF cache when present (~3.3M real entity names). If you
# haven't downloaded it, we fall back to a tiny inline DataFrame so the
# walkthrough still runs.

cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "name_cluster"
gleif_files = sorted((cache_root / "gleif" / "parquet").glob("lei2-2*.parquet"))

if gleif_files:
    full = pl.read_parquet(gleif_files[-1])
    # Block by country to keep the walkthrough fast + meaningful. UK has lots
    # of distinct entity types and a large alias surface; swap to "DE" or
    # "CN" to see different normalization rules fire.
    df = full.filter(pl.col("country") == "GB").sample(n=2000, seed=0)
    name_col = "legal_name"
    print(f"loaded {df.height:,} GB names from {gleif_files[-1].name}")
else:
    df = pl.DataFrame({
        "name": [
            "Acme Corporation",
            "ACME Corp Inc",
            "Acme Corp.",
            "Apple Computer Co.",
            "Apple Inc",
            "Microsoft Corporation, USA",
            "Microsoft Corp",
            "International Business Machines",
            "I.B.M. Inc",
            "IBM Corp",
            "Foothill Industries",
            "Foothill Inds Limited",
            "Sherwin-Williams Co",
            "Sherwin Williams Company",
            None,
        ],
    })
    name_col = "name"
    print(f"GLEIF cache not found, using {df.height} inline names")
    print("(run `uv run scripts/download_corpora.py gleif` to populate)")

print(df.head(5))


# ---------------------------------------------------------------------------
# 2. Normalize one name at a time
# ---------------------------------------------------------------------------
# `normalize()` previews exactly what the lib compares. Useful when a pair
# you expect to merge isn't merging — see what each side looks like post-
# normalize.

samples = df[name_col].drop_nulls().head(8).to_list()
for raw in samples:
    print(f"  {raw!r:50} -> {nc.normalize(raw)!r}")


# ---------------------------------------------------------------------------
# 3. Cluster the column
# ---------------------------------------------------------------------------
# Default threshold (0.85) is high-precision. The output adds two columns;
# row order is preserved. Null/empty-post-normalize rows get null in both.

result = nc.cluster(df, name_col=name_col)
print(result.head(10))

# How many clusters?
n_clusters = result["cluster_id"].drop_nulls().n_unique()
print(f"  -> {n_clusters} clusters across {df.height} rows")


# ---------------------------------------------------------------------------
# 4. Inspect cluster sizes
# ---------------------------------------------------------------------------
sizes = (
    result.drop_nulls("cluster_id")
    .group_by("cluster_id", "canonical_name")
    .len()
    .sort("len", descending=True)
)
print("top 10 clusters by size:")
print(sizes.head(10))


# ---------------------------------------------------------------------------
# 5. Threshold sweep — see how the partition shifts
# ---------------------------------------------------------------------------
# Higher = more clusters (precision); lower = fewer (recall). Sweep on a
# labeled sample to find your domain's sweet spot.

print(f"  {'threshold':>10}  {'n_clusters':>10}")
for t in (0.70, 0.80, 0.85, 0.90, 0.95):
    r = nc.cluster(df, name_col=name_col, threshold=t)
    n = r["cluster_id"].drop_nulls().n_unique()
    print(f"  {t:>10.2f}  {n:>10}")


# ---------------------------------------------------------------------------
# 6. Debug: candidates() — what pairs scored close, and at what cosine?
# ---------------------------------------------------------------------------
# `min_score=0.0` returns every LSH-blocked pair so you can scan for
# borderline calls. Bump min_score to focus on near-misses around your
# threshold of interest.

cand = nc.candidates(df, name_col=name_col, min_score=0.7)
print(f"  {cand.height} pairs with score >= 0.7")
print(cand.sort("score", descending=True).head(10))


# ---------------------------------------------------------------------------
# 7. Debug: explain() — what's inside one cluster?
# ---------------------------------------------------------------------------
# Pick the largest cluster and dig in. Edges show every pair of unique
# members and their n-gram overlap (count-cosine, not the IDF-weighted
# score the lib used to merge them — IDF on a tiny subset is degenerate).

biggest_cid = sizes[0, "cluster_id"]
info = nc.explain(result, biggest_cid, name_col=name_col)
print(f"  cluster {biggest_cid}: canonical={info['canonical']!r}  size={info['size']}  hub_radius={info['hub_radius']}")
for member in info["members"][:5]:
    print(f"    member: {member}")
for a, b, score in info["edges"][:5]:
    print(f"    edge:   {score:.3f}  {a!r:35} <-> {b!r}")


# ---------------------------------------------------------------------------
# 8. Aliases — force-merge an acronym with its expansion
# ---------------------------------------------------------------------------
# char-n-gram cosine never bridges "IBM" with "International Business
# Machines". User-supplied aliases canonicalize both sides to the same
# normalized string before MinHash/LSH/TF-IDF run.

ibm_demo = pl.DataFrame({
    "name": [
        "IBM Corp",
        "I.B.M. Inc",
        "International Business Machines",
        "International Business Machines Corp",
        "Apple Inc",
    ],
})

print("without aliases:")
print(nc.cluster(ibm_demo, name_col="name").select("name", "cluster_id"))

print("with aliases:")
print(nc.cluster(
    ibm_demo, name_col="name",
    aliases={"International Business Machines": ["IBM", "I.B.M."]},
).select("name", "cluster_id"))


# ---------------------------------------------------------------------------
# 9. lsh_calibrate — pick (bands, rows) for a target jaccard / recall
# ---------------------------------------------------------------------------
# When the default 32×4 (num_perm=128) doesn't match your similarity
# regime, this picks a configuration that hits target recall at your
# target jaccard while minimising candidate-collision noise.

print(nc.lsh_calibrate(target_jaccard=0.6, target_recall=0.95))
print(nc.lsh_calibrate(target_jaccard=0.85, target_recall=0.99))


# ---------------------------------------------------------------------------
# 10. score_clusters — quality metrics against ground truth
# ---------------------------------------------------------------------------
# Round-trip on synthetic data: generator emits known true_entity_id, then
# score_clusters compares predicted vs true with ARI + pairwise F1.

ds = nc.generate_examples(n_entities=40, difficulty="medium", seed=42)
synthetic_result = nc.cluster(ds, name_col="variant_name")
metrics = nc.score_clusters(
    synthetic_result["cluster_id"].to_pylist(),
    ds["true_entity_id"].to_pylist(),
)
print(metrics)


# ---------------------------------------------------------------------------
# Variables left in the REPL after this script runs:
#   df, name_col          — your loaded input
#   result                — clustered output
#   sizes                 — per-cluster size table
#   cand                  — candidate-pair scores
#   info                  — explain() dict for the biggest cluster
#   ibm_demo, ds          — small demo frames
# ---------------------------------------------------------------------------
print("\nready — paste more `nc.*` calls or inspect the variables above.")
