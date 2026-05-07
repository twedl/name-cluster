#!/usr/bin/env python3
"""Build a stratified-by-score eval set of candidate pairs for manual labelling.

Pipeline::

    parquet input
       │
       ▼  optional sub-sample (--sample-n)
       ▼
    nc.candidates()         ← cached as parquet (--candidates-cache) so
       │                       re-runs that change only sampling/buckets
       │                       skip the expensive LSH+TF-IDF step
       ▼
    bucket by score (equal-width, --buckets)
       │
       ▼  --per-bucket pairs each
       ▼
    join context columns from input by name (--context-cols)
       │
       ▼
    pairs CSV  (name_a, name_b, score, bucket, [<col>_a, <col>_b...],
                normalized_a, normalized_b)

Output is ready for ``python scripts/label_pairs.py``.

Match ``--lsh-bands`` / ``--lsh-rows`` to your ``cluster()`` invocation
so the candidate pool reflects what production sees. The candidates
cache is keyed only by file path — delete it after changing input,
``--name-col``, ``--lsh-*``, ``--min-score``, ``--sample-n``, or
``--seed``, otherwise you'll resample stale pairs.

Usage
-----
    python scripts/build_eval_set.py us_data.parquet eval/us_pairs.csv

    python scripts/build_eval_set.py us_data.parquet eval/us_pairs.csv \\
        --sample-n 500000 --buckets 7 --per-bucket 50 \\
        --context-cols country,address \\
        --candidates-cache eval/us_candidates.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import namecluster as nc
import polars as pl


def load_input(
    path: Path, name_col: str, *, sample_n: int | None, seed: int
) -> pl.DataFrame:
    df = pl.read_parquet(path)
    if name_col not in df.columns:
        sys.exit(
            f"error: --name-col '{name_col}' not in {path}\n"
            f"  available: {', '.join(df.columns)}"
        )
    df = df.filter(pl.col(name_col).is_not_null())
    if sample_n and df.height > sample_n:
        df = df.sample(n=sample_n, seed=seed)
    return df


def get_candidates(
    df: pl.DataFrame,
    *,
    name_col: str,
    min_score: float,
    lsh_bands: int,
    lsh_rows: int,
    seed: int,
    cache: Path | None,
) -> pl.DataFrame:
    if cache and cache.exists():
        cand = pl.read_parquet(cache)
        print(f"  loaded {cand.height:,} candidates from cache: {cache}")
        return cand
    print(f"  running nc.candidates() on {df.height:,} rows  ", end="", flush=True)
    cand = nc.candidates(
        df,
        name_col=name_col,
        min_score=min_score,
        seed=seed,
        lsh_bands=lsh_bands,
        lsh_rows=lsh_rows,
    )
    print(f"→ {cand.height:,} candidate pairs")
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cand.write_parquet(cache)
        print(f"  cached to {cache}")
    return cand


def add_bucket_column(
    cand: pl.DataFrame,
    *,
    min_score: float,
    max_score: float,
    n_buckets: int,
) -> pl.DataFrame:
    """Equal-width buckets in [min_score, max_score). Bucket value = lower
    edge, rounded to 4 places for stable CSV output. Pairs outside the
    range are dropped."""
    step = (max_score - min_score) / n_buckets
    return cand.filter(
        (pl.col("score") >= min_score) & (pl.col("score") < max_score)
    ).with_columns(
        bucket=(
            min_score + ((pl.col("score") - min_score) / step).floor() * step
        ).round(4)
    )


def sample_per_bucket(
    cand: pl.DataFrame, *, per_bucket: int, seed: int
) -> pl.DataFrame:
    """Take up to `per_bucket` random pairs from each bucket. Buckets with
    fewer pairs use what they have."""
    parts: list[pl.DataFrame] = []
    for sub in cand.partition_by("bucket"):
        n = min(per_bucket, sub.height)
        if n > 0:
            parts.append(sub.sample(n=n, seed=seed, with_replacement=False))
    if not parts:
        return cand.head(0)
    return pl.concat(parts).sort("bucket", "score")


def join_context_columns(
    sampled: pl.DataFrame,
    df: pl.DataFrame,
    *,
    name_col: str,
    context_cols: list[str],
) -> pl.DataFrame:
    """Attach context-column values for each side of the pair by joining on
    raw name. For names appearing in multiple input rows, take the first
    non-null value per (name, context_col). Names absent from the lookup
    get null context values (left join)."""
    if not context_cols:
        return sampled
    lookup = df.group_by(name_col, maintain_order=True).agg(
        [pl.col(c).drop_nulls().first().alias(c) for c in context_cols]
    )
    side_a = lookup.rename({name_col: "name_a", **{c: f"{c}_a" for c in context_cols}})
    side_b = lookup.rename({name_col: "name_b", **{c: f"{c}_b" for c in context_cols}})
    return sampled.join(side_a, on="name_a", how="left").join(
        side_b, on="name_b", how="left"
    )


def reorder_columns(out: pl.DataFrame, context_cols: list[str]) -> pl.DataFrame:
    head = ["name_a", "name_b", "score", "bucket"]
    ctx: list[str] = []
    for c in context_cols:
        ctx.extend([f"{c}_a", f"{c}_b"])
    tail = [c for c in ("normalized_a", "normalized_b") if c in out.columns]
    return out.select(head + ctx + tail)


def print_distribution(
    out: pl.DataFrame, *, min_score: float, max_score: float, n_buckets: int
) -> None:
    step = (max_score - min_score) / n_buckets
    counts = out.group_by("bucket").agg(pl.len().alias("n")).sort("bucket")
    print()
    print("bucket distribution:")
    seen: set[float] = set()
    for row in counts.iter_rows(named=True):
        lo = row["bucket"]
        seen.add(round(lo, 4))
        print(f"  [{lo:.2f}, {lo + step:.2f})   n={row['n']}")
    # Surface empty buckets so user sees the gap.
    expected = [round(min_score + i * step, 4) for i in range(n_buckets)]
    empties = [e for e in expected if e not in seen]
    if empties:
        print(f"  empty buckets: {', '.join(f'{e:.2f}' for e in empties)}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("input", type=Path, help="parquet file with name column")
    p.add_argument("output", type=Path, help="output CSV (overwritten)")
    p.add_argument("--name-col", default="name")
    p.add_argument(
        "--context-cols",
        default="",
        help="comma-separated columns to include alongside each name "
        "(e.g. country,address). For names appearing on multiple input "
        "rows, the first non-null value per column is used.",
    )
    p.add_argument(
        "--sample-n",
        type=int,
        default=None,
        help="random-subsample input rows to this many before candidates() "
        "(default: use all rows)",
    )
    p.add_argument(
        "--buckets", type=int, default=7, help="# of equal-width score buckets"
    )
    p.add_argument(
        "--per-bucket", type=int, default=50, help="pairs sampled per bucket"
    )
    p.add_argument(
        "--min-score", type=float, default=0.3, help="lower bound of score range"
    )
    p.add_argument(
        "--max-score",
        type=float,
        default=1.0,
        help="upper bound of score range (exclusive)",
    )
    p.add_argument("--lsh-bands", type=int, default=32)
    p.add_argument("--lsh-rows", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--candidates-cache",
        type=Path,
        default=None,
        help="parquet path to read/write nc.candidates() output. Re-used "
        "as-is on subsequent runs; delete after changing input/--name-col/"
        "--lsh-*/--min-score/--sample-n/--seed.",
    )
    args = p.parse_args()

    if args.min_score >= args.max_score:
        sys.exit("error: --min-score must be < --max-score")
    if args.buckets < 1:
        sys.exit("error: --buckets must be >= 1")

    context_cols = [c.strip() for c in args.context_cols.split(",") if c.strip()]

    print(f"input:    {args.input}")
    print(f"output:   {args.output}")
    print(
        f"params:   buckets={args.buckets}  per-bucket={args.per_bucket}  "
        f"score=[{args.min_score}, {args.max_score})  "
        f"lsh={args.lsh_bands}x{args.lsh_rows}  seed={args.seed}"
    )

    df = load_input(args.input, args.name_col, sample_n=args.sample_n, seed=args.seed)
    print(f"loaded:   {df.height:,} rows from input")

    for c in context_cols:
        if c not in df.columns:
            sys.exit(
                f"error: --context-col '{c}' not in input "
                f"(available: {', '.join(df.columns)})"
            )

    cand = get_candidates(
        df,
        name_col=args.name_col,
        min_score=args.min_score,
        lsh_bands=args.lsh_bands,
        lsh_rows=args.lsh_rows,
        seed=args.seed,
        cache=args.candidates_cache,
    )
    if cand.height == 0:
        sys.exit(
            f"error: 0 candidate pairs at --min-score {args.min_score}; "
            f"lower it or check your input"
        )

    cand = add_bucket_column(
        cand,
        min_score=args.min_score,
        max_score=args.max_score,
        n_buckets=args.buckets,
    )
    sampled = sample_per_bucket(cand, per_bucket=args.per_bucket, seed=args.seed)
    out = join_context_columns(
        sampled, df, name_col=args.name_col, context_cols=context_cols
    )
    out = reorder_columns(out, context_cols)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(args.output)
    print(f"\nwrote {out.height:,} pairs to {args.output}")
    print_distribution(
        out,
        min_score=args.min_score,
        max_score=args.max_score,
        n_buckets=args.buckets,
    )

    print()
    print(
        f"label with: python scripts/label_pairs.py {args.output} "
        f"--hide-cols normalized_a,normalized_b"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
