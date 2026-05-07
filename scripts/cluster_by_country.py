#!/usr/bin/env python3
"""Cluster names per country, write a Hive-partitioned parquet output.

For each distinct value in the partition column, runs ``nc.cluster()``
over the unique names in that group and writes the result under::

    <output>/<partition_col>=<value>/data.parquet

Each per-partition file has columns ``<name_col>``, ``cluster_id``,
``canonical_name``. The partition column is encoded in the path (Hive
convention) — read back with::

    pl.scan_parquet("<output>", hive_partitioning=True).collect()

Notes
-----
- Reads the input parquet once at startup (only ``--name-col`` and
  ``--partition-col`` are projected, so other columns don't compete
  for RAM). The full df stays resident across the whole run.
- Cluster IDs are call-local: each country starts at 0. If you need
  globally-unique IDs across the whole output, offset post-hoc by
  scanning the partitioned dataset and computing per-partition offsets.
- ``--skip-existing`` makes a long run resumable: re-run with the same
  args and finished partitions are skipped.
- Memory ceiling per country still applies — pass ``--lsh-bands`` /
  ``--lsh-rows`` to dial it down for the largest groups (US is the
  pain point on customs-style data).

Usage
-----
    python scripts/cluster_by_country.py data/names.parquet output/clusters
    python scripts/cluster_by_country.py data/names.parquet output/clusters \\
        --threshold 0.85 --lsh-bands 16 --countries US,DE,FR \\
        --skip-existing
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import namecluster as nc
import polars as pl


def discover_countries(df: pl.DataFrame, partition_col: str) -> list[str]:
    """Return sorted distinct non-null partition-column values."""
    return sorted(c for c in df[partition_col].unique().to_list() if c is not None)


def cluster_one_country(
    df: pl.DataFrame,
    country: str,
    *,
    partition_col: str,
    name_col: str,
    threshold: float,
    seed: int,
    lsh_bands: int,
    lsh_rows: int,
    n_threads: int | None,
) -> pl.DataFrame:
    """Filter to one country, dedup names, cluster, return clustered df."""
    names = df.filter(pl.col(partition_col) == country).select(name_col).unique()
    return nc.cluster(
        names,
        name_col=name_col,
        threshold=threshold,
        seed=seed,
        lsh_bands=lsh_bands,
        lsh_rows=lsh_rows,
        n_threads=n_threads,
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("input", type=Path, help="parquet with names + partition col")
    p.add_argument(
        "output", type=Path, help="output directory (Hive-partitioned written here)"
    )
    p.add_argument("--name-col", default="vendor_name")
    p.add_argument("--partition-col", default="country_of_origin")
    p.add_argument("--threshold", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lsh-bands", type=int, default=32)
    p.add_argument("--lsh-rows", type=int, default=4)
    p.add_argument(
        "--n-threads",
        type=int,
        default=None,
        help="rayon worker count (default: num_cpus). Note: cgroup-unaware",
    )
    p.add_argument(
        "--countries",
        default="",
        help="comma-separated subset of partition values to process "
        "(default: all distinct values from the input)",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip partitions whose output file already exists "
        "(makes interrupted runs resumable)",
    )
    args = p.parse_args()

    if not args.input.exists():
        sys.exit(f"error: {args.input} not found")

    print(f"loading {args.input}")
    t_load = time.perf_counter()
    df = pl.read_parquet(args.input, columns=[args.name_col, args.partition_col])
    print(
        f"  {df.height:,} rows  ({df.estimated_size('mb'):.0f} MB)  "
        f"{time.perf_counter() - t_load:.1f}s"
    )

    if args.countries:
        countries = [c.strip() for c in args.countries.split(",") if c.strip()]
    else:
        countries = discover_countries(df, args.partition_col)
        print(f"  found {len(countries)} partitions")

    args.output.mkdir(parents=True, exist_ok=True)

    summary: list[dict] = []
    grand_t0 = time.perf_counter()
    for i, country in enumerate(countries, 1):
        partition_dir = args.output / f"{args.partition_col}={country}"
        partition_path = partition_dir / "data.parquet"
        if args.skip_existing and partition_path.exists():
            print(f"[{i:>3}/{len(countries)}] {country}: skip (exists)")
            continue
        partition_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        clusters = cluster_one_country(
            df,
            country,
            partition_col=args.partition_col,
            name_col=args.name_col,
            threshold=args.threshold,
            seed=args.seed,
            lsh_bands=args.lsh_bands,
            lsh_rows=args.lsh_rows,
            n_threads=args.n_threads,
        )
        clusters.write_parquet(partition_path)
        elapsed = time.perf_counter() - t0

        n_names = clusters.height
        n_clusters = clusters["cluster_id"].drop_nulls().n_unique()
        n_null = clusters["cluster_id"].null_count()
        # Drop the per-country df + force collection before the next country's
        # cluster() call peaks. Matters at 6M-name scale where a held df is
        # hundreds of MB.
        del clusters
        gc.collect()

        print(
            f"[{i:>3}/{len(countries)}] {country}: "
            f"{n_names:>9,} names → {n_clusters:>8,} clusters  "
            f"({n_null:,} null)  {elapsed:>6.1f}s"
        )
        summary.append(
            {
                "country": country,
                "n_names": n_names,
                "n_clusters": n_clusters,
                "n_null": n_null,
                "elapsed_s": elapsed,
            }
        )

    grand_elapsed = time.perf_counter() - grand_t0
    total_names = sum(s["n_names"] for s in summary)
    total_clusters = sum(s["n_clusters"] for s in summary)
    print()
    print(
        f"done — {len(summary)} partitions, {total_names:,} names → "
        f"{total_clusters:,} clusters in {grand_elapsed:.1f}s"
    )
    print(f"output: {args.output}")
    print(
        "read back: "
        f"pl.scan_parquet({str(args.output)!r}, hive_partitioning=True).collect()"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
