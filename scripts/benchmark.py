#!/usr/bin/env python3
"""Scaling benchmark for nc.cluster() — wall time + peak RSS by input size.

Loads GLEIF if cached, otherwise the synthetic generator. Runs cluster()
at each requested input size, prints a markdown table, and writes it to
docs/benchmark.md.

Run from the project venv (where namecluster is maturin-developed):

    uv pip install psutil polars
    python scripts/benchmark.py
    python scripts/benchmark.py --sizes 1000,10000,100000
    python scripts/benchmark.py --include-large    # adds 3M (full GLEIF)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import threading
import time
from datetime import date
from pathlib import Path

import namecluster as nc
import polars as pl
import psutil

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS = REPO_ROOT / "docs"
CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "name_cluster"


def load_corpus(min_size: int) -> tuple[pl.DataFrame, str, str]:
    """Return (full_df, name_col, source_label). Tries GLEIF first."""
    gleif = sorted((CACHE / "gleif" / "parquet").glob("lei2-2*.parquet"))
    if gleif:
        df = pl.read_parquet(gleif[-1]).filter(pl.col("legal_name").is_not_null())
        return df, "legal_name", f"GLEIF ({gleif[-1].stem})"
    print("  GLEIF cache missing — falling back to synthetic generator")
    n_entities = max(min_size // 10, 50)
    df = nc.generate_examples(n_entities=n_entities, difficulty="medium", seed=42)
    return df, "variant_name", f"synthetic generator (n_entities={n_entities})"


def measure_call(fn, *args, **kwargs) -> tuple[object, float, float, float]:
    """Call fn(*args, **kwargs); return (result, wall_seconds, peak_rss_mb, peak_vms_mb).

    Both peaks are deltas from baseline measured just before the call,
    sampled in a 10ms background loop. RSS = resident set size (what's
    in physical memory). VMS = virtual memory size (everything mapped),
    a better proxy for total allocation when the OS pages aggressively.
    """
    proc = psutil.Process(os.getpid())
    gc.collect()
    base_info = proc.memory_info()
    base_rss, base_vms = base_info.rss, base_info.vms
    peak_rss, peak_vms = base_rss, base_vms
    stop = threading.Event()

    def sampler():
        nonlocal peak_rss, peak_vms
        while not stop.is_set():
            info = proc.memory_info()
            if info.rss > peak_rss:
                peak_rss = info.rss
            if info.vms > peak_vms:
                peak_vms = info.vms
            time.sleep(0.01)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    start = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
    finally:
        elapsed = time.perf_counter() - start
        stop.set()
        t.join(timeout=1.0)
    return (
        result,
        elapsed,
        (peak_rss - base_rss) / 1024 / 1024,
        (peak_vms - base_vms) / 1024 / 1024,
    )


def run(sizes: list[int], thread_configs: list[int | None]) -> list[dict]:
    full, name_col, source_label = load_corpus(min(sizes))
    print(f"corpus: {source_label}  ({full.height:,} rows)")
    print()

    rows: list[dict] = []
    for n in sizes:
        if n > full.height:
            print(f"  skip n={n:,}: corpus has only {full.height:,} rows")
            continue
        sample = full.sample(n=n, seed=42) if n < full.height else full
        for nt in thread_configs:
            result, elapsed, peak_rss_mb, peak_vms_mb = measure_call(
                nc.cluster,
                sample,
                name_col=name_col,
                n_threads=nt,
            )
            cids = result["cluster_id"].to_list()
            n_clusters = len({c for c in cids if c is not None})
            n_null = sum(1 for c in cids if c is None)
            throughput = n / elapsed if elapsed > 0 else float("inf")
            rows.append(
                {
                    "n_input": n,
                    "n_threads": nt,  # None serialises to JSON null
                    "n_clusters": n_clusters,
                    "n_null": n_null,
                    "wall_seconds": elapsed,
                    "peak_rss_delta_mb": peak_rss_mb,
                    "peak_vms_delta_mb": peak_vms_mb,
                    "throughput": throughput,
                }
            )
            label = "default" if nt is None else str(nt)
            print(
                f"  n={n:>10,}  threads={label:>7}  t={elapsed:>7.2f}s  "
                f"rss Δ={peak_rss_mb:>7.0f} MB  vms Δ={peak_vms_mb:>7.0f} MB  "
                f"thr={throughput:>10,.0f} names/s  "
                f"-> {n_clusters:,} clusters ({n_null:,} null)"
            )
            del result
            gc.collect()
        del sample
        gc.collect()
    return rows


def write_report(rows: list[dict], source_label: str) -> Path:
    DOCS.mkdir(exist_ok=True)
    today = date.today().isoformat()
    md = DOCS / "benchmark.md"
    js = DOCS / "benchmark.json"
    js_snapshot = DOCS / f"benchmark-{today}.json"

    info = {
        "machine": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "namecluster": nc.__version__,
        "date": date.today().isoformat(),
        "corpus": source_label,
    }

    with md.open("w") as f:
        f.write("# Performance benchmarks\n\n")
        f.write(f"`nc.cluster()` scaling on {info['corpus']}.\n\n")
        f.write(
            f"- Machine: {info['machine']} ({info['processor']})\n"
            f"- Python: {info['python']}, namecluster: {info['namecluster']}\n"
            f"- Date: {info['date']}\n\n"
        )
        f.write("## Default settings (threshold=0.85, num_perm=128)\n\n")
        f.write(
            "| input names | threads | clusters | null | wall time | peak Δ RSS | peak Δ VMS | throughput |\n"
        )
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            thr_label = "default" if r["n_threads"] is None else str(r["n_threads"])
            f.write(
                f"| {r['n_input']:,} | {thr_label} | {r['n_clusters']:,} | {r['n_null']:,} | "
                f"{r['wall_seconds']:.2f} s | {r['peak_rss_delta_mb']:.0f} MB | "
                f"{r['peak_vms_delta_mb']:.0f} MB | "
                f"{r['throughput']:,.0f} names/s |\n"
            )
        f.write("\n## Methodology\n\n")
        f.write(
            "- Single `nc.cluster()` call (default `threshold=0.85`, "
            "`num_perm=128`) against a seeded random sample of the corpus.\n"
            "- `threads` column: `n_threads=` value passed to `cluster()`. "
            "`default` = rayon's global pool (`RAYON_NUM_THREADS` if set, else "
            "`available_parallelism()`). `1` forces single-threaded.\n"
            "- Wall time: `time.perf_counter()` around the call.\n"
            "- Peak Δ RSS: resident memory sampled in a 10ms background loop; "
            "max-during minus baseline-just-before-call. macOS pages aggressively, "
            "so RSS underestimates total allocation.\n"
            "- Peak Δ VMS: virtual-memory-mapped delta over the same window — a "
            "better proxy for total working memory at scale.\n"
            "- Cluster count excludes null IDs (rows whose name was null or "
            "normalized to empty).\n"
            "- Numbers are unblocked single-call performance. "
            "`group_by('country').map_groups(cluster)` cuts both time and memory "
            "meaningfully at multi-million scale.\n"
            "- Phase 1 parallelism only covers the TF-IDF rerank scoring + "
            "per-name vectorisation stages; the `add()` loop and downstream "
            "cluster-stage code remain serial. Amdahl's law caps the practical "
            "speedup at 1.5–2× until Phase 2 lands.\n\n"
        )
        f.write(
            "## Reproduce\n\n"
            "```bash\n"
            "uv pip install psutil polars\n"
            "python scripts/benchmark.py                  # default sizes + thread comparison\n"
            "python scripts/benchmark.py --include-large  # also bench at 3M (full GLEIF)\n"
            "python scripts/benchmark.py --threads 1,4,0  # custom thread sweep\n"
            "```\n"
        )

    payload = {"info": info, "rows": rows}
    for path in (js, js_snapshot):
        with path.open("w") as f:
            json.dump(payload, f, indent=2)

    return md


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--sizes",
        default="1000,10000,100000,1000000",
        help="comma-separated input sizes (default 1k,10k,100k,1M)",
    )
    p.add_argument(
        "--include-large",
        action="store_true",
        help="also benchmark at 3,000,000 names (full GLEIF)",
    )
    p.add_argument(
        "--threads",
        default="1,0",
        help="comma-separated thread counts (0 = rayon default = all cores "
        "available to the process); "
        "default '1,0' compares single-threaded vs all-cores",
    )
    args = p.parse_args()
    sizes = [int(x) for x in args.sizes.split(",")]
    if args.include_large:
        sizes.append(3_000_000)
    sizes = sorted(set(sizes))
    thread_configs: list[int | None] = [
        None if int(x) == 0 else int(x) for x in args.threads.split(",")
    ]

    rows = run(sizes, thread_configs)
    if not rows:
        print("no rows measured")
        return 1
    full, _, source_label = load_corpus(min(sizes))
    md = write_report(rows, source_label)
    print(f"\nwrote {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
