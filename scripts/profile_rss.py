#!/usr/bin/env python3
"""RSS timeline profiler for nc.cluster_names().

Samples resident set size every ``--interval`` seconds (default 100 ms)
in a background thread while clustering runs, then prints a summary and
optionally writes per-run CSV timelines for plotting.

Phase markers are limited to what Python can see (input staging vs the
single rust call) — sub-stage attribution comes from reading the timeline
plot, since the rust core runs as one opaque step from here.

Examples
--------
    # Sweep below the OOM ceiling, write CSVs, default settings:
    uv pip install psutil polars
    python scripts/profile_rss.py --sweep 250000,500000,1000000,2000000 \\
        --csv-dir docs/rss_timelines

    # Vary LSH config to see how much it shifts peak:
    python scripts/profile_rss.py --n 1000000 --lsh-bands 16

    # Real corpus instead of synthetic generator:
    python scripts/profile_rss.py --input names.parquet --col name --n 1000000
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import platform
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

import psutil

REPO_ROOT = Path(__file__).resolve().parent.parent


def fmt_bytes(b: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    i = 0
    while b >= 1024 and i < len(units) - 1:
        b /= 1024
        i += 1
    return f"{b:.1f} {units[i]}"


class RssSampler:
    """Background sampler with phase markers.

    `mark(name)` switches the phase tag attached to subsequent samples
    (and takes one immediately, so the marker shows up even when the
    interval is coarser than the phase).
    """

    def __init__(self, interval_s: float = 0.1):
        self.interval_s = interval_s
        self.proc = psutil.Process(os.getpid())
        self.samples: list[tuple[float, int, int, str]] = []  # (t, rss, vms, phase)
        self._phase = "init"
        self._t0 = time.monotonic()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._sample()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._sample()

    def mark(self, phase: str) -> None:
        self._phase = phase
        self._sample()

    def _sample(self) -> None:
        info = self.proc.memory_info()
        self.samples.append(
            (time.monotonic() - self._t0, info.rss, info.vms, self._phase)
        )

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def report(self, baseline_rss: int) -> dict:
        peak_t, peak_rss, _, peak_phase = max(self.samples, key=lambda s: s[1])
        per_phase: dict[str, int] = {}
        for _, rss, _, phase in self.samples:
            if rss > per_phase.get(phase, 0):
                per_phase[phase] = rss
        return {
            "baseline_rss": baseline_rss,
            "peak_rss": peak_rss,
            "peak_rss_delta": peak_rss - baseline_rss,
            "peak_at_s": peak_t,
            "peak_phase": peak_phase,
            "per_phase_peak_rss": per_phase,
            "n_samples": len(self.samples),
            "duration_s": self.samples[-1][0] if self.samples else 0.0,
        }

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "rss_bytes", "vms_bytes", "phase"])
            for t, rss, vms, phase in self.samples:
                w.writerow([f"{t:.4f}", rss, vms, phase])


def load_names(args: argparse.Namespace, n: int) -> list[str | None]:
    """Return up to `n` names from --input parquet, --gleif cache, or the
    synthetic generator. Names beyond what the source can provide is OK —
    we just return what's available and warn."""
    if args.input:
        import pyarrow.parquet as pq

        tbl = pq.read_table(args.input, columns=[args.col])
        names = tbl.column(args.col).to_pylist()
        return names[:n]

    if args.gleif:
        import polars as pl

        cache = (
            Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
            / "name_cluster"
            / "gleif"
            / "parquet"
        )
        files = sorted(cache.glob("lei2-2*.parquet"))
        if not files:
            sys.exit(f"no GLEIF parquet found under {cache}")
        df = pl.read_parquet(files[-1]).filter(pl.col("legal_name").is_not_null())
        if df.height > n:
            df = df.sample(n=n, seed=args.seed)
        return df["legal_name"].to_list()

    # Synthetic fallback. The generator emits a power-law of variants per
    # entity, so we oversize n_entities then trim to exactly n.
    import namecluster as nc

    n_entities = max(n, 1000)
    tbl = nc.generate_examples(
        n_entities=n_entities,
        difficulty="medium",
        seed=args.seed,
    )
    names = tbl["variant_name"].to_pylist()
    return names[:n]


def run_one(
    names: Sequence[str | None],
    *,
    threads: int | None,
    lsh_bands: int,
    lsh_rows: int,
    threshold: float,
    interval_s: float,
    csv_path: Path | None,
) -> dict:
    """One profiling run. The sampler starts before the cluster call and
    stops after; CSV (if requested) is written even if the call raises,
    so an OOM still leaves a partial timeline behind."""
    import namecluster as nc

    proc = psutil.Process(os.getpid())
    gc.collect()
    baseline_rss = proc.memory_info().rss

    sampler = RssSampler(interval_s=interval_s)
    sampler.start()
    sampler.mark("ready")
    err: BaseException | None = None
    n_clusters = -1
    elapsed = 0.0
    try:
        sampler.mark("cluster_call")
        t0 = time.perf_counter()
        ids = nc.cluster_names(
            list(names),
            threshold=threshold,
            lsh_bands=lsh_bands,
            lsh_rows=lsh_rows,
            n_threads=threads,
        )
        elapsed = time.perf_counter() - t0
        sampler.mark("cluster_done")
        n_clusters = len({i for i in ids if i is not None})
        del ids
    except BaseException as e:  # MemoryError lives outside Exception
        err = e
        sampler.mark("error")
    finally:
        sampler.stop()
        if csv_path:
            sampler.write_csv(csv_path)

    rep = sampler.report(baseline_rss)
    rep["n_input"] = len(names)
    rep["n_clusters"] = n_clusters
    rep["elapsed_s"] = elapsed
    rep["error"] = type(err).__name__ if err else None
    rep["error_msg"] = str(err) if err else None
    return rep


def print_report(rep: dict) -> None:
    err = rep.get("error")
    head = f"  n_input        = {rep['n_input']:,}"
    if err:
        head += f"  [{err}: {rep['error_msg']}]"
    print(head)
    if rep["n_clusters"] >= 0:
        print(f"  n_clusters     = {rep['n_clusters']:,}")
    print(f"  duration       = {rep['duration_s']:.1f} s")
    if rep["elapsed_s"]:
        print(f"  cluster call   = {rep['elapsed_s']:.1f} s")
    print(f"  baseline RSS   = {fmt_bytes(rep['baseline_rss'])}")
    print(
        f"  peak RSS       = {fmt_bytes(rep['peak_rss'])} "
        f"(Δ {fmt_bytes(rep['peak_rss_delta'])})  "
        f"@ t={rep['peak_at_s']:.1f}s phase={rep['peak_phase']}"
    )
    print("  per-phase peak RSS:")
    for phase, b in rep["per_phase_peak_rss"].items():
        print(f"    {phase:14}  {fmt_bytes(b)}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--input", type=Path, help="parquet file with --col")
    src.add_argument(
        "--gleif",
        action="store_true",
        help="use cached GLEIF parquet (~/.cache/name_cluster/gleif)",
    )
    p.add_argument("--col", default="name", help="name column when using --input")

    p.add_argument("--n", type=int, default=500_000, help="single-run input size")
    p.add_argument("--sweep", help="comma-separated sizes; overrides --n if given")

    p.add_argument(
        "--threads", type=int, default=None, help="0 / unset = rayon default"
    )
    p.add_argument("--lsh-bands", type=int, default=32)
    p.add_argument("--lsh-rows", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.85)
    p.add_argument("--interval", type=float, default=0.1, help="sampler period (s)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--csv-dir",
        type=Path,
        default=None,
        help="write per-run RSS timelines here (one CSV per N)",
    )
    args = p.parse_args()

    sizes = [int(s) for s in args.sweep.split(",")] if args.sweep else [args.n]

    print(
        f"profile_rss: source={'input=' + str(args.input) if args.input else 'gleif' if args.gleif else 'synthetic'}  "
        f"threads={args.threads}  lsh={args.lsh_bands}x{args.lsh_rows}  "
        f"threshold={args.threshold}  interval={args.interval}s"
    )
    print(
        f"  host: {platform.platform()}  cpus={os.cpu_count()}  "
        f"total_ram={fmt_bytes(psutil.virtual_memory().total)}"
    )
    print()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    summary: list[dict] = []
    for n in sizes:
        print(f"--- N = {n:,} ---")
        names = load_names(args, n)
        actual = len(names)
        if actual < n:
            print(f"  warning: source had only {actual:,} names")
        csv_path = args.csv_dir / f"rss-{stamp}-n{actual}.csv" if args.csv_dir else None
        rep = run_one(
            names,
            threads=args.threads,
            lsh_bands=args.lsh_bands,
            lsh_rows=args.lsh_rows,
            threshold=args.threshold,
            interval_s=args.interval,
            csv_path=csv_path,
        )
        print_report(rep)
        if csv_path:
            print(f"  csv: {csv_path}")
        summary.append(rep)
        print()
        del names
        gc.collect()
        if rep.get("error") == "MemoryError":
            print("OOM — stopping sweep")
            break

    if len(summary) > 1:
        print("=== sweep summary ===")
        print(
            f"  {'N':>10}  {'peak RSS':>10}  {'Δ peak':>10}  {'phase':<14}  {'time':>7}"
        )
        for rep in summary:
            print(
                f"  {rep['n_input']:>10,}  {fmt_bytes(rep['peak_rss']):>10}  "
                f"{fmt_bytes(rep['peak_rss_delta']):>10}  "
                f"{rep['peak_phase']:<14}  {rep['elapsed_s']:>6.1f}s"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
