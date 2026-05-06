# Performance benchmarks

`nc.cluster()` scaling on GLEIF (lei2-2026-05-05).

- Machine: macOS-26.4.1-arm64-arm-64bit (arm)
- Python: 3.12.12, name_cluster: 0.1.0
- Date: 2026-05-05

## Default settings (threshold=0.85, num_perm=128)

| input names | threads | clusters | null | wall time | peak Δ RSS | peak Δ VMS | throughput |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 1 | 963 | 37 | 0.02 s | 4 MB | 22 MB | 40,854 names/s |
| 1,000 | default | 963 | 37 | 0.01 s | 0 MB | 16 MB | 68,163 names/s |
| 10,000 | 1 | 9,561 | 420 | 0.14 s | 37 MB | 56 MB | 70,360 names/s |
| 10,000 | default | 9,561 | 420 | 0.13 s | 0 MB | 16 MB | 76,309 names/s |
| 100,000 | 1 | 94,648 | 4,043 | 2.24 s | 326 MB | 343 MB | 44,655 names/s |
| 100,000 | default | 94,648 | 4,043 | 1.83 s | 271 MB | 300 MB | 54,557 names/s |
| 1,000,000 | 1 | 886,004 | 40,555 | 123.08 s | 6125 MB | 11472 MB | 8,125 names/s |
| 1,000,000 | default | 886,004 | 40,555 | 75.71 s | 6428 MB | 10803 MB | 13,208 names/s |

## Methodology

- Single `nc.cluster()` call (default `threshold=0.85`, `num_perm=128`) against a seeded random sample of the corpus.
- `threads` column: `n_threads=` value passed to `cluster()`. `default` = rayon's `num_cpus`. `1` forces single-threaded.
- Wall time: `time.perf_counter()` around the call.
- Peak Δ RSS: resident memory sampled in a 10ms background loop; max-during minus baseline-just-before-call. macOS pages aggressively, so RSS underestimates total allocation.
- Peak Δ VMS: virtual-memory-mapped delta over the same window — a better proxy for total working memory at scale.
- Cluster count excludes null IDs (rows whose name was null or normalized to empty).
- Numbers are unblocked single-call performance. `group_by('country').map_groups(cluster)` cuts both time and memory meaningfully at multi-million scale.
- Phase 1 parallelism only covers the TF-IDF rerank scoring + per-name vectorisation stages; the `add()` loop and downstream cluster-stage code remain serial. Amdahl's law caps the practical speedup at 1.5–2× until Phase 2 lands.

## Reproduce

```bash
uv pip install psutil polars
python scripts/benchmark.py                  # default sizes + thread comparison
python scripts/benchmark.py --include-large  # also bench at 3M (full GLEIF)
python scripts/benchmark.py --threads 1,4,0  # custom thread sweep
```
