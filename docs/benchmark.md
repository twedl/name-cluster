# Performance benchmarks

`nc.cluster()` scaling on GLEIF (lei2-2026-05-05).

- Machine: macOS-26.4.1-arm64-arm-64bit (arm)
- Python: 3.12.12, name_cluster: 0.1.0
- Date: 2026-05-05

## Default settings (threshold=0.85, num_perm=128)

| input names | threads | clusters | null | wall time | peak Δ RSS | peak Δ VMS | throughput |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100,000 | 1 | 94,648 | 4,043 | 2.21 s | 436 MB | 460 MB | 45,261 names/s |
| 100,000 | default | 94,648 | 4,043 | 1.32 s | 247 MB | 280 MB | 75,779 names/s |
| 1,000,000 | 1 | 886,004 | 40,555 | 121.26 s | 5546 MB | 11782 MB | 8,247 names/s |
| 1,000,000 | default | 886,004 | 40,555 | 66.58 s | 3437 MB | 7886 MB | 15,020 names/s |

## Methodology

- Single `nc.cluster()` call (default `threshold=0.85`, `num_perm=128`) against a seeded random sample of the corpus.
- `threads` column: `n_threads=` value passed to `cluster()`. `default` = rayon's global pool (`RAYON_NUM_THREADS` if set, else `available_parallelism()`). `1` forces single-threaded.
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
