# name_cluster

Cluster business names into groups representing the same legal entity.

Given customs-form-style data where the same exporter appears as
`"IBM USA"`, `"International Business Machines Inc"`, `"00 IBM"`, etc., the
library produces stable cluster IDs grouping these as one entity, plus a
canonical name per cluster. Designed for millions of names on CPU-only
deployments with limited memory; no GPU, no model downloads, no network.

Rust core (PyO3 binding via maturin) + Python adapter over
[narwhals](https://github.com/narwhals-dev/narwhals), so the same call
works on polars, pandas, and pyarrow inputs.

## Install

PyPI publishing is pending. From source (requires rust toolchain):

```bash
git clone https://github.com/jessetweedle/name-cluster && cd name-cluster
uv venv .venv && source .venv/bin/activate
uv pip install maturin
maturin develop --release
```

Runtime deps (resolved automatically): `narwhals`, `pyarrow`. The lib has
no model weights, no data files, no network calls.

## Quickstart

A runnable end-to-end demo lives at [`examples/quickstart.py`](./examples/quickstart.py)
— exercises every public symbol on synthetic data, plus an optional
real-data section that activates if you've populated the dev cache via
`scripts/download_*.py`. Run it with `python examples/quickstart.py`.

The minimal version:

```python
import name_cluster as nc
import polars as pl

df = pl.DataFrame({
    "name": [
        "Acme Corporation",
        "ACME Corp",
        "Acme Corporation Inc",
        "Apple Computer Co.",
        "Apple Inc",
        None,
    ],
})

result = nc.cluster(df, name_col="name")
```

```
shape: (6, 3)
┌──────────────────────┬────────────┬────────────────┐
│ name                 ┆ cluster_id ┆ canonical_name │
╞══════════════════════╪════════════╪════════════════╡
│ Acme Corporation     ┆ 0          ┆ acme           │
│ ACME Corp            ┆ 0          ┆ acme           │
│ Acme Corporation Inc ┆ 0          ┆ acme           │
│ Apple Computer Co.   ┆ 2          ┆ apple computer │
│ Apple Inc            ┆ 1          ┆ apple          │
│ null                 ┆ null       ┆ null           │
└──────────────────────┴────────────┴────────────────┘
```

(Cluster IDs are assigned `0..N-1` sorted by canonical name ascending, so
`acme` < `apple` < `apple computer`. Deterministic given the same seed.)

Same call works on a pandas `DataFrame` or a pyarrow `Table` — narwhals
detects the input type and returns the same type.

## Public API

```python
import name_cluster as nc

# Main entry — cluster a name column on any dataframe
nc.cluster(data, name_col="name", threshold=0.85, seed=0, ...)

# REPL/notebook convenience for a flat list of strings
nc.cluster_names(["IBM Corp", "IBM Inc", "Apple Inc"])  # -> [0, 0, 1]

# Single-name normalization (debug what the lib actually compares)
nc.normalize("00 IBM Corp.")  # -> "ibm"

# Synthetic data + cluster-quality metrics for evaluation
ds = nc.generate_examples(n_entities=100, difficulty="medium", seed=0)
metrics = nc.score_clusters(predicted, true)  # ARI, F1, precision, recall

# LSH config picker for a target Jaccard + recall
nc.lsh_calibrate(target_jaccard=0.6, target_recall=0.95)
# -> {"bands": ..., "rows": ..., "num_perm": ..., "p_at_target": ..., "p_at_fp": ...}

# Debug: what pairs did LSH propose, and at what cosine score?
nc.candidates(df, name_col="name", min_score=0.5)
# -> df with name_a, name_b, normalized_a, normalized_b, score

# Debug: what's inside one cluster — members, edges, hub eccentricity?
nc.explain(result, cluster_id=42)
# -> {"canonical": "...", "members": [...], "edges": [(a, b, score), ...],
#     "hub_radius": int, "size": int}
```

## Common patterns

### Block by country (recommended for cross-country corpora)

The library doesn't take a `country_col` kwarg — users do hard blocking
themselves to keep the API tight. Polars idiom:

```python
result = (
    df.group_by("country", maintain_order=True)
      .map_groups(lambda g: nc.cluster(g, name_col="name"))
)
# Note: cluster_ids are call-local (each group starts at 0). If you
# concatenate groups, offset cluster_ids per group to make them unique.
```

### Tune the threshold

Higher `threshold` = more clusters (precision-favoring); lower = fewer
(recall-favoring). Default `0.85` is high precision. Sweep on a labeled
sample to find your domain's sweet spot:

```python
for t in [0.75, 0.80, 0.85, 0.90, 0.95]:
    r = nc.cluster(df, name_col="name", threshold=t)
    metrics = nc.score_clusters(r["cluster_id"], known_labels)
    print(t, metrics)
```

### Generator + round-trip evaluation

```python
# Cap n_entities <= 50 (the embedded toy-canonical pool size) to avoid
# the wrap-around disambig suffix; or pass canonicals=[...] to use your own.
ds = nc.generate_examples(n_entities=40, difficulty="easy", seed=42)
result = nc.cluster(ds, name_col="variant_name")
metrics = nc.score_clusters(
    result["cluster_id"].to_pylist(),
    ds["true_entity_id"].to_pylist(),
)
# Easy difficulty target: ARI > 0.85, recall > 0.95
```

Difficulty levels (per `ARCHITECTURE.md`):

| level  | edits per variant                                              | typical cosine |
|--------|----------------------------------------------------------------|----------------|
| easy   | case + punct + suffix swap                                     | ≥ 0.95         |
| medium | + abbr expansion + leading garbage + accent + `THE` toggle     | ≥ 0.85         |
| hard   | + char typos + word drop + geo suffix + spacing oddities       | ≥ 0.70         |

### Custom suffix lists / abbreviations

Pass `extra_suffixes=[...]` and `extra_canonical={...}` to extend the
shipped lists for niche jurisdictions or industry abbreviations. Defaults
already cover ~45 international legal-form suffixes plus 15 descriptor
canonicalizations (see `ARCHITECTURE.md` § Normalization).

> Note: `extra_suffixes` / `extra_canonical` kwargs are spec'd in
> ARCHITECTURE.md but not yet plumbed through the public API. Tracked as
> a v1.x follow-up.

## Configuration

All knobs are flat kwargs on `cluster()`:

| kwarg | default | what it controls |
|---|---|---|
| `threshold` | 0.85 | cosine cutoff for the TF-IDF rerank |
| `seed` | 0 | deterministic RNG seed (MinHash + LSH bucket hashing) |
| `ngram_size` | 3 | char-n-gram window for vectors |
| `lsh_bands` | 32 | LSH band count |
| `lsh_rows` | 4 | LSH rows/band; `num_perm = bands × rows` |
| `hub_radius_max` | 2 | per-cluster diameter check threshold (flag-only in v1) |
| `diameter_check_min_size` | 5 | skip diameter check on small clusters |
| `max_name_length` | 256 | truncate raw input names beyond this many bytes |

## Scope

- **Input language:** English with light non-ASCII accents (`Café` → `cafe`).
  Non-Latin-script names (CJK, Arabic, Cyrillic, Hebrew, etc.) post-normalize
  to empty strings and are returned with `cluster_id=null`.
- **Performance target:** millions of names per CPU-only k8s notebook,
  < 16 GB peak RAM at 10M-name scale with country blocking.
- **Determinism:** same input + same seed → byte-identical output, on the
  same wheel. Guaranteed within a lib version; cluster IDs may differ
  across `0.x` → `0.y` releases.
- **Out of v1:** soft-scoring side-features (country / products as
  signals rather than block keys); incremental fit/predict; cross-language
  synonym translation (e.g. acronym ↔ expansion). See `ARCHITECTURE.md`
  § Future work for the parked-task list.

## How it works (brief)

```
input names
   │
   ▼
 normalize    NFKD → lower → punct policy → bidirectional legal-form strip
   │          → multi-token compound canonicalize → descriptor abbreviate
   ▼
 char-n-gram  default n=3, byte-level on ASCII-guaranteed normalized strings
   │
   ▼
 MinHash      universal hashing (Mersenne-prime 2^61-1 reduction),
   │          deterministic via rand_chacha
   ▼
 LSH          banded bucketing, default 32×4
   │
   ▼
 candidates   pairs that collide in any band
   │
   ▼
 TF-IDF       sparse char-n-gram vectors, L2-normalized at build time
 rerank       cosine on sorted-merge dot product
   │
   ▼
 threshold    drop pairs below `threshold`
   │
   ▼
 union-find   connected components
   │
   ▼
 hub          max-degree node per CC = canonical
   │
   ▼
 sort + relabel  cluster IDs assigned by canonical name asc
```

Full design rationale, audit findings, and the decision history live in
[`ARCHITECTURE.md`](./ARCHITECTURE.md).

## Development

Run all tests (rust + python integration):

```bash
cargo test --lib                          # 63 rust unit tests
maturin develop --release                  # rebuild + reinstall extension
pytest tests/test_public_api.py            # 11 python integration tests
```

Re-run the normalization audit against real corpora (downloads ~600 MB on
first run; auth required for SAM.gov):

```bash
uv run scripts/download_corpora.py all
uv run scripts/download_sam.py             # requires SAM_API_KEY
uv run scripts/validate_normalization.py
```

## License

MIT — see [`LICENSE`](./LICENSE).
