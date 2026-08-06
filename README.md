# name-cluster

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

```bash
pip install name-cluster   # Linux x86_64 / aarch64 wheels
```

Other platforms (macOS, Windows) — build from source, requires the Rust toolchain:

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
import namecluster as nc
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
import namecluster as nc

# Main entry — cluster a name column on any dataframe
nc.cluster(data, name_col="name", threshold=0.85, seed=0, ...)

# REPL/notebook convenience for a flat list of strings
nc.cluster_names(["IBM Corp", "IBM Inc", "Apple Inc"])  # -> [0, 0, 1]

# Single-name normalization (debug what the lib actually compares)
nc.normalize("00 IBM Corp.")  # -> "ibm"

# Batched, parallel normalize over a list (None / np.nan → None passthrough)
nc.normalize_names(["Acme Corp", "ACME Inc", None])  # -> ["acme", "acme", None]

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

# Discover acronym↔expansion candidates from the corpus (feeds `aliases=`)
nc.acronym_map(df, name_col="name")
# -> df with (acronym, expansion_count, expansions, acronym_examples)
```

## Common patterns

### Add a normalized column to a dataframe

`nc.normalize_names(seq)` is the batched (rayon-parallel) form of
`nc.normalize`. The polars idiom:

```python
df = df.with_columns(
    pl.col("name")
      .map_batches(
          lambda s: pl.Series(nc.normalize_names(s.to_list())),
          return_dtype=pl.Utf8,
      )
      .alias("normalized")
)
```

`map_batches` hands the whole series to the callback in one shot, so
rayon parallelizes the Rust loop across all available cores. Useful for
inspecting what `cluster()` actually compares, or pre-deduping before
clustering. On pandas, `df["name"].map(nc.normalize)` works but is
serial — go through `normalize_names` for the batched path.

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

For large corpora (millions of names, memory-constrained pods),
[`scripts/cluster_by_country.py`](./scripts/cluster_by_country.py) does
the same partitioned clustering as a standalone, resumable job: reads a
parquet input once, clusters each country group, and writes a
Hive-partitioned parquet output (`--skip-existing` makes a long run
resumable across restarts).

```bash
uv run scripts/cluster_by_country.py data/names.parquet output/clusters \
    --threshold 0.85 --lsh-bands 16 --skip-existing
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

### Build a labeled eval set from real data

When you don't have ground-truth labels for your corpus, the two scripts
in `scripts/` give you a workflow for hand-labeling a stratified sample
of candidate pairs.

**1. Sample candidate pairs across the score range.**
`scripts/build_eval_set.py` runs `nc.candidates()` over your parquet
input, buckets pairs by cosine score, and writes a labeling-ready CSV.

```bash
uv run scripts/build_eval_set.py us_data.parquet eval/us_pairs.csv \
    --sample-n 1000000 --buckets 7 --per-bucket 50 \
    --context-cols country,address \
    --candidates-cache eval/us_candidates.parquet
```

Match `--lsh-bands` / `--lsh-rows` to your production `cluster()` call so
the candidate pool reflects what production sees. The cache parquet
skips the expensive LSH+TF-IDF step on re-runs that change only
sampling/buckets — pass `--rebuild-cache` to force recompute.

**2. Adjudicate each pair.** `scripts/label_pairs.py` is a single-
keystroke terminal labeler. Reads the CSV, prompts `s` / `d` / `u`
(same / different / unsure) per row, autosaves on every keystroke,
supports undo (`b`) and resume (re-running picks up where you left off):

```bash
uv run scripts/label_pairs.py eval/us_pairs.csv \
    --hide-cols normalized_a,normalized_b
```

The labeled CSV's `label` column then drives the threshold sweep above:
build a `(score, label)` table from the file and read precision-vs-
threshold straight off it.

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

### Acronym / expansion aliases

Pass `aliases={canonical: [alias, ...]}` to force-merge an acronym with
its expansion (or any other variant pair the lib's char-n-gram cosine
won't bridge by itself):

```python
nc.cluster(
    df, name_col="name",
    aliases={"International Business Machines": ["IBM", "I.B.M."]},
)
# All four — "IBM Corp", "I.B.M. Inc", "International Business Machines",
# "International Business Machines Inc" — collapse into one cluster with
# canonical_name="intl business machines".
```

Each canonical and each alias is normalized; post-normalize, every alias
form is rewritten to the canonical's form before MinHash/LSH/TF-IDF run.
If two canonicals map the same alias, the last one wins.

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
| `hub_radius_max` | 2 | per-cluster diameter check threshold for the hub-radius split |
| `diameter_check_min_size` | 5 | skip diameter check on small clusters |
| `max_name_length` | 256 | truncate raw input names beyond this many bytes |
| `aliases` | `None` | acronym/expansion override map: `{canonical: [alias, ...]}` |
| `n_threads` | `None` | worker threads for parallelised stages; `None` = rayon default (CPU count) |

## Scope

- **Input language:** English with light non-ASCII accents (`Café` → `cafe`).
  Non-Latin-script names (CJK, Arabic, Cyrillic, Hebrew, etc.) post-normalize
  to empty strings and are returned with `cluster_id=null`.
- **Performance target:** millions of names per CPU-only k8s notebook,
  < 16 GB peak RAM at 10M-name scale with country blocking.
- **Determinism:** same input + same seed → byte-identical output, on the
  same wheel. Guaranteed within a lib version only. Pre-1.0, cluster IDs
  may shift on **any** release, patch included — a normalization change
  moves every affected name to a new ID. Re-cluster after upgrading rather
  than comparing IDs across versions. (0.1.1 shifts IDs: legal-form codes
  in head position are no longer stripped.)
- **Minimum corpus size:** at least 3 unique post-normalize names are
  required for the TF-IDF rerank to produce meaningful cosine scores.
  Inputs with ≤ 2 unique normalized names always return one cluster per
  unique name (no merging) — IDF collapses to zero when every n-gram
  appears in every document. Realistic corpora are never near this floor;
  it bites REPL/test usage with toy inputs.
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
cargo test --lib                          # 83 rust unit tests
maturin develop --release                  # rebuild + reinstall extension
pytest tests/test_public_api.py            # 25 python integration tests
```

### Pre-push hook (local CI)

`scripts/ci.sh` runs ruff (check + format), `cargo fmt`, `cargo clippy
-D warnings`, `cargo test --lib`, `maturin develop` (only if rust changed),
and `pytest`. Wire it as a pre-push hook once:

```bash
git config core.hooksPath .githooks
```

Subsequent `git push` runs the script and aborts on failure. Bypass with
`git push --no-verify`. The script can also be run manually: `scripts/ci.sh`.

Re-run the normalization audit against real corpora (downloads ~600 MB on
first run; auth required for SAM.gov):

```bash
uv run scripts/download_corpora.py all
uv run scripts/download_sam.py             # requires SAM_API_KEY
uv run scripts/validate_normalization.py
```

## License

MIT — see [`LICENSE`](./LICENSE).
