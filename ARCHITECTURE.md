# Architecture

Locked design decisions for v1 of the business-name clustering library.
Origin goals live in `PLAN.md`; this doc captures *how* we'll meet them.

Audience: contributors and the primary user. Decisions here can change, but
require revisiting the trade-off captured next to the lock.

---

## Scope

**v1 goal:** cluster a list of English business names (millions-scale) into
groups representing the same legal entity, on CPU only, inside a firewalled
k8s notebook with PyPI access via an internal artifactory mirror.

**Out of scope for v1:** non-English names, soft scoring on side-features
(country/products), incremental fit-then-predict, GPU paths, online learning.
See `Future work` for what's parked.

**Preference ordering** (from PLAN): UX > correctness/accuracy > performance >
verifiability/observability.

---

## Public API surface

Eight symbols. Tight on purpose.

```
name_cluster.cluster(data, ...)             # main entry
name_cluster.cluster_names(names, ...)      # list[str] convenience for REPL
name_cluster.candidates(data, ...)          # debug: candidate pairs + scores
name_cluster.explain(result, cluster_id)    # per-cluster explanation
name_cluster.normalize(name)                # debug: normalization preview
name_cluster.generate_examples(...)         # synthetic perturbation generator
name_cluster.score_clusters(predicted, true)# accuracy metrics (ARI, F1)
name_cluster.lsh_calibrate(jaccard, recall) # band/row config helper
```

### `cluster()` signature

Flat kwargs (chosen over a `Config` dataclass for discoverability via
`help()` and to avoid an extra exported type):

```python
def cluster(
    data,                                # nw-supported df (polars/pandas/pyarrow/...)
    name_col: str = "name",
    threshold: float = 0.85,             # cosine threshold on TF-IDF rerank
    seed: int = 0,
    extra_suffixes: list[str] | None = None,
    extra_canonical: dict[str, str] | None = None,
    with_diagnostics: bool = False,      # adds cluster_size, score_to_canonical, normalized_name
    strict_nulls: bool = False,          # raise on null name (default: skip)
    ngram_size: int = 3,
    lsh_bands: int = 32,
    lsh_rows: int = 4,                   # num_perm = lsh_bands * lsh_rows = 128
    hub_radius_max: int = 2,             # diameter check threshold for CC split
    diameter_check_min_size: int = 5,    # skip diameter check on small CCs
    max_name_length: int = 256,          # truncate normalized names beyond this
):
    ...
```

Returns input df with `cluster_id` (nullable int) and `canonical_name`
(nullable str) columns appended. Row order preserved. `cluster_id` is
`0..N-1`, sorted by canonical name ascending.

`candidates()` mirrors the relevant subset (no cluster-stage knobs).

### Defaults are provisional

The `threshold`, suffix lists, and canonicalize map are tuned by the audit
(see `Validation & audit`). Numbers above are starting points.

---

## Pipeline

### Stages

```
Input names (Arrow)
       │
       ▼
   Normalize        (NFKD → lower → punct → ws → strip → canonicalize)
       │
       ▼
   Internal dedup   (group by normalized name, run pipeline on uniques)
       │
       ▼
   Char n-gram      (ngram_size, default 3)
       │
       ▼
   MinHash sigs     (num_perm = lsh_bands × lsh_rows)
       │
       ▼
   LSH bucketing    (bands × rows; deterministic seeded hash)
       │
       ▼
   Candidate pairs  (above-threshold buckets only)
       │
       ▼
   TF-IDF rerank    (sparse char-n-gram cosine on candidates only)
       │
       ▼
   Threshold filter (≥ cosine threshold)
       │
       ▼
   Connected comps  (union-find)
       │
       ▼
   Diameter check   (per CC of size ≥ min: BFS hub eccentricity;
                     if > hub_radius_max, attempt min-cut split)
       │
       ▼
   Hub selection    (max-degree node per cluster = canonical)
       │
       ▼
   Re-broadcast     (cluster_id back to original (pre-dedup) rows)
       │
       ▼
   Sort + relabel   (cluster_id 0..N-1 by canonical_name asc)
       │
       ▼
Output Arrow with cluster_id + canonical_name
```

### Why this pipeline (B'' hybrid)

| Family | Verdict | Why |
|---|---|---|
| A. Hand-tuned fuzzy | Used as preprocessor only | Brittle across edits; doesn't scale alone |
| B. Char-n-gram TF-IDF + ANN | Backbone | IDF auto-downweights legal-form noise; cosine forgives length asymmetry |
| B'. MinHash-LSH only | Used as blocker | Formal recall guarantee, light memory, set-based — needs preprocessing for biz suffixes |
| C. Fellegi-Sunter probabilistic | Parked v1.5 | Pays off with many side-features; we have one (country) and we punt it from lib |
| D. Embeddings (small CPU model) | Parked v1.5+ | English-only + light accents kills the cross-lingual gain; CPU cost without benefit |

Hybrid: MinHash-LSH for cheap high-recall blocking + TF-IDF cosine for high-precision rerank. Best of both.

### Connected components vs alternatives

Plain CC (single-linkage / union-find) is the default. **Critical reason:**
real big-company clusters in customs data are *star-shaped* (canonical hub
+ many spokes; spokes share the hub but often nothing with each other —
`"IBM"` and `"INT BUS MACH"` share ~0 char-3-grams). This kills:

- Complete-linkage HAC (requires all pairs ≥ threshold → shatters stars)
- Average-linkage HAC (mean dragged down by spoke-spoke near-zeros)
- Density-based post-hoc splits (star = low-density by construction)

CC handles stars correctly because chaining-via-hub is the *feature*. The
risk is bad-bridge chaining between unrelated entities. Diagnostic:
**graph diameter**, not density. Star = diameter 2, chain = diameter k.

**Diameter-aware split** (per-CC, only if size ≥ `diameter_check_min_size`):

1. Find max-degree node = candidate hub.
2. BFS from hub → hub eccentricity (single BFS, O(E) per CC).
3. If eccentricity ≤ `hub_radius_max`: keep CC as-is (star or near-star).
4. Else: attempt split via min-cut / high-betweenness edge removal.

**Side benefit:** the hub IS the canonical name. Free output column. No
separate canonicalization pass. Also serves as anchor for the future
incremental-fit mode (parked).

### Memory architecture

- **Rust core** = builder pattern: `ClusterBuilder::new() → add_chunk() →
  finalize() → (ids, canonical)`. Streaming-shaped, supports future
  incremental + soft-feature modes without API change.
- **Python orchestrator** = single call to builder. No country logic in
  the lib. Users opt into hard country blocking via:

  ```python
  result = (
      df.group_by("country")
        .map_groups(cluster)
      # caller handles cluster_id offset across groups; doc-only helper
  )
  ```

- **Memory ceiling** at default settings: ~12-16 GB for 10M global rows
  (MinHash sigs + LSH index + TF-IDF rerank + UF). 100M rows without
  blocking will OOM on typical k8s notebook pods. Doc the threshold and
  encourage `group_by` for hard blocking.

### Internal dedup

Customs data has heavy exact duplicates (same exporter on many shipments).
Group rows by normalized name, run the pipeline on unique normalized
names only, broadcast cluster_id back to original rows by lookup. Big
perf win when dedup ratio is high (typical for shipment-level data).

Dedup key = normalized name only. Passthrough columns remain row-level.

---

## Normalization

### Pipeline (in order)

1. **Unicode NFKD + strip combining marks** (`Café` → `Cafe`)
1b. **Atomic Latin-extension char map** — letters NFKD doesn't decompose
   (no combining-mark form exists) get explicit ASCII fallbacks:
   `Ł→l, Ø→o, Æ→ae, Œ→oe, ß→ss, Þ→th, Ð/Đ→d, ı/İ→i`. Without this,
   `SPÓŁKA` would split mid-token into `spo ka` because `ł` is treated
   as non-ASCII space. Covers Polish, Danish/Norwegian, German, French,
   Icelandic, Croatian, Turkish.
2. **Lowercase**
3. **Punctuation collapse — split-rule:**
   - `&` → ` and `
   - **Drop entirely:** `.`  `'`  `-`  `_`  `+`
     (chars that often appear/disappear between recordings of the same
     entity: `S.A.`→`sa`, `WAL-MART`→`walmart`, `MCDONALD'S`→`mcdonalds`,
     `H+M`→`hm`)
   - **Replace with space:** `,`  `;`  `:`  `/`  `\`  `|`  `(`  `)`  `[`  `]`
     `{`  `}` and any other non-alnum char (separators between meaningful
     tokens: `IMPORT/EXPORT`→`import export`, `ACME (USA) INC`→`acme usa inc`)
4. **Whitespace collapse**
5. **Strip leading garbage**: `^0+\s+` and `^[#*]+\s+` only
   (matches PLAN's stated `00 IBM`, `000 IBM`; preserves `3M`, `7-Eleven`,
   `123 Textile Mfg`)
6. **Strip leading `THE`**, only if ≥3 tokens remain after
7. **List 3 compound canonicalize (anywhere in name)**: multi-token legal-form
   phrases collapsed to a single canonical token (e.g.
   `LIMITED LIABILITY COMPANY` → `llc`,
   `OBSHCHESTVO S OGRANICHENNOI OTVETSTVENNOSTYU` → `ooo`). Longest-match-first.
   Resulting canonical token is then handled by step 8 (List 1 strip).
8. **Iterative bidirectional strip List 1 (legal-form suffixes)** —
   tail multi-word, then tail single, then head single. Loop to fixed point.
   Always preserves ≥1 token. (Head-strip catches Russian-style prefix forms:
   `JSC ROSNEFT`, `OOO X`, `OAO Y`.)
9. **Token-by-token canonicalize via List 2 map**
10. **Whitespace collapse** again

Steps 1-4 are character-level; 5-10 are token-level on `tokens = s.split()`.

### List 1: legal-form suffixes (always strip, head + tail)

Confirmed by GLEIF/UKCH/OFAC audit (task #13) plus per-jurisdiction
trailing-token analysis. Stripped iteratively from BOTH ends; preserves
≥1 token.

```
# English / Western
INC, LLC, LTD, LIMITED, CORP, CORPORATION, CO, COMPANY, COMPANIES,
LP, LLP, PLC, PTY,

# German (DE/AT/CH)
GMBH, AG, KG, KGAA, OHG, GBR, EV, EG, GGMBH, MBH,
AKTIENGESELLSCHAFT, KOMMANDITGESELLSCHAFT, GESELLSCHAFT,

# French / Belgian / Swiss-French
SA, SAS, SARL, SPRL, SAGL,

# Italian
SRL, SPA, SNC, SS,

# Spanish / Portuguese
SL, SLU, SLP, CB, LDA, LTDA,

# Dutch / Belgian-Dutch
BV, NV, BVBA, VOF, UA,

# Nordic
OY, AB, AS, ASA, OYJ, APS,

# East Asian
KK, GK, TMK, PTE,

# Russian / Slavic (head-strip primary use case)
OOO, OAO, OJSC, PJSC, CJSC, JSC, ZAO, PAO, AO,

# Polish (post-List-3 compound canonicalization)
SPZOO, PSA, SKA, SPK, SPJ,

# Middle East
FZE,

# Turkish
SIRKETI,

# Initialisms preserved through period-drop
CV,

# Multi-word
"CO LTD"
```

User extends via `extra_suffixes=[...]`. Polish forms deferred to
task #15 (compound forms with non-ASCII characters require their own
audit pass).

### List 3: compound legal-form canonicalize (always-on, anywhere in name)

Multi-token legal-form phrases mapped to a single canonical short token.
Applied BEFORE List 1 strip (step 7 in pipeline). The resulting canonical
token is then handled by the normal List 1 head/tail strip — same end
effect as if the multi-word phrase were stripped, but expressed as a
canonicalize-then-strip composition for clarity and future extensibility
(see task #14 cross-language synonyms).

Anchored anywhere in the name; longest-match-first to avoid partial
collisions. Risk of mid-name false-match on legitimate names is low —
phrases like `LIMITED LIABILITY COMPANY` are not natural mid-name content.

```
# English
LIMITED LIABILITY COMPANY        → llc
LIMITED LIABILITY                → llc   (when COMPANY pre-stripped)
LIMITED LIABILITY PARTNERSHIP    → llp
LIMITED PARTNERSHIP              → lp
GENERAL PARTNERSHIP              → gp
JOINT STOCK COMPANY              → jsc
PUBLIC JOINT STOCK COMPANY       → pjsc
OPEN JOINT STOCK COMPANY         → ojsc
CLOSED JOINT STOCK COMPANY       → cjsc

# Russian (transliterated)
OBSHCHESTVO S OGRANICHENNOI OTVETSTVENNOSTYU   → ooo
AKTSIONERNOE OBSHCHESTVO                       → ao

# Polish (post atomic-Latin map; NFKD strips ą/ć/ę/ń/ó/ś/ź/ż diacritics)
SPÓŁKA Z OGRANICZONĄ ODPOWIEDZIALNOŚCIĄ        → spzoo   (LLC)
SP. Z O.O.  /  SP Z O O                        → spzoo   (abbreviated forms)
SPÓŁKA AKCYJNA                                 → sa      (joint stock)
PROSTA SPÓŁKA AKCYJNA                          → psa     (simple JSC, 2021+)
SPÓŁKA KOMANDYTOWO-AKCYJNA                     → ska
SPÓŁKA KOMANDYTOWA                             → spk
SPÓŁKA JAWNA                                   → spj
SPÓŁKA PARTNERSKA                              → spp
SPÓŁKA CYWILNA                                 → sc
SPÓŁKA EUROPEJSKA                              → se

# Spanish / French
SOCIEDAD ANONIMA                 → sa
SOCIEDAD LIMITADA                → sl
SOCIETE ANONYME                  → sa

# Japanese (transliterated)
KABUSHIKI KAISHA                 → kk
```

User extends via `extra_compound_legal={...}` (TBD kwarg).

### List 2: descriptor-noise canonicalization (always-on)

Map all spelling variants to one short canonical form. Preserves the
token (so `ACME HOLDINGS` ≠ `ACME OPERATING`) but eliminates spelling
noise (so `ACME MANUFACTURING` = `ACME MFG`). Always-on by default; no
opt-in flag — there's no false-merge risk because the operation is
deterministic and consistent across all rows.

```
MANUFACTURING, MANUFACT, MFR  → MFG
INTERNATIONAL                 → INTL
IMPORT,  IMPORTS              → IMP
EXPORT,  EXPORTS              → EXP
IMPORT EXPORT, IMP EXP        → IMPEXP
HOLDINGS, HOLDING             → HLDG
GROUP                         → GRP
ENTERPRISES, ENTERPRISE       → ENT
INDUSTRIES, INDUSTRY          → IND
SERVICES,   SVCS              → SVC
SOLUTIONS                     → SLN
TRADING, TRADE                → TRD
TECHNOLOGY, TECHNOLOGIES, TECHS → TECH
DEVELOPMENT                   → DEV
ASSOCIATES                    → ASSOC
```

Synonyms (`GLOBAL` ↔ `WORLDWIDE`) deliberately *not* equated — only
abbreviation pairs of the same word. User extends via `extra_canonical={...}`.

### What's not stripped

- **City/province prefixes** — Chinese-style `SHENZHEN XXX TECH CO LTD`
  is preserved. City often disambiguates entities. Char n-grams plus
  TF-IDF handle the merge naturally via shared core tokens.
- **Numeric tokens** other than zero-prefix garbage — `STORE 47`, `3M`,
  `7-Eleven` keep their digits.
- **In-name abbreviations** (`MFG` mid-name) — already covered by List 2
  canonicalization.

---

## Data interop

`@nw.narwhalify` decorator on the public fn. Caller passes any
narwhals-supported df type (polars, pandas, pyarrow, modin); decorator
unwraps, our code uses one API, output re-wraps to caller's type.

Internal canonical = PyArrow record batch. Zero-copy from polars/pyarrow
into rust via `arrow-rs`. Returned Arrow array attached as new columns.

DuckDB users do `cluster(rel.arrow(), ...)` (zero-copy). Documented one-liner.

`cluster_names(names: list[str], ...) → list[int]` is a thin wrapper that
synthesizes a 1-col `pa.Table` for REPL convenience.

---

## Determinism

**Policy: deterministic clusters with canonical IDs (option (c)).** Same
input + same seed → bit-identical output, including cluster_id assignment.
No single-thread sacrifice — parallelism preserved via sort-then-process.

### Rules

1. One `seed: int = 0` parameter; single source of all randomness.
2. MinHash hash params derived from `rand_chacha::ChaCha8Rng::seed_from_u64(seed)`.
3. Iteration over LSH buckets and candidate pairs uses sorted key order.
4. Rayon reductions affecting output: `par_iter().fold().collect()` then sort,
   or ordered fold.
5. Union-find iterates edges sorted by `(min(a, b), max(a, b))`.
6. Hub selection ties broken by `(degree desc, normalized_name asc)`.
7. Cluster ID relabel: sort components by `(canonical_name asc, size desc)`,
   assign IDs `0..N`.
8. Sparse cosine sums in fixed (sorted) order to remove FP nondeterminism
   within a single pair.

### Cross-machine guarantee

Bit-exact across machines on the same wheel/binary. Across LLVM/rustc
versions: same clusters, possibly different last-bit FP scores at the
boundary (rare to cross threshold). Documented as "deterministic within a
lib version" — `0.1.0` clusters may differ from `0.2.0`.

---

## Observability

### Output columns

Default: `cluster_id` + `canonical_name`.
With `with_diagnostics=True`: + `cluster_size`, `score_to_canonical`,
`normalized_name`.

### `explain(result, cluster_id) -> dict`

Returns `{canonical, members, edges, hub_radius, size}` for one cluster.
Edges and scores recomputed on demand from members (cheap, O(k²) for
cluster of size k; most clusters tiny). Avoids retaining full edge state
post-clustering.

### Debug surfaces

- `normalize(name) -> str` — preview what the lib actually compares.
- `candidates(data, ...) -> df[(idx_a, idx_b, score)]` — return
  candidate pairs without clustering. Threshold-tuning aid.

### Logging

Standard `logging` module, namespace `name_cluster`. Default level
WARNING. Rust core bridges via `pyo3-log` to honor py-side level. INFO =
stage timings, DEBUG = pair-level decisions.

---

## Edge cases (input contract)

| Case | Behavior |
|---|---|
| Null name | Skip; `cluster_id=null`, `canonical_name=null` in output. `strict_nulls=True` raises. |
| Empty / whitespace-only post-norm | Same as null. |
| Exact duplicate names | Internal dedup; same `cluster_id` re-broadcast to all original rows. |
| 0 rows | Return empty df with new cols added. |
| 1 row | `cluster_id=0`, canonical = normalized name. |
| Single-char or 2-char names | Processed. May produce singleton clusters (too few n-grams to candidate-match anything). No warning. |
| Name > 256 chars | Truncate post-normalization. Warn with row indices. Cap configurable via `max_name_length`. |
| Non-UTF8 input | Arrow contract violation; fails at narwhals/arrow boundary, not our concern. |
| Threshold out of [0, 1] | `ValueError` at API entry. |
| Same name with different passthrough columns | Both rows get same cluster_id; passthroughs preserved per row. |

---

## Generator and shipped data

### `generate_examples()`

Pure-Python generator. Public API. Pulls canonical names from a small
embedded list (~50 obviously-fake names baked into Python source — `Acme
Corp`, `Initech LLC`, etc.) by default. User overrides with `canonicals=`.

```python
generate_examples(
    n_entities: int = 1000,
    variants_per_entity: tuple[int, int] = (1, 8),  # power-law
    difficulty: Literal["easy", "medium", "hard"] = "medium",
    seed: int = 0,
    canonicals: list[str] | None = None,
) -> pa.Table  # cols: variant_name, true_entity_id, true_canonical
```

### Difficulty levels

| Level | Edits | Expected match threshold |
|---|---|---|
| easy   | case + punct + whitespace + suffix swap | ≥ 0.95 |
| medium | + abbr expansion/contraction + leading garbage + THE add/drop + accents | ≥ 0.85 |
| hard   | + char typos (≤2/name) + word drop + word add (city/region) + spacing oddities | ≥ 0.70 |

Each level is a superset of the previous.

### Star-topology generation

For each entity, draw `N` variants via power-law (most entities 1-2,
few 50+). Each variant generated independently from the canonical, NOT
from prior variants. Mirrors the customs-form star topology that drove
the cluster-algorithm decisions.

### `score_clusters(predicted, true) -> dict`

Returns `{adjusted_rand, f1_pairs, precision, recall}`. Standard cluster
metrics. ARI handles cluster-id label permutation correctly.

### Shipped data

**Wheel ships zero real-name data.** Just code + the embedded toy
canonicals (~50 names, ~3 KB inline in Python source). Wheel size stays
~200 KB.

Real-data corpora (GLEIF, OFAC SDN, UK Companies House) live in a
**separate repo or release tag**, hosted publicly under their respective
licenses (CC BY 4.0 / public domain / OGL — all redistributable with
attribution). Used only by `scripts/` in the dev repo to:

- Run the normalization rules audit (task #13)
- Tune threshold defaults
- Build regression test sets

**No `datasets` submodule in the lib.** No download code at runtime. Lib
stays pure algorithm.

---

## Validation & audit (task #13)

Real-name corpora drive empirical lockdown of provisional defaults.

| Source | License | Coverage | Use |
|---|---|---|---|
| GLEIF Golden Copy (full) | CC BY 4.0 | 3.3M LEIs (95.2% Latin-script) | Norm rule audit, threshold sweep |
| GLEIF (CN-filter, Latin) | CC BY 4.0 | 703 (English-name CN entities) | CN city/province pattern stress-test |
| OFAC SDN Enhanced (entities) | US public domain | 9,648 entities, 22.8K names, 5,374 multi-alias | Adversarial holdout (multilingual, hard) |
| UK Companies House bulk | OGL v3.0 | 5.7M (99.9% Latin-script) | Norm rule audit (English-heavy) |
| SAM.gov Public Monthly V2 | US public domain | 790K active US entities, 148K legal/DBA pairs | US small-business coverage; bulk download via sam.gov Data Bank, parsed by `scripts/parse_sam_bulk.py` |

Workflow:
1. `scripts/download_corpora.py` fetches snapshots, parses, caches as parquet
   under `~/.cache/name_cluster/<source>/parquet/`.
2. `scripts/validate_normalization.py` applies provisional rules from
   `scripts/_norm.py` (a Python mirror of the eventual rust impl), writes
   `audit/report-YYYY-MM-DD.md` + JSON metrics.
3. Audit findings → final suffix/canonicalize lists, default threshold.
4. Edge cases become regression tests in repo.

**Latin-script filter** at audit boundary: keep names with ≥3 ASCII alpha
chars AND ≥50% ASCII-alpha-or-digit ratio post-NFKD. Mirrors the lib's
runtime "empty-post-norm → null" behavior. v1 scope is English + light
accents; non-Latin names are correctly excluded.

Re-runnable on each rule change. Not bundled in wheel.

### Audit findings driving the final rule set

The provisional rules went through three iterations against the corpora.
Pair-recall measured on OFAC alias holdout (5,374 entities, 46,939 alias
pairs) — the % of known-same-entity pairs whose *normalized forms exactly
match*. Lower bound: the rest depend on the MinHash+cosine fuzzy layer.

| Iteration | Pair-recall (OFAC) | Fully-collapsed | GLEIF substantive |
|---|---|---|---|
| Baseline (drop-all-punct-as-space, end-strip-only, ~30 suffixes) | 974 / 46,939 = 2.08% | 243 | — |
| + refined punct (drop set vs space set) | 1,089 = 2.32% | 268 | — |
| + bidirectional strip (head + tail) + intl suffixes | 1,684 = 3.59% | 313 | 77.71% |
| + List 3 compound canonicalize | 2,760 = 5.88% | 578 | 77.71% |
| + Tier 1+2 European/Asian forms (DE/IT/ES/NL/BE/CH/JP) | 2,770 = 5.90% | 579 | 80.52% |
| + atomic-Latin char map + Polish List 3/1 (task #15) | **2,771 = 5.90%** | **581** | **80.51%** |

Baseline → final = ~3× matched pairs on OFAC, ~2.4× fully-collapsed
entities. The Tier 1+2 European/Asian additions specifically target
GLEIF coverage (~88K more names correctly stripped) — they don't move
OFAC pair-recall much because OFAC's alias holdout is dominated by
Russian/Middle East entities (already covered by Fixes 2+3).
Substantive-change rate at full-corpus scale: OFAC 70.6%, GLEIF 80.5%,
UKCH 98.3%, SAM 76.0%. Empty / one-char alarms <0.005% on all four
sources.

**SAM legal/DBA pair-recall:** 82,874 / 178,588 = **46.41%** with 62,952
entities fully collapsed via normalization alone. Much higher than OFAC's
5.90% because SAM legal-vs-DBA pairs are mostly punct-only or suffix-only
variations (`Acme, Inc.` vs `Acme Inc`) — the ideal normalization target.
The remaining 53% requires the MinHash+cosine fuzzy layer (e.g.
`WORKSOFT, INC.` vs `CERTIFY WORKSOFT, INC.`).

Notable patterns SAM revealed:
- **Multi-location franchise entities**: Sherwin-Williams alone has ~4,500
  separate UEIs across SAM (one per store/location). Each registers
  independently. Three normalized variants exist (`the sherwinwilliams`,
  `sherwin williams`, `sherwinwilliams company the`) due to comma /
  hyphen / `THE` placement differences. Char-n-gram cosine handles these
  reliably. Whether to merge franchise locations into one cluster vs
  keep separate is a use-case judgment, not a normalization concern.
- **`THE` placement**: `THE SHERWIN-WILLIAMS COMPANY` and
  `SHERWIN-WILLIAMS COMPANY, THE` both appear in SAM. Current rule
  strips leading THE only when ≥3 tokens remain after; trailing THE not
  stripped. Possible refinement: also strip trailing THE (if any). Filed
  as audit follow-up but low priority — the cosine layer bridges these.

Notable patterns surfaced by the audit:
- **Russian leading legal-forms** (`JSC X`, `OAO Y`, `OOO Z`) drove
  Fix 2 (head-strip) — accounts for the largest single jump.
- **Long compound forms** (`LIMITED LIABILITY COMPANY`,
  `OBSHCHESTVO S OGRANICHENNOI OTVETSTVENNOSTYU`) drove List 3.
- **Initialism punct** (`S.A.`, `C.V.`, `L.L.C.`, `S.R.L.`) drove the
  drop-set / space-set split in step 3.
- **Documented v1 limitations** (task #14): cross-language synonym
  translation (`OPYTNO KONSTRUKTORSKOE BYURO` ≠ `OKB`), Cyrillic Latin-
  transliteration variants (`CRYOTRADE` vs `KRIOTREID`), Arabic-name
  variant spellings (`HANIFA/HANIFEH/HANIFAH/HUNAIFA`) — all fall to the
  fuzzy layer or remain v1 limitations.

---

## Rust crate inventory

```toml
[dependencies]
pyo3                  = { version = "0.22", features = ["abi3-py39", "extension-module"] }
arrow                 = { version = "53",   features = ["pyarrow"] }
unicode-normalization = "0.1"
iddqd                 = "0.4"     # cluster registry: BiHashMap<id, canonical>
ahash                 = "0.8"     # MinHash + LSH bucket key hashing
rayon                 = "1.10"
rand_chacha           = "0.3"     # seedable deterministic PRNG
rand                  = "0.8"
thiserror             = "1"
log                   = "0.4"
pyo3-log              = "0.11"

[dev-dependencies]
proptest              = "1"
criterion             = { version = "0.5", features = ["html_reports"] }
```

Hand-rolled (no crate): MinHash + LSH bucketing (~300 LOC), char-n-gram
extraction (~30 LOC), sparse cosine on candidate pairs (~50 LOC),
union-find with path compression + rank (~50 LOC), BFS for hub
eccentricity (~30 LOC).

Build: `maturin` with `abi3-py39` → single wheel for Python 3.9+.
CI: `cibuildwheel` for manylinux2014 + macOS arm64/x86_64 + Windows.

Deliberately **not** picked: `gaoya` (hand-roll wins on control + debug
surfaces), `tantivy` (search engine, overkill), `polars` rust dep (lib
stays df-agnostic), `tokio` (CPU-bound batch), `regex` (rules simple
enough to hand-write), `petgraph` / `sprs` / `union-find` crate (each
overkill for our narrow needs).

---

## Distribution

- **PyPI** wheel built by `maturin` with `pyo3` `abi3-py39`.
- **Single wheel per platform** (no per-Python-version builds), works on
  3.9+.
- **Platform matrix**: manylinux2014 x86_64 + aarch64, macOS arm64 +
  x86_64, Windows x64 via `cibuildwheel`.
- **Dependencies** at runtime: `narwhals`, `pyarrow` (transitive via
  narwhals).
- **Firewall context**: PLAN states k8s notebook with PyPI access via
  internal artifactory mirror. All deps above are PyPI-mirrored;
  artifactory must mirror `narwhals`, `pyarrow`, this lib.

---

## Future work (parked)

| Item | Trigger to revisit |
|---|---|
| **Incremental fit/predict** (`Clusterer().fit().predict()`) | Customs data arrives in streams; users want to label new arrivals against historical corpus. Builder API already supports this — needs Python-side stateful wrapper. |
| **Soft-scoring side-features** (country, products) | After v1 validates and users want richer signal. Adds optional kwargs `country_col`, `country_mode='block'/'soft'`, similar for products. d-core builder accepts these without surgery. |
| **Leiden / Louvain community detection** | If diameter-aware CC split shows weakness on dense candidate graphs. Replace step in pipeline; keep API. |
| **Fellegi-Sunter probabilistic scoring** | Pays off when many side-features are added. Ties to soft-feature work. |
| **Embeddings (model2vec, fastText)** | If non-English/cross-script support is needed. Adds model artifact handling. |
| **Auto-threshold tuner** (`suggest_threshold`) | After enough users hit the manual sweep workflow. Cheap helper. |
| **External-memory / spill-to-disk** | If users hit RAM limits at >100M rows without a block key. Builder pattern allows mmap'd sig storage. |
| ~~**Polish compound legal-form audit (task #15)**~~ | DONE. Atomic-Latin char map (Ł→l etc.) + Polish List 3 entries (spzoo, sa, psa, ska, spk, spj, spp, sc, se) + List 1 strip of canonicalized tokens. Full 41K PL GLEIF: 0% → 89.4% substantive change rate; 0 parity mismatches with Python prototype. Char-map fix also benefits German (ß), Danish (ø), French (œ), Icelandic (þ), Croatian (đ), Turkish (ı). |
| **Asian-jurisdiction corpora (task #16)** | GLEIF JP/KR coverage is weak (6-8% Latin). Supplement with JPX listed (~3.8K), KRX listed (~2.3K) when cross-Asian-jurisdiction clustering becomes a real need. |
| **Cross-language synonym translation (task #14)** | Same legal concept, different language root: `OPYTNO KONSTRUKTORSKOE BYURO` ≠ `OKB`, `LIMITED LIABILITY COMPANY` ≠ `OBSHCHESTVO S OGRANICHENNOI OTVETSTVENNOSTYU`. Requires per-jurisdiction translation table or embedding-based similarity. |
| **Acronym ↔ expansion matching (task #17)** | `IBM` vs `INTERNATIONAL BUSINESS MACHINES` share zero char-n-grams. Approach: builder takes `(canonical, [aliases])`; alias = first-letter acronym sig for names with ≥3 tokens. Plus acronym-aware scoring branch (`max(cosine, w·acronym_match)`). Opt-in flag `acronym_aliases=False`. Collision risk on common acronyms — needs disambiguation, ties to soft-scoring side-features. |
| **Corpus-derived acronym map (task #18)** | Utility fn `acronym_map(data, name_col) -> df[(acronym, expansion_count, expansions)]`. Standalone introspection of user's domain data; feeds task #17 with high-confidence single-expansion aliases. Cheap to write independently of #17. |

---

## File layout

```
name_cluster/                        # repo root, also PyPI package name
├── src/                              # rust core
│   ├── lib.rs
│   ├── normalize.rs
│   ├── ngram.rs
│   ├── minhash.rs
│   ├── lsh.rs
│   ├── tfidf.rs
│   ├── cluster.rs                    # union-find + BFS + diameter split
│   └── builder.rs                    # ClusterBuilder
├── python/
│   └── name_cluster/
│       ├── __init__.py               # public surface
│       ├── _toy_canonicals.py        # ~50 fake names for generator default
│       ├── generator.py              # generate_examples + difficulty grammars
│       └── score.py                  # score_clusters (ARI, F1)
├── tests/
│   ├── test_normalize.py
│   ├── test_cluster.py
│   ├── test_generator.py
│   └── test_determinism.py
├── scripts/                          # NOT shipped in wheel
│   ├── download_corpora.py
│   ├── validate_normalization.py
│   └── publish_data_release.py
├── pyproject.toml
├── Cargo.toml
├── README.md
├── PLAN.md                           # original goals
└── ARCHITECTURE.md                   # this file
```

Hosted real-data corpora live either in `data-vYYYY-MM-DD` GitHub
Release tags of this repo, or in a separate `name_cluster-data` repo —
decision deferred to first data publication.
