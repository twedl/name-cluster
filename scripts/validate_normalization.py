#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "polars>=1.20",
#     "tqdm>=4.66",
#     "unidecode>=1.3",
# ]
# ///
"""Audit provisional normalization rules against real-name corpora.

Reads cached parquet from scripts/download_corpora.py. Writes a markdown
report under audit/report-YYYY-MM-DD.md plus a JSON metrics dump.

Metrics per source:
  - n_input, n_normalized_nonempty, n_normalized_empty
  - distribution of post-norm token-count and char-length
  - top 100 most-common normalized strings (collision candidates)
  - sample 20 names that became empty after norm (alarms)
  - sample 20 names where post-norm length is 1 char
  - sample 20 names changed substantively by stripping (preview before/after)
  - top 50 trailing tokens in raw names (suffix-list inclusion candidates)
  - top 50 leading tokens in raw names (THE-strip candidates, etc.)

OFAC-specific (recall proxy):
  - For each entity, fraction of {alias} pairs that share a normalized form.
    A higher number means our rules collapse known same-entity variants more
    often. (Upper bound: variants that *should* match by structure alone.)

Usage:
  uv run scripts/validate_normalization.py
  uv run scripts/validate_normalization.py --source ofac
  uv run scripts/validate_normalization.py --sample 200000  # subsample big sources
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

# Make sibling _norm.py importable
sys.path.insert(0, str(Path(__file__).parent))
from _norm import normalize  # noqa: E402

import polars as pl  # noqa: E402
from tqdm import tqdm  # noqa: E402
from unidecode import unidecode  # noqa: E402

CACHE_ROOT = Path(
    os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
) / "name_cluster"
REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIT_DIR = REPO_ROOT / "audit"


def latest_parquet(source: str, glob: str) -> Path | None:
    pq_dir = CACHE_ROOT / source / "parquet"
    if not pq_dir.exists():
        return None
    files = sorted(pq_dir.glob(glob))
    return files[-1] if files else None


@dataclass(frozen=True)
class SourceSpec:
    """Per-source schema differences for the audit pipeline."""
    glob: str            # parquet filename glob in <cache>/<source>/parquet/
    src_col: str         # raw name column
    translit_col: str    # transliterated name column (added by download_corpora.py)
    entity_col: str      # column to use as entity_id
    has_aliases: bool    # True if the parquet has is_primary/alias_type rows
    entity_type_filter: str | None = None  # filter pl.col("entity_type") == this if set


SOURCE_SPECS: dict[str, SourceSpec] = {
    "ofac": SourceSpec(
        glob="sdn-aliases-*.parquet",
        src_col="name",
        translit_col="name_translit",
        entity_col="entity_id",
        has_aliases=True,
        entity_type_filter="Entity",
    ),
    # Date-prefixed only; excludes derived files like lei2-cn-latin-*.parquet
    "gleif": SourceSpec(
        glob="lei2-2*.parquet",
        src_col="legal_name",
        translit_col="legal_name_translit",
        entity_col="lei",
        has_aliases=False,
    ),
    "ukch": SourceSpec(
        glob="basic-*.parquet",
        src_col="legal_name",
        translit_col="legal_name_translit",
        entity_col="company_number",
        has_aliases=False,
    ),
    "sam": SourceSpec(
        glob="sam-aliases-*.parquet",
        src_col="name",
        translit_col="name_translit",
        entity_col="entity_id",
        has_aliases=True,
    ),
}


def _select_audit_name(df: pl.DataFrame, src: str, translit: str) -> pl.Series:
    """Pick `translit` if column present, else compute unidecode on the fly.

    Audit funnel: non-Latin scripts (CJK, Cyrillic, Greek, ...) get a Latin
    transliteration so the normalize() pipeline produces meaningful tokens
    instead of collapsing them to empty. Pre-computed in download_corpora.py
    when present; recomputed here as fallback for legacy parquet caches.
    """
    if translit in df.columns:
        return df[translit]
    print(f"  no {translit} col found — computing unidecode on the fly")
    return pl.Series(
        translit,
        [unidecode(n) if n is not None else None for n in df[src].to_list()],
    )


def load_names(source: str, sample: int | None) -> pl.DataFrame:
    """Return a df with columns (entity_id, name, role) for the source.

    role:
      - "primary": canonical/legal name
      - "alias": known-same-entity alias (OFAC/SAM only)
    """
    spec = SOURCE_SPECS.get(source)
    if spec is None:
        raise ValueError(f"unknown source: {source}")
    path = latest_parquet(source, spec.glob)
    if path is None:
        raise FileNotFoundError(
            f"{source.upper()} parquet not found; run download_corpora.py {source}"
        )
    raw = pl.read_parquet(path)
    if spec.entity_type_filter is not None:
        raw = raw.filter(pl.col("entity_type") == spec.entity_type_filter)
    audit_name = _select_audit_name(raw, spec.src_col, spec.translit_col)
    role_expr = (
        pl.when(pl.col("is_primary")).then(pl.lit("primary")).otherwise(pl.lit("alias"))
        if spec.has_aliases
        else pl.lit("primary")
    )
    df = raw.with_columns(audit_name=audit_name).select(
        entity_id=pl.col(spec.entity_col),
        name=pl.col("audit_name"),
        role=role_expr,
    )
    df = df.filter(pl.col("name").is_not_null() & (pl.col("name").str.len_chars() > 0))
    if sample and df.height > sample:
        df = df.sample(n=sample, seed=42)
    return df


def normalize_column(names: Iterable[str], desc: str) -> list[str]:
    return [normalize(n) for n in tqdm(names, desc=desc, unit="name", leave=False)]


def audit_source(source: str, sample: int | None) -> dict:
    print(f"\n=== {source.upper()} ===")
    df = load_names(source, sample)
    n = df.height
    print(f"loaded {n:,} names")

    raw_names = df["name"].to_list()
    norm_names = normalize_column(raw_names, desc=f"{source}: normalize")

    df = df.with_columns(normalized=pl.Series(norm_names))

    n_empty = sum(1 for s in norm_names if not s)
    n_one_char = sum(1 for s in norm_names if len(s) == 1)
    print(f"  empty after norm: {n_empty:,}  ({100 * n_empty / n:.2f}%)")
    print(f"  one-char after norm: {n_one_char:,}  ({100 * n_one_char / n:.3f}%)")

    # Substantive changes (any difference after lowercase + collapse)
    n_changed = sum(1 for r, n_ in zip(raw_names, norm_names) if r.casefold().strip() != n_)
    print(f"  changed substantively: {n_changed:,}  ({100 * n_changed / n:.2f}%)")

    # Length distribution
    lens = [len(s) for s in norm_names if s]
    tok_counts = [len(s.split()) for s in norm_names if s]
    len_p = pl.Series(lens).describe()
    tok_p = pl.Series(tok_counts).describe()
    print("  char-length describe:")
    print(len_p)
    print("  token-count describe:")
    print(tok_p)

    # Top normalized strings (collision candidates)
    top_norm = (
        df.filter(pl.col("normalized") != "")
          .group_by("normalized")
          .len()
          .sort("len", descending=True)
          .head(100)
    )
    print(f"  top-5 collisions: {top_norm.head(5).to_dicts()}")

    # Sample alarms
    sample_empty = [r for r, n_ in zip(raw_names, norm_names) if not n_][:20]
    sample_one_char = [(r, n_) for r, n_ in zip(raw_names, norm_names) if len(n_) == 1][:20]
    sample_changed = [
        (r, n_) for r, n_ in zip(raw_names, norm_names)
        if r.casefold().strip() != n_ and n_
    ]
    # Pick 20 evenly-spaced from changed for diversity
    if len(sample_changed) > 20:
        step = max(1, len(sample_changed) // 20)
        sample_changed = sample_changed[::step][:20]

    # Trailing token stats (suffix-list candidates)
    trailing = Counter()
    leading = Counter()
    for r in raw_names:
        toks = r.lower().split()
        if not toks:
            continue
        # strip trailing punct
        last = toks[-1].rstrip(".,;:")
        first = toks[0].lstrip(".,;:")
        if last:
            trailing[last] += 1
        if first:
            leading[first] += 1

    return {
        "source": source,
        "n_input": n,
        "n_normalized_empty": n_empty,
        "n_normalized_one_char": n_one_char,
        "n_changed_substantively": n_changed,
        "char_length_describe": _describe_to_dict(len_p),
        "token_count_describe": _describe_to_dict(tok_p),
        "top_normalized_collisions": top_norm.to_dicts(),
        "sample_empty_after_norm": sample_empty,
        "sample_one_char_after_norm": sample_one_char,
        "sample_substantively_changed": sample_changed,
        "top_trailing_tokens": trailing.most_common(50),
        "top_leading_tokens": leading.most_common(50),
    }


def _describe_to_dict(d: pl.DataFrame) -> dict:
    return {row["statistic"]: row["value"] for row in d.iter_rows(named=True)}


def audit_alias_recall(source: str) -> dict:
    """For each entity with >=2 aliases, measure how many pairs share a
    normalized form. Recall proxy for the normalization step alone.

    Works for any source whose aliases parquet has the OFAC-shape schema
    (entity_id, entity_type, is_primary, alias_type, name).
    """
    spec = SOURCE_SPECS.get(source)
    if spec is None or not spec.has_aliases:
        return {}
    path = latest_parquet(source, spec.glob)
    if path is None:
        print(f"[{source}-recall] skip: no parquet")
        return {}
    print(f"\n=== {source.upper()} alias-recall proxy ===")
    df = pl.read_parquet(path)
    if spec.entity_type_filter is not None:
        df = df.filter(pl.col("entity_type") == spec.entity_type_filter)
    audit_name = _select_audit_name(df, spec.src_col, spec.translit_col)
    df = df.with_columns(
        audit_name=audit_name,
        normalized=pl.Series([normalize(n) for n in audit_name.to_list()]),
    )
    df = df.filter(pl.col("normalized") != "")

    grouped = (
        df.group_by("entity_id")
          .agg(pl.col("normalized"))
          .with_columns(n=pl.col("normalized").list.len())
          .filter(pl.col("n") >= 2)
    )
    print(f"  entities with >=2 normalized aliases: {grouped.height:,}")

    total_pairs = 0
    matching_pairs = 0
    n_full_collapse = 0
    for row in grouped.iter_rows(named=True):
        forms = row["normalized"]
        k = len(forms)
        n_pairs = k * (k - 1) // 2
        total_pairs += n_pairs
        # Pairs that share a normalized string
        c = Counter(forms)
        matched = sum(v * (v - 1) // 2 for v in c.values())
        matching_pairs += matched
        if len(set(forms)) == 1:
            n_full_collapse += 1

    pct = 100 * matching_pairs / total_pairs if total_pairs else 0
    print(f"  pair-recall via exact-normalized-match: {matching_pairs:,} / {total_pairs:,} ({pct:.2f}%)")
    print(f"  entities collapsed to single normalized form: {n_full_collapse:,} / {grouped.height:,}")

    # Sample entities NOT collapsing (where rules failed to merge)
    not_collapsed = (
        grouped.with_columns(n_unique=pl.col("normalized").list.unique().list.len())
               .filter(pl.col("n_unique") > 1)
               .head(10)
    )
    samples = []
    for row in not_collapsed.iter_rows(named=True):
        eid = row["entity_id"]
        raws = df.filter(pl.col("entity_id") == eid).select(
            "name", "audit_name", "normalized"
        ).to_dicts()
        samples.append({"entity_id": eid, "names": raws})

    return {
        "n_entities_with_aliases": int(grouped.height),
        "total_pairs": int(total_pairs),
        "matching_pairs": int(matching_pairs),
        "pair_recall_pct": pct,
        "entities_collapsed_fully": int(n_full_collapse),
        "samples_not_collapsing": samples,
    }


def write_report(per_source: dict[str, dict], recalls: dict[str, dict]) -> Path:
    AUDIT_DIR.mkdir(exist_ok=True)
    today = date.today().isoformat()
    md = AUDIT_DIR / f"report-{today}.md"
    js = AUDIT_DIR / f"report-{today}.json"

    with md.open("w") as f:
        f.write(f"# Normalization audit — {today}\n\n")
        f.write("Provisional rules per `scripts/_norm.py`. Audit per ARCHITECTURE.md task #13.\n\n")

        for src, m in per_source.items():
            f.write(f"## {src}\n\n")
            f.write(f"- input names: **{m['n_input']:,}**\n")
            f.write(f"- empty after norm: {m['n_normalized_empty']:,} "
                    f"({100*m['n_normalized_empty']/m['n_input']:.3f}%)\n")
            f.write(f"- one-char after norm: {m['n_normalized_one_char']:,} "
                    f"({100*m['n_normalized_one_char']/m['n_input']:.4f}%)\n")
            f.write(f"- changed substantively: {m['n_changed_substantively']:,} "
                    f"({100*m['n_changed_substantively']/m['n_input']:.2f}%)\n\n")

            f.write("### char length\n\n")
            f.write("| stat | value |\n|---|---|\n")
            for k, v in m["char_length_describe"].items():
                f.write(f"| {k} | {v} |\n")
            f.write("\n### token count\n\n")
            f.write("| stat | value |\n|---|---|\n")
            for k, v in m["token_count_describe"].items():
                f.write(f"| {k} | {v} |\n")

            f.write("\n### top 20 normalized collisions\n\n")
            f.write("| normalized | count |\n|---|---|\n")
            for row in m["top_normalized_collisions"][:20]:
                f.write(f"| `{row['normalized']}` | {row['len']} |\n")

            f.write("\n### sample names that became EMPTY after norm\n\n")
            for n in m["sample_empty_after_norm"]:
                f.write(f"- `{n}`\n")

            f.write("\n### sample names that became ONE CHAR after norm\n\n")
            for raw, norm in m["sample_one_char_after_norm"]:
                f.write(f"- `{raw}` → `{norm}`\n")

            f.write("\n### sample substantive changes\n\n")
            for raw, norm in m["sample_substantively_changed"]:
                f.write(f"- `{raw}` → `{norm}`\n")

            f.write("\n### top 30 trailing raw tokens (suffix-list inclusion candidates)\n\n")
            f.write("| token | count |\n|---|---|\n")
            for tok, cnt in m["top_trailing_tokens"][:30]:
                f.write(f"| `{tok}` | {cnt} |\n")

            f.write("\n### top 30 leading raw tokens\n\n")
            f.write("| token | count |\n|---|---|\n")
            for tok, cnt in m["top_leading_tokens"][:30]:
                f.write(f"| `{tok}` | {cnt} |\n")

            f.write("\n")

        for src_name, r in recalls.items():
            f.write(f"## {src_name.upper()} alias-recall proxy\n\n")
            f.write(f"- entities with >=2 normalized aliases: **{r['n_entities_with_aliases']:,}**\n")
            f.write(f"- alias pairs sharing normalized form: **{r['matching_pairs']:,} / {r['total_pairs']:,}** "
                    f"({r['pair_recall_pct']:.2f}%)\n")
            f.write(f"- entities fully collapsed to one normalized form: {r['entities_collapsed_fully']:,}\n\n")
            f.write("### samples NOT collapsing (rule blind spots)\n\n")
            for s in r["samples_not_collapsing"]:
                f.write(f"- entity `{s['entity_id']}`:\n")
                for n in s["names"]:
                    if n["name"] != n["audit_name"]:
                        f.write(f"  - `{n['name']}` → translit `{n['audit_name']}` → `{n['normalized']}`\n")
                    else:
                        f.write(f"  - `{n['name']}` → `{n['normalized']}`\n")
            f.write("\n")

    with js.open("w") as f:
        json.dump({"per_source": per_source, "recalls": recalls}, f, indent=2, default=str)

    return md


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["ofac", "gleif", "ukch", "sam", "all"], default="all")
    p.add_argument("--sample", type=int, default=None,
                   help="randomly subsample to this many names per source")
    args = p.parse_args()

    sources = ["ofac", "gleif", "ukch", "sam"] if args.source == "all" else [args.source]
    per_source = {}
    for s in sources:
        try:
            per_source[s] = audit_source(s, args.sample)
        except FileNotFoundError as e:
            print(f"  skip {s}: {e}")

    recalls: dict[str, dict] = {}
    for s in sources:
        if s in {"ofac", "sam"}:
            r = audit_alias_recall(s)
            if r:
                recalls[s] = r
    md = write_report(per_source, recalls)
    print(f"\nwrote {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
