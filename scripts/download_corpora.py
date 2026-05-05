#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx>=0.27",
#     "polars>=1.20",
#     "lxml>=5",
#     "tqdm>=4.66",
#     "unidecode>=1.3",
# ]
# ///
"""Download real-name corpora for normalization rule validation.

Sources (all publicly redistributable):
  - GLEIF Golden Copy LEI-CDF       CC BY 4.0
  - UK Companies House Basic Data   OGL v3.0
  - OFAC SDN Enhanced XML           US public domain (17 USC 105)

Cache layout:
  ~/.cache/name_cluster/<source>/raw/<dated-original-file>
  ~/.cache/name_cluster/<source>/parquet/<dated>.parquet
  ~/.cache/name_cluster/<source>/parquet/<dated>-aliases.parquet   (when applicable)

Idempotent: skips re-download/re-parse if outputs exist.
Override with --force.

Usage:
  uv run scripts/download_corpora.py ofac
  uv run scripts/download_corpora.py gleif
  uv run scripts/download_corpora.py ukch
  uv run scripts/download_corpora.py all
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import httpx
import polars as pl
from lxml import etree
from tqdm import tqdm
from unidecode import unidecode

UA = "name_cluster-corpora/0.1 (+https://github.com/jessetweedle/name-cluster)"

CACHE_ROOT = Path(
    os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
) / "name_cluster"


# ---------------------------------------------------------------------------
# generic helpers
# ---------------------------------------------------------------------------

def stream_download(url: str, dest: Path, *, follow_redirects: bool = True) -> None:
    """Download `url` to `dest` with progress bar. Atomic via .tmp rename."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    headers = {"User-Agent": UA}
    with httpx.stream(
        "GET", url, headers=headers, follow_redirects=follow_redirects, timeout=120.0
    ) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with tmp.open("wb") as f, tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            desc=dest.name,
            leave=False,
        ) as bar:
            for chunk in r.iter_bytes(chunk_size=1024 * 256):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)


def cache_dir(source: str) -> tuple[Path, Path]:
    raw = CACHE_ROOT / source / "raw"
    parquet = CACHE_ROOT / source / "parquet"
    raw.mkdir(parents=True, exist_ok=True)
    parquet.mkdir(parents=True, exist_ok=True)
    return raw, parquet


def add_translit(df: pl.DataFrame, src_col: str, dst_col: str) -> pl.DataFrame:
    """Add a Latin-script transliteration column via unidecode.

    For ASCII input this is essentially identity (diacritics stripped); for
    non-Latin scripts (CJK, Cyrillic, Greek, Arabic, ...) this produces a
    best-effort Latin form that survives the normalize() pipeline. Audit-
    only — not consumed by the wheel.
    """
    return df.with_columns(
        pl.col(src_col)
        .map_elements(
            lambda x: unidecode(x) if x is not None else None,
            return_dtype=pl.Utf8,
        )
        .alias(dst_col)
    )


def parquet_has_col(path: Path, col: str) -> bool:
    """Return True if `col` is present in the parquet schema (no full read)."""
    try:
        return col in pl.scan_parquet(path).collect_schema().names()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# OFAC SDN
# ---------------------------------------------------------------------------

OFAC_URL = (
    "https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/SDN_ENHANCED.XML"
)
# PublicationPreview Enhanced XML default namespace
OFAC_NS_URI = "https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/ENHANCED_XML"
OFAC_NS = {"x": OFAC_NS_URI}


def download_ofac(*, force: bool = False) -> None:
    raw, pq = cache_dir("ofac")
    today = date.today().isoformat()
    raw_path = raw / f"sdn_enhanced-{today}.xml"
    parquet_path = pq / f"sdn-{today}.parquet"
    aliases_path = pq / f"sdn-aliases-{today}.parquet"

    if not raw_path.exists() or force:
        print(f"[ofac] downloading {OFAC_URL}")
        stream_download(OFAC_URL, raw_path)
    else:
        print(f"[ofac] cached {raw_path.name}")

    if (
        parquet_path.exists()
        and aliases_path.exists()
        and parquet_has_col(aliases_path, "name_translit")
        and not force
    ):
        print(f"[ofac] parsed already: {parquet_path.name}, {aliases_path.name}")
        return

    print(f"[ofac] parsing {raw_path.name}")
    parties: list[dict] = []
    aliases: list[dict] = []

    # Stream-parse <entity> elements. Each entity has a generalInfo (entityType:
    # Entity / Individual / Vessel / Aircraft) and 1+ <name> children, each with
    # translations -> formattedFullName.
    context = etree.iterparse(
        str(raw_path), events=("end",), tag=f"{{{OFAC_NS_URI}}}entity"
    )
    for _, elem in context:
        entity_id = elem.get("id")
        entity_type = elem.findtext("x:generalInfo/x:entityType", "", OFAC_NS).strip()

        primary_name = None
        for name in elem.findall("x:names/x:name", OFAC_NS):
            is_primary = name.findtext("x:isPrimary", "false", OFAC_NS) == "true"
            alias_type = name.findtext("x:aliasType", "", OFAC_NS).strip()  # blank for primary
            # Take the Latin-script translation; fall back to first available
            full = ""
            for trans in name.findall("x:translations/x:translation", OFAC_NS):
                script = trans.findtext("x:script", "", OFAC_NS).strip()
                ffn = trans.findtext("x:formattedFullName", "", OFAC_NS).strip()
                if ffn and (script == "Latin" or not full):
                    full = ffn
                if full and script == "Latin":
                    break
            if not full:
                continue
            if is_primary and primary_name is None:
                primary_name = full
            aliases.append({
                "entity_id": entity_id,
                "entity_type": entity_type,
                "is_primary": is_primary,
                "alias_type": alias_type or ("PRIMARY" if is_primary else ""),
                "name": full,
            })
        if primary_name:
            parties.append({
                "entity_id": entity_id,
                "entity_type": entity_type,
                "primary_name": primary_name,
            })
        elem.clear()
        # Also clear preceding siblings to keep memory bounded during iterparse
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    parties_df = add_translit(pl.DataFrame(parties), "primary_name", "primary_name_translit")
    aliases_df = add_translit(pl.DataFrame(aliases), "name", "name_translit")
    parties_df.write_parquet(parquet_path)
    aliases_df.write_parquet(aliases_path)
    n_entity = sum(1 for p in parties if p["entity_type"] == "Entity")
    n_alias_entity = sum(1 for a in aliases if a["entity_type"] == "Entity")
    print(
        f"[ofac] wrote {parquet_path.name} ({len(parties)} total / {n_entity} entities), "
        f"{aliases_path.name} ({len(aliases)} alias rows / {n_alias_entity} entity-alias rows)"
    )


# ---------------------------------------------------------------------------
# GLEIF Golden Copy LEI-CDF
# ---------------------------------------------------------------------------

GLEIF_API = "https://goldencopy.gleif.org/api/v2/golden-copies/publishes/latest"


def download_gleif(*, force: bool = False) -> None:
    raw, pq = cache_dir("gleif")

    print(f"[gleif] querying {GLEIF_API}")
    with httpx.Client(headers={"User-Agent": UA}, timeout=60.0) as client:
        meta = client.get(GLEIF_API).raise_for_status().json()
    csv_meta = meta["data"]["lei2"]["full_file"]["csv"]
    publish_date = meta["data"]["publish_date"][:10]  # YYYY-MM-DD
    csv_url = csv_meta["url"]
    expected_size = csv_meta["size"]
    print(
        f"[gleif] publish={publish_date} records={csv_meta['record_count']} "
        f"size={csv_meta['size_human_readable']}"
    )

    raw_path = raw / f"lei2-{publish_date}.csv.zip"
    parquet_path = pq / f"lei2-{publish_date}.parquet"

    if not raw_path.exists() or raw_path.stat().st_size != expected_size or force:
        print(f"[gleif] downloading {csv_url}")
        stream_download(csv_url, raw_path)
    else:
        print(f"[gleif] cached {raw_path.name}")

    if (
        parquet_path.exists()
        and parquet_has_col(parquet_path, "legal_name_translit")
        and not force
    ):
        print(f"[gleif] parsed already: {parquet_path.name}")
        return

    print(f"[gleif] parsing {raw_path.name}")
    # LEI-CDF CSV is one big CSV inside the ZIP; pick relevant cols only.
    # Field names follow the LEI-CDF v3.1 spec.
    keep_cols = [
        "LEI",
        "Entity.LegalName",
        "Entity.LegalAddress.Country",
        "Entity.LegalJurisdiction",
        "Entity.EntityCategory",
        "Entity.EntityStatus",
    ]
    with zipfile.ZipFile(raw_path) as zf:
        # Single CSV inside
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not names:
            raise RuntimeError(f"no CSV inside {raw_path}")
        with zf.open(names[0]) as fh:
            df = pl.read_csv(
                fh.read(),
                infer_schema_length=0,  # all utf8 — LEI-CDF cells can be quoted weirdly
                ignore_errors=False,
            )
    have = [c for c in keep_cols if c in df.columns]
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        print(f"[gleif] WARNING: missing cols (CDF schema drift?): {missing}")
    df = df.select(have).rename({
        "LEI": "lei",
        "Entity.LegalName": "legal_name",
        "Entity.LegalAddress.Country": "country",
        "Entity.LegalJurisdiction": "jurisdiction",
        "Entity.EntityCategory": "category",
        "Entity.EntityStatus": "status",
    })
    print(f"[gleif] transliterating {df.height:,} legal names")
    df = add_translit(df, "legal_name", "legal_name_translit")
    df.write_parquet(parquet_path, compression="zstd", compression_level=3)
    print(f"[gleif] wrote {parquet_path.name} ({df.height} rows, {parquet_path.stat().st_size//1_000_000} MB)")


# ---------------------------------------------------------------------------
# UK Companies House Basic Data
# ---------------------------------------------------------------------------

def _ukch_snapshot_url(today: date | None = None) -> tuple[str, str]:
    """Return (url, snapshot_date_iso) for the most recent monthly snapshot.

    UKCH publishes within 5 working days of the previous month-end. Try the
    1st of the current month; fall back to the 1st of the previous month
    if 404.
    """
    today = today or date.today()
    candidates = []
    first_this = today.replace(day=1)
    candidates.append(first_this.isoformat())
    prev_month_end = first_this - timedelta(days=1)
    first_prev = prev_month_end.replace(day=1)
    candidates.append(first_prev.isoformat())
    return candidates  # type: ignore[return-value]


def download_ukch(*, force: bool = False) -> None:
    raw, pq = cache_dir("ukch")

    candidates = _ukch_snapshot_url()
    snapshot_date = None
    url = None
    with httpx.Client(headers={"User-Agent": UA}, timeout=60.0) as client:
        for cand in candidates:
            test_url = (
                f"https://download.companieshouse.gov.uk/"
                f"BasicCompanyDataAsOneFile-{cand}.zip"
            )
            r = client.head(test_url)
            if r.status_code == 200:
                url = test_url
                snapshot_date = cand
                break
    if url is None:
        raise RuntimeError(f"no UKCH snapshot found in candidates {candidates}")

    raw_path = raw / f"basic-{snapshot_date}.csv.zip"
    parquet_path = pq / f"basic-{snapshot_date}.parquet"

    if not raw_path.exists() or force:
        print(f"[ukch] downloading {url}")
        stream_download(url, raw_path)
    else:
        print(f"[ukch] cached {raw_path.name}")

    if (
        parquet_path.exists()
        and parquet_has_col(parquet_path, "legal_name_translit")
        and not force
    ):
        print(f"[ukch] parsed already: {parquet_path.name}")
        return

    print(f"[ukch] parsing {raw_path.name}")
    # UKCH headers have inconsistent leading spaces. Match by stripped name.
    want = {
        "CompanyName": "legal_name",
        "CompanyNumber": "company_number",
        "RegAddress.Country": "country",
        "CompanyCategory": "category",
        "CompanyStatus": "status",
    }
    with zipfile.ZipFile(raw_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not names:
            raise RuntimeError(f"no CSV inside {raw_path}")
        with zf.open(names[0]) as fh:
            df = pl.read_csv(fh.read(), infer_schema_length=0)
    # Map raw -> renamed by matching on stripped header
    by_stripped = {c.strip(): c for c in df.columns}
    pairs = [(by_stripped[w], r) for w, r in want.items() if w in by_stripped]
    missing = [w for w in want if w not in by_stripped]
    if missing:
        print(f"[ukch] WARNING: missing cols (UKCH schema drift?): {missing}")
    df = df.select([raw for raw, _ in pairs]).rename(dict(pairs))
    print(f"[ukch] transliterating {df.height:,} legal names")
    df = add_translit(df, "legal_name", "legal_name_translit")
    df.write_parquet(parquet_path, compression="zstd", compression_level=3)
    print(f"[ukch] wrote {parquet_path.name} ({df.height} rows, {parquet_path.stat().st_size//1_000_000} MB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DISPATCH = {
    "ofac": download_ofac,
    "gleif": download_gleif,
    "ukch": download_ukch,
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", choices=[*DISPATCH.keys(), "all"])
    p.add_argument("--force", action="store_true", help="re-download + re-parse")
    args = p.parse_args()

    print(f"cache root: {CACHE_ROOT}")
    sources = list(DISPATCH.keys()) if args.source == "all" else [args.source]
    for s in sources:
        DISPATCH[s](force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
