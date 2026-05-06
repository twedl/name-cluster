#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx>=0.27",
#     "polars>=1.20",
#     "tqdm>=4.66",
# ]
# ///
"""Download SAM.gov Entity Registration data — US small-business coverage.

Source: https://api.sam.gov/entity-information/v3/entities
License: US government public domain (17 USC §105)

WHY THIS IS A SAMPLER, NOT A FULL DOWNLOADER
---------------------------------------------
The SAM entity API caps page size at **10 records**. With ~700K active
entities and a default 1000-req/day public rate limit, full coverage via
this API is impractical (~70 days).

For full coverage, use the **SAM Public Extract bulk download** instead:
  1. Log in at https://sam.gov/ (login.gov SSO).
  2. Navigate to Data Bank -> Public Data Files.
  3. Download "SAM Public Monthly" (one-shot ~100MB ZIP of full corpus).
  4. Parse locally (separate script TBD; tracked as task).

This script is the API-path SAMPLER — useful for adding ~10K real US
small-business names to the audit corpus quickly, without manual download.

Auth: free API key required.
  1. Sign up at https://sam.gov/ (login.gov SSO).
  2. Account Details -> Request Public API Key.
  3. export SAM_API_KEY=<your_key>

Filters (server-side):
  registrationStatus=A      — active registrations only
  samRegistered=Yes         — required by API

Pagination: 10 entities/page (API limit). Default cap: 10,000 records.
Wall time: ~2 min at 10 req/sec for 10K records.

Resumable: appends to jsonl.gz as it goes. Cursor file tracks pages done.

Outputs (mirroring scripts/download_corpora.py layout):
  ~/.cache/name_cluster/sam/raw/sam-<date>.jsonl.gz       # one entity/line
  ~/.cache/name_cluster/sam/raw/sam-<date>.cursor.json    # progress
  ~/.cache/name_cluster/sam/parquet/sam-<date>.parquet    # selected cols
  ~/.cache/name_cluster/sam/parquet/sam-aliases-<date>.parquet
                                                          # OFAC-shape aliases

Usage:
  uv run scripts/download_sam.py                  # default 10K-record sample
  uv run scripts/download_sam.py --max-records 50000
  uv run scripts/download_sam.py --max-pages 10   # smoke test (100 records)
  uv run scripts/download_sam.py --force          # re-download from page 0
  uv run scripts/download_sam.py --parse-only     # skip fetch, only re-parse
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import httpx
import polars as pl
from tqdm import tqdm

UA = "name_cluster-corpora/0.1 (+https://github.com/jessetweedle/name-cluster)"

CACHE_ROOT = (
    Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "name_cluster"
)

API_BASE = "https://api.sam.gov/entity-information/v3/entities"
PAGE_SIZE = 10  # server max for entity-information v3
RPS_TARGET = 8.0  # req/sec; conservative inside 10/sec burst limit
DEFAULT_MAX_RECORDS = 10_000  # ~1000 requests; ~2 min wall time at 8 rps


def cache_dirs() -> tuple[Path, Path]:
    raw = CACHE_ROOT / "sam" / "raw"
    pq = CACHE_ROOT / "sam" / "parquet"
    raw.mkdir(parents=True, exist_ok=True)
    pq.mkdir(parents=True, exist_ok=True)
    return raw, pq


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def fetch_all(
    api_key: str,
    raw_path: Path,
    cursor_path: Path,
    *,
    max_pages: int | None = None,
    max_records: int | None = None,
    force: bool = False,
) -> None:
    if force:
        raw_path.unlink(missing_ok=True)
        cursor_path.unlink(missing_ok=True)

    if cursor_path.exists():
        cursor = json.loads(cursor_path.read_text())
        next_page = cursor["next_page"]
        total = cursor.get("total_records")
        print(f"[sam] resuming from page {next_page} (cursor present)")
    else:
        next_page = 0
        total = None
        print("[sam] starting fresh fetch")

    # SAM.gov accepts api_key as query param or header; send both for safety.
    headers = {"User-Agent": UA, "X-Api-Key": api_key}
    params_base = {
        "api_key": api_key,
        "registrationStatus": "A",
        "samRegistered": "Yes",  # required filter on some endpoints
        "size": str(PAGE_SIZE),
    }

    # Open jsonl.gz append-binary so we can resume
    out = gzip.open(raw_path, "ab")
    last_req_t = 0.0
    interval = 1.0 / RPS_TARGET

    pbar: tqdm | None = None
    pages_done_this_run = 0
    try:
        with httpx.Client(headers=headers, timeout=60.0) as client:
            while True:
                if max_pages is not None and pages_done_this_run >= max_pages:
                    print(f"[sam] hit --max-pages {max_pages}, stopping")
                    break

                # rate-limit
                now = time.monotonic()
                wait = (last_req_t + interval) - now
                if wait > 0:
                    time.sleep(wait)
                last_req_t = time.monotonic()

                params = {**params_base, "page": str(next_page)}
                try:
                    r = client.get(API_BASE, params=params)
                except httpx.HTTPError as e:
                    print(
                        f"[sam] network error on page {next_page}: {e}; retrying in 30s"
                    )
                    time.sleep(30)
                    continue

                if r.status_code == 429:
                    retry_after = int(r.headers.get("Retry-After", "60"))
                    print(f"[sam] 429 rate-limited, sleeping {retry_after}s")
                    time.sleep(retry_after)
                    continue
                if r.status_code in (500, 502, 503, 504):
                    print(
                        f"[sam] server {r.status_code} on page {next_page}, backoff 30s"
                    )
                    time.sleep(30)
                    continue
                if r.status_code in (401, 403):
                    raise SystemExit(
                        f"[sam] auth failed ({r.status_code}). Check SAM_API_KEY. Body: {r.text[:500]}"
                    )
                if r.status_code >= 400:
                    raise SystemExit(
                        f"[sam] HTTP {r.status_code} on page {next_page}\n"
                        f"  URL:  {r.url}\n"
                        f"  Body: {r.text[:1000]}"
                    )

                payload = r.json()
                entities = payload.get("entityData", [])
                if total is None:
                    total = int(payload.get("totalRecords", 0))
                    pbar = tqdm(
                        total=total,
                        desc="sam entities",
                        unit="ent",
                        initial=next_page * PAGE_SIZE,
                    )
                    print(f"[sam] total active entities: {total:,}")

                if not entities:
                    print(f"[sam] empty page {next_page}; assuming end of stream")
                    break

                for e in entities:
                    out.write((json.dumps(e, separators=(",", ":")) + "\n").encode())
                out.flush()

                next_page += 1
                pages_done_this_run += 1
                if pbar:
                    pbar.update(len(entities))
                cursor_path.write_text(
                    json.dumps(
                        {
                            "next_page": next_page,
                            "total_records": total,
                        }
                    )
                )

                if next_page * PAGE_SIZE >= total:
                    print(f"[sam] fetched all {total:,} entities")
                    break
                if max_records is not None and next_page * PAGE_SIZE >= max_records:
                    print(f"[sam] hit --max-records {max_records:,}, stopping")
                    break
    finally:
        out.close()
        if pbar:
            pbar.close()


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------


def _get(d: dict, *path: str, default=None):
    for k in path:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
        if d is None:
            return default
    return d


def parse(raw_path: Path, parquet_path: Path, aliases_path: Path) -> None:
    print(f"[sam] parsing {raw_path.name}")
    rows: list[dict] = []
    aliases: list[dict] = []

    with gzip.open(raw_path, "rt") as f:
        for line in tqdm(f, desc="parse", unit="ent"):
            e = json.loads(line)
            er = e.get("entityRegistration", {}) or {}
            cd = e.get("coreData", {}) or {}
            assertions = e.get("assertions", {}) or {}
            gs = assertions.get("goodsAndServices", {}) or {}

            uei = er.get("ueiSAM") or er.get("uei")
            legal = (er.get("legalBusinessName") or "").strip()
            dba = (er.get("dbaName") or "").strip()
            phys = cd.get("physicalAddress", {}) or {}

            naics_list = gs.get("naicsList") or []
            naics_codes = [
                (n.get("naicsCode") or "") for n in naics_list if isinstance(n, dict)
            ]
            primary_naics = next(
                (
                    n.get("naicsCode")
                    for n in naics_list
                    if isinstance(n, dict) and n.get("isPrimary") in ("Y", True, "true")
                ),
                naics_codes[0] if naics_codes else None,
            )

            rows.append(
                {
                    "uei": uei,
                    "legal_name": legal,
                    "dba_name": dba or None,
                    "state": phys.get("stateOrProvinceCode"),
                    "country": phys.get("countryCode"),
                    "primary_naics": primary_naics,
                    "naics_codes": naics_codes,
                    "entity_structure": _get(
                        cd, "entityInformation", "entityStructureCode"
                    ),
                    "registration_date": er.get("initialRegistrationDate"),
                    "entity_status": er.get("registrationStatus"),
                }
            )

            if uei and legal:
                aliases.append(
                    {
                        "entity_id": uei,
                        "entity_type": "Entity",
                        "is_primary": True,
                        "alias_type": "PRIMARY",
                        "name": legal,
                    }
                )
                if dba and dba != legal:
                    aliases.append(
                        {
                            "entity_id": uei,
                            "entity_type": "Entity",
                            "is_primary": False,
                            "alias_type": "DBA",
                            "name": dba,
                        }
                    )

    df = pl.DataFrame(rows)
    df.write_parquet(parquet_path, compression="zstd", compression_level=3)
    aliases_df = pl.DataFrame(aliases)
    aliases_df.write_parquet(aliases_path, compression="zstd", compression_level=3)

    n_with_dba = sum(1 for r in rows if r["dba_name"])
    print(
        f"[sam] wrote {parquet_path.name} ({df.height:,} rows; {n_with_dba:,} have DBA), "
        f"{aliases_path.name} ({aliases_df.height:,} alias rows)"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--force", action="store_true", help="re-download from page 0")
    p.add_argument(
        "--parse-only", action="store_true", help="skip fetch, only re-parse cached raw"
    )
    p.add_argument(
        "--max-pages", type=int, default=None, help="cap pages fetched (smoke test)"
    )
    p.add_argument(
        "--max-records",
        type=int,
        default=DEFAULT_MAX_RECORDS,
        help=f"cap total records (default {DEFAULT_MAX_RECORDS:,}; 0 = unlimited)",
    )
    args = p.parse_args()

    raw_dir, pq_dir = cache_dirs()
    today = date.today().isoformat()
    raw_path = raw_dir / f"sam-{today}.jsonl.gz"
    cursor_path = raw_dir / f"sam-{today}.cursor.json"
    parquet_path = pq_dir / f"sam-{today}.parquet"
    aliases_path = pq_dir / f"sam-aliases-{today}.parquet"

    if not args.parse_only:
        api_key = os.environ.get("SAM_API_KEY")
        if not api_key:
            raise SystemExit(
                "SAM_API_KEY env var not set. Get a key at https://sam.gov/ "
                "(Account Details -> Request Public API Key)."
            )
        max_records = args.max_records if args.max_records > 0 else None
        fetch_all(
            api_key,
            raw_path,
            cursor_path,
            max_pages=args.max_pages,
            max_records=max_records,
            force=args.force,
        )

    if not raw_path.exists():
        raise SystemExit(f"no raw data at {raw_path}; run without --parse-only")

    parse(raw_path, parquet_path, aliases_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
