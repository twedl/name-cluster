#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "polars>=1.20",
#     "tqdm>=4.66",
# ]
# ///
"""Parse the SAM Public Monthly bulk extract.

The SAM v3 entity API is rate-limited (10 records/page, 1000/day default),
so full coverage (~880K active entities) requires the bulk extract route:

  1. Log into https://sam.gov/ via login.gov.
  2. Navigate to Data Bank -> Public Data Files -> Entity Management Public V2.
  3. Download SAM_PUBLIC_UTF-8_MONTHLY_V2_<YYYYMMDD>.ZIP (or pre-extracted .dat).
  4. Place in ~/.cache/name_cluster/sam/raw/

This script auto-discovers the .zip or .dat file in the cache dir and writes
parquet output matching the schema produced by scripts/download_sam.py
(API-path sampler) — so validate_normalization.py picks it up transparently
via --source sam.

File format
-----------
- UTF-8 pipe-delimited, no header row.
- Line 1: BOF marker (e.g. `BOF PUBLIC V2 00000000 20260503 0884203 0008237`).
- Line N: data records, 142 fields each.
- Last line: `!end` marker (col 142 of last record actually contains "!end").

Column positions used (by inspection of records; full layout in the SAM
data dictionary docx):
   1: UEI                                12: Legal Business Name
   6: Registration Status (A = Active)   13: DBA Name
   8: Initial Registration Date          19: Physical Address State
   22: Physical Address Country          28: Entity Structure Code
   33: Primary NAICS                     35: All NAICS (tilde-delimited)

Outputs
-------
~/.cache/name_cluster/sam/parquet/sam-<date>.parquet
~/.cache/name_cluster/sam/parquet/sam-aliases-<date>.parquet  (OFAC-shape)

Usage
-----
  uv run scripts/parse_sam_bulk.py
  uv run scripts/parse_sam_bulk.py --input ~/Downloads/SAM_...20260503.dat
  uv run scripts/parse_sam_bulk.py --force
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile
from datetime import date
from pathlib import Path

import polars as pl
from tqdm import tqdm

CACHE_ROOT = Path(
    os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
) / "name_cluster"

# Column positions are 1-indexed in the data dictionary; polars new_columns
# is 0-indexed names. We give names to ALL 142 cols (mostly placeholders) and
# select what we want.
N_COLS = 142
COL_NAMES = [f"f{i:03d}" for i in range(1, N_COLS + 1)]

# Mapping from logical names -> column positions (1-indexed)
FIELD_MAP = {
    "uei": 1,
    "cage_code": 4,
    "status": 6,
    "purpose_of_registration": 7,
    "initial_registration_date": 8,
    "expiration_date": 9,
    "legal_name": 12,
    "dba_name": 13,
    "physical_addr_street1": 16,
    "physical_addr_city": 18,
    "physical_addr_state": 19,
    "physical_addr_zip": 20,
    "physical_addr_country": 22,
    "entity_structure_code": 28,
    "entity_country_of_incorp": 30,
    "primary_naics": 33,
    "all_naics": 35,
    "psc_codes": 37,
}


def cache_dirs() -> tuple[Path, Path]:
    raw = CACHE_ROOT / "sam" / "raw"
    pq = CACHE_ROOT / "sam" / "parquet"
    raw.mkdir(parents=True, exist_ok=True)
    pq.mkdir(parents=True, exist_ok=True)
    return raw, pq


def find_input(raw_dir: Path, override: Path | None) -> tuple[Path, str]:
    """Return (path, snapshot_date_iso) for the bulk extract to parse.

    Override wins if given. Otherwise: prefer .dat (already extracted),
    else .zip in raw/.
    """
    if override:
        return override, _date_from_filename(override)
    candidates = sorted(raw_dir.glob("SAM_PUBLIC_*MONTHLY_V2_*.dat"))
    candidates += sorted(raw_dir.glob("SAM_PUBLIC_*MONTHLY_V2_*.zip"))
    if not candidates:
        raise SystemExit(
            f"No SAM monthly extract in {raw_dir}.\n"
            f"  Download SAM_PUBLIC_UTF-8_MONTHLY_V2_*.zip from sam.gov "
            f"Data Bank and place in {raw_dir}."
        )
    p = candidates[0]
    return p, _date_from_filename(p)


def _date_from_filename(p: Path) -> str:
    # SAM_PUBLIC_UTF-8_MONTHLY_V2_20260503.{zip,dat}
    name = p.stem
    digits = "".join(c for c in name.split("_")[-1] if c.isdigit())
    if len(digits) == 8:
        return f"{digits[0:4]}-{digits[4:6]}-{digits[6:8]}"
    return date.today().isoformat()


def read_dat(input_path: Path) -> pl.DataFrame:
    """Read the pipe-delimited .dat (or .dat inside .zip), skip BOF row."""
    print(f"[sam-bulk] reading {input_path.name}")
    if input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path) as zf:
            dats = [n for n in zf.namelist() if n.endswith(".dat") or n.endswith(".DAT")]
            if not dats:
                raise SystemExit(f"no .dat inside {input_path}")
            with zf.open(dats[0]) as fh:
                raw = fh.read()
    else:
        raw = input_path.read_bytes()

    # Drop BOF marker line. (EOF marker `!end` lives in the LAST record's last
    # field, so the row count is correct without filtering at the row level.)
    nl = raw.find(b"\n")
    if nl < 0:
        raise SystemExit("no newline in file")
    raw_no_bof = raw[nl + 1:]

    df = pl.read_csv(
        raw_no_bof,
        separator="|",
        has_header=False,
        new_columns=COL_NAMES,
        infer_schema_length=0,            # all utf8
        truncate_ragged_lines=True,
        ignore_errors=False,
        quote_char=None,                  # SAM doesn't quote pipe-separated fields
    )
    print(f"[sam-bulk] loaded {df.height:,} rows x {df.width} cols")
    return df


def to_entities_parquet(df: pl.DataFrame, out_path: Path) -> pl.DataFrame:
    cols = [pl.col(f"f{FIELD_MAP[name]:03d}").alias(name) for name in FIELD_MAP]
    out = df.select(cols)
    out = out.filter(pl.col("status") == "A")
    out.write_parquet(out_path, compression="zstd", compression_level=3)
    print(f"[sam-bulk] wrote {out_path.name} ({out.height:,} active entities, "
          f"{out_path.stat().st_size // 1_000_000} MB)")
    return out


def to_aliases_parquet(entities: pl.DataFrame, out_path: Path) -> None:
    """OFAC-shape aliases parquet: (entity_id, entity_type, is_primary,
    alias_type, name). One PRIMARY row per entity + DBA row when distinct."""
    # PRIMARY rows
    primary = (
        entities.filter(
            pl.col("legal_name").is_not_null() & (pl.col("legal_name").str.len_chars() > 0)
        )
        .select(
            entity_id=pl.col("uei"),
            entity_type=pl.lit("Entity"),
            is_primary=pl.lit(True),
            alias_type=pl.lit("PRIMARY"),
            name=pl.col("legal_name"),
        )
    )
    # DBA rows (only when DBA is non-null and differs from legal_name)
    dba = (
        entities.filter(
            pl.col("dba_name").is_not_null()
            & (pl.col("dba_name").str.len_chars() > 0)
            & (pl.col("dba_name") != pl.col("legal_name"))
        )
        .select(
            entity_id=pl.col("uei"),
            entity_type=pl.lit("Entity"),
            is_primary=pl.lit(False),
            alias_type=pl.lit("DBA"),
            name=pl.col("dba_name"),
        )
    )
    aliases = pl.concat([primary, dba])
    aliases.write_parquet(out_path, compression="zstd", compression_level=3)
    n_primary = primary.height
    n_dba = dba.height
    print(f"[sam-bulk] wrote {out_path.name} ({aliases.height:,} alias rows: "
          f"{n_primary:,} primary + {n_dba:,} DBA-distinct)")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", type=Path, default=None,
                   help="path to SAM_PUBLIC_..._MONTHLY_V2_YYYYMMDD.{zip,dat}")
    p.add_argument("--force", action="store_true", help="re-parse even if outputs exist")
    args = p.parse_args()

    raw_dir, pq_dir = cache_dirs()
    input_path, snapshot_date = find_input(raw_dir, args.input)
    print(f"[sam-bulk] input: {input_path}")
    print(f"[sam-bulk] snapshot date: {snapshot_date}")

    parquet_path = pq_dir / f"sam-{snapshot_date}.parquet"
    aliases_path = pq_dir / f"sam-aliases-{snapshot_date}.parquet"

    if parquet_path.exists() and aliases_path.exists() and not args.force:
        print(f"[sam-bulk] outputs exist; pass --force to re-parse")
        return 0

    df = read_dat(input_path)
    entities = to_entities_parquet(df, parquet_path)
    to_aliases_parquet(entities, aliases_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
