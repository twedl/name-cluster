"""Provisional Python normalization — mirror of the eventual rust impl.

Used by scripts/validate_normalization.py to audit rules against real corpora
before we encode them in rust. Throwaway-quality; not shipped in the wheel.

Pipeline (per ARCHITECTURE.md §Normalization):

  1. NFKD + strip combining marks
  2. Lowercase
  3. Punct collapse: '&' -> ' and ', others -> space, drop non-alphanum
  4. Whitespace collapse
  5. Strip leading ^0+\\s+ and ^[#*]+\\s+
  6. Strip leading 'THE' (only if >=3 tokens remain after)
  7. Iterative end-strip List 1 legal-form suffixes (only if >=1 token remains)
  8. Token-by-token canonicalize via List 2 map
  9. Whitespace collapse
"""
from __future__ import annotations

import re
import unicodedata

# --- List 1: legal-form suffixes (always end-strip) ---
# Tokens compared case-insensitively, post-punct. Multi-word entries match as
# adjacent tokens at the end of the name.
LIST1_SUFFIXES_SINGLE: frozenset[str] = frozenset({
    # English / Western
    "inc", "llc", "ltd", "limited", "corp", "corporation", "co", "company",
    "companies", "lp", "llp", "plc", "pty",
    # German (DE/AT/CH)
    "gmbh", "ag", "kg", "kgaa", "ohg", "gbr", "ev", "eg", "ggmbh", "mbh",
    "aktiengesellschaft", "kommanditgesellschaft", "gesellschaft",
    # French / Belgian / Swiss-French
    "sa", "sas", "sarl", "sprl", "sagl",
    # Italian
    "srl", "spa", "snc", "ss",
    # Spanish / Portuguese
    "sl", "slu", "slp", "cb", "lda", "ltda",
    # Dutch / Belgian-Dutch
    "bv", "nv", "bvba", "vof", "ua",
    # Nordic
    "oy", "ab", "as", "asa", "oyj", "aps",
    # East Asian
    "kk", "gk", "tmk", "pte",
    # Russian / Slavic (head-strip primary use case post-Fix 2)
    "ooo", "oao", "ojsc", "pjsc", "cjsc", "jsc", "zao", "pao", "ao",
    # Polish (post-List-3 compound canonicalization)
    "spzoo", "psa", "ska", "spk", "spj",
    # Middle East
    "fze",
    # Turkish
    "sirketi",
    # Common initialisms preserved through period-drop
    "cv",
})
LIST1_SUFFIXES_MULTI: tuple[tuple[str, ...], ...] = (
    ("co", "ltd"),  # 'CO LTD' is one logical suffix in Chinese-style names
)

# --- List 3: compound legal-form canonicalize (anywhere in name) ---
# Multi-token legal-form phrases mapped to their standard short form. Applied
# BEFORE List 1 strip, so the resulting canonical token (e.g. "llc") is then
# end/head-stripped by List 1 in the normal way.
# Anchored anywhere in the name (handles head, tail, and rare mid-name).
# Matched longest-first to avoid partial collisions.
LIST3_COMPOUND_LEGAL: dict[tuple[str, ...], str] = {
    # English
    ("limited", "liability", "company"): "llc",
    ("limited", "liability"): "llc",
    ("limited", "liability", "partnership"): "llp",
    ("limited", "partnership"): "lp",
    ("general", "partnership"): "gp",
    ("joint", "stock", "company"): "jsc",
    ("public", "joint", "stock", "company"): "pjsc",
    ("open", "joint", "stock", "company"): "ojsc",
    ("closed", "joint", "stock", "company"): "cjsc",
    # Russian (transliterated)
    ("obshchestvo", "s", "ogranichennoi", "otvetstvennostyu"): "ooo",
    ("aktsionernoe", "obshchestvo"): "ao",
    # Polish (post-atomic-Latin map: ł->l, accents stripped via NFKD)
    # spółka z ograniczoną odpowiedzialnością = LLC = "sp. z o.o."
    ("spolka", "z", "ograniczona", "odpowiedzialnoscia"): "spzoo",
    # Abbreviated forms after period-drop. Source variants:
    #   "Sp. z o.o."     -> "sp z oo"   (3 tokens — periods drop adjacent o's)
    #   "Sp Z O O"       -> "sp z o o"  (4 tokens — letters were already spaced)
    ("sp", "z", "oo"): "spzoo",
    ("sp", "z", "o", "o"): "spzoo",
    ("spolka", "akcyjna"): "sa",
    ("prosta", "spolka", "akcyjna"): "psa",
    # Limited stock partnership: source can be "komandytowo-akcyjna" (hyphen
    # drop concatenates) or "komandytowo akcyjna" (already spaced).
    ("spolka", "komandytowoakcyjna"): "ska",
    ("spolka", "komandytowo", "akcyjna"): "ska",
    ("spolka", "komandytowa"): "spk",
    ("spolka", "jawna"): "spj",
    ("spolka", "partnerska"): "spp",
    ("spolka", "cywilna"): "sc",
    ("spolka", "europejska"): "se",
    # Spanish / French / Italian
    ("sociedad", "anonima"): "sa",
    ("sociedad", "limitada"): "sl",
    ("societe", "anonyme"): "sa",
    # Japanese (transliterated)
    ("kabushiki", "kaisha"): "kk",
}

# --- List 2: descriptor-noise canonicalization (always-on, both forms -> short) ---
LIST2_CANONICAL: dict[str, str] = {
    "manufacturing": "mfg",
    "manufact": "mfg",
    "mfr": "mfg",
    "international": "intl",
    "import": "imp",
    "imports": "imp",
    "export": "exp",
    "exports": "exp",
    "holdings": "hldg",
    "holding": "hldg",
    "group": "grp",
    "enterprises": "ent",
    "enterprise": "ent",
    "industries": "ind",
    "industry": "ind",
    "services": "svc",
    "svcs": "svc",
    "solutions": "sln",
    "trading": "trd",
    "trade": "trd",
    "technology": "tech",
    "technologies": "tech",
    "techs": "tech",
    "development": "dev",
    "associates": "assoc",
}

# Compound canonicalizations (multi-token in -> single-token out). Applied
# before the single-token map to avoid partial matches.
LIST2_COMPOUND: dict[tuple[str, ...], str] = {
    ("import", "export"): "impexp",
    ("imp", "exp"): "impexp",
}

# Atomic Latin-extension letters that NFKD doesn't decompose.
# Without explicit mapping the punct-collapse step would treat these as non-
# ASCII and replace with space, splitting words mid-token (Polish "SPÓŁKA"
# would become "spo ka"). Map to their conventional ASCII fallbacks.
_ATOMIC_LATIN_MAP = {
    "Ł": "L", "ł": "l",
    "Ø": "O", "ø": "o",
    "Æ": "AE", "æ": "ae",
    "Œ": "OE", "œ": "oe",
    "ß": "ss",
    "Þ": "TH", "þ": "th",
    "Ð": "D", "ð": "d", "Đ": "D", "đ": "d",
    "ı": "i", "İ": "i",
}
_ATOMIC_LATIN_TABLE = str.maketrans(_ATOMIC_LATIN_MAP)

_RE_LEADING_ZERO_GARBAGE = re.compile(r"^0+\s+")
_RE_LEADING_HASH_GARBAGE = re.compile(r"^[#*]+\s+")
_RE_AMP = re.compile(r"&")
# Drop chars: appear/disappear between recordings of same entity.
#   . initialism / abbreviation (S.A., Inc.)
#   ' apostrophe (MCDONALD'S -> MCDONALDS)
#   - hyphen (WAL-MART, COCA-COLA, H-E-B)
#   _ underscore (rare)
#   + (H+M)
_RE_DROP = re.compile(r"[.'\-_+]")
# Everything else not alnum/space -> space (commas, parens, slashes, ...)
_RE_NON_ALNUM_SPACE = re.compile(r"[^a-z0-9\s]+")
_RE_WS = re.compile(r"\s+")


def normalize(name: str) -> str:
    if name is None:
        return ""
    s = name

    # 1. Unicode NFKD + strip combining marks
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    # 1b. Map atomic Latin-extension letters that NFKD doesn't decompose.
    s = s.translate(_ATOMIC_LATIN_TABLE)

    # 2. Lowercase
    s = s.lower()

    # 3. Punct collapse: & -> ' and ', drop set -> nothing, others -> space
    s = _RE_AMP.sub(" and ", s)
    s = _RE_DROP.sub("", s)
    s = _RE_NON_ALNUM_SPACE.sub(" ", s)

    # 4. Whitespace collapse
    s = _RE_WS.sub(" ", s).strip()

    # 5. Strip leading garbage
    s = _RE_LEADING_ZERO_GARBAGE.sub("", s)
    s = _RE_LEADING_HASH_GARBAGE.sub("", s)

    if not s:
        return ""

    tokens = s.split()

    # 6. Strip leading THE if >=3 tokens remain after
    if len(tokens) >= 4 and tokens[0] == "the":
        tokens = tokens[1:]

    # 6b. Compound legal-form canonicalize (List 3, anywhere in name).
    # Multi-token phrases collapse to a single canonical token (e.g. "llc"),
    # which the next List 1 pass can then head/tail-strip normally.
    if LIST3_COMPOUND_LEGAL:
        compound_patterns = sorted(
            LIST3_COMPOUND_LEGAL.items(), key=lambda kv: -len(kv[0])
        )
        out: list[str] = []
        i = 0
        while i < len(tokens):
            matched = False
            for pat, canon in compound_patterns:
                plen = len(pat)
                if i + plen <= len(tokens) and tuple(tokens[i:i + plen]) == pat:
                    out.append(canon)
                    i += plen
                    matched = True
                    break
            if not matched:
                out.append(tokens[i])
                i += 1
        tokens = out

    # 7. Iterative bidirectional strip List 1, keeping >=1 token.
    # Order: multi-word tail, single-word tail, single-word head. Loop to fixed point.
    # (Multi-word head -- e.g. "JOINT STOCK COMPANY X" -- handled later by List 3
    # canonicalize once that exists; for now leading single-token strips catch
    # OOO/OAO/JSC/LLC etc.)
    changed = True
    while changed and len(tokens) > 1:
        changed = False
        for multi in LIST1_SUFFIXES_MULTI:
            mlen = len(multi)
            if len(tokens) > mlen and tuple(tokens[-mlen:]) == multi:
                tokens = tokens[:-mlen]
                changed = True
                break
        if changed:
            continue
        if len(tokens) > 1 and tokens[-1] in LIST1_SUFFIXES_SINGLE:
            tokens = tokens[:-1]
            changed = True
            continue
        if len(tokens) > 1 and tokens[0] in LIST1_SUFFIXES_SINGLE:
            tokens = tokens[1:]
            changed = True

    # 8. Canonicalize List 2 (compounds first, then singles)
    if LIST2_COMPOUND:
        out: list[str] = []
        i = 0
        while i < len(tokens):
            matched = False
            for compound, canon in LIST2_COMPOUND.items():
                clen = len(compound)
                if tuple(tokens[i:i + clen]) == compound:
                    out.append(canon)
                    i += clen
                    matched = True
                    break
            if not matched:
                out.append(tokens[i])
                i += 1
        tokens = out
    tokens = [LIST2_CANONICAL.get(t, t) for t in tokens]

    # 9. Whitespace collapse (join with single space)
    return " ".join(t for t in tokens if t)
