"""Synthetic perturbation generator.

Public API: `generate_examples()`. Given a pool of canonical names, produce
variants that mirror the kind of noise found in real customs records:
suffix swaps, abbreviation drift, garbage prefixes, typos, casing/spacing
oddities. Output is a `pa.Table` with `(variant_name, true_entity_id,
true_canonical)` columns — directly consumable by `cluster()` and
`score_clusters()`.

Design (per ARCHITECTURE.md):
- Star topology: variants are generated INDEPENDENTLY from the canonical,
  not from each other (mirrors real customs star structure).
- Power-law distribution over per-entity variant counts: most entities
  get 1-2 variants, few get many.
- Three difficulty levels (easy / medium / hard) compose: medium is
  superset of easy, hard of medium.
- Deterministic given seed.
"""
from __future__ import annotations

import random
import string
import unicodedata
from typing import Callable, Literal

import pyarrow as pa

from ._toy_canonicals import TOY_CANONICALS

Difficulty = Literal["easy", "medium", "hard"]

# ---------------------------------------------------------------------------
# Edit primitives
# ---------------------------------------------------------------------------

LEGAL_SUFFIXES_INTERCHANGEABLE = [
    "Inc", "LLC", "Ltd", "Limited", "Corp", "Corporation", "Co", "Company",
    "LLP", "LP", "PLC",
]
_LEGAL_SUFFIXES_LOWER = frozenset(s.lower() for s in LEGAL_SUFFIXES_INTERCHANGEABLE)

ABBREVIATION_PAIRS = [
    ("International", "Intl"),
    ("Manufacturing", "Mfg"),
    ("Industries", "Ind"),
    ("Holdings", "Hldg"),
    ("Group", "Grp"),
    ("Services", "Svc"),
    ("Technology", "Tech"),
    ("Technologies", "Tech"),
    ("Trading", "Trd"),
    ("Solutions", "Sln"),
    ("Enterprises", "Ent"),
    ("Development", "Dev"),
    ("Associates", "Assoc"),
    ("Import", "Imp"),
    ("Export", "Exp"),
]
# Lower-cased lookup: token-form -> replacement-form.
_ABBREV_LOOKUP: dict[str, str] = {}
for _long, _short in ABBREVIATION_PAIRS:
    _ABBREV_LOOKUP[_long.lower()] = _short
    _ABBREV_LOOKUP[_short.lower()] = _long

GARBAGE_PREFIXES = ["00 ", "000 ", "# ", "* ", "## "]

GEO_SUFFIXES = ["USA", "America", "Global", "Worldwide", "International"]

TYPO_VOWELS = "aeiou"


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )


def _swap_suffix(name: str, rng: random.Random) -> str:
    parts = name.rsplit(maxsplit=1)
    if len(parts) != 2:
        return name
    head, last = parts
    last_lower = last.rstrip(".,").strip().lower()
    if last_lower not in _LEGAL_SUFFIXES_LOWER:
        return name
    candidates = [s for s in LEGAL_SUFFIXES_INTERCHANGEABLE if s.lower() != last_lower]
    return f"{head} {rng.choice(candidates)}"


def _drop_suffix(name: str, rng: random.Random) -> str:
    parts = name.rsplit(maxsplit=1)
    if len(parts) != 2:
        return name
    head, last = parts
    if last.rstrip(".,").lower() in _LEGAL_SUFFIXES_LOWER:
        return head
    return name


def _swap_abbreviation(name: str, rng: random.Random) -> str:
    tokens = name.split()
    candidates: list[tuple[int, str]] = []  # (token_idx, replacement)
    for i, tok in enumerate(tokens):
        clean = tok.strip(".,;:").lower()
        if clean in _ABBREV_LOOKUP:
            candidates.append((i, _ABBREV_LOOKUP[clean]))
    if not candidates:
        return name
    i, repl = rng.choice(candidates)
    new = list(tokens)
    new[i] = repl
    return " ".join(new)


def _add_garbage_prefix(name: str, rng: random.Random) -> str:
    return rng.choice(GARBAGE_PREFIXES) + name


def _add_geo_suffix(name: str, rng: random.Random) -> str:
    return f"{name} {rng.choice(GEO_SUFFIXES)}"


def _toggle_the(name: str, rng: random.Random) -> str:
    if name.lower().startswith("the "):
        return name[4:]
    return f"The {name}"


def _change_case(name: str, rng: random.Random) -> str:
    return rng.choice([name.upper, name.lower, name.title])()


def _typo_char(name: str, rng: random.Random) -> str:
    """Insert / delete / transpose / substitute one char."""
    if len(name) < 4:
        return name
    op = rng.choice(["insert", "delete", "transpose", "substitute"])
    # Bias edits to the middle: avoid messing with first or last char
    pos = rng.randrange(2, len(name) - 1)
    if op == "insert":
        ch = rng.choice(string.ascii_lowercase)
        return name[:pos] + ch + name[pos:]
    if op == "delete":
        return name[:pos] + name[pos + 1:]
    if op == "transpose" and pos < len(name) - 1:
        return name[:pos] + name[pos + 1] + name[pos] + name[pos + 2:]
    if op == "substitute":
        ch = rng.choice(string.ascii_lowercase)
        return name[:pos] + ch + name[pos + 1:]
    return name


def _drop_internal_word(name: str, rng: random.Random) -> str:
    tokens = name.split()
    if len(tokens) <= 2:
        return name
    # Drop one internal token (not first or last)
    idx = rng.randrange(1, len(tokens) - 1)
    return " ".join(tokens[:idx] + tokens[idx + 1:])


def _add_internal_spacing_oddity(name: str, rng: random.Random) -> str:
    """E.g. `IBM` -> `I B M` (mostly relevant for short tokens)."""
    tokens = name.split()
    candidates = [i for i, t in enumerate(tokens) if 2 <= len(t.strip(".,")) <= 5 and t.strip(".,").isalpha()]
    if not candidates:
        return name
    i = rng.choice(candidates)
    spaced = " ".join(tokens[i].strip(".,"))
    new = list(tokens)
    new[i] = spaced
    return " ".join(new)


def _toggle_punct(name: str, rng: random.Random) -> str:
    if "," in name and rng.random() < 0.5:
        return name.replace(",", "")
    for s in LEGAL_SUFFIXES_INTERCHANGEABLE:
        sfx = " " + s
        if name.endswith(sfx) and not name.endswith(", " + s):
            return name[: -len(sfx)] + ", " + s
    return name


# ---------------------------------------------------------------------------
# Difficulty-leveled edit menus
# ---------------------------------------------------------------------------

# Each entry is (op_name, fn(name, rng) -> name). Difficulty menus reference
# these names. Adding an op = one entry here + one entry in the difficulty
# tier(s). The dispatcher is a dict lookup, no if/elif chain.
def _strip_accents_op(name: str, rng: random.Random) -> str:
    return _strip_accents(name)


_OPS: dict[str, Callable[[str, random.Random], str]] = {
    "case":           _change_case,
    "punct":          _toggle_punct,
    "suffix_swap":    _swap_suffix,
    "drop_suffix":    _drop_suffix,
    "abbrev":         _swap_abbreviation,
    "garbage_prefix": _add_garbage_prefix,
    "the_toggle":     _toggle_the,
    "strip_accents":  _strip_accents_op,
    "typo":           _typo_char,
    "drop_word":      _drop_internal_word,
    "add_geo":        _add_geo_suffix,
    "spacing_oddity": _add_internal_spacing_oddity,
}

EASY_OPS = ["case", "punct", "suffix_swap", "drop_suffix"]
MEDIUM_OPS = EASY_OPS + ["abbrev", "garbage_prefix", "the_toggle", "strip_accents"]
HARD_OPS = MEDIUM_OPS + ["typo", "drop_word", "add_geo", "spacing_oddity"]

DIFFICULTY_OPS: dict[Difficulty, list[str]] = {
    "easy": EASY_OPS,
    "medium": MEDIUM_OPS,
    "hard": HARD_OPS,
}

# Approximate # of edits per variant per difficulty (Bernoulli-ish per op).
DIFFICULTY_EDIT_RATE: dict[Difficulty, float] = {
    "easy": 0.6,    # ~1-2 edits per variant
    "medium": 1.2,  # ~2-3 edits per variant
    "hard": 2.0,    # ~3-5 edits per variant
}


def _generate_one_variant(
    canonical: str,
    difficulty: Difficulty,
    rng: random.Random,
) -> str:
    ops = DIFFICULTY_OPS[difficulty]
    edit_rate = DIFFICULTY_EDIT_RATE[difficulty]
    name = canonical
    p = min(edit_rate / max(len(ops), 1), 1.0)
    fires = 0
    for op in rng.sample(ops, len(ops)):
        if fires >= 5:
            break
        if rng.random() < p:
            name = _OPS[op](name, rng)
            fires += 1
    if not name.strip():
        return canonical
    return name


def _power_law_count(low: int, high: int, rng: random.Random) -> int:
    """1/k weighting favors the low end (most entities get the minimum)."""
    population = range(low, high + 1)
    weights = [1.0 / (k - low + 1) for k in population]
    return rng.choices(population, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def generate_examples(
    n_entities: int = 1000,
    variants_per_entity: tuple[int, int] = (1, 8),
    difficulty: Difficulty = "medium",
    seed: int = 0,
    canonicals: list[str] | None = None,
) -> pa.Table:
    """Generate a synthetic dataset of business-name variants.

    Args:
        n_entities: number of distinct true entities to produce.
        variants_per_entity: (low, high) range for per-entity variant count.
            Drawn power-law (most entities get the low end).
        difficulty: 'easy' | 'medium' | 'hard'. Higher = more edits per variant.
        seed: deterministic RNG seed.
        canonicals: pool of canonical names to sample from. Defaults to
            embedded TOY_CANONICALS (~50 fake names).

    Returns:
        pa.Table with columns:
          - variant_name: str
          - true_entity_id: int (0..n_entities-1)
          - true_canonical: str

    Each entity gets >=1 variant, including potentially the canonical itself.
    Variants are generated INDEPENDENTLY from the canonical (star topology) —
    spokes diverge from the hub but not necessarily from each other.
    """
    if difficulty not in DIFFICULTY_OPS:
        raise ValueError(f"difficulty must be one of {list(DIFFICULTY_OPS)}, got {difficulty!r}")
    low, high = variants_per_entity
    if low < 1 or high < low:
        raise ValueError(f"variants_per_entity must be (low>=1, high>=low), got {variants_per_entity!r}")

    pool = canonicals if canonicals is not None else TOY_CANONICALS
    if not pool:
        raise ValueError("canonicals pool is empty")

    rng = random.Random(seed)

    rows_variant: list[str] = []
    rows_id: list[int] = []
    rows_canonical: list[str] = []

    for entity_id in range(n_entities):
        canonical = pool[entity_id % len(pool)]
        # Add slight uniqueness when n_entities > pool size
        if entity_id >= len(pool):
            disambig = f" {entity_id // len(pool) + 1}"
            canonical = canonical + disambig
        n_variants = _power_law_count(low, high, rng)
        # Within-entity dedup only: cross-entity name collisions are real-world
        # signal (two distinct entities with the same recorded name) so kept.
        seen: set[str] = set()
        for _ in range(n_variants):
            v = _generate_one_variant(canonical, difficulty, rng)
            if v in seen:
                continue
            seen.add(v)
            rows_variant.append(v)
            rows_id.append(entity_id)
            rows_canonical.append(canonical)

    return pa.table({
        "variant_name": pa.array(rows_variant, type=pa.string()),
        "true_entity_id": pa.array(rows_id, type=pa.int64()),
        "true_canonical": pa.array(rows_canonical, type=pa.string()),
    })
