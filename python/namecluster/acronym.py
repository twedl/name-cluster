"""`acronym_map(data, name_col)` — corpus-derived acronym candidates.

Scans the user's corpus for names that look like acronyms and names that
look like their expansions, matches them by first-letter signature, and
returns a dataframe of `(acronym, expansion_count, expansions, acronym_examples)`.

Single-expansion rows are high-confidence candidates for the `aliases=`
kwarg on `cluster()`. Multi-expansion rows are ambiguous (e.g. "AA" might
match "American Airlines" AND "Alcoholics Anonymous" — caller decides).

Standalone of cluster() — only depends on `normalize()` for tokenisation.
"""

from __future__ import annotations

import narwhals as nw

from ._lowlevel import normalize as _normalize


# Connective words usually skipped when forming an acronym from an expansion.
# "NASA" leaves out the "and" in "National Aeronautics AND Space Administration";
# "USA" leaves out the "of" in "United States OF America". Including both the
# strict signature and the stopword-skipped one lets us catch both styles.
_ACRONYM_STOPWORDS = frozenset({"and", "of", "the", "for", "in", "by", "to", "a"})


@nw.narwhalify
def acronym_map(
    data,
    name_col: str = "name",
    min_acronym_len: int = 2,
    max_acronym_len: int = 6,
    min_expansion_tokens: int = 2,
):
    """Find acronym↔expansion pairs co-occurring in the input corpus.

    Parameters
    ----------
    data : DataFrame
        Any narwhals-supported df (polars, pandas, pyarrow Table).
    name_col : str
        Column holding the name strings.
    min_acronym_len, max_acronym_len : int
        A name is an acronym candidate if it normalizes to one token of
        length within this range. Defaults catch typical 2- to 6-letter
        acronyms (IBM, NASA, AT&T's "atandt", etc.) without picking up
        every 1-character outlier or long single-word brand.
    min_expansion_tokens : int
        A name is an expansion candidate if it normalizes to at least this
        many tokens. Default 2 catches "General Electric" → "GE";
        bump to 3 to focus on stricter multi-word expansions like IBM.

    Returns
    -------
    Same df type as input, sorted by `expansion_count` ascending then
    `acronym` ascending. Columns:

    - acronym : str          — the normalized acronym (lowercase)
    - expansion_count : int  — how many distinct normalized expansions matched
    - expansions : list[str] — RAW input names whose first-letter signature
                               equals `acronym` (capped at 10 examples)
    - acronym_examples : list[str]
                             — RAW input names that normalize to `acronym`
                               (capped at 5 examples)

    Single-expansion rows are high-confidence candidates for the `aliases=`
    kwarg on `cluster()`:

        ac = nc.acronym_map(df, name_col="name")
        high = ac.filter(pl.col("expansion_count") == 1)
        aliases = {
            row["expansions"][0]: row["acronym_examples"]
            for row in high.iter_rows(named=True)
        }
        nc.cluster(df, name_col="name", aliases=aliases)
    """
    if min_acronym_len < 1 or max_acronym_len < min_acronym_len:
        raise ValueError(
            f"need 1 <= min_acronym_len <= max_acronym_len, "
            f"got ({min_acronym_len}, {max_acronym_len})"
        )
    if min_expansion_tokens < 2:
        raise ValueError(
            f"min_expansion_tokens must be >= 2, got {min_expansion_tokens}"
        )

    raw_names = data[name_col].to_list()
    acronym_to_raws: dict[str, list[str]] = {}
    expansion_to_raws: dict[str, list[str]] = {}

    for raw in raw_names:
        if not isinstance(raw, str):
            continue
        norm = _normalize(raw)
        if not norm:
            continue
        tokens = norm.split()
        if len(tokens) == 1:
            tok = tokens[0]
            if min_acronym_len <= len(tok) <= max_acronym_len:
                acronym_to_raws.setdefault(tok, []).append(raw)
        elif len(tokens) >= min_expansion_tokens:
            sig_strict = "".join(t[0] for t in tokens)
            sig_loose = "".join(t[0] for t in tokens if t not in _ACRONYM_STOPWORDS)
            for sig in {sig_strict, sig_loose}:
                if min_acronym_len <= len(sig) <= max_acronym_len:
                    expansion_to_raws.setdefault(sig, []).append(raw)

    rows = []
    for ac in sorted(acronym_to_raws.keys()):
        if ac not in expansion_to_raws:
            continue
        expansions = expansion_to_raws[ac]
        rows.append(
            {
                "acronym": ac,
                "expansion_count": len(set(_normalize(e) for e in expansions)),
                "expansions": expansions[:10],
                "acronym_examples": acronym_to_raws[ac][:5],
            }
        )

    rows.sort(key=lambda r: (r["expansion_count"], r["acronym"]))

    backend = data.implementation
    return nw.from_dict(
        {
            "acronym": [r["acronym"] for r in rows],
            "expansion_count": [r["expansion_count"] for r in rows],
            "expansions": [r["expansions"] for r in rows],
            "acronym_examples": [r["acronym_examples"] for r in rows],
        },
        backend=backend,
    )
