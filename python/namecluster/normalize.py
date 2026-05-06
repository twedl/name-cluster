"""Public normalize API.

`normalize(name)`         — single-name normalization (rust binding)
`normalize_names(seq)`    — batched, parallel normalize over a sequence

The rust core lives in `_lowlevel.{normalize, normalize_lists}`; this layer
adds the type-coercion the dataframe path expects (np.nan / non-str → None,
matching `cluster()`).
"""

from __future__ import annotations

from typing import Sequence

from ._lowlevel import normalize, normalize_lists as _normalize_lists


def normalize_names(names: Sequence[str | None]) -> list[str | None]:
    """Normalize each name in `names` in parallel.

    `None` (and any non-str like `np.nan`) passes through to `None`. Empty
    string and names that normalize to empty (e.g. ``"00"``) return ``""`` —
    matching the scalar `normalize()`.

    Apply to a polars column:

    >>> import polars as pl, namecluster as nc
    >>> df = pl.DataFrame({"name": ["Acme Corp", "ACME Inc", None]})
    >>> df.with_columns(
    ...     pl.col("name")
    ...       .map_batches(lambda s: pl.Series(nc.normalize_names(s.to_list())),
    ...                    return_dtype=pl.Utf8)
    ...       .alias("normalized")
    ... )
    """
    coerced = [n if isinstance(n, str) else None for n in names]
    return _normalize_lists(coerced)


__all__ = ["normalize", "normalize_names"]
