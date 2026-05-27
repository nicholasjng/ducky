"""Conversions from a ducky result into DataFrame/array libraries.

Each function takes any object implementing the Arrow C stream interface
(``__arrow_c_stream__``) — i.e. a ducky ``Result`` or ``Connection`` — and
imports its target library lazily, so none of these are hard dependencies. The
extension itself only knows how to produce the Arrow stream; everything here is
plain Python on top of it.

Note: consuming the stream drains the underlying result, so call exactly one of
these (or the ``fetch*`` methods) per executed query.
"""

from __future__ import annotations

from typing import Any


def arrow(source: Any) -> Any:
    """Return the result as a ``pyarrow.Table``."""
    import pyarrow as pa

    return pa.table(source)


def df(source: Any) -> Any:
    """Return the result as a ``pandas.DataFrame`` (via Arrow)."""
    return arrow(source).to_pandas()


def pl(source: Any, lazy: bool = False) -> Any:
    """Return the result as a polars ``DataFrame`` (or ``LazyFrame`` if ``lazy``)."""
    import polars

    frame = polars.from_arrow(source)
    return frame.lazy() if lazy else frame


def fetchnumpy(source: Any) -> dict[str, Any]:
    """Return the result as a dict of column name -> ``numpy`` array."""
    table = arrow(source)
    return {
        name: column.to_numpy(zero_copy_only=False)
        for name, column in zip(table.column_names, table.columns)
    }
