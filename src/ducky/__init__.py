"""ducky: tiny, fast nanobind bindings for DuckDB's C API."""

from __future__ import annotations

import sys
from typing import cast

if sys.version_info >= (3, 12):
    from typing import Unpack
else:
    from typing_extensions import Unpack

from ._conversions import ArrowSource
from ._core import (
    Appender,
    Connection,
    Error,
    Result,
    __duckdb_version__,
    config_options,
)
from ._core import connect as _connect
from ._dataset import (
    Dataset,
    Feature,
    Fold,
    Split,
    Target,
    dataset,
    feature,
    split,
    target,
)
from ._typing import DuckDBConfig


def connect(
    database: str = ":memory:",
    **config: Unpack[DuckDBConfig],
) -> Connection:
    """Open `database` (default in-memory) and return a Connection.

    DuckDB settings are passed as keyword arguments using the keys declared
    in :class:`DuckDBConfig` (autocomplete-friendly). Any keyword unknown to
    DuckDB raises :class:`Error` at open time; see :func:`config_options`
    for the full runtime list.

    Examples
    --------
    >>> con = ducky.connect(memory_limit="2GB", threads="4")
    >>> con = ducky.connect("mydb.duckdb", access_mode="READ_ONLY")
    """
    # `config` is a plain dict at runtime (kwargs collection) — the cast
    # bridges the TypedDict view to the concrete dict shape that `_connect`
    # advertises. TypedDicts aren't structurally assignable to dict[str, str]
    # in PEP 589, so this purely-static narrowing is the cleanest workaround.
    return _connect(database, cast("dict[str, str]", config) if config else None)


__all__ = [
    "Appender",
    "ArrowSource",
    "Connection",
    "Dataset",
    "DuckDBConfig",
    "Error",
    "Feature",
    "Fold",
    "Result",
    "Split",
    "Target",
    "__duckdb_version__",
    "config_options",
    "connect",
    "dataset",
    "feature",
    "split",
    "target",
]

__version__ = "0.1.0"
