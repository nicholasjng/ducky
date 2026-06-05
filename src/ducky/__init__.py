"""ducky: tiny, fast nanobind bindings for DuckDB's C API."""

from __future__ import annotations

import sys

if sys.version_info >= (3, 12):
    from typing import Unpack
else:
    from typing_extensions import Unpack

from ._conversions import ArrowSource
from ._core import (
    Appender,
    Connection,
    Error,
    PreparedStatement,
    Result,
    __duckdb_version__,
    config_options,
)
from ._core import connect as _connect
from ._dataset import (
    Dataset,
    Feature,
    Fold,
    Matrix,
    Split,
    Target,
    Vector,
    dataset,
    feature,
    matrix,
    split,
    target,
    vector,
)
from ._progress import progress_bar
from ._typing import DuckDBConfig


def connect(
    database: str = ":memory:",
    **config: Unpack[DuckDBConfig],
) -> Connection:
    """Open `database` (default in-memory) and return a Connection.

    DuckDB settings are passed as keyword arguments using the keys declared in
    :class:`DuckDBConfig` (autocomplete-friendly), as their natural Python type
    — `ducky.connect(memory_limit="2GB", threads=4, enable_object_cache=True)`.
    Values are coerced to strings for DuckDB; any key or value DuckDB rejects
    raises :class:`Error` at open time. See :func:`config_options` for the full
    runtime list.

    Examples
    --------
    >>> con = ducky.connect(memory_limit="2GB", threads=4)
    >>> con = ducky.connect("mydb.duckdb", access_mode="READ_ONLY")
    """

    def coerce(value: object) -> str:
        # Coerce a config value to the string form duckdb_set_config expects.
        # bool is checked before int (it is an int subclass) so flags become
        # DuckDB's "true"/"false" rather than "1"/"0". Strings pass through; any
        # other scalar is stringified. DuckDB validates the result at open time.
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str):
            return value
        return str(value)

    coerced = {key: coerce(value) for key, value in config.items()}
    return _connect(database, coerced or None)


__all__ = [
    "Appender",
    "ArrowSource",
    "Connection",
    "Dataset",
    "DuckDBConfig",
    "Error",
    "Feature",
    "Fold",
    "Matrix",
    "PreparedStatement",
    "Result",
    "Split",
    "Target",
    "Vector",
    "__duckdb_version__",
    "config_options",
    "connect",
    "dataset",
    "feature",
    "matrix",
    "progress_bar",
    "split",
    "target",
    "vector",
]

__version__ = "0.1.0"
