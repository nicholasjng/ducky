"""ducky: tiny, fast nanobind bindings for DuckDB's C API."""

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
from ._profile import (
    ProfileConfig,
    format_profiling_info,
    jsonl_profile_sink,
    profile,
)
from ._progress import progress_bar
from ._typing import DuckDBConfig


def connect(
    database: str = ":memory:",
    **config: Unpack[DuckDBConfig],
) -> Connection:
    """Open ``database`` and return a :class:`Connection`.

    DuckDB settings are passed as natural-typed keyword arguments using the keys declared in :class:`DuckDBConfig`.
    Values are coerced to strings for DuckDB; an invalid key or value raises :class:`Error` at open time.

    Parameters
    ----------
    database : str, default ':memory:'
        File path, ``':memory:'``, or any DuckDB connection string.
    **config : DuckDBConfig
        DuckDB settings applied at open.
        See :func:`config_options` for the full list.

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
    con = _connect(database, coerced or None)

    # if DUCKY_PROFILE_DIR is set, attach a JSONL sink so every execute()/sql()
    # on this connection lands in {dir}/profile-{pid}.jsonl with no code change.
    if cfg := ProfileConfig.from_env():
        con.set_profile_sink(cfg.sink, sample=cfg.sample, mode=cfg.mode)

    return con


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
    "ProfileConfig",
    "Result",
    "Split",
    "Target",
    "Vector",
    "__duckdb_version__",
    "config_options",
    "connect",
    "dataset",
    "feature",
    "format_profiling_info",
    "jsonl_profile_sink",
    "matrix",
    "profile",
    "progress_bar",
    "split",
    "target",
    "vector",
]

__version__ = "0.1.0"
