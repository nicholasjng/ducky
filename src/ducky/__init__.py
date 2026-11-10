"""ducky: tiny, fast nanobind bindings for DuckDB's C API."""

from typing import Unpack

from ._conversions import ArrowSource
from ._core import (
    DEFAULT,
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
    """Open a DuckDB database with optional configuration settings."""

    def coerce(value: object) -> str:
        # DuckDB expects lowercase Boolean strings.
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str):
            return value
        return str(value)

    coerced = {key: coerce(value) for key, value in config.items()}
    con = _connect(database, coerced or None)

    if cfg := ProfileConfig.from_env():
        con.set_profile_sink(cfg.sink, sample=cfg.sample, mode=cfg.mode)

    return con


__all__ = [
    "DEFAULT",
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
