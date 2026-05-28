"""ducky: tiny, fast nanobind bindings for DuckDB's C API."""

from __future__ import annotations

from ._core import (
    Appender,
    Connection,
    Error,
    Result,
    __duckdb_version__,
    connect,
)
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

__all__ = [
    "Appender",
    "Connection",
    "Dataset",
    "Error",
    "Feature",
    "Fold",
    "Result",
    "Split",
    "Target",
    "__duckdb_version__",
    "connect",
    "dataset",
    "feature",
    "split",
    "target",
]

__version__ = "0.1.0"
