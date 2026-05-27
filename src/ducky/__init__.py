"""ducky: tiny, fast nanobind bindings for DuckDB's C API."""

from __future__ import annotations

from ._core import (
    Connection,
    Error,
    Result,
    __duckdb_version__,
    connect,
)

__all__ = [
    "Connection",
    "Error",
    "Result",
    "__duckdb_version__",
    "connect",
]

__version__ = "0.1.0"
