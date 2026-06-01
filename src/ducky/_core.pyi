"""ducky: tiny nanobind bindings for the DuckDB C API"""

from collections.abc import Callable, Iterable, Iterator
from types import TracebackType
from typing import Any, Literal, overload

import jax
import numpy
import pandas
import pyarrow
import torch
from polars import DataFrame as _PolarsDataFrame
from polars import LazyFrame as _PolarsLazyFrame

__duckdb_version__: str = "v1.6.2-dev4685"

class Error(Exception):
    pass

class Chunk:
    """
    A single DuckDB data chunk: a column-major batch of up to STANDARD_VECTOR_SIZE rows. Numeric and temporal columns are exposed as zero-copy ndarrays.
    """

    def __len__(self) -> int:
        """Number of rows in this chunk."""

    @property
    def columns(self) -> list[str]:
        """Column names."""

    @property
    def types(self) -> list[str]:
        """Column type names."""

    def column(self, key: int | str) -> numpy.ndarray:
        """
        Return the column at `key` (int index or str name) as a 1-D ndarray view over the chunk's buffer.
        """

    def validity(self, key: int | str) -> numpy.ndarray | None:
        """
        Return a uint8 ndarray (1=valid, 0=null) of length len(self), or None if the column has no nulls.
        """

class Result:
    """
    A query result. Iterate it, or use the fetch* methods, to pull rows as tuples.
    """

    @property
    def columns(self) -> list[str]:
        """Column names."""

    @property
    def types(self) -> list[str]:
        """Column type names."""

    @property
    def description(self) -> list[tuple] | None:
        """PEP 249 result description."""

    def fetchone(self) -> tuple | None:
        """Return the next row, or None."""

    def fetchmany(self, size: int = 1) -> list[tuple]:
        """Return up to `size` rows."""

    def fetchall(self) -> list[tuple]:
        """Return all remaining rows."""

    def fetch_chunk(self) -> Chunk | None:
        """Pull the next data chunk as a Chunk, or None at end of stream."""

    def __arrow_c_stream__(self, requested_schema: Any = None) -> Any:
        """Export the result via the Arrow C stream (PyCapsule) interface."""

    def arrow(self) -> pyarrow.Table:
        """Return the result as a pyarrow.Table."""

    def df(self) -> pandas.DataFrame:
        """Return the result as a pandas.DataFrame."""

    @overload
    def pl(self, lazy: Literal[False] = False) -> _PolarsDataFrame: ...
    @overload
    def pl(self, lazy: Literal[True]) -> _PolarsLazyFrame: ...
    def pl(self, lazy: bool = False) -> _PolarsDataFrame | _PolarsLazyFrame:
        """Return the result as a polars DataFrame (or LazyFrame)."""

    def fetchnumpy(self) -> dict[str, numpy.ndarray]:
        """Return the result as a dict of column name -> numpy array."""

    def chunks(self) -> Iterator[Chunk]:
        """Iterate over the result one Chunk at a time. Drains the result."""

    @overload
    def iter_batches(
        self, columns: Iterable[str] | None = None, with_validity: Literal[False] = False
    ) -> Iterator[dict[str, numpy.ndarray]]: ...
    @overload
    def iter_batches(
        self, columns: Iterable[str] | None = None, *, with_validity: Literal[True]
    ) -> Iterator[dict[str, tuple[numpy.ndarray, numpy.ndarray | None]]]: ...
    def iter_batches(
        self, columns: Iterable[str] | None = None, with_validity: bool = False
    ) -> Iterator[dict[str, numpy.ndarray] | dict[str, tuple[numpy.ndarray, numpy.ndarray | None]]]:
        """
        Yield one dict per chunk: {name: ndarray}, or {name: (values, mask)} if with_validity=True.
        """

    def to_numpy(self, columns: Iterable[str] | None = None) -> dict[str, numpy.ndarray]:
        """Eagerly concatenate all chunks into {name: numpy.ndarray}."""

    def to_torch(
        self, columns: Iterable[str] | None = None, device: torch.device | str | int | None = None
    ) -> dict[str, torch.Tensor]:
        """Eagerly concatenate all chunks into {name: torch.Tensor}."""

    def to_jax(
        self, columns: Iterable[str] | None = None, device: jax.Device | None = None
    ) -> dict[str, jax.Array]:
        """Eagerly concatenate all chunks into {name: jax.Array}."""

    def __iter__(self) -> Iterator[tuple]: ...
    def __next__(self) -> tuple: ...

class Appender:
    """
    Bulk-insert handle for a target table. Supports a row-at-a-time API for mixed/typed writes and a columnar fast path for numeric/temporal ndarrays.
    """

    @property
    def columns(self) -> list[str]:
        """Target column names."""

    @property
    def types(self) -> list[str]:
        """Target column type names."""

    def append_row(self, *values: object) -> None:
        """Append one row of positional values, matching the target column order."""

    def append_columns(
        self, columns: dict[str, numpy.ndarray], masks: dict[str, numpy.ndarray] | None = None
    ) -> None:
        """
        Append a batch of columns. Missing columns are filled with NULL. `masks` maps column name -> 1-D bool/uint8 array (True = valid).
        """

    def flush(self) -> None:
        """Flush pending rows to the table."""

    def close(self) -> None:
        """Flush and tear down the appender."""

    def __enter__(self) -> Appender: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        pass

class Connection:
    """A connection to a DuckDB database."""

    def execute(self, query: str, parameters: list | dict[str, Any] | None = None) -> Connection:
        """
        Execute a query, optionally with positional parameters, and return self for chaining.
        """

    def sql(self, query: str) -> Result:
        """Run a query and return its Result."""

    def query(self, query: str) -> Result:
        """Alias for sql()."""

    def fetchone(self) -> tuple | None: ...
    def fetchmany(self, size: int = 1) -> list[tuple]: ...
    def fetchall(self) -> list[tuple]: ...
    @property
    def description(self) -> list[tuple] | None: ...
    @property
    def columns(self) -> list[str] | None: ...
    def arrow(self) -> pyarrow.Table:
        """Return the last result as a pyarrow.Table."""

    def df(self) -> pandas.DataFrame:
        """Return the last result as a pandas.DataFrame."""

    @overload
    def pl(self, lazy: Literal[False] = False) -> _PolarsDataFrame: ...
    @overload
    def pl(self, lazy: Literal[True]) -> _PolarsLazyFrame: ...
    def pl(self, lazy: bool = False) -> _PolarsDataFrame | _PolarsLazyFrame:
        """Return the last result as a polars DataFrame (or LazyFrame)."""

    def fetchnumpy(self) -> dict[str, numpy.ndarray]:
        """Return the last result as a dict of column name -> numpy array."""

    def chunks(self) -> Iterator[Chunk]:
        """Iterate over the last result one Chunk at a time."""

    @overload
    def iter_batches(
        self, columns: Iterable[str] | None = None, with_validity: Literal[False] = False
    ) -> Iterator[dict[str, numpy.ndarray]]: ...
    @overload
    def iter_batches(
        self, columns: Iterable[str] | None = None, *, with_validity: Literal[True]
    ) -> Iterator[dict[str, tuple[numpy.ndarray, numpy.ndarray | None]]]: ...
    def iter_batches(
        self, columns: Iterable[str] | None = None, with_validity: bool = False
    ) -> Iterator[dict[str, numpy.ndarray] | dict[str, tuple[numpy.ndarray, numpy.ndarray | None]]]:
        """Yield one dict per chunk for the last result."""

    def to_numpy(self, columns: Iterable[str] | None = None) -> dict[str, numpy.ndarray]:
        """Return the last result as {name: numpy.ndarray}."""

    def to_torch(
        self, columns: Iterable[str] | None = None, device: torch.device | str | int | None = None
    ) -> dict[str, torch.Tensor]:
        """Return the last result as {name: torch.Tensor}."""

    def to_jax(
        self, columns: Iterable[str] | None = None, device: jax.Device | None = None
    ) -> dict[str, jax.Array]:
        """Return the last result as {name: jax.Array}."""

    def create_function(
        self,
        name: str,
        fn: Callable,
        parameters: list[str] | dict[str, str] | None = None,
        return_type: str | None = None,
        varargs: str | None = None,
    ) -> None:
        """
        Register a Python callable as a DuckDB scalar function. `parameters` is a list of type strings (positional call) or a dict of {name: type_string} (dict-style call). Inputs arrive as zero-copy 1-D ndarrays; `fn` must return one ndarray of length chunk_size and matching dtype. If `parameters` or `return_type` is omitted, they are inferred from `fn`'s annotations (bool/int/float → BOOLEAN/BIGINT/DOUBLE). Pass `varargs="TYPE"` (mutually exclusive with `parameters`) to register a variable-arity function; `fn` is then called as `fn(*args)` with one ndarray per SQL argument.
        """

    def appender(
        self, table: str, schema: str | None = None, catalog: str | None = None
    ) -> Appender:
        """
        Open an Appender for bulk inserts into `table`. The appender shares the connection's database handle, so the database stays open until the appender is closed.
        """

    def interrupt(self) -> None:
        """
        Best-effort cancel of any query currently running on this connection. Safe to call from another thread while execute() is in flight.
        """

    def progress(self) -> tuple[float, int, int]:
        """
        Snapshot of the current query's progress: (percentage, rows_processed, total_rows_to_process). `percentage` is -1.0 until DuckDB has an estimate.
        """

    def register_arrow(self, name: str, obj: Any) -> None:
        """
        Register a Python object exposing `__arrow_c_stream__` (pyarrow Table, polars DataFrame, pandas-3 DataFrame, ...) as a table named `name`. The data is materialized into DuckDB at registration; the source object is not retained.
        """

    def close(self) -> None:
        """Close the connection."""

    def __enter__(self) -> Connection: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        pass

def connect(database: str = ":memory:", config: dict[str, str] | None = None) -> Connection:
    """
    Open `database` (default in-memory) and return a Connection. `config` is an optional dict of DuckDB settings applied at open time (see ducky.config_options() for the full list of keys).
    """

def config_options() -> list[tuple[str, str]]:
    """
    Return the (name, description) of every DuckDB config option settable via connect(config=...).
    """
