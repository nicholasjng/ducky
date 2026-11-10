"""ducky: tiny nanobind bindings for the DuckDB C API"""

import enum
from collections.abc import Callable, Coroutine, Iterable, Iterator
from types import TracebackType
from typing import Any, Literal, Self, overload

import jax
import numpy
import pandas
import polars
import pyarrow
import torch
from mlx import core as mx

__duckdb_version__: str = "v2.1.0-dev0"

class Error(Exception): ...

class Default:
    """Sentinel type for appender DEFAULT values; use ``ducky.DEFAULT``."""

DEFAULT: Default = ...

class Chunk:
    """A column-major DuckDB result batch."""

    def __len__(self) -> int:
        """Number of rows."""

    @property
    def columns(self) -> list[str]:
        """Column names."""

    @property
    def types(self) -> list[str]:
        """Column type names."""

    def column(self, key: int | str) -> numpy.ndarray:
        """Return a zero-copy 1-D ndarray view by index or name."""

    def validity(self, key: int | str) -> numpy.ndarray | None:
        """
        Return a uint8 validity mask (1 = valid, 0 = NULL), or None if no nulls.
        """

    def decimal_scale(self, key: int | str) -> int:
        """Return a DECIMAL column's scale."""

    def dlpack(self, key: int | str) -> Any:
        """Return a numeric or temporal column as a DLPack object."""

    def mlx(self, key: int | str) -> mx.array:
        """Return a numeric or temporal column as an MLX array."""

class Result:
    """A query result. Iterate or call ``fetch*`` to pull rows."""

    @property
    def columns(self) -> list[str]:
        """Column names."""

    @property
    def types(self) -> list[str]:
        """Column type names."""

    @property
    def description(self) -> list[tuple] | None:
        """PEP 249 result description, or None if the result is closed."""

    def fetchone(self) -> tuple | None:
        """Return the next row as a tuple, or None at end of result."""

    def fetchmany(self, size: int = 1) -> list[tuple]:
        """Return up to ``size`` rows."""

    def fetchall(self) -> list[tuple]:
        """Return all remaining rows."""

    def fetchitem(self) -> Any:
        """
        Return the lone scalar of a 1x1 result, for ``COUNT(*)``-style queries. Raises :class:`Error` on any other shape.
        """

    @property
    def rows_changed(self) -> int:
        """Rows affected by INSERT, UPDATE, or DELETE; otherwise 0."""

    def fetch_chunk(self) -> Chunk | None:
        """Pull the next :class:`Chunk`, or None at end of stream."""

    def fetch_numpy_chunk(self) -> dict[str, numpy.ndarray] | None:
        """Pull the next chunk as NumPy arrays, or None at end."""

    def to_numpy(self, columns: Iterable[str] | None = None) -> dict[str, numpy.ndarray]:
        """
        Drain the result into a {name: numpy.ndarray} dict. NULLs become NaN in FLOAT/DOUBLE columns and raise elsewhere. Raises on VARCHAR/LIST/STRUCT/MAP; use arrow() for those.
        """

    def __arrow_c_stream__(self, requested_schema: Any = None) -> Any:
        """Export the result via the Arrow PyCapsule stream interface."""

    def arrow(self) -> pyarrow.Table:
        """Return the result as a pyarrow.Table."""

    def arrow_batches(self) -> pyarrow.RecordBatchReader:
        """Stream one Arrow record batch per DuckDB chunk."""

    def df(self) -> pandas.DataFrame:
        """Return the result as a pandas.DataFrame."""

    @overload
    def pl(self, lazy: Literal[False] = False) -> polars.DataFrame: ...
    @overload
    def pl(self, lazy: Literal[True]) -> polars.LazyFrame: ...
    def pl(self, lazy: bool = False) -> polars.DataFrame | polars.LazyFrame:
        """Return the result as a polars DataFrame (or LazyFrame)."""

    def fetchnumpy(self) -> dict[str, numpy.ndarray]:
        """
        Return the result as a dict of column name -> numpy array. Numeric and temporal columns only; use arrow() for strings and nested types. NULLs become NaN in FLOAT/DOUBLE columns and raise elsewhere.
        """

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

    def iter_batches_torch(
        self, columns: Iterable[str] | None = None, device: torch.device | str | int | None = None
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Yield Torch tensors per chunk."""

    def iter_batches_jax(
        self, columns: Iterable[str] | None = None, device: jax.Device | None = None
    ) -> Iterator[dict[str, jax.Array]]:
        """Yield one dict per chunk: {name: jax.Array}."""

    def iter_batches_mlx(
        self, columns: Iterable[str] | None = None
    ) -> Iterator[dict[str, mx.array]]:
        """Yield MLX arrays per chunk."""

    def to_torch(
        self, columns: Iterable[str] | None = None, device: torch.device | str | int | None = None
    ) -> dict[str, torch.Tensor]:
        """Return selected columns as Torch tensors."""

    def to_jax(
        self, columns: Iterable[str] | None = None, device: jax.Device | None = None
    ) -> dict[str, jax.Array]:
        """Return selected columns as JAX arrays."""

    def to_mlx(self, columns: Iterable[str] | None = None) -> dict[str, mx.array]:
        """Return selected columns as MLX arrays."""

    def __iter__(self) -> Iterator[tuple]: ...
    def __next__(self) -> tuple: ...

class Appender:
    """Buffered bulk-insert handle for a table."""

    @property
    def columns(self) -> list[str]:
        """Target column names."""

    @property
    def types(self) -> list[str]:
        """Target column type names."""

    def append_row(self, *values: object) -> None:
        """Append one row of positional values, in target column order."""

    def append_columns(
        self, columns: dict[str, numpy.ndarray], masks: dict[str, numpy.ndarray] | None = None
    ) -> None:
        """
        Append equal-length numeric or temporal arrays with optional validity masks. Missing columns become NULL.
        """

    def append_arrow(self, source: Any) -> None:
        """
        Append an Arrow stream or array. Columns must match the table by position and type.
        """

    def flush(self) -> None:
        """Flush pending rows to the table."""

    def close(self) -> None:
        """Flush and tear down the appender."""

    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

class PreparedStatement:
    """A reusable compiled SQL statement."""

    def execute(
        self, parameters: list | tuple | dict[str, Any] | None = None, streaming: bool = False
    ) -> Result:
        """
        Bind parameters and return the result. Streaming results are single-pass.
        """

    def executemany(self, parameters: Iterable[list | tuple | dict[str, Any]]) -> None:
        """Run once per parameter set and discard the results."""

    @property
    def num_parameters(self) -> int:
        """Number of bind parameters."""

    def parameter_name(self, index: int) -> str | None:
        """
        Name of the parameter at 1-based ``index``, or None for positional / out-of-range.
        """

    @property
    def columns(self) -> list[str]:
        """
        Result column names, known ahead of execution (empty for non-result statements).
        """

    @property
    def types(self) -> list[str]:
        """Result column type names, known ahead of execution."""

    @property
    def statement_type(self) -> str:
        """Statement kind: ``'SELECT'``, ``'INSERT'``, ``'UPDATE'``, ..."""

    def close(self) -> None:
        """Destroy the prepared statement."""

    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

class PendingState(enum.Enum):
    """State of an async query's executor after a tick."""

    READY = 0
    """The result is ready to materialize."""

    NOT_READY = 1
    """More tasks remain; tick again."""

    ERROR = 2
    """Execution failed; see ``.error()``."""

    NO_TASKS = 3
    """Workers own the outstanding tasks; yield, then tick again."""

class PendingResult:
    """Steppable in-flight query handle."""

    def execute_task(self) -> PendingState:
        """
        Advance the executor by one task (GIL released) and return the new state.
        """

    def execute_tasks(self, budget: float = 0.005) -> PendingState:
        """
        Run tasks for up to ``budget`` seconds (GIL released), stopping once ready or failed.
        """

    def error(self) -> str:
        """DuckDB's message for the most recent error, may be empty."""

    def materialize(self) -> Result:
        """
        Materialize the finished result; call once ``execute_task`` reports ``READY``.
        """

    def drain(self) -> None:
        """Interrupt the query and drain the executor (cancellation teardown)."""

class Connection:
    """A connection to a DuckDB database."""

    def execute(
        self,
        query: str,
        parameters: list | tuple | dict[str, Any] | None = None,
        streaming: bool = False,
    ) -> Connection:
        """
        Execute one statement and retain its result for fetch methods. Streaming results are single-pass. Returns ``self``.
        """

    def executemany(self, query: str, parameters: Iterable[list | tuple | dict[str, Any]]) -> None:
        """Run once per parameter set and discard the results."""

    def sql(self, query: str, streaming: bool = False) -> Result:
        """Run a query and return its result directly."""

    def query(self, query: str, streaming: bool = False) -> Result:
        """Alias for :meth:`sql`."""

    def aexecute(
        self,
        query: str,
        parameters: list | tuple | dict[str, Any] | None = None,
        streaming: bool = False,
    ) -> Coroutine[Any, Any, Result]:
        """
        Execute asynchronously and return a Result without setting current_result.
        """

    def asql(self, query: str, streaming: bool = False) -> Coroutine[Any, Any, Result]:
        """Async variant of :meth:`sql`. See :meth:`aexecute`."""

    def make_pending(
        self,
        query: str,
        parameters: list | tuple | dict[str, Any] | None = None,
        streaming: bool = False,
    ) -> PendingResult:
        """Build a steppable pending result."""

    def prepare(self, query: str) -> PreparedStatement:
        """Compile a reusable prepared statement."""

    def fetchone(self) -> tuple | None:
        """Fetch one row from the latest execute result."""

    def fetchmany(self, size: int = 1) -> list[tuple]:
        """Fetch up to ``size`` rows from the latest result."""

    def fetchall(self) -> list[tuple]:
        """Fetch all remaining rows from the latest result."""

    def fetchitem(self) -> Any:
        """Return the lone scalar from the latest 1x1 result."""

    @property
    def rows_changed(self) -> int:
        """Rows affected by the latest INSERT, UPDATE, or DELETE."""

    @property
    def description(self) -> list[tuple] | None:
        """PEP 249 result description for the most recent query, or None."""

    @property
    def columns(self) -> list[str] | None:
        """Column names of the most recent result, or None."""

    @property
    def current_result(self) -> Result:
        """The latest execute result; raises if no query has run."""

    @overload
    def arrow(self) -> pyarrow.Table:
        """Return the result as a pyarrow.Table."""

    @overload
    def arrow(
        self, query: str, parameters: list | tuple | dict[str, Any] | None = None
    ) -> pyarrow.Table:
        """Execute a query and return its complete Arrow table."""

    def arrow_batches(self) -> pyarrow.RecordBatchReader:
        """Stream one Arrow record batch per DuckDB chunk."""

    def df(self) -> pandas.DataFrame:
        """Return the result as a pandas.DataFrame."""

    @overload
    def pl(self, lazy: Literal[False] = False) -> polars.DataFrame: ...
    @overload
    def pl(self, lazy: Literal[True]) -> polars.LazyFrame: ...
    def pl(self, lazy: bool = False) -> polars.DataFrame | polars.LazyFrame:
        """Return the result as a polars DataFrame (or LazyFrame)."""

    def fetchnumpy(self) -> dict[str, numpy.ndarray]:
        """
        Return the result as a dict of column name -> numpy array. Numeric and temporal columns only; use arrow() for strings and nested types. NULLs become NaN in FLOAT/DOUBLE columns and raise elsewhere.
        """

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

    def iter_batches_torch(
        self, columns: Iterable[str] | None = None, device: torch.device | str | int | None = None
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Yield Torch tensors per chunk."""

    def iter_batches_jax(
        self, columns: Iterable[str] | None = None, device: jax.Device | None = None
    ) -> Iterator[dict[str, jax.Array]]:
        """Yield one dict per chunk: {name: jax.Array}."""

    def iter_batches_mlx(
        self, columns: Iterable[str] | None = None
    ) -> Iterator[dict[str, mx.array]]:
        """Yield MLX arrays per chunk."""

    def to_torch(
        self, columns: Iterable[str] | None = None, device: torch.device | str | int | None = None
    ) -> dict[str, torch.Tensor]:
        """Return selected columns as Torch tensors."""

    def to_jax(
        self, columns: Iterable[str] | None = None, device: jax.Device | None = None
    ) -> dict[str, jax.Array]:
        """Return selected columns as JAX arrays."""

    def to_mlx(self, columns: Iterable[str] | None = None) -> dict[str, mx.array]:
        """Return selected columns as MLX arrays."""

    def arrow_reader(
        self,
        query: str,
        parameters: list | tuple | dict[str, Any] | None = None,
        rows_per_batch: int = 1000000,
    ) -> pyarrow.RecordBatchReader:
        """Execute a query and stream batches of up to ``rows_per_batch`` rows."""

    def to_numpy(self, columns: Iterable[str] | None = None) -> dict[str, numpy.ndarray]:
        """
        Drain the latest result into a {name: numpy.ndarray} dict. Raises on VARCHAR/LIST/STRUCT/MAP; use arrow() for those.
        """

    def create_function(
        self,
        name: str,
        fn: Callable,
        parameters: list[str] | dict[str, str] | None = None,
        return_type: str | None = None,
        varargs: str | None = None,
        init: Callable[[], Any] | None = None,
        *,
        volatile: bool = False,
        special_handling: bool = False,
    ) -> None:
        """
        Register a chunk-wise NumPy scalar UDF. Types may be explicit or inferred. Rows with a NULL argument are not passed to fn and yield NULL; with special_handling, inputs and output use ``(values, validity)`` tuples instead.
        """

    def create_arrow_function(
        self,
        name: str,
        fn: Callable,
        parameters: list[str] | dict[str, str],
        return_type: str,
        record_batch: bool = False,
    ) -> None:
        """
        Register an Arrow scalar UDF. The callable receives Arrow arrays, a named array dict, or a RecordBatch, and returns an Arrow array.
        """

    def create_aggregate_function(
        self, name: str, cls: type, parameters: list[str] | dict[str, str], return_type: str
    ) -> None:
        """
        Register an aggregate class defining update, combine and finalize. Rows with a NULL argument are skipped. combine(other) merges another partial state into self without modifying other.
        """

    def create_table_function(
        self, name: str, factory: Callable, parameters: list[str], columns: dict[str, str]
    ) -> None:
        """Register a table function yielding rows matching ``columns``."""

    def appender(
        self, table: str, schema: str | None = None, catalog: str | None = None
    ) -> Appender:
        """Open a bulk appender for ``table``."""

    def interrupt(self) -> None:
        """Request cancellation of a running query; thread-safe."""

    def progress(self) -> tuple[float, int, int]:
        """
        Return ``(percentage, rows_processed, total_rows)``; percentage is -1 until estimated.
        """

    def get_profiling_info(self) -> dict[str, Any] | None:
        """Return the latest profiling tree, or None when profiling is disabled."""

    def set_profile_sink(
        self,
        sink: Callable[[str, dict[str, Any]], None] | None,
        *,
        sample: int = 1,
        mode: str = "standard",
    ) -> None:
        """
        Call ``sink(query, info)`` after every Nth materialized query. Pass None to detach. Streaming queries are skipped.
        """

    def register_arrow(self, name: str, obj: Any) -> None:
        """
        Register an Arrow C stream source as table ``name`` (matched case-insensitively). Tables and data frames are re-read by every query; readers and query results are one-shot and can be scanned once.
        """

    def load_extension(self, name: str) -> None:
        """
        Load an extension by name or path. Unsigned extensions require allow_unsigned_extensions=True when connecting.
        """

    def close(self) -> None:
        """Close the connection."""

    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

def connect(database: str = ":memory:", config: dict[str, str] | None = None) -> Connection:
    """Open a database with an optional string-valued configuration."""

def config_options() -> list[tuple[str, str]]:
    """
    Return ``(name, description)`` for every DuckDB setting accepted by :func:`connect`.
    """
