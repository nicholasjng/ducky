"""ducky: tiny nanobind bindings for the DuckDB C API"""

import enum
from collections.abc import Callable, Coroutine, Iterable, Iterator
from types import TracebackType
from typing import Any, Literal, overload

import jax
import mlx.core
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

    def decimal_scale(self, key: int | str) -> int:
        """
        Return the scale (digits after the decimal point) for a DECIMAL column. Raises Error if the column is not DECIMAL.
        """

    def dlpack(self, key: int | str) -> Any:
        """
        Return the column at `key` as an object implementing the DLPack protocol (__dlpack__ / __dlpack_device__), without going through numpy. Flat numeric/temporal types only.
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

    def fetchitem(self) -> Any:
        """
        Return the lone scalar of a 1-row x 1-column result (e.g. a COUNT(*) query). Raises a ducky.Error unless the result has exactly one column and yields exactly one row.
        """

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

    def iter_batches_torch(
        self, columns: Iterable[str] | None = None, device: torch.device | str | int | None = None
    ) -> Iterator[dict[str, torch.Tensor]]:
        """
        Yield one dict per chunk: {name: torch.Tensor}. Zero-copy on CPU via DLPack.
        """

    def iter_batches_jax(
        self, columns: Iterable[str] | None = None, device: jax.Device | None = None
    ) -> Iterator[dict[str, jax.Array]]:
        """Yield one dict per chunk: {name: jax.Array}."""

    def iter_batches_mlx(
        self, columns: Iterable[str] | None = None
    ) -> Iterator[dict[str, mlx.core.array]]:
        """
        Yield one dict per chunk: {name: mlx.core.array}. Zero-copy on CPU via DLPack.
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

    def to_mlx(self, columns: Iterable[str] | None = None) -> dict[str, mlx.core.array]:
        """Eagerly concatenate all chunks into {name: mlx.core.array}."""

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

    def append_arrow(self, source: Any) -> None:
        """
        Append from an Arrow PyCapsule object (__arrow_c_stream__ — pyarrow Table / RecordBatchReader / polars / pandas-3 / a ducky Result — or __arrow_c_array__ — a pyarrow RecordBatch). Columns must line up positionally with the target table and match its types. Covers VARCHAR / LIST / STRUCT, unlike the ndarray path of append_columns.
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

class PreparedStatement:
    """
    A SQL statement compiled once via Connection.prepare(), executable repeatedly with different parameters without re-parsing the query.
    """

    def execute(
        self, parameters: list | tuple | dict[str, Any] | None = None, streaming: bool = False
    ) -> Result:
        """
        Bind `parameters` (positional list/tuple, named dict, or None) and run the statement, returning its Result. Pass `streaming=True` for a lazy streaming result whose chunks are produced on demand — peak memory stays bounded to one chunk, useful for iter_batches_torch / iter_batches_jax / iter_batches_mlx on large queries. A streaming result is single-pass; consume it via fetch* or iter_batches*, not both.
        """

    def executemany(self, parameters: Iterable[list | tuple | dict[str, Any]]) -> None:
        """
        Run the statement once per parameter set, discarding each result. The fast path for batched INSERT/UPDATE/DELETE.
        """

    @property
    def num_parameters(self) -> int:
        """Number of bind parameters in the statement."""

    def parameter_name(self, index: int) -> str | None:
        """
        Name of the parameter at `index` (1-based), or None if the index is out of range or the parameter is positional.
        """

    @property
    def columns(self) -> list[str]:
        """
        Result column names, known ahead of execution (empty for statements that produce no result set).
        """

    @property
    def types(self) -> list[str]:
        """Result column type names, known ahead of execution."""

    @property
    def statement_type(self) -> str:
        """The statement kind: 'SELECT', 'INSERT', 'UPDATE', etc."""

    def close(self) -> None:
        """Destroy the prepared statement."""

    def __enter__(self) -> PreparedStatement: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        pass

class PendingState(enum.Enum):
    """State of an async query's executor after a tick."""

    READY = 0
    """The result is ready to materialize."""

    NOT_READY = 1
    """More tasks remain; tick again."""

    ERROR = 2
    """Execution failed; see .error()."""

    NO_TASKS = 3
    """Workers own the outstanding tasks; yield, then tick again."""

class PendingResult:
    """
    A steppable handle over an in-flight async query. Internal: built by Connection.make_pending and driven by ducky._aio.
    """

    def execute_task(self) -> PendingState:
        """
        Advance the executor by one task (GIL released) and return the new state.
        """

    def error(self) -> str:
        """DuckDB's message for the most recent error."""

    def materialize(self) -> Result:
        """Materialize the finished result; call once execute_task reports READY."""

    def drain(self) -> None:
        """Cancellation teardown: interrupt the query and drain the executor."""

class Connection:
    """A connection to a DuckDB database."""

    def execute(
        self,
        query: str,
        parameters: list | tuple | dict[str, Any] | None = None,
        streaming: bool = False,
    ) -> Connection:
        """
        Execute a query, optionally with positional parameters, and return self for chaining. Pass `streaming=True` for a lazy streaming result whose chunks are produced on demand — peak memory stays bounded to one chunk, useful for iter_batches_torch / iter_batches_jax / iter_batches_mlx on large queries. A streaming result is single-pass; consume it via fetch* or iter_batches*, not both.
        """

    def sql(self, query: str, streaming: bool = False) -> Result:
        """Run a query and return its Result. See execute() for `streaming`."""

    def query(self, query: str, streaming: bool = False) -> Result:
        """Alias for sql()."""

    def aexecute(
        self,
        query: str,
        parameters: list | tuple | dict[str, Any] | None = None,
        streaming: bool = False,
    ) -> Coroutine[Any, Any, Result]:
        """
        Async variant of execute(): drives the query on the event loop, ticking the executor one task at a time off-thread so the loop stays responsive and a cancelled await interrupts the query. Resolves to the Result directly (it does not set current_result). Requires a running event loop.
        """

    def asql(self, query: str, streaming: bool = False) -> Coroutine[Any, Any, Result]:
        """Async variant of sql(); see aexecute()."""

    def make_pending(
        self,
        query: str,
        parameters: list | tuple | dict[str, Any] | None = None,
        streaming: bool = False,
    ) -> PendingResult:
        """
        Low-level: build a steppable PendingResult for the ducky._aio drivers. Prefer aexecute() / asql().
        """

    def prepare(self, query: str) -> PreparedStatement:
        """
        Compile `query` once into a PreparedStatement for repeated execution with different parameters, avoiding per-call re-parsing and planning.
        """

    def fetchone(self) -> tuple | None: ...
    def fetchmany(self, size: int = 1) -> list[tuple]: ...
    def fetchall(self) -> list[tuple]: ...
    def fetchitem(self) -> Any:
        """
        Return the lone scalar of a 1-row x 1-column result (e.g. a COUNT(*) query). Raises a ducky.Error unless the result has exactly one column and yields exactly one row.
        """

    @property
    def description(self) -> list[tuple] | None: ...
    @property
    def columns(self) -> list[str] | None: ...
    @property
    def current_result(self) -> Result:
        """
        The most recent result produced by execute(); raises Error if no query has run yet. The conversion accessors (arrow/df/pl/...) delegate to it.
        """

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

    def iter_batches_torch(
        self, columns: Iterable[str] | None = None, device: torch.device | str | int | None = None
    ) -> Iterator[dict[str, torch.Tensor]]:
        """
        Yield one dict per chunk: {name: torch.Tensor}. Zero-copy on CPU via DLPack.
        """

    def iter_batches_jax(
        self, columns: Iterable[str] | None = None, device: jax.Device | None = None
    ) -> Iterator[dict[str, jax.Array]]:
        """Yield one dict per chunk: {name: jax.Array}."""

    def iter_batches_mlx(
        self, columns: Iterable[str] | None = None
    ) -> Iterator[dict[str, mlx.core.array]]:
        """
        Yield one dict per chunk: {name: mlx.core.array}. Zero-copy on CPU via DLPack.
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

    def to_mlx(self, columns: Iterable[str] | None = None) -> dict[str, mlx.core.array]:
        """Eagerly concatenate all chunks into {name: mlx.core.array}."""

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

    def create_arrow_function(
        self,
        name: str,
        fn: Callable,
        parameters: list[str] | dict[str, str],
        return_type: str,
        record_batch: bool = False,
    ) -> None:
        """
        Register a Python callable as a DuckDB scalar function backed by the Arrow C-API path; supports VARCHAR, LIST, STRUCT, DECIMAL, MAP, and any other DuckDB type. `parameters` is a list of DuckDB type strings or a dict of {name: type_string}. By default `fn` is called with one `pyarrow.Array` per column (positional, or a {name: Array} dict when `parameters` is a dict). Pass `record_batch=True` to receive a single `pyarrow.RecordBatch` instead. `fn` must return a `pyarrow.Array`.
        """

    def create_aggregate_function(
        self, name: str, cls: type, parameters: list[str] | dict[str, str], return_type: str
    ) -> None:
        """
        Register a Python class as a DuckDB aggregate function. `cls` must have `__init__`, `update(self, *arrays)`, and `finalize(self) -> scalar` methods. An optional `combine(self, other)` method enables parallel aggregate execution.
        """

    def create_table_function(
        self, name: str, factory: Callable, parameters: list[str], columns: dict[str, str]
    ) -> None:
        """
        Register a Python generator function as a DuckDB table function. `factory` is called with the SQL arguments and must return a generator that yields one tuple per row. `columns` is an ordered dict {column_name: type_string} declaring the output schema.
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
