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
    A column-major batch of up to STANDARD_VECTOR_SIZE rows.

    Numeric and temporal columns are exposed as zero-copy ndarrays over DuckDB's internal buffers.
    """

    def __len__(self) -> int:
        """Number of rows."""

    @property
    def columns(self) -> list[str]:
        """Column names."""

    @property
    def types(self) -> list[str]:
        """Column type names."""

    def column(self, key: int | str) -> numpy.ndarray:
        """
        Return the column at `key` as a 1-D ndarray view.

        Parameters
        ----------
        key : int or str
            Column index or name.
        """

    def validity(self, key: int | str) -> numpy.ndarray | None:
        """
        Return a uint8 validity mask (1 = valid, 0 = NULL), or None if no nulls.
        """

    def decimal_scale(self, key: int | str) -> int:
        """Return the scale (digits after the decimal point) of a DECIMAL column."""

    def dlpack(self, key: int | str) -> Any:
        """
        Return the column as a DLPack object (``__dlpack__`` / ``__dlpack_device__``), without going through numpy.
        Flat numeric / temporal types only.
        """

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
        Return the lone scalar of a 1-row × 1-column result.

        Useful for ``COUNT(*)``-style queries.Raises :class:`Error` if the result is not exactly one row and one column.
        """

    def fetch_chunk(self) -> Chunk | None:
        """Pull the next :class:`Chunk`, or None at end of stream."""

    def to_numpy(self, columns: Iterable[str] | None = None) -> dict[str, numpy.ndarray]:
        """
        Drain the result into a {name: numpy.ndarray} dict. Raises on VARCHAR/LIST/STRUCT/MAP — use .arrow() for those.
        """

    def __arrow_c_stream__(self, requested_schema: Any = None) -> Any:
        """Export the result via the Arrow PyCapsule stream interface."""

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
        """
        Return the result as a dict of column name -> numpy array. Numeric / temporal columns only; use arrow() for strings / nested types.
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
    Bulk-insert handle for a target table.

    Supports row-at-a-time inserts, a columnar fast path for numeric / temporal ndarrays, and an Arrow ingest path that covers VARCHAR / LIST / STRUCT.
    """

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
        Append a batch of named columns; missing columns are filled with NULL.

        Parameters
        ----------
        columns : dict[str, numpy.ndarray]
            Column-name → 1-D ndarray of equal length.
        masks : dict[str, numpy.ndarray], optional
            Per-column validity (1-D bool / uint8; True = valid).
        """

    def append_arrow(self, source: Any) -> None:
        """
        Append from any object exposing the Arrow PyCapsule interface.

        Accepts ``__arrow_c_stream__`` (pyarrow Table / RecordBatchReader / polars / pandas-3 / a ducky :class:`Result`) or ``__arrow_c_array__`` (a pyarrow RecordBatch).
        Columns must line up positionally with the target table and match its types.
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
    A SQL statement compiled once via :meth:`Connection.prepare`.

    Executable repeatedly with different parameters without re-parsing.
    """

    def execute(
        self, parameters: list | tuple | dict[str, Any] | None = None, streaming: bool = False
    ) -> Result:
        """
        Bind ``parameters`` and run the statement, returning its :class:`Result`.

        Parameters
        ----------
        parameters : list | tuple | dict[str, Any], optional
            Positional sequence or named dict; ``None`` runs without binding.
        streaming : bool, default False
            Return a streaming result whose chunks are pulled lazily (peak memory stays bounded to one chunk).
            Single-pass; consume via ``fetch*`` or ``iter_batches_*``, not both.
        """

    def executemany(self, parameters: Iterable[list | tuple | dict[str, Any]]) -> None:
        """
        Run the statement once per parameter set, discarding each result.

        The fast path for batched ``INSERT`` / ``UPDATE`` / ``DELETE``.
        """

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
        """Statement kind: ``'SELECT'``, ``'INSERT'``, ``'UPDATE'``, …."""

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
    """Execution failed; see ``.error()``."""

    NO_TASKS = 3
    """Workers own the outstanding tasks; yield, then tick again."""

class PendingResult:
    """
    Steppable handle over an in-flight async query.

    Internal: built by :meth:`Connection.make_pending` and driven by the ``ducky._aio`` event-loop helpers.
    Prefer :meth:`Connection.aexecute` / :meth:`Connection.asql`.
    """

    def execute_task(self) -> PendingState:
        """
        Advance the executor by one task (GIL released) and return the new state.
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
    """
    A connection to a DuckDB database.

    Each :func:`connect` returns one connection that owns its own database handle; two ``connect(":memory:")`` calls are isolated.
    """

    def execute(
        self,
        query: str,
        parameters: list | tuple | dict[str, Any] | None = None,
        streaming: bool = False,
    ) -> Connection:
        """
        Execute a SQL query and stash the result for the ``fetch*`` methods.

        Repeated query text reuses a per-connection cache of prepared statements, skipping re-parsing and re-planning.

        Parameters
        ----------
        query : str
            SQL text.
            Single statement only.
        parameters : list | tuple | dict[str, Any], optional
            Positional sequence or named dict; ``None`` runs without binding.
        streaming : bool, default False
            Return a streaming result whose chunks are pulled lazily; useful for ``iter_batches_*`` on large queries.
            Single-pass; consume via ``fetch*`` or ``iter_batches_*``, not both.

        Returns
        -------
        Connection
            ``self``, so calls can be chained (e.g. ``con.execute(sql).fetchall()``).

        Examples
        --------
        >>> rows = con.execute("SELECT * FROM t WHERE id = ?", [42]).fetchall()
        """

    def sql(self, query: str, streaming: bool = False) -> Result:
        """
        Run a query and return its :class:`Result` directly.

        Unlike :meth:`execute`, ``sql`` does not stash the result on the connection.
        See :meth:`execute` for ``streaming``.
        """

    def query(self, query: str, streaming: bool = False) -> Result:
        """Alias for :meth:`sql`."""

    def aexecute(
        self,
        query: str,
        parameters: list | tuple | dict[str, Any] | None = None,
        streaming: bool = False,
    ) -> Coroutine[Any, Any, Result]:
        """
        Async variant of :meth:`execute`.

        Ticks the executor one task at a time off-thread so the event loop stays responsive; cancelling the await interrupts the query.
        Resolves to the :class:`Result` directly — does not set ``current_result``.
        """

    def asql(self, query: str, streaming: bool = False) -> Coroutine[Any, Any, Result]:
        """Async variant of :meth:`sql`. See :meth:`aexecute`."""

    def make_pending(
        self,
        query: str,
        parameters: list | tuple | dict[str, Any] | None = None,
        streaming: bool = False,
    ) -> PendingResult:
        """
        Build a steppable :class:`PendingResult` for the ``ducky._aio`` drivers.
        Low-level; prefer :meth:`aexecute` / :meth:`asql`.
        """

    def prepare(self, query: str) -> PreparedStatement:
        """
        Compile ``query`` once into a :class:`PreparedStatement` for repeated execution, avoiding per-call re-parsing and planning.
        """

    def fetchone(self) -> tuple | None:
        """
        Fetch one row from the most recent :meth:`execute` result. Delegates to :meth:`Result.fetchone`.
        """

    def fetchmany(self, size: int = 1) -> list[tuple]:
        """Fetch up to ``size`` rows. Delegates to :meth:`Result.fetchmany`."""

    def fetchall(self) -> list[tuple]:
        """Fetch all remaining rows. Delegates to :meth:`Result.fetchall`."""

    def fetchitem(self) -> Any:
        """
        Return the lone scalar of a 1-row × 1-column result.
        Delegates to :meth:`Result.fetchitem`.
        """

    @property
    def description(self) -> list[tuple] | None:
        """PEP 249 result description for the most recent query, or None."""

    @property
    def columns(self) -> list[str] | None:
        """Column names of the most recent result, or None."""

    @property
    def current_result(self) -> Result:
        """
        The most recent :class:`Result` from :meth:`execute`.

        Raises :class:`Error` if no query has run yet.
        The conversion accessors (``arrow`` / ``df`` / ``pl`` / …) delegate to this.
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
        """
        Return the result as a dict of column name -> numpy array. Numeric / temporal columns only; use arrow() for strings / nested types.
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

    def to_numpy(self, columns: Iterable[str] | None = None) -> dict[str, numpy.ndarray]:
        """
        Drain the latest result into a {name: numpy.ndarray} dict. Raises on VARCHAR/LIST/STRUCT/MAP — use .arrow() for those.
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
        Register a Python callable as a DuckDB scalar UDF.

        Inputs arrive as zero-copy 1-D ndarrays; ``fn`` must return one ndarray of matching dtype.

        Parameters
        ----------
        name : str
            Name to register the function under.
        fn : callable
            Called as ``fn(*args)`` (positional ``parameters``) or ``fn(kwargs)`` (dict ``parameters``).
        parameters : list[str] | dict[str, str], optional
            DuckDB type strings — list for positional, dict for keyword.
            If omitted, types are inferred from ``fn``'s annotations (``bool``/``int``/``float`` → ``BOOLEAN``/``BIGINT``/``DOUBLE``).
        return_type : str, optional
            Inferred from annotations if omitted.
        varargs : str, optional
            DuckDB type for a variable-arity function.
            Mutually exclusive with ``parameters``.
        init : callable, optional
            Zero-arg factory for per-worker-thread state.
            Its return value becomes the first positional arg of ``fn`` on every chunk.
        volatile : bool, default False
            Mark non-deterministic (e.g. RNG, clock) so the optimizer won't fold or cache it.
        special_handling : bool, default False
            NULL-aware: arguments arrive as ``(values, mask)`` tuples (``mask`` is a 1-D uint8 ndarray, 1 = valid) and the UDF emits NULLs via the same return shape.
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
        Register a scalar UDF backed by the Arrow C-API path.

        Supports VARCHAR, LIST, STRUCT, DECIMAL, MAP — anything DuckDB can round-trip through Arrow.
        ``fn`` is called with one ``pyarrow.Array`` per column (positional, or ``{name: Array}`` dict if ``parameters`` is a dict); set ``record_batch=True`` to receive one ``pyarrow.RecordBatch`` instead.
        Must return a ``pyarrow.Array``.
        """

    def create_aggregate_function(
        self, name: str, cls: type, parameters: list[str] | dict[str, str], return_type: str
    ) -> None:
        """
        Register a Python class as a DuckDB aggregate UDF.

        ``cls`` must define ``__init__``, ``update(self, *arrays)``, and ``finalize(self) -> scalar``.
        An optional ``combine(self, other)`` enables parallel aggregate execution.
        """

    def create_table_function(
        self, name: str, factory: Callable, parameters: list[str], columns: dict[str, str]
    ) -> None:
        """
        Register a Python generator as a DuckDB table function.

        ``factory`` is called with the SQL arguments and must return a generator that yields one tuple per row.
        ``columns`` is an ordered dict ``{column_name: type_string}`` declaring the output schema.
        """

    def appender(
        self, table: str, schema: str | None = None, catalog: str | None = None
    ) -> Appender:
        """
        Open an :class:`Appender` for bulk inserts into ``table``.

        The appender shares the connection's database handle, so the database stays open until the appender is closed.
        """

    def interrupt(self) -> None:
        """
        Best-effort cancel of any query running on this connection.

        Safe to call from another thread while :meth:`execute` is in flight.
        """

    def progress(self) -> tuple[float, int, int]:
        """
        Snapshot of the current query's progress.

        Returns
        -------
        tuple[float, int, int]
            ``(percentage, rows_processed, total_rows_to_process)``.
            ``percentage`` is ``-1.0`` until DuckDB has an estimate.
        """

    def get_profiling_info(self) -> dict[str, Any] | None:
        """
        Programmatic EXPLAIN ANALYZE for the most recent query.

        Returns a nested ``{"metrics": {str: str}, "children": [...]}`` dict, or ``None`` if profiling isn't enabled.
        For ergonomics use :func:`ducky.profile` (context manager) or :func:`ducky.format_profiling_info` (pretty-printer).
        Metric values are strings per the DuckDB C API; coerce numerics on the caller side.
        """

    def set_profile_sink(
        self,
        sink: Callable[[str, dict[str, Any]], None] | None,
        *,
        sample: int = 1,
        mode: str = "standard",
    ) -> None:
        """
        Install an always-on profile sink for every :meth:`execute` / :meth:`sql` on this connection.

        Parameters
        ----------
        sink : callable or None
            Called as ``sink(query, info)`` after every materialized query with the same nested dict :meth:`get_profiling_info` returns.
            Pass ``None`` to detach.
        sample : int, default 1
            Fire the sink every Nth query (useful in tight training loops).
        mode : str, default 'standard'
            DuckDB ``profiling_mode``. ``'detailed'`` adds per-operator counters like ``cpu_time``.

        Notes
        -----
        Streaming results (``streaming=True``) are skipped — their chunks haven't been pulled when ``execute`` returns.
        Sink exceptions are reported via ``PyErr_WriteUnraisable`` and do not propagate.
        """

    def register_arrow(self, name: str, obj: Any) -> None:
        """
        Register an Arrow-PyCapsule source as a table named ``name``.

        Accepts anything exposing ``__arrow_c_stream__`` (pyarrow Table, polars / pandas-3 DataFrame, …).
        The source is kept and re-streamed on each query via a replacement scan, so it must support being streamed more than once. Queries scan it in parallel; primitive columns are read zero-copy from the Arrow buffers, and each execution buffers the stream's batches (cheap chunk wraps for an in-memory table, a full materialization for an unbounded reader).
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
    Open ``database`` (default in-memory) and return a :class:`Connection`.

    Low-level entrypoint.
    The Python wrapper :func:`ducky.connect` accepts DuckDB settings as typed keyword arguments and is what most users want.

    Parameters
    ----------
    database : str, default ':memory:'
        File path, ``':memory:'``, or any DuckDB connection string.
    config : dict[str, str], optional
        DuckDB settings applied at open time.
        See :func:`config_options`.
    """

def config_options() -> list[tuple[str, str]]:
    """
    Return ``(name, description)`` for every DuckDB setting accepted by :func:`connect`.
    """
