"""ducky: tiny nanobind bindings for the DuckDB C API"""

import types
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import numpy

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

    def arrow(self) -> Any:
        """Return the result as a pyarrow.Table."""

    def df(self) -> Any:
        """Return the result as a pandas.DataFrame."""

    def pl(self, lazy: bool = False) -> Any:
        """Return the result as a polars DataFrame (or LazyFrame)."""

    def fetchnumpy(self) -> dict[str, numpy.ndarray]:
        """Return the result as a dict of column name -> numpy array."""

    def chunks(self) -> Iterator[Chunk]:
        """Iterate over the result one Chunk at a time. Drains the result."""

    def iter_batches(
        self, columns: Iterable[str] | None = None, with_validity: bool = False
    ) -> Iterator[dict[str, Any]]:
        """
        Yield one dict per chunk: {name: ndarray}, or {name: (values, mask)} if with_validity=True.
        """

    def to_numpy(self, columns: Iterable[str] | None = None) -> dict[str, numpy.ndarray]:
        """Eagerly concatenate all chunks into {name: numpy.ndarray}."""

    def to_torch(self, columns: Iterable[str] | None = None, device: Any = None) -> dict[str, Any]:
        """Eagerly concatenate all chunks into {name: torch.Tensor}."""

    def to_jax(self, columns: Iterable[str] | None = None, device: Any = None) -> dict[str, Any]:
        """Eagerly concatenate all chunks into {name: jax.Array}."""

    def __iter__(self) -> Iterator[tuple]: ...
    def __next__(self) -> tuple: ...

class Connection:
    """A connection to a DuckDB database."""

    def execute(self, query: str, parameters: list | None = None) -> Connection:
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
    def arrow(self) -> Any:
        """Return the last result as a pyarrow.Table."""

    def df(self) -> Any:
        """Return the last result as a pandas.DataFrame."""

    def pl(self, lazy: bool = False) -> Any:
        """Return the last result as a polars DataFrame (or LazyFrame)."""

    def fetchnumpy(self) -> dict[str, numpy.ndarray]:
        """Return the last result as a dict of column name -> numpy array."""

    def chunks(self) -> Iterator[Chunk]:
        """Iterate over the last result one Chunk at a time."""

    def iter_batches(
        self, columns: Iterable[str] | None = None, with_validity: bool = False
    ) -> Iterator[dict[str, Any]]:
        """Yield one dict per chunk for the last result."""

    def to_numpy(self, columns: Iterable[str] | None = None) -> dict[str, numpy.ndarray]:
        """Return the last result as {name: numpy.ndarray}."""

    def to_torch(self, columns: Iterable[str] | None = None, device: Any = None) -> dict[str, Any]:
        """Return the last result as {name: torch.Tensor}."""

    def to_jax(self, columns: Iterable[str] | None = None, device: Any = None) -> dict[str, Any]:
        """Return the last result as {name: jax.Array}."""

    def create_function(
        self, name: str, fn: Callable, parameters: list[str] | dict[str, str], return_type: str
    ) -> None:
        """
        Register a Python callable as a DuckDB scalar function. `parameters` is a list of type strings (positional call) or a dict of {name: type_string} (dict-style call). Inputs arrive as zero-copy 1-D ndarrays; `fn` must return one ndarray of length chunk_size and matching dtype.
        """

    def close(self) -> None:
        """Close the connection."""

    def __enter__(self) -> Connection: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None: ...

def connect(database: str = ":memory:") -> Connection:
    """Open `database` (default in-memory) and return a Connection."""
