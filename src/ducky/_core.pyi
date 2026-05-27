"""ducky: tiny nanobind bindings for the DuckDB C API"""

import types

__duckdb_version__: str = "v1.6.2-dev4685"

class Error(Exception):
    pass

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
    def description(self) -> object:
        """PEP 249 result description."""

    def fetchone(self) -> object:
        """Return the next row, or None."""

    def fetchmany(self, size: int = 1) -> list:
        """Return up to `size` rows."""

    def fetchall(self) -> list:
        """Return all remaining rows."""

    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object:
        """Export the result via the Arrow C stream (PyCapsule) interface."""

    def arrow(self) -> object:
        """Return the result as a pyarrow.Table."""

    def df(self) -> object:
        """Return the result as a pandas.DataFrame."""

    def pl(self, lazy: bool = False) -> object:
        """Return the result as a polars DataFrame (or LazyFrame)."""

    def fetchnumpy(self) -> object:
        """Return the result as a dict of column name -> numpy array."""

    def __iter__(self) -> object: ...
    def __next__(self) -> object: ...

class Connection:
    """A connection to a DuckDB database."""

    def execute(self, query: str, parameters: object | None = None) -> Connection:
        """
        Execute a query, optionally with positional parameters, and return self for chaining.
        """

    def sql(self, query: str) -> Result:
        """Run a query and return its Result."""

    def query(self, query: str) -> Result:
        """Alias for sql()."""

    def fetchone(self) -> object: ...
    def fetchmany(self, size: int = 1) -> list: ...
    def fetchall(self) -> list: ...
    @property
    def description(self) -> object: ...
    @property
    def columns(self) -> object: ...
    def arrow(self) -> object:
        """Return the last result as a pyarrow.Table."""

    def df(self) -> object:
        """Return the last result as a pandas.DataFrame."""

    def pl(self, lazy: bool = False) -> object:
        """Return the last result as a polars DataFrame (or LazyFrame)."""

    def fetchnumpy(self) -> object:
        """Return the last result as a dict of column name -> numpy array."""

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
