"""Typing surface for :func:`ducky.connect`.

:class:`DuckDBConfig` is a curated :class:`~typing.TypedDict` of common DuckDB settings, used via :pep:`692` ``Unpack[DuckDBConfig]`` on the ``**config`` parameter of :func:`ducky.connect` so call sites read as plain keyword arguments::

    ducky.connect(memory_limit="2GB", threads=4)

The runtime accepts any string-keyed mapping; DuckDB validates names and values at open time.
See :func:`ducky.config_options` for the full runtime list of accepted keys.
"""

from typing import Literal, TypedDict


class DuckDBConfig(TypedDict, total=False):
    """A curated subset of DuckDB global configuration options.

    Values are given as their natural Python type — ``int`` for counts, ``bool`` for flags, ``str`` for sizes / paths / enums (e.g. ``"2GB"``, ``"READ_ONLY"``).
    :func:`ducky.connect` coerces them to the string form ``duckdb_set_config`` expects (``True`` → ``"true"``, ``4`` → ``"4"``); DuckDB validates names and values at open time.
    See :func:`ducky.config_options` for the full runtime list.
    """

    access_mode: Literal["AUTOMATIC", "READ_ONLY", "READ_WRITE"]
    """How the database file is opened."""

    default_order: Literal["ASC", "DESC"]
    """Default sort order when unspecified."""

    default_null_order: Literal["NULLS_FIRST", "NULLS_LAST"]
    """NULL placement in default sort order."""

    preserve_insertion_order: bool
    """Keep result rows in insertion order when possible."""

    enable_object_cache: bool
    """Cache Parquet metadata across queries."""

    enable_external_access: bool
    """Gate httpfs / s3 / file:// scans."""

    autoload_known_extensions: bool
    """Load extensions on first reference."""

    autoinstall_known_extensions: bool
    """Install extensions on first reference."""

    enable_progress_bar: bool
    """Show a progress bar for long-running queries."""

    threads: int
    """Number of worker threads.
    ``1`` disables parallelism."""

    worker_threads: int
    """Alias of :attr:`threads`."""

    memory_limit: str
    """e.g. ``"2GB"``, ``"512MB"``."""

    max_memory: str
    """Alias of :attr:`memory_limit`."""

    temp_directory: str
    """Directory for spilled intermediates."""
