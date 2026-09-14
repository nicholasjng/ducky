"""Types for DuckDB connection settings."""

from typing import Literal, TypedDict


class DuckDBConfig(TypedDict, total=False):
    """Common DuckDB settings accepted by :func:`ducky.connect`."""

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

    allow_unsigned_extensions: bool
    """Permit unsigned extensions. This must be set when connecting."""

    enable_progress_bar: bool
    """Show a progress bar for long-running queries."""

    threads: int
    """Number of worker threads; ``1`` disables parallelism."""

    worker_threads: int
    """Alias of :attr:`threads`."""

    memory_limit: str
    """e.g. ``"2GB"``, ``"512MB"``."""

    max_memory: str
    """Alias of :attr:`memory_limit`."""

    temp_directory: str
    """Directory for spilled intermediates."""
