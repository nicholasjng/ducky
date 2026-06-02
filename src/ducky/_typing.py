"""Typing surface for `ducky.connect(...)`.

`DuckDBConfig` declares the most common keys DuckDB accepts on
`duckdb_set_config`. The runtime accepts any string-keyed mapping — DuckDB
validates names at open time, and `ducky.config_options()` returns the full
runtime list. This module exists so IDEs and type checkers can autocomplete
the curated keys (and, where DuckDB itself defines a closed set, the
allowed values) and catch typos at static-analysis time.

Used via :pep:`692` `Unpack[DuckDBConfig]` on the `**config` parameter of
`ducky.connect`, so call sites read as plain keyword arguments:

    ducky.connect(memory_limit="2GB", threads=4)
"""

from __future__ import annotations

from typing import Literal, TypedDict


class DuckDBConfig(TypedDict, total=False):
    """A curated subset of DuckDB global configuration options.

    Values are given as their natural Python type — `int` for counts, `bool`
    for flags, `str` for sizes / paths / enums (e.g. `"2GB"`, `"READ_ONLY"`).
    `ducky.connect` coerces them to the string form `duckdb_set_config`
    expects (`True` → `"true"`, `4` → `"4"`); DuckDB itself validates the
    names and values at open time. See `ducky.config_options()` for the full
    runtime list of accepted keys.
    """

    # ── Closed value sets (Literal) ───────────────────────────────────────
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

    # ── Counts ────────────────────────────────────────────────────────────
    threads: int
    """Number of worker threads. `1` disables parallelism."""

    worker_threads: int
    """Alias of `threads`."""

    # ── Free-form string values ───────────────────────────────────────────
    memory_limit: str
    """e.g. `"2GB"`, `"512MB"`."""

    max_memory: str
    """Alias of `memory_limit`."""

    temp_directory: str
    """Directory for spilled intermediates."""
