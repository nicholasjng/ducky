# DuckDB API gaps for data loading and wrangling

*Reviewed 2026-11-10 against DuckDB `77fa3df`.*

This note tracks additions that would materially improve ducky's data-wrangling
and ML data-loading workflows. It excludes extension-author APIs whose main use
would be reproducing DuckDB internals in Python.

## Shared databases and connection pools

**Available C API:** `duckdb_create_instance_cache`,
`duckdb_get_or_create_from_cache`, `duckdb_destroy_instance_cache`, and
`duckdb_connect`.

Each ducky `Connection` currently owns both a database and one DuckDB
connection. Separate `connect(":memory:")` calls therefore see separate data,
and queries on one connection run one at a time, so concurrent `aexecute()`
calls on it take turns.

Expose a `Database` owner with `Database.connect()` and an optional bounded
pool. Each active query or loader worker would get its own connection while
sharing the catalog, buffer manager, and in-memory data. This also gives
`Connection.close()` simpler ownership semantics and provides the foundation
for asynchronous query concurrency.

## Batch-producing table functions

**Available C API:** table-function bind, global and local initialization,
named parameters, projection selection, cardinality estimates, and maximum
worker counts.

The current Python table function consumes one generator row at a time, runs on
one thread, and disables projection pushdown. Add an Arrow or NumPy batch
producer with explicit global and per-worker state. It should receive the
projected columns, advertise cardinality when known, and allow independent
partitions to run in parallel. This reduces Python calls and avoids constructing
values for columns the query does not use.

C API v1 exposes projection pushdown but no filter pushdown. C API v2, which is
unstable (see [capi-v2.md](capi-v2.md)), has a filter pushdown callback, and
claim batch, order preservation, partition data, and stats callbacks that fit a
parallel batch producer. Filter pushdown would let Python data sources prune
files, row groups, or remote requests before producing batches.

## Parallel Arrow conversion

**Available C API:** Arrow schema and batch conversion plus custom table
functions.

The registered Arrow scan pulls batches lazily, re-streams reusable sources on
every execution, and refuses a second scan of a one-shot reader. Workers emit
zero-copy windows of the current batch in parallel, but a single worker pulls
and converts each batch while holding the scan's lock, so conversion does not
overlap with production or with other conversions.

Convert outside the lock, keeping a bounded number of batches in flight, so
wide or nested batches convert on several threads. A Python callable that opens
a fresh stream for each execution would also make generated or remote sources
rescannable without a reusable object. C API v2's claim batch callback and
Arrow importer fit this design.

## Structured types and nested values

**Available C API:** logical-type constructors and inspectors for LIST, ARRAY,
MAP, STRUCT, UNION, ENUM, and DECIMAL, nested `duckdb_value` constructors, and
`duckdb_bind_value`.

Ducky currently exposes schema types as strings and binds scalar Python values.
Introduce an immutable `LogicalType` tree and translate Python lists, tuples,
dictionaries, decimals, and UUIDs into typed `duckdb_value` objects. Structured
schema inspection would let loaders decide whether a column can use a dense
tensor, needs offsets and values, or should stay in Arrow. Nested parameter
binding would also remove SQL-literal workarounds for list and struct filters.

## Python-backed filesystems

**C++ only:** `FileSystem::RegisterSubSystem` registers handlers for path
prefixes. The C API can use an existing configured filesystem but cannot
register one.

A filesystem adapter could make fsspec URLs directly readable by DuckDB and
unify authentication, caching, and file discovery with the surrounding Python
training stack. It needs either a stable C callback API upstream or a small C++
adapter tied to the pinned DuckDB revision. Python calls on every small read
would be expensive, so the adapter should favor large range reads and release
the GIL while waiting on storage.

This is lower priority than using DuckDB's native `httpfs` and cloud extensions.
It becomes worthwhile for protocols or credential systems those extensions do
not support.

## Lower-priority C API surface

- `duckdb_extract_statements` and `duckdb_prepare_extracted_statement` would
  support SQL scripts and statement-by-statement results. Useful for tooling,
  but peripheral to loaders.
- `duckdb_appender_create_query` and `duckdb_append_value` could extend
  appender projections and nested rows. Arrow ingestion already covers the
  important columnar path, and `append_columns` uses
  `duckdb_appender_add_column` to give omitted columns their defaults.
- External task-state APIs could let caller-owned threads participate in
  DuckDB's scheduler. A connection pool and DuckDB's internal workers are a
  simpler first solution for Python training code.
- Custom copy functions, casts, configuration options, catalogs, and log
  storage mainly serve extension authors and do not justify public ducky APIs
  yet.

## Suggested order

1. Add `Database`, multiple connections, and a bounded pool.
2. Add batch-producing table functions and parallel Arrow conversion.
3. Expose structured logical types and nested parameter binding.
4. Revisit filesystem registration after native extensions prove insufficient.
