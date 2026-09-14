# DuckDB API gaps for data loading and wrangling

*Reviewed 2026-09-14 against DuckDB `625f75f`.*

This note tracks additions that would materially improve ducky's data-wrangling
and ML data-loading workflows. It excludes extension-author APIs whose main use
would be reproducing DuckDB internals in Python.

## Shared databases and connection pools

**Available C API:** `duckdb_create_instance_cache`,
`duckdb_get_or_create_from_cache`, `duckdb_destroy_instance_cache`, and
`duckdb_connect`.

Each ducky `Connection` currently owns both a database and one DuckDB
connection. Separate `connect(":memory:")` calls therefore see separate data,
and overlapping pending queries on one connection are invalid. In practice,
concurrent `aexecute()` calls can close or invalidate one another.

Expose a `Database` owner with `Database.connect()` and an optional bounded
pool. Each active query or loader worker would get its own connection while
sharing the catalog, buffer manager, and in-memory data. This also gives
`Connection.close()` simpler ownership semantics and provides the foundation
for asynchronous query concurrency.

## Streaming datasets

**Available C API:** pending and streaming execution, `duckdb_stream_fetch_chunk`,
Arrow conversion, progress, and interruption already provide the primitives.

`dataset()` currently materializes the complete query, creates backend arrays,
and gathers each fold into more arrays. Peak memory can therefore be several
times the source data, and training cannot start until the query finishes.

Add a rescannable loader that yields configurable batches and supports worker
and rank sharding, bounded shuffle buffers, epochs, prefetch, and cancellation.
NumPy batches can be filled directly from chunks; accelerator backends can
transfer coalesced batches from pinned host buffers. This is primarily a ducky
API gap rather than a missing DuckDB primitive.

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

DuckDB's C++ `TableFunction` also has filter-pushdown hooks and complex-filter
callbacks. The C API exposes projection pushdown but no equivalent filter
callback. A C API addition would let Python data sources prune files, row
groups, or remote requests before producing batches.

## Incremental Arrow registration

**Available C API:** Arrow schema and batch conversion plus custom table
functions.

The registered Arrow scan drains the entire source during table-function
initialization. That permits parallel, GIL-free reads after initialization, but
removes backpressure and buffers an unbounded reader before the query can make
progress.

Keep the buffered parallel mode and add a sequential incremental mode that owns
one live stream and converts batches on demand. Registration should optionally
take a callable that opens a fresh stream for each execution. This makes cached
and prepared statements safely rescannable and supports remote or generated
sources whose complete contents should never reside in memory.

## Structured types and nested values

**Available C API:** logical-type constructors and inspectors for LIST, ARRAY,
MAP, STRUCT, UNION, ENUM, and DECIMAL; nested `duckdb_value` constructors; and
`duckdb_bind_value`.

Ducky currently exposes schema types as strings and binds scalar Python values.
Introduce an immutable `LogicalType` tree and translate Python lists, tuples,
dictionaries, decimals, and UUIDs into typed `duckdb_value` objects. Structured
schema inspection would let loaders decide whether a column can use a dense
tensor, needs offsets and values, or should stay in Arrow. Nested parameter
binding would also remove SQL-literal workarounds for list and struct filters.

## Stable split keys

No new DuckDB API is required. `split()` hashes `row_number() OVER ()`, so fold
membership depends on scan order. Parallel scans, file ordering, or a changed
plan can move rows between folds despite an unchanged seed.

Accept a key expression and hash it with the seed. For sources without a stable
key, document that positional splits reproduce only when input order does. A
stable key is necessary for comparable training and evaluation runs.

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
- `duckdb_appender_create_query`, `duckdb_appender_add_column`, and
  `duckdb_append_value` could extend appender projections and nested rows. Arrow
  ingestion already covers the important columnar path.
- External task-state APIs could let caller-owned threads participate in
  DuckDB's scheduler. A connection pool and DuckDB's internal workers are a
  simpler first solution for Python training code.
- Custom copy functions, casts, configuration options, catalogs, and log
  storage mainly serve extension authors and do not justify public ducky APIs
  yet.

## Suggested order

1. Add `Database`, multiple connections, and a bounded pool.
2. Build a streaming loader on separate pooled connections.
3. Add stable split keys and slice unshuffled in-memory batches.
4. Add batch-producing table functions and incremental Arrow sources.
5. Expose structured logical types and nested parameter binding.
6. Revisit filesystem registration after native extensions prove insufficient.
