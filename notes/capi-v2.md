# DuckDB C API v2

*Reviewed 2026-11-14 against DuckDB `5c8008b4`.*

DuckDB C API v2 is a redesigned C interface in `duckdb_v2.h`. Its version is
independent of the DuckDB engine version: a DuckDB 2.1 development build can
provide C API v1 and C API v2 at the same time. Ducky is migrating to v2: see
[Migration status](#migration-status).

## What v2 changes

V1 grew incrementally and mixes conventions: some calls return `duckdb_state`,
some return null handles, some attach errors to results, and strings may be
owned, borrowed, or null-terminated. Several public structures also retain
historical fields.

V2 applies one model across the API:

- Resources are opaque handles with explicit destroy functions.
- Calls return a categorized `DUCKDB_V2_ERROR` and can provide structured error
  information.
- Strings are length-delimited, so embedded NUL bytes are unambiguous.
- An environment owns database creation and shared file-conflict detection.
- Logical types, values, options, schemas, catalogs, functions, and filesystems
  use consistent handle and error conventions.

These changes mainly improve correctness, diagnostics, and maintainability.
They do not make vector or chunk access inherently faster.

## Stability

Every v2 function is unstable. DuckDB PR #27410 ("Mark C API v2 unstable",
2026-10-16) demoted the whole surface, which had been marked stable at v2.0.0,
because it is still being worked on; the `v2.0-cyanoptera` release branch
matches. Consumers must define `DUCKDB_V2_API_ALLOW_UNSTABLE=1` to see any
declaration, and the surface keeps changing (`duckdb_v2_str` parameters were
removed on 2026-11-05).

Ducky statically links its pinned DuckDB revision, so it can compile against
the unstable surface, but each pin bump may break v2 code. V1 remains supported
and mature.

## Query and result ownership

A v2 result is an incremental cursor over one connection. `result_step` runs a
bounded unit of execution without blocking and reports `CHUNK`, `WAITING`,
`FINISHED`, or `CANCELLED`; `result_wait` blocks until a step can make progress,
and `result_fetch_chunk` and `result_drain` are blocking conveniences.
Cancellation is a status rather than an error. While a result is live, the
connection rejects another query with `DUCKDB_V2_ERROR_RESOURCE_IN_USE`.

This fits ducky's asynchronous execution better than v1, which ducky drives
with pending results, `execute_tasks` budgets, and a sleep backoff while workers
own the remaining tasks. Actual concurrency still requires separate
connections, created by the shared `Database` and pool proposed in
[capi-gaps.md](capi-gaps.md).

## Relevant additions for ducky

- **Arrow results.** `prepared_statement_execute_arrow` and
  `statement_execute_arrow` take a rows-per-batch limit and produce Arrow arrays
  through DuckDB's native Arrow result format. The result steps like a chunk
  result and converts to an `ArrowArrayStream` with
  `arrow_result_to_arrow_c_stream`. It replaced the v1 Arrow collector patch
  ducky carried.
- **Arrow importer and exporter.** An importer resolves a schema once and splits
  long arrays into chunks of a given batch size, carrying short remainders into
  the next array. It covers the windowing that `append_arrow` and the Arrow scan
  build on `patches/0007-capi-vector-reference-slice.patch`.
- **Table functions.** A claim batch callback hands work to threads one batch
  at a time; `bind_set_order_preservation(NO_ORDER)` lets such a scan run in
  parallel into results, INSERT, and COPY, and a partition data callback keeps
  insertion order while running in parallel. A stats callback lets the
  optimizer prune scans and answer `min()` and `max()`; a filter pushdown
  callback lets the function inspect and accept the query's predicates.
- **Instances and connections.** Environments, instances, and
  `connection_create` support one database shared by several connections.
- **Collections as statement inputs.** `statement_add_collection` exposes a
  column data collection to one statement as a named common table expression,
  read again on every execution. Collections support parallel append and scan.
- **Type parsing.** `factory_create_type_from_text` parses SQL type text,
  including nested, enum, and catalog types. It would replace `TypeParser`'s
  `SELECT CAST(NULL AS ...)` query on a private connection.
- **Smaller additions.** Typed query metrics (`connection_get_query_metrics`),
  a thread-safe progress snapshot (`connection_progress_get`), scalar functions
  whose return type depends on their arguments (a resolve types callback),
  `vector_cast` and `value_cast` with SQL `CAST` and `TRY_CAST` semantics,
  generated-column introspection (covered for v1 by
  `patches/0004-capi-column-is-generated.patch`), multi-file reader functions,
  and filesystem path helpers such as glob.

## Gaps

- V2 has no appender. `statement_add_collection` with `INSERT ... SELECT`
  covers bulk columnar inserts, but not row-at-a-time appends or `DEFAULT`
  handling.
- `result_fetch_chunk` and `result_step` return a new chunk per call; nothing
  fetches into a reused chunk as `patches/0002-capi-fetch-chunk-into.patch`
  does for v1.
- Vectors have no offset-and-count reference; zero-copy windows of a chunk need
  the Arrow importer or a patch.
- Registering a Python-backed filesystem still needs a C++ adapter.
- V1 and v2 handles do not convert into each other through the API.
  `patches/0010-capi-v2-v1-bridge.patch` gets the v1 handles of a v2 connection
  and instance, so v1 functions keep working on a database that v2 owns.

## Migration cost

A complete migration would touch nearly every binding subsystem:

- database and connection ownership;
- prepared, pending, and streaming query execution;
- result, chunk, vector, type, and value conversion;
- Arrow import and export;
- appenders, which v2 lacks, and Python UDF registration;
- errors, progress, profiling, and extension loading;
- the carried DuckDB patches.

DuckDB exports both symbol families. With the bridge patch, subsystems move one
at a time on a database that v2 owns.

## Migration status

Ducky creates every database as a v2 instance in one process-wide environment,
attaches the path as its default database, and connects through v2. The bridge
hands the v1 handles to everything not yet migrated. Because the environment
detects file conflicts across instances, opening a file that the process
already has open now fails; v1 opened a second instance on it.

`Connection.arrow()` runs through `statement_execute_arrow` with parameters
converted to v2 values, and steps the result to its end with the GIL released.
Against the retired v1 collector, 10M-row results take about 10% longer. A v2
result drains rather than retains, so `max_streaming_buffer_size` (10 MiB by
default) caps its queued arrays and parks workers until the consumer pops one;
with the cap lifted, v2 matches v1. Lifting the cap for each Arrow query
through the connection's config would recover this. A trivial query takes about
64 µs instead of 31 µs because each call parses and plans again; a v2
prepared-statement cache would recover that.

The appender reads its target's columns through `connection_describe_table`,
which reports generated columns, and the Arrow scan is a v2 table function and
replacement scan. Each scan worker converts vector-size windows of the current
batch with its own Arrow importer, outside the scan's lock; windows are Arrow
slices of the batch, so neither the importer nor `append_arrow`, which windows
long batches the same way for the v1 appender, needs a vector slice. Against
the v1 scan, string columns convert 3.4x faster and numeric scans match.
`count(*)` converts one column it does not read, since a v2 table function
cannot tell that projection from a scan of its first column.

v2 table functions let a prepared plan be reused, where v1 rebound on every
execution, so a plan keeps the Arrow schema it was bound with. A ducky
`PreparedStatement` therefore prepares again after any Arrow registration on its
connection; SQL `PREPARE` reports the changed schema instead.

Next candidates:

- `result_step` for the asynchronous driver, replacing pending results, the
  `execute_tasks` budget, and the sleep backoff.
- Python UDFs as v2 scalar, aggregate, and table functions, with
  `factory_create_type_from_text` replacing `TypeParser`'s private connection.
  That retires `duckdb_v2_instance_open_v1`.
- Query execution and results, which would leave the v1 connection to the
  appender alone.

Patch 0002 (fetch into a reused chunk) has no v2 replacement. Patch 0008
changes the engine and applies to both APIs.
