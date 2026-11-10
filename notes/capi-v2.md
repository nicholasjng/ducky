# DuckDB C API v2

*Reviewed 2026-11-10 against DuckDB `77fa3df`.*

DuckDB C API v2 is a redesigned C interface in `duckdb_v2.h`. Its version is
independent of the DuckDB engine version: a DuckDB 2.1 development build can
provide C API v1 and C API v2 at the same time. Ducky currently uses v1 through
`duckdb.h`.

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
  `arrow_result_to_arrow_c_stream`. This covers what
  `patches/0001-capi-arrow-collector.patch` adds to v1.
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
- V1 and v2 handles do not convert into each other, so both cannot share one
  database instance.

## Migration cost

A complete migration would touch nearly every binding subsystem:

- database and connection ownership;
- prepared, pending, and streaming query execution;
- result, chunk, vector, type, and value conversion;
- Arrow import and export;
- appenders, which v2 lacks, and Python UDF registration;
- errors, progress, profiling, and extension loading;
- the carried DuckDB patches.

DuckDB exports both symbol families, but without handle conversion a v2 path
must own its database end to end. A narrow vertical prototype is safer than
converting isolated helpers.

## Recommendation

Keep the v1 implementation while v2 is unstable.

Prototype behind a flag where v2 removes code or a patch: Arrow results, which
would retire patch 0001, and `result_step` for the asynchronous driver.
Benchmark against the v1 path and verify error and lifetime behavior under
sanitizers. Revisit batch-producing table functions once their callbacks
stabilize; they are the largest functional gain.
