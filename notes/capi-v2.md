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

A result of a SELECT always streams. A stream holds at most
`max_streaming_buffer_size` of rows the consumer has not taken and parks the
query's workers until the consumer steps; a statement that keeps the source
order of its rows then runs mostly on the consumer's thread. Statements without
rows are retained: they run to their end without a consumer. v2 has no call to
retain a SELECT's result, which reading it to its end in parallel needs;
`patches/0011-capi-v2-result-retain.patch` adds `result_retain`. See
[What ducky needs from v2](#what-ducky-needs-from-v2).

Actual concurrency still requires separate connections, created by the shared
`Database` and pool proposed in [capi-gaps.md](capi-gaps.md).

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
  including nested, enum, and catalog types, without the
  `SELECT CAST(NULL AS ...)` query ducky ran on a private connection.
- **Smaller additions.** Typed query metrics (`connection_get_query_metrics`),
  a thread-safe progress snapshot (`connection_progress_get`), scalar functions
  whose return type depends on their arguments (a resolve types callback),
  `vector_cast` and `value_cast` with SQL `CAST` and `TRY_CAST` semantics,
  generated-column introspection (covered for v1 by
  `patches/0004-capi-column-is-generated.patch`), multi-file reader functions,
  and filesystem path helpers such as glob.

## What ducky needs from v2

Ordered by what each would gain ducky. Each names the workaround ducky uses
until then.

1. **An appender.** Row-at-a-time appends with `DEFAULT` handling, and appending
   chunks or Arrow arrays to a table. `statement_add_collection` with
   `INSERT ... SELECT` covers bulk columnar inserts only. The appender is
   ducky's last user of the v1 connection, and the reason the bridge
   (`patches/0010-capi-v2-v1-bridge.patch`) exists at all.
2. **Retaining a result.** A SELECT's result always streams: it holds at most
   `max_streaming_buffer_size` of rows the consumer has not taken and parks the
   workers until the consumer steps, so an order-preserving query runs mostly
   on the consumer's thread. A heavy projection over a table, read in full,
   takes 645 ms streamed and 173 ms retained on 8 threads.
   `patches/0011-capi-v2-result-retain.patch` adds `result_retain`; a flag on
   the execute functions would do as well.
3. **Arrow conversion outside callbacks.** An Arrow exporter needs a callback's
   context, so a chunk result cannot be converted to Arrow with v2:
   `Result.arrow()` converts through v1 with the connection's Arrow options, on
   the bridged chunk and types. An exporter created from a connection would
   retire that, and with the appender, the bridge.
4. **Profiling metrics with their text.** `connection_get_query_metrics` keeps
   only numeric metrics. Against v1's profiling info for the same query, it
   lacks the SQL text, each operator's own name (`SEQ_SCAN` under a
   `TABLE_SCAN` node) and its `extra_info`, which carries the table, the pushed
   filters, projections, aggregates and estimated cardinality: what a profile
   is read for. Its names differ (`query.total_time`, `operator.timing`) and it
   has no grouped summary nodes, which a format change in ducky could absorb.
   String-valued metrics would let ducky move profiling to v2.
5. **Fetching into a caller-owned chunk.** `result_fetch_chunk` and
   `result_step` allocate a chunk per call. Reading a retained 10M-row result of
   two columns chunk by chunk, or into one Arrow batch, takes 15–20% longer than
   v1 with the fetch-into-chunk patch ducky carried. Chunks handed to Python
   cannot be reused, so this only pays for conversions that copy each chunk
   out: NumPy, Arrow, rows.
6. **A wake-up for asynchronous consumers.** `result_wait` blocks a thread, so
   an asyncio driver steps on a worker thread and hops back per tick. A callback
   or file descriptor signalled when a step can make progress would let the
   event loop step a result without a thread.
7. **Prepared-statement metadata.** A prepared statement does not report its
   parameter types or result schema; ducky keeps the parsed statement to bind it
   again with `statement_bind`, which costs a second bind per prepare.
8. **Smaller ones.**
   - A vector reference to an offset and count of another vector, for
     zero-copy windows of a chunk; ducky windows through the Arrow importer.
   - Telling a table function's `count(*)` from a scan of its first column;
     ducky's Arrow scan converts a column it does not read.
   - Registering a filesystem without a C++ adapter.
   - `value_get_ubigint` converts a query metric's DOUBLE duration, such as
     0.0003, to 0 without an error, so a caller that tries the integer getters
     first reads every timing as 0.

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

`Connection.arrow()` runs through `prepared_statement_execute_arrow` on the
cached statement of the query, with parameters converted to v2 values, retains
the result (patch 0011) and steps it to its end with the GIL released. A trivial
query takes 33 µs, as with v1's collector, and 10M rows of two columns take
28 ms.

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

Python table functions, scalar UDFs, and aggregates are v2 functions. Their types parse
through the connection's factory, which resolves catalog types in its open
transaction and runs no query, so the private connection ducky parsed types on,
and the v1 database handle it needed, are gone. A table function writes each
chunk through cached data pointers, about 1.4x faster than through v1's
per-cell vector calls. V2 hands a scalar function its arguments unflattened, so
the NumPy trampoline flattens them; an Arrow UDF converts through an exporter
and importer that each worker creates once. Both match their v1 speed, as do
aggregates. A v2 aggregate's destroy callback is documented to run only on
states discarded without being finalized, but it is the engine's state
destructor, as v1's is, so it runs on every state.

Queries run through v2 prepared statements, cached per query text as before;
`PreparedStatement` holds one too. A cached statement keeps the parameter types
its plan assumed, read once with `statement_bind`, and executions convert ints
and floats to them, so a plan is reused instead of bound again: a
parameterized INSERT takes 76 µs instead of 152. v2 binds each execution's
values afresh, so nothing carries over between executemany rows.

A materialized result runs to its end with the GIL released, its result
retained (patch 0011) and stepped into chunks that the `Result` then hands out
without copying. Retaining matters: a heavy projection over a table read in
full took 645 ms streamed and takes 173 ms retained, on 8 threads. A streaming
result runs as it is fetched, so runtime errors surface on fetch, as with v1.
Async queries step the same way: `execute_tasks` runs steps and waits in
`result_wait` while workers own the remaining tasks, replacing v1's pending
results and the sleep backoff. Interrupting is v2's `connection_interrupt`,
which a result reports as cancelled rather than failed.

Against v1, materialized and streaming reads of 10M rows match or are faster
(a streaming `to_numpy` takes 30 ms instead of 46), except that reading a
retained result chunk by chunk or as one Arrow batch takes 15–20% longer: v2
allocates each chunk, where ducky's v1 patch fetched into a reused one. A
trivial query takes 34 µs instead of 30; most of a query's time goes to waking
DuckDB's workers for its few tasks and waiting for them, as v1 also did.

Rows are still read through v1: the result's chunks and types reach the
existing conversions through the bridge, and Arrow export converts with v1
using the connection's Arrow options, since only a callback can create a v2
exporter.

`Connection.progress()` reads `connection_progress_get`, the same snapshot
v1's `query_progress` reads.

Next candidates:

- Reading chunks and types through v2 vector views, which would retire the
  chunk and type bridge.

Patch 0008 changes the engine and applies to both APIs.
