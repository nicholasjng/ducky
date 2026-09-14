# DuckDB C API v2

*Reviewed 2026-09-14 against DuckDB `625f75f`.*

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
- Strings are length-delimited views, so embedded NUL bytes are unambiguous.
- An environment owns database creation and shared file-conflict detection.
- Logical types, values, options, schemas, catalogs, functions, and filesystems
  use consistent handle and error conventions.
- The stable surface is versioned independently, with separate switches for
  deprecated and unstable additions.

These changes mainly improve correctness, diagnostics, and maintainability.
They do not make vector or chunk access inherently faster.

## Query and result ownership

A v2 result is an incremental cursor over one connection. The caller advances
it with the result step and chunk-fetch operations, then destroys it. While the
result is live, the connection rejects another query with
`DUCKDB_V2_ERROR_RESOURCE_IN_USE`.

This contract fits ducky's asynchronous execution better than v1. Ducky can
currently create several pending results on one connection; they interfere and
usually leave only one query successful. V2 would turn that misuse into an
immediate typed error. Actual concurrency would still require separate
connections, preferably created by the shared `Database` and pool proposed in
[capi-gaps.md](capi-gaps.md).

The unified result state machine also combines pending execution, streaming,
cancellation, metadata, and finalization. Ducky currently composes these from
several v1 APIs and carries patches for Arrow result collection and reusable
chunk fetching.

## Relevant additions for ducky

V2's environment and database ownership are a natural base for multiple
connections. Its structured error codes could map to a useful Python exception
hierarchy without parsing DuckDB messages.

The following areas also align with ducky's data-loading work:

- Column-data collections support parallel append and scan state.
- Logical types and values cover nested data through one consistent interface.
- Arrow import and export have explicit importer/exporter handles and ownership.
- Function properties unify volatility, NULL behavior, fallibility, and other
  planning metadata.
- Typed database and connection options avoid constructing configuration SQL.
- Filesystem handles expose configured storage with structured options and
  errors.

Some desired features remain absent or incomplete. In particular, C++ table
functions have filter-pushdown hooks that v2 does not expose as a general C
callback, and registering a Python-backed filesystem still needs a C++ adapter
or a future C API.

## Stability

The generated `duckdb_v2.h` marks its normal surface stable at API v2.0.0 and
hides unstable declarations unless explicitly enabled. Consumers can target an
older v2 surface with version macros.

The separate `duckdb_extension_v2.h` still warns that the extension ABI is
guaranteed only for the matching DuckDB version. That matters most to loadable
binary extensions. Ducky statically links its pinned DuckDB revision, so it does
not need cross-version extension ABI compatibility, but it would still absorb
changes whenever the pin advances.

V1 remains supported and mature. V2 should currently be treated as a cleaner
new foundation rather than a required upgrade.

## Migration cost

A complete migration would touch nearly every binding subsystem:

- database and connection ownership;
- prepared, pending, and streaming query execution;
- result, chunk, vector, type, and value conversion;
- Arrow import and export;
- appenders and Python UDF registration;
- errors, progress, profiling, and extension loading;
- both carried DuckDB patches.

Mixing v1 and v2 temporarily is possible because DuckDB exports both symbol
families, but sharing ownership across their handle types would require care.
A narrow vertical prototype is safer than converting isolated helpers.

## Recommendation

Keep the existing v1 implementation for now. It has broad coverage, established
tests, and working optimized conversion paths.

Reassess v2 while designing shared databases and streaming loaders. Start with
an experimental path that implements environment creation, database open,
multiple connections, query execution, chunk fetching, cancellation, and
cleanup. Benchmark it against the v1 path and verify error and lifetime behavior
under sanitizers. If that core is simpler and stable across the next DuckDB pin,
migrate the remaining binding surface in coherent subsystems.
