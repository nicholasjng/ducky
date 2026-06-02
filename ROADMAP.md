# Roadmap

The ndarray and UDF paths in `README.md` are a deliberate v1: numeric and
temporal types only, matching what `nb::ndarray` can flatly represent.

Legend: ✅ shipped · 🟡 partially shipped · ⬜ not started.

## v2: UDFs

- ✅ **String / nested types.** `Connection.create_arrow_function` takes a
  `pyarrow.RecordBatch` per call and returns a `pyarrow.Array`. Built on
  DuckDB's Arrow C API (`duckdb_data_chunk_to_arrow` /
  `duckdb_data_chunk_from_arrow` + `duckdb_vector_reference_vector`), this
  covers `VARCHAR`, `LIST`, `STRUCT`, `DECIMAL`, `MAP` — i.e. anything the
  ndarray path refuses. Type expressions are parsed by DuckDB itself via a
  `SELECT CAST(NULL AS <t>)` round-trip, so the full type grammar is
  supported.
- ✅ **Nullable outputs.** A UDF may return `(values, mask)` where `mask` is a
  1D uint8/bool ndarray (1=valid, 0=null); the trampoline ensures the output
  validity bitset and flips the null bits via
  `duckdb_validity_set_row_invalid`.
- ✅ **Varargs UDFs** via `duckdb_scalar_function_set_varargs`. Pass
  `varargs="TYPE"` on registration (mutually exclusive with `parameters`); the
  trampoline reads arity from `duckdb_data_chunk_get_column_count` and calls
  `fn(*args)` with one ndarray per SQL argument.
- ✅ **Type inference from Python hints.** `parameters` / `return_type` are
  optional; when omitted, `inspect.signature` + `typing.get_type_hints` derive
  them from `fn`'s annotations (`bool`→BOOLEAN, `int`→BIGINT, `float`→DOUBLE).
  Anything else raises a clear error pointing the user at the explicit form.
- ✅ **Aggregate and table UDFs** (`duckdb_create_aggregate_function`,
  `duckdb_create_table_function`). `Connection.create_aggregate_function`
  takes a class with `__init__` / `update(*arrays)` / `finalize()` and an
  optional `combine(other)` for parallel execution; state is a live Python
  instance stored as a `PyObject*` in DuckDB's aggregate state slot. Without
  `combine`, a `__dict__`-copy fallback keeps serial queries correct.
  `Connection.create_table_function` wraps a Python generator factory: DuckDB
  calls bind / init / produce; the produce loop drives `next()` up to 2048
  rows per chunk and writes values into DuckDB vectors via typed dispatch.

## v2: ndarray export

- ✅ **Richer dtypes.** `HUGEINT` / `UHUGEINT` / `INTERVAL` / `DECIMAL` are
  now exposed via `Chunk.column()` as numpy structured arrays (zero-copy via
  `.view()`). `Chunk.decimal_scale()` returns the exponent so callers can
  reconstruct the exact value.
- ✅ **Streaming `to_torch` / `to_jax`.** `Result.iter_batches_torch` /
  `iter_batches_jax` yield one `{name: Tensor/Array}` dict per chunk
  (zero-copy on CPU via DLPack), keeping peak memory bounded to one chunk.
  `to_torch` / `to_jax` now use these iterators internally, eliminating the
  intermediate numpy materialization of the old `to_numpy`-then-convert path.
- ✅ **Direct DLPack export from `Chunk`.** `Chunk.dlpack(key)` returns a
  `nanobind.nb_ndarray` with `__dlpack__` / `__dlpack_device__` (using
  nanobind's `array_api` framework, which requires no external library).
  `iter_batches_torch` / `iter_batches_jax` use this path, so numpy is no
  longer in the tensor/array hot path.

## v2: round-trip data path

- ✅ **Appender API** (`duckdb_appender_*`). Fastest bulk-insert path in DuckDB.
  A natural mirror of `Result.to_numpy`: an appender that consumes an ndarray
  per column closes the loop for writing predictions / eval metrics / feature
  batches back into a table without the prepared-statement round-trip.
  - ⬜ **Arrow ingest into the appender.** Once the ndarray path lands, extend
    `Appender.append_columns` to accept any object exposing
    `__arrow_c_array__` / `__arrow_c_stream__` and forward it through
    DuckDB's Arrow append path. Subsumes pandas / polars / pyarrow in one
    code path and covers VARCHAR / LIST / STRUCT for free.
- 🟡 **Replacement scans** (`duckdb_add_replacement_scan`). Lets
  `SELECT * FROM my_df` resolve to a Python object (NumPy / pandas / Arrow).
  Closes the input side the same way the ndarray/Arrow exporters close the
  output side; today users have to materialize via an explicit register step.
  - **Status.** `Connection.register_arrow(name, obj)` ships today but
    materializes via `duckdb_arrow_scan` + `CREATE OR REPLACE TABLE`, paying
    one full copy at registration. The "real" version is a lazy, zero-copy
    Arrow ingest exposed as a custom table function registered through
    `duckdb_create_table_function`, with a replacement scan that calls
    `__arrow_c_stream__` fresh on every plan so repeated queries replay
    against the source object. The shortcut (reuse DuckDB's built-in
    `arrow_scan` with its internal `FactoryGetNext` / `FactoryGetSchema`)
    is not viable from the C API: those factories live in DuckDB's
    anonymous C++ namespace
    (`ext/duckdb/src/main/capi/arrow-c.cpp` ~lines 358 and 392) and their
    signatures take internal C++ types. The clean path is to implement our
    own `arrow_scan` table function (~250 LOC: bind / init / produce / cleanup
    plus Arrow → DuckDB type conversion mirroring `chunk.cpp` in the output
    direction); the fast path — linking against DuckDB's C++ internals to
    reuse the factories (~30 LOC) — couples us to an ABI the rest of the
    binding deliberately avoids.
- ✅ **Richer parameter binding.** Today `bind` covers bool/int/float/str/None.
  Add `DATE`/`TIME`/`TIMESTAMP`/`BLOB`/`DECIMAL`/`HUGEINT` and named-parameter
  lookup via `duckdb_bind_parameter_index` to remove a class of surprises at
  the SQL boundary.

## v2: runtime control

- ✅ **Config API** (`duckdb_create_config`, `duckdb_set_config`). Expose
  `threads`, `memory_limit`, `access_mode`, etc. on `connect(...)` instead of
  forcing users through `PRAGMA`.
- ⬜ **Runtime extension loading** (`duckdb_extension_install` /
  `duckdb_extension_load`). Load `httpfs` / `icu` / `parquet` declaratively at
  runtime rather than baking them into `BUILD_EXTENSIONS` at compile time.
- ✅ **Interrupt + progress** (`duckdb_interrupt`, `duckdb_query_progress`).
  Cancel long-running queries and surface progress — big quality-of-life win
  in notebooks and inside training loops where a hung scan currently kills
  the kernel.
  - ✅ **Built-in progress-bar helper.** `with ducky.progress_bar(con): ...`
    (see `examples/progress_bar.py`) enables progress tracking, suppresses
    DuckDB's own printed bar, polls `Connection.progress()` from a daemon
    thread, and restores the connection's progress settings on exit. Renders
    through tqdm when installed and degrades to a minimal stderr
    `desc [####    ]  42.0%` printer otherwise (`use_tqdm=False` forces the
    fallback) — importing the helper never hard-fails on a missing tqdm.
- ⬜ **Pending / streaming results** (`duckdb_pending_prepared`,
  `duckdb_execute_pending`, `duckdb_pending_execute_task`). The proper
  substrate for releasing the GIL during `execute` (see UDF section) and for
  a true streaming `to_torch` / `to_jax` that avoids the intermediate copy.

## Refactors not gated on v2

- ⬜ Move the `.arrow/.df/.pl/.to_numpy/.to_torch/.to_jax/.chunks/.iter_batches`
  wrappers off the C++ class definitions and into a Python-side mixin, using
  a Python-visible `Connection.current_result` property. ~80 lines of
  `ducky.cpp` go away and the Result-vs-Connection duplication collapses.
  See the discussion attached to step 2.
