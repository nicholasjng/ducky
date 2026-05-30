# Roadmap

The ndarray and UDF paths in `README.md` are a deliberate v1: numeric and
temporal types only, matching what `nb::ndarray` can flatly represent.

Legend: ✅ shipped · 🟡 partially shipped · ⬜ not started.

## v2: UDFs

- ⬜ **String / nested types.** A second registration variant takes a
  `pyarrow.RecordBatch` per call and returns a `pyarrow.Array`. Built on
  DuckDB's existing Arrow stream path, this covers `VARCHAR`, `LIST`,
  `STRUCT`, `DECIMAL`, `MAP` — i.e. anything the ndarray path refuses.
- ⬜ **Nullable outputs.** Optional `(values, mask)` return convention so UDFs
  can produce NULLs; wires up to `duckdb_vector_get_validity` on the output.
- ⬜ **Varargs UDFs** via `duckdb_scalar_function_set_varargs`.
- ⬜ **Type inference from Python hints.** Optional sugar:
  `def f(x: float, y: int) -> float` infers
  `parameters=["DOUBLE", "BIGINT"]`, `return_type="DOUBLE"`.
- ⬜ **Aggregate and table UDFs** (`duckdb_create_aggregate_function`,
  `duckdb_create_table_function`). The scalar trampoline is the template.

## v2: ndarray export

- ⬜ **Richer dtypes.** `DECIMAL` (as int128 + scale), `HUGEINT` (as a pair of
  int64), `INTERVAL` (struct of months/days/micros).
- ⬜ **Streaming `to_torch` / `to_jax`.** Today they materialize via `to_numpy`
  and concatenate; a chunk-by-chunk variant would avoid the intermediate
  copy for large results.
- ⬜ **Direct DLPack export from `Chunk`.** Numpy already exposes `__dlpack__`,
  so torch / JAX consumers work today via that bridge; a native DLPack
  capsule would let non-numpy consumers skip the round-trip.

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
  - ⬜ **Built-in progress-bar helper.** Today, wiring `Connection.progress()`
    into a tqdm bar takes ~20 lines of threading boilerplate (see
    `examples/progress_bar.py`). A `Connection.execute_progress(query, ...)`
    method (or `with ducky.progress_bar(con): ...` context manager) would
    bake the pattern in. Should degrade gracefully when tqdm isn't
    installed — fall back to a minimal stderr `[####    ] 42%` printer
    rather than hard-failing the import.
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
