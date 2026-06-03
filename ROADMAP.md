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
- ✅ **Pending / streaming results** (`duckdb_pending_prepared`,
  `duckdb_pending_prepared_streaming`, `duckdb_pending_execute_task`,
  `duckdb_execute_pending`). `Connection.execute` / `Connection.sql` /
  `PreparedStatement.execute` all route through a single `run_pending` helper
  in `connection.cpp` that drives the executor one task at a time with the GIL
  released between ticks. Two wins over the old `gil_scoped_release` wrap of
  `duckdb_execute_prepared`: (1) `PyErr_CheckSignals()` runs between ticks, so
  `KeyboardInterrupt` lands mid-query instead of parking until the query
  finishes — on signal we call `duckdb_interrupt`, drain the pending result,
  and re-raise; (2) opt-in `streaming=True` on those three entry points
  returns a streaming `duckdb_result` whose chunks are pulled lazily via
  `duckdb_fetch_chunk`, so `iter_batches_torch` / `iter_batches_jax` /
  `iter_batches_mlx` can stay bounded to one chunk of peak memory. The
  parameterless `duckdb_query` fast path is gone — everything routes through
  `duckdb_prepare` for one unified pending path; multi-statement strings are
  deliberately not supported (`duckdb_extract_statements` is the roadmap path
  if we ever want them back, see *Multi-statement scripts* below).
  - ⬜ **Auto-streaming for iter_batches\_\* / to_torch / to_jax.** Today the
    caller has to remember `con.execute(sql, streaming=True).iter_batches_torch()`.
    A `Connection.iter_torch(sql, ...)` (and friends) that runs the query with
    `streaming=True` internally would close the ergonomic gap; alternatively,
    have `iter_batches_*` warn when the source isn't streaming so the bounded-
    memory claim above stays honest.
  - ✅ **`async def execute`.** `Connection.aexecute` / `asql` drive the pending
    executor from a coroutine (`ducky._aio`): each task tick is offloaded via
    `asyncio.to_thread` (GIL released in C++) with `asyncio.sleep(0)` between
    ticks so the event loop stays responsive, and `CancelledError` /
    `KeyboardInterrupt` triggers `duckdb_interrupt` + drain. Built on a steppable
    `PendingResult` handle (`Connection.make_pending`) whose `std::mutex` keeps
    the cancellation drain from racing an in-flight worker tick.

## Dataset / feature API

- ⬜ **Named output fields (beyond a single `X` / `y`).** `ducky.dataset()`
  today materialises exactly two output arrays per fold — features stacked into
  `X` and the target column as `y` (`Fold.tensors()`). The split,
  standardisation (train-fold stats) and backend materialisation already operate
  on an arbitrary per-column dict (`Fold._arrays`); only the *assembly* step
  hardcodes "stack the features → `X`, take the target → `y`". Generalise the
  output into a dict of named **fields**, each either a **matrix** (several
  columns stacked into `(n, d)`) or a **vector** (one column → `(n,)`):

  ```python
  ds = ducky.dataset(
      source,
      fields={
          "X": ducky.matrix({"age": ducky.feature("Age", standardize=True), ...}),
          "y": ducky.vector("Survived"),
          "w": ducky.vector("sample_weight"),          # sample weights
          "ids": ducky.vector("PassengerId", dtype="i64"),
          # a multi-output target is just another matrix: "Y": ducky.matrix({...})
      },
      split=ducky.split(0.8), backend="jax",
  )
  Xtr, wtr = ds.train["X"], ds.train["w"]
  ```

  Unlocks sample weights, multi-output / multi-task targets, group / id columns
  (passthrough today, and the substrate for future *group-aware* splitting), and
  separate dense / categorical feature blocks — all as uniform fields, with no
  per-role special cases.

  **Shape of the change** (mostly assembly, not the SQL / streaming core):
  - New spec helpers `ducky.matrix(columns)` / `ducky.vector(expr, *, dtype,
    standardize)` alongside `feature` / `target` / `split`.
  - `_compile_sql` collects column exprs across *all* fields under
    field-qualified aliases (e.g. `X__age`, `y`, `w`) so names stay unique;
    standardisation stats and the hash-bucket split are otherwise unchanged.
  - Assembly per fold reuses the existing on-device `_stack` / `_gather`: stack a
    matrix field's columns, take a vector field's lone column. `Fold` holds
    `_fields: dict[str, ArrayT]` and gains `Fold.__getitem__(name) -> ArrayT`, so
    `ds["train"]["X"]` reads naturally.
  - Typing is unaffected: every field is the same backend type `ArrayT` (the
    `AbstractArray` bound only needs `.shape`), so a 2-D `X` and a 1-D `w` are
    both members of one `Fold[ArrayT]`, and the `dataset()` backend overloads
    keep mapping `backend=` → element type.
  - **Backward compatible.** Keep `columns=` / `target=` as a shorthand that
    desugars to `fields={"X": matrix(columns), "y": vector(target)}`, and
    `Fold.tensors()` as sugar returning `(self["X"], self["y"])` when both exist
    — so existing call sites and the Titanic examples stay untouched.

## QOL improvements

- ✅ **`Result.fetchitem()` / `Connection.fetchitem()`.** A scalar-fetch
  helper that returns the single value of the single result row, raising a
  clear `ducky.Error` if the result isn't exactly 1 row × 1 column. Named after
  numpy's `ndarray.item()` for the same "collapse a container to its lone
  scalar" intent. `COUNT(*)`-style queries previously forced every call site
  through `row = ...fetchone(); assert row is not None; (x,) = row` purely to
  satisfy the type checker (`fetchone() -> tuple | None`); `fetchitem()`
  makes `n = con.execute(...).fetchitem()` both correct and type-clean.
  `Result::fetchitem` reuses the existing `convert_value` row decoder (so every
  type `fetchone` handles works, `NULL`→`None` included), checks
  `column_count_ == 1`, decodes the lone cell, then confirms no second row
  remains; `Connection.fetchitem` delegates through `current_result` like the
  other `fetch*` methods. The `assert ... is not None` + `[0]` dance was removed
  from the scalar-fetch call sites in `tests/` (aggregate-UDF, bind, runtime,
  progress-bar).

## Refactors not gated on v2

- ✅ Collapsed the duplicated `.arrow/.df/.pl/.to_numpy/.to_torch/.to_jax/
  .chunks/.iter_batches*` wrappers. Rather than a Python-side mixin (which
  would drop the methods from the auto-generated `_core.pyi`), a single
  `def_conversions()` template in `ducky.cpp` registers all eleven once and is
  instantiated for both `Result` and `Connection`, parameterised by a "source
  extractor" (identity for `Result`, `current_result()` for `Connection`). The
  methods stay compiled — so the stub generator still emits them — and a
  Python-visible `Connection.current_result` property was added. Net ~90 lines
  off `ducky.cpp`.

## Free-threading (CPython 3.13t+)

- 🟡 **Parallel UDFs + concurrent decode on no-GIL builds.** The heavy DuckDB
  work already runs GIL-free, so the GIL only bottlenecks the Python-side hot
  paths — and DuckDB already fans those across worker threads. On a free-threaded
  interpreter, Python scalar / arrow / aggregate UDFs (invoked from every
  `threads=N` worker, each doing `gil_scoped_acquire`) run in parallel instead of
  serialising, and result decoding / `to_numpy` / `to_torch` across connections
  overlap. The extension now builds with nanobind's `FREE_THREADED` flag (emits
  `Py_MOD_GIL_NOT_USED`), and shared caches already use `nb::ft_mutex`. Still
  open: the thread-safety audit — chiefly that the per-`Connection` /
  per-`Result` "one thread" contract holds as a *memory*-safety invariant (FT
  turns those logical races into real ones), e.g. guarding `Result` cursor state
  and `Connection.last_result_`.

## Unbound C API: candidates

A survey of `ext/duckdb/src/include/duckdb.h` (≈548 functions) against what the
bindings call (≈172) leaves ≈379 unbound. Most are low-level plumbing or
already covered above (pending/streaming results, replacement scans, Arrow
appender ingest). The items below are the gaps that look worth their weight for
ducky's data-science / ML audience, roughly highest-value first.

- ✅ **Prepared statements as first-class objects.**
  `Connection.prepare(sql) -> PreparedStatement` compiles the query once
  (parse + bind + plan) and exposes `.execute(params) -> Result` /
  `.executemany(rows)`, so loops over a query reuse the plan instead of
  re-preparing on every call as `connection.cpp:run()` does for ad-hoc
  parameterised `execute()`. The execution releases the GIL (same as `run`),
  reuses the existing `bind_parameters` machinery, and the returned `Result`
  shares the `DuckDBHandle` so it outlives the connection. Introspection
  without executing: `num_parameters`, `parameter_name(i)` (1-based),
  `columns` / `types` (result schema via
  `duckdb_prepared_statement_column_*`), and `statement_type`
  (`duckdb_prepared_statement_type`). Context-manager aware. Per-index
  `param_type` was left out for now — easy to add on top if a use case wants it.

- ⬜ **Query profiling access** (`duckdb_get_profiling_info` +
  `duckdb_profiling_info_get_metrics` / `_get_value` / `_get_child(_count)`).
  Programmatic `EXPLAIN ANALYZE`: walk the operator tree and pull timing /
  cardinality metrics into a Python dict (gated by the `enable_profiling`
  setting), instead of scraping `EXPLAIN` text. A natural fit for perf tuning
  in notebooks and inside training loops.

- ⬜ **Table introspection + appender DEFAULTs** (`duckdb_table_description_*`,
  `duckdb_append_default` / `duckdb_append_default_to_chunk`). The
  `table_description` API gives column names / types / default-ness directly,
  letting the appender drop its `SELECT * FROM t LIMIT 0` column-discovery hack
  (`appender.cpp:discover_column_names`). It also unlocks appending rows that
  fall back to a column's `DEFAULT` (e.g. autoincrement / `now()` columns)
  instead of requiring every column — a common ergonomic gap in bulk insert.

- ⬜ **Stateful / volatile / NULL-aware scalar UDFs**
  (`duckdb_scalar_function_set_bind` / `set_init` / `get_state` /
  `set_bind_data`, `set_volatile`, `set_special_handling`). Rounds out the v2
  UDF engine: `set_volatile` marks non-deterministic functions so the optimizer
  won't fold or cache them (needed for `random`/`now`-style UDFs);
  `set_special_handling` lets a UDF observe and emit NULLs (current ndarray UDFs
  are NULL-oblivious — see the NULL-handling note in `_conversions.py`);
  bind/init add per-statement and per-thread state.

- ⬜ **Faithful table-function parameters** (the `duckdb_value` getter family:
  `duckdb_get_date` / `_get_timestamp` / `_get_decimal` / `_get_blob` / nested
  getters). `table.cpp:duckdb_value_to_python` currently decodes only
  bool/int/float and falls back to `str()` (varchar) for everything else, so a
  `DATE` / `TIMESTAMP` / `DECIMAL` argument reaches the Python factory as text.
  A complete `duckdb_value` → Python decoder (shared with the scalar bind path
  in `connection.cpp`) would pass these through with their real types.

- ⬜ **Multi-statement scripts + statement-type introspection**
  (`duckdb_extract_statements` / `_error`, `duckdb_prepared_statement_type`,
  `duckdb_result_statement_type`). Run a multi-statement `.sql` script in one
  call, and expose the statement kind (SELECT / INSERT / UPDATE / …) so callers
  can branch — e.g. only materialise a result for statements that produce one.

- ⬜ **DuckDB filesystem access from Python** (`duckdb_file_system_open`,
  `duckdb_file_handle_read` / `_seek` / `_tell` / `_size` / `_write` / `_close`
  via `duckdb_connection_get_client_context` →
  `duckdb_client_context_get_file_system`). Read and write files through
  DuckDB's *configured* filesystem — including extensions like `httpfs` / S3 —
  from Python, without a separate IO stack. Note this is a consumer API
  (open/read/seek on DuckDB's FS); it does not register a Python-backed
  filesystem, so it complements `fsspec` rather than replacing it.

- ⬜ **Shared in-memory databases via the instance cache**
  (`duckdb_create_instance_cache`, `duckdb_get_or_create_from_cache`,
  `duckdb_destroy_instance_cache`). Each ducky `Connection` currently owns its
  own database, so two `connect(":memory:")` calls are fully isolated (see the
  note in `connection.hpp`). Routing opens through an instance cache would let
  several connections share one in-memory database — useful for multi-threaded
  readers or a writer + reader split over the same transient data.

### Deliberately skipped (niche / high effort)

- **Custom COPY functions** (`duckdb_copy_function_*`, ~30 calls) — Python-backed
  `COPY TO/FROM` for bespoke import/export formats. Large surface, narrow
  audience versus just reading/writing Arrow or Parquet.
- **Custom CAST functions** (`duckdb_create_cast_function` + `duckdb_cast_function_*`)
  — register Python cast logic between types; rarely needed once UDFs exist.
- **Custom log storage** (`duckdb_create_log_storage`, `duckdb_log_*`) — pluggable
  log sinks; operational rather than analytical.
- **Extension install/load** — the roadmap's "runtime extension loading" item
  has no matching C API surface in this DuckDB pin (no `duckdb_extension_*`
  functions in `duckdb.h`); it would need a `PRAGMA`/SQL approach instead.
