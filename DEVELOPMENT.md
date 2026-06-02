# Development

Notes for hacking on ducky locally.

## Prerequisites

- CMake ≥ 3.29 and Ninja
- [uv](https://docs.astral.sh/uv/)
- A C++ compiler toolchain capable of building DuckDB (must support C++17).

## Clone with a sparse DuckDB submodule

The DuckDB submodule's full working tree is ~280 MB; we only need ~50 MB of it
(source, third_party, the two essential extensions, and a handful of CMake
glue files). `scripts/init-duckdb.sh` configures the sparse checkout *before*
the first checkout so unused paths are never written to disk (and, with
`--filter=blob:none`, their blobs are never fetched either):

```sh
jj git clone --colocate https://github.com/nicholasjunge/ducky
cd ducky
scripts/init-duckdb.sh
```

The script is idempotent: if you already ran `git submodule update --init
ext/duckdb` and pulled the full tree, re-running it just trims the already-fat
working tree rather than avoiding the download up front. The exact path list
lives in the script.

## Build

Builds run **without build isolation** (configured in `pyproject.toml` via
`tool.uv.no-build-isolation-package`) so nanobind's headers in the project
`.venv` stay visible to CMake across rebuilds.

```sh
uv sync   # editable install of ducky + build deps
```

The first build compiles DuckDB from source and takes a while; subsequent
builds reuse `build/`.

To bundle additional DuckDB extensions, pass them through at configure time:

```sh
uv sync -C cmake.define.BUILD_EXTENSIONS="core_functions;parquet;json;icu"
```

## Tests

```sh
uv run pytest
```

## Linting

```sh
uvx prek run --all-files --show-diff-on-failure
```

## Layout

```
src/cpp/        nanobind + DuckDB C API sources
  ducky.*         module entry point (NB_MODULE), class registration, shared helpers
  connection.*    Connection: open/execute/prepare/close, config, interrupt/progress
  result.*        Result: lazy, chunk-based row decoding
  chunk.*         Chunk: zero-copy ndarray views over DuckDB vectors + validity masks
  function.*      Scalar UDF registration and the per-chunk Python trampoline
  aggregate.*     Aggregate UDF registration (Python class as live state)
  table.*         Table UDF registration (Python generator factory)
  appender.*      Appender: bulk-insert path consuming ndarrays per column
  database.hpp    DuckDBHandle: shared database/connection ownership
  arrow_abi.h     Arrow C-data-interface struct definitions
src/ducky/      the Python package (re-exports from the _core extension)
  __init__.py     public API surface (connect, progress_bar, ...)
  _core.pyi       type stub for the compiled _core extension
  _conversions.py Arrow / pandas / polars interop helpers
  _dataset.py     Dataset/Feature/Split/Target ML helpers
  _progress.py    progress_bar context manager (tqdm + stderr fallback)
  _typing.py      DuckDBConfig TypedDict for connect(**config)
examples/       end-to-end demos
tests/          pytest suite
scripts/        developer helpers (init-duckdb.sh: sparse submodule setup)
ext/duckdb/     DuckDB submodule, built from source
CMakeLists.txt  builds DuckDB from the submodule and links the C API statically
```
