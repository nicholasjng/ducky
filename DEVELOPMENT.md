# Development

Notes for hacking on ducky locally.

## Prerequisites

- CMake ≥ 3.29 and Ninja
- [uv](https://docs.astral.sh/uv/)
- A C++ compiler toolchain capable of building DuckDB (must support C++17).

## Clone with a sparse DuckDB submodule

The DuckDB submodule's full working tree is ~280 MB; we only need ~50 MB of it
(source, third_party, the two essential extensions, and a handful of CMake
glue files). Configure the sparse checkout *before* the first checkout so
unused paths are never written to disk (and, with `--filter=blob:none`, their
blobs are never fetched either):

```sh
jj git clone --colocate https://github.com/nicholasjunge/ducky
cd ducky
git submodule update --init --no-checkout --filter=blob:none ext/duckdb

git -C ext/duckdb sparse-checkout init --no-cone
git -C ext/duckdb sparse-checkout set \
    '/CMakeLists.txt' \
    '/DuckDBConfig.cmake.in' \
    '/DuckDBConfigVersion.cmake.in' \
    '/LICENSE' \
    '/src/' \
    '/third_party/' \
    '/scripts/' \
    '/tools/CMakeLists.txt' \
    '/tools/utils/' \
    '/extension/CMakeLists.txt' \
    '/extension/*.cmake' \
    '/extension/*.in' \
    '/extension/loader/' \
    '/extension/core_functions/' \
    '/extension/parquet/' \
    '/extension/json/' \
    '/.github/config/extensions/httpfs.cmake' \
    '/.github/patches/extensions/httpfs/'

git -C ext/duckdb checkout HEAD
```

If you already ran `git submodule update --init ext/duckdb`, the same
`sparse-checkout init` / `set` / `checkout HEAD` sequence still works — it
just trims an already-fat working tree rather than avoiding the download up
front.

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
  ducky.cpp       module entry point (NB_MODULE), class registration
  connection.*    Connection: open/execute/prepare/close
  result.*        Result: lazy, chunk-based row decoding
  chunk.*         Chunk: zero-copy ndarray views over DuckDB vectors + validity masks
  function.*      Scalar UDF registration and the per-chunk Python trampoline
src/ducky/      the Python package (re-exports from the _core extension)
examples/       end-to-end demos
tests/          pytest suite
ext/duckdb/     DuckDB submodule, built from source
CMakeLists.txt  builds DuckDB from the submodule and links the C API statically
```
