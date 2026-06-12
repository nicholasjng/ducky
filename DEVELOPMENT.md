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

## DuckDB patch stack

Local DuckDB edits live as a working-tree overlay in `patches/`
(`NNNN-slug.patch`, applied in lexical order).
`scripts/bump-duckdb.sh` re-applies the whole stack from a clean checkout
(`--3way`, so patches survive context drift);
applying is the default since the build needs them.

```sh
scripts/bump-duckdb.sh                # re-apply patches/ at the current pin (after init-duckdb.sh)
scripts/bump-duckdb.sh --no-patches   # build against pristine upstream
```

Author a patch by editing the submodule tree and exporting with the next
numeric prefix (prepend a header explaining why; `git apply` skips the
preamble):

```sh
git -C ext/duckdb diff -- <paths> > patches/0002-my-change.patch
```

When a bump breaks a patch the script hard-fails and names it — either retire
it (`rm patches/000X-*.patch`, if upstreamed) or refresh it (`git -C ext/duckdb
apply --reject`, fix the `.rej` hunks, re-export). Past a handful of patches,
use `quilt`.

## Bumping the DuckDB submodule

`scripts/bump-duckdb.sh <ref>` keeps the three pins in sync:

1. The **gitlink** — the submodule commit in this repo's tree (`git ls-tree
   HEAD ext/duckdb`). The canonical pin; `.gitmodules` holds only the URL.
2. **`OVERRIDE_GIT_DESCRIBE`** in `CMakeLists.txt` — the version string for
   `duckdb_library_version()`. **Don't remove it**: the checkout is shallow and
   tagless, so without it DuckDB's `git describe` falls back to `v0.0.1` on
   every fresh/CI clone (and sdist builds have no `.git` at all). It's set with
   `CACHE STRING "" FORCE` — **keep the `FORCE`**, or the value is ignored
   against an existing build dir and the bump never reaches the compiler. The
   string is `git describe`'s output for the pin; the script derives it with
   `git -C ext/duckdb describe --tags --long --match 'v[0-9]*'` (after fetching
   tags). (Same reason to leave `shallow = true` in `.gitmodules` — it only
   affects `git submodule update`, which we don't use.)
3. The **`patches/` stack** — re-applied on the new revision (see above).

```sh
scripts/bump-duckdb.sh v1.6.3
```

The first bump unshallows the commit/tag graph (still blobless + sparse, ~50 MB)
so `git describe` can derive the new version. Then:

```sh
uv sync --all-groups --reinstall-package=ducky
uv run --no-sync pytest -q
jj describe   # record the gitlink + CMakeLists bump
```

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

### AddressSanitizer

Build a separate ASAN wheel (lands in `build/asan/`, leaving the Release wheel
alone; a plain `uv sync` afterwards swaps the editable install back to Release):

```sh
DUCKY_ASAN=1 uv sync --all-groups --reinstall-package=ducky
```

`uv run pytest` alone does *not* preload the ASAN runtime, so the test process
aborts at the first import of an ASAN-built extension. Preload it explicitly:

```sh
# Linux: preload the ASAN runtime *and* libc++. libc++ must be preloaded too,
# otherwise its container-overflow annotations don't line up with the
# instrumented build and ASAN reports false positives.
DUCKY_ASAN=1 LD_PRELOAD="$(clang -print-file-name=libclang_rt.asan.so) $(clang -print-file-name=libc++.so)" \
  ASAN_OPTIONS="detect_leaks=0:halt_on_error=1" uv run --all-groups pytest

# macOS: use the wrapper script (dyld drops DYLD_INSERT_LIBRARIES across the
# Homebrew python shim, so the script re-execs the real interpreter directly).
scripts/asan-pytest.sh
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
