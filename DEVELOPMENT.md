# Development

Requires CMake 3.29+, Ninja, uv, picky, and a C++17 compiler.

## Setup

```sh
jj git clone --colocate https://github.com/nicholasjunge/ducky
cd ducky
picky init ext/duckdb
uv sync --all-groups --reinstall-package=ducky
uv run --no-sync python scripts/configure-clangd.py
uv run --no-sync pytest -q
```

Picky keeps DuckDB sparse. Its paths, patches, and update hook are declared in
`.gitmodules`.

The clangd configuration uses nanobind from `.venv` and copies its compilation
database to the repository root. Package builds remain isolated.

## Updating DuckDB

```sh
picky update ext/duckdb main
uv sync --all-groups --reinstall-package=ducky
uv run --no-sync pytest -q
```

Picky updates the gitlink, applies `patches/*.patch`, and writes
`cmake/duckdb-describe.txt`. Rebase or remove any failed patch.

To refresh a patch, edit `ext/duckdb`, then export its diff:

```sh
git -C ext/duckdb diff -- <paths> > patches/0003-name.patch
```

## Stubs

Normal builds generate `_core.pyi` in the build tree. Update or check the
repository copy explicitly:

```sh
build_dir="build/clangd"
cmake --build "$build_dir" --target update_ducky_core_stub
cmake --build "$build_dir" --target check_ducky_core_stub
```

## Checks

```sh
uv run --no-sync pytest -q
uv run --no-sync ty check src/ducky tests
uvx prek run --all-files
```

## AddressSanitizer

```sh
DUCKY_ASAN=1 uv sync --all-groups --reinstall-package=ducky
```

On macOS:

```sh
VIRTUAL_ENV="$PWD/.venv" scripts/asan-pytest.sh -q
```

On Linux:

```sh
LD_PRELOAD="$(clang -print-file-name=libclang_rt.asan.so) $(clang -print-file-name=libc++.so)" \
ASAN_OPTIONS="detect_leaks=0:halt_on_error=1" uv run --no-sync pytest --capture no
```

Restore Release with `uv sync --all-groups --reinstall-package=ducky`.
