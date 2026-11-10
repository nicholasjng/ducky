# Development

Requires CMake 3.29+, Ninja, uv, picky, and a C++17 compiler.

## Setup

```sh
jj git clone --colocate https://github.com/nicholasjunge/ducky
cd ducky
picky init ext/duckdb
picky update ext/duckdb "$(git ls-tree HEAD ext/duckdb | awk '{print $3}')"
uv sync --all-groups --reinstall-package=ducky
uv run --no-sync python scripts/configure-clangd.py
uv run --no-sync pytest -q
```

Picky keeps DuckDB sparse. Its paths, patches, and update hook are declared in
`.gitmodules`. `picky init` checks out the pinned commit but does not apply
`patches/`; the `picky update` to the same pin applies them. Run that `update`
line again after pulling a commit that moves the pin: without a ref, `picky
update` re-checks out the submodule's current commit, not the new pin.

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

If CMake then fails to update `_deps/httpfs_extension_fc-src` ("Failed to
unstash changes"), the build tree still holds the patched httpfs checkout of the
previous pin; delete `build/<wheel tag>/_deps/httpfs_extension_fc-*` and build
again.

Patches apply in lexical order, each on top of the previous ones. To refresh a
patch, edit `ext/duckdb` with the earlier patches applied, then export the
difference:

```sh
git -C ext/duckdb diff -- <paths> > patches/NNNN-name.patch
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

## Symbols

The extension module exports only `PyInit__core`, so another DuckDB in the
process (duckdb-python, say) cannot bind to ducky's. Release builds also strip
the hidden symbols' names, which leaves profilers and DuckDB's internal-error
stack traces with bare addresses. Keep the names, at the cost of a larger
module:

```sh
DUCKY_SYMBOLS=1 uv sync --all-groups --reinstall-package=ducky
```

AddressSanitizer builds keep them as well. Loadable extensions do not need
ducky's symbols, because DuckDB links each one against its own copy of DuckDB.
Only an extension built with `EXTENSION_STATIC_BUILD=OFF` cannot load.

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
