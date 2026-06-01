# ducky: duckdb Python bindings with nanobind

This repo is called ducky, and hosts nanobind bindings of duckdb.

The goal of this project is to expose duckdb's API in Python, and make use of nanobind's improvements over the current set of pybind11 bindings.
DuckDB is built from source via the `ext/duckdb` git submodule (a shallow pin of duckdb/duckdb). The official pybind11 bindings (duckdb/duckdb-python) are no longer vendored locally — refer to them on GitHub if needed.

## Goals

The resulting bindings should expose an API similar to duckdb's Python API.
If you find that some choices do not make sense anymore or could be handled better than in the original bindings, feel free to improve, but ask first.

## Version control

This repo uses [jj](https://github.com/jj-vcs/jj) (Jujutsu) as its primary VCS, colocated with git. Docs: https://docs.jj-vcs.dev/latest/. Prefer `jj` for most version-control work — `jj status`, `jj diff`, `jj log`, `jj new`, `jj describe`, `jj squash`, `jj file track`, etc. Use plain `git` only for things jj doesn't cover (e.g. `git submodule update`, `git remote`-only operations).

## Style

Use `uv sync --all-groups --reinstall-package=ducky` for building the Python wheel.
Run tests with pytest using `uv run --all-groups pytest`. Tests should live in the tests/ folder.
Export a compile commands database using CMake, and make sure to build the project without build isolation to keep nanobind's include paths alive.
Use prek as a linter and formatter. Run `uvx prek run --all-files --show-diff-on-failure`.
Use C-style casts everywhere in C++ code (e.g. `(uint8_t)x`, `(const char*)p`). Do not use `static_cast`, `reinterpret_cast`, or `const_cast`.

### AddressSanitizer build

```bash
# Build (output goes to build/asan/, separate from the Release wheel).
# DUCKY_ASAN=1 also re-points the editable install at build/asan; a plain
# `uv sync` afterwards swaps the editable install back to Release.
DUCKY_ASAN=1 uv sync --all-groups --reinstall-package=ducky
```

Running the tests under ASAN differs by platform — `uv run pytest` alone does *not* preload the ASAN runtime, so the test process aborts at the first import of an ASAN-built extension.

```bash
# Linux: preload the runtime via LD_PRELOAD.
DUCKY_ASAN=1 LD_PRELOAD=$(clang -print-file-name=libclang_rt.asan.so) \
  ASAN_OPTIONS="detect_leaks=0:halt_on_error=1" uv run --all-groups pytest

# macOS: use the wrapper script. dyld strips DYLD_INSERT_LIBRARIES from
# Homebrew's bin/python3.13 (which .venv/bin/python symlinks to), so the
# wrapper resolves the underlying Python.app/Contents/MacOS/Python binary
# via _NSGetExecutablePath and launches it through `cmake -E env` with
# __PYVENV_LAUNCHER__ set so the venv stays activated.
scripts/asan-pytest.sh
```

`CMakePresets.json` has a matching `asan` preset for direct CMake invocations (`cmake --preset asan`). The flag is plumbed through `-DDUCKY_ASAN=ON` rather than `-DCMAKE_CXX_FLAGS=-fsanitize=address` — nanobind's stubgen sanitizer-detector (PR #1000) only walks per-target `COMPILE_OPTIONS` / `LINK_OPTIONS`, so flags set on the global `CMAKE_CXX_FLAGS` are invisible to it and stubgen runs without the ASAN preload.

## Docs

The duckdb Python API docs live in https://duckdb.org/docs/current/clients/python/overview.
The nanobind documentation lives at https://nanobind.readthedocs.io.
