# ducky: duckdb Python bindings with nanobind

This repo is called ducky, and hosts nanobind bindings of duckdb.

The goal of this project is to expose duckdb's API in Python, and make use of nanobind's improvements over the current set of pybind11 bindings.
DuckDB is built from source via the `ext/duckdb` git submodule (a shallow pin of duckdb/duckdb). The official pybind11 bindings (duckdb/duckdb-python) are no longer vendored locally — refer to them on GitHub if needed.

## Goals

The resulting bindings should expose an API similar to duckdb's Python API.
If you find that some choices do not make sense anymore or could be handled better than in the original bindings, feel free to improve, but ask first.

## Style

Use CMake and scikit-build-core for building the Python wheel.
Run tests with pytest. Tests should live in the tests/ folder.
Export a compile commands database using CMake, and make sure to build the project without build isolation to keep nanobind's include paths alive.
Use prek as a linter and formatter. Run `uvx prek run --all-files --show-diff-on-failure`.

## Docs

The duckdb Python API docs live in https://duckdb.org/docs/current/clients/python/overview.
The nanobind documentation lives at https://nanobind.readthedocs.io.
