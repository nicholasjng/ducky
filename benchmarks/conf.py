"""mew setup file: records which engine ``bench_engine.py`` measured.

mew imports this file for every command, so the engine is imported only in
``context()``, which ``mew run`` calls once before the first benchmark.
"""

from __future__ import annotations

import os


def context() -> dict:
    engine = os.environ.get("DUCKY_BENCH_ENGINE", "ducky")
    if engine == "duckdb":
        import duckdb

        return {"engine": f"duckdb {duckdb.__version__}"}
    import ducky

    return {"engine": f"ducky {ducky.__version__} (DuckDB {ducky.__duckdb_version__})"}
