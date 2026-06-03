"""Benchmark Python-UDF parallelism under free-threading.

DuckDB invokes a Python scalar UDF from every worker thread (`threads=N`). Under
the GIL those calls serialize; on a free-threaded interpreter (3.13t+, with the
extension built via ducky's `FREE_THREADED` flag) they run in parallel. This
script runs a deliberately CPU-bound *pure-Python* UDF over a materialized table
at several thread counts and reports the speedup — so the difference between a
GIL and a no-GIL build is visible directly.

Run it against an already-built environment (`--no-sync` so uv doesn't rebuild).
numpy is in the dev group, so it's already present in the default `.venv`; the
free-threaded `.venv-ft` needs it installed once (see below):

    uv run --no-sync scripts/ft-udf-bench.py                                  # GIL   -> flat
    UV_PROJECT_ENVIRONMENT=.venv-ft uv run --no-sync scripts/ft-udf-bench.py  # 3.13t -> scales

ducky builds without isolation (see pyproject's no-build-isolation-package), so it
can't be built into an isolated PEP 723 script env — hence the prebuilt-env route
above rather than inline script metadata. The free-threaded env is set up with:

    uv venv --python 3.13t .venv-ft
    UV_PROJECT_ENVIRONMENT=.venv-ft uv sync --no-default-groups --group build \
        --reinstall-package=ducky --python 3.13t
    VIRTUAL_ENV=.venv-ft uv pip install numpy
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

import ducky


def busy(x: np.ndarray) -> np.ndarray:
    # The inner loop is pure Python, so it holds the GIL and only parallelizes
    # once the GIL is gone. (numpy vectorization releases the GIL and would
    # muddy the comparison, so we deliberately avoid it here.)
    xl = x.tolist()
    out = [0.0] * len(xl)
    for i in range(len(xl)):
        v = xl[i]
        s = 0.0
        for _ in range(80):
            s += v * 1.0000001
        out[i] = s
    return np.array(out, dtype=np.float64)


def run(threads: int, rows: int) -> float:
    con = ducky.connect(threads=threads)
    con.create_function("busy", busy, parameters=["DOUBLE"], return_type="DOUBLE")
    # A materialized table scan parallelizes across row groups (~122,880 rows
    # each), so DuckDB calls the UDF from multiple workers. (A bare range() scan
    # stays single-threaded and would show no speedup regardless of the GIL.)
    con.execute(f"CREATE TABLE t AS SELECT (i % 1000)::DOUBLE AS x FROM range({rows}) t(i)")
    start = time.perf_counter()
    con.sql("SELECT sum(busy(x)) FROM t").fetchitem()
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description="Python-UDF parallelism benchmark.")
    parser.add_argument("--rows", type=int, default=800_000, help="table rows (default 800,000)")
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="thread counts to sweep (default: 1 2 4 8)",
    )
    args = parser.parse_args()

    gil = sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else True
    print(
        f"python {sys.version.split()[0]}  |  GIL {'enabled' if gil else 'DISABLED'}  "
        f"|  rows {args.rows:,}"
    )

    run(1, 50_000)  # warm up: prime module imports / type caches
    baseline: float | None = None
    for n in args.threads:
        dt = run(n, args.rows)
        if baseline is None:
            baseline = dt
        print(f"  threads={n:<3} {dt:6.2f}s   speedup {baseline / dt:4.2f}x")


if __name__ == "__main__":
    main()
