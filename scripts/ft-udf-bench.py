"""Benchmark Python UDF scaling across DuckDB thread counts."""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

import ducky


def busy(x: np.ndarray) -> np.ndarray:
    # Deliberately pure Python — numpy vectorization would release the GIL
    # and muddy the with/without-GIL comparison.
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
    # Materialize so the scan parallelizes across row groups; a bare range()
    # scan stays single-threaded and would show no speedup either way.
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
