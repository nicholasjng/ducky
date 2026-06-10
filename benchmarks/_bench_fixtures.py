"""Engine-agnostic data fixtures shared by bench_ducky.py and bench_duckdb.py.

We can't import ducky and the upstream duckdb Python module in the same
process — both statically link DuckDB and the symbols collide. Engine-specific
benchmark files import only their own engine; this file imports neither.
"""

from __future__ import annotations

import functools

import numpy as np
import pyarrow as pa

SIZES = [10_000, 100_000, 1_000_000]
# executemany binds one Python value at a time per row; 1M rows takes minutes.
BIND_SIZES = [10_000, 100_000]


@functools.cache
def arrays(n: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed=42)
    return {
        "a": rng.integers(0, 1 << 30, size=n, dtype=np.int32),
        "b": rng.integers(0, 1 << 60, size=n, dtype=np.int64),
        "c": rng.standard_normal(size=n, dtype=np.float64),
    }


@functools.cache
def arrow_table(n: int) -> pa.Table:
    cols = arrays(n)
    return pa.table(
        {
            "a": pa.array(cols["a"], type=pa.int32()),
            "b": pa.array(cols["b"], type=pa.int64()),
            "c": pa.array(cols["c"], type=pa.float64()),
        }
    )


@functools.cache
def row_tuples(n: int) -> list[tuple[int, int, float]]:
    cols = arrays(n)
    return list(zip(cols["a"].tolist(), cols["b"].tolist(), cols["c"].tolist(), strict=True))
