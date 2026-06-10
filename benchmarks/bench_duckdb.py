"""duckdb-side workloads. Run with `mew run benchmarks/bench_duckdb.py`.

Pair with bench_ducky.py (run in a separate process) for cross-engine
comparison — see benchmarks/run_vs_duckdb.py.
"""

from __future__ import annotations

import duckdb
import mew
from _bench_fixtures import BIND_SIZES, SIZES, arrow_table, row_tuples

_PARAMS = [{"n": n} for n in SIZES]
_IDS = [f"n={n}" for n in SIZES]
_BIND_PARAMS = [{"n": n} for n in BIND_SIZES]
_BIND_IDS = [f"n={n}" for n in BIND_SIZES]
_TAGS = ("duckdb",)


def _populate(n: int):
    con = duckdb.connect()
    con.execute("CREATE TABLE t (a INT, b BIGINT, c DOUBLE)")
    con.from_arrow(arrow_table(n)).insert_into("t")
    return con


@mew.parametrize(_PARAMS, ids=_IDS, tags=(*_TAGS, "select"), min_time=0.2)
def bench_select_fetchall(state: mew.State, n: int) -> None:
    con = _populate(n)
    for _ in state:
        con.execute("SELECT * FROM t").fetchall()


@mew.parametrize(_PARAMS, ids=_IDS, tags=(*_TAGS, "select"), min_time=0.2)
def bench_select_arrow(state: mew.State, n: int) -> None:
    con = _populate(n)
    for _ in state:
        con.execute("SELECT * FROM t").fetch_arrow_table()


@mew.parametrize(_PARAMS, ids=_IDS, tags=(*_TAGS, "select"), min_time=0.2)
def bench_select_numpy(state: mew.State, n: int) -> None:
    """fetchnumpy: native dict[str, ndarray] (duckdb-python's own path)."""
    con = _populate(n)
    for _ in state:
        con.execute("SELECT * FROM t").fetchnumpy()


# Map the ducky-specific `to_numpy` (chunk-direct concat) to duckdb's fetchnumpy
# so the side-by-side table shows both engines' fastest column-major path.
@mew.parametrize(_PARAMS, ids=_IDS, tags=(*_TAGS, "ml"), min_time=0.2)
def bench_select_to_numpy(state: mew.State, n: int) -> None:
    con = _populate(n)
    for _ in state:
        con.execute("SELECT * FROM t").fetchnumpy()


@mew.parametrize(_PARAMS, ids=_IDS, tags=(*_TAGS, "ml"), min_time=0.2)
def bench_select_iter_batches(state: mew.State, n: int) -> None:
    """duckdb's chunked path: fetch_record_batch (Arrow RecordBatchReader)."""
    con = _populate(n)
    for _ in state:
        reader = con.execute("SELECT * FROM t").fetch_record_batch()
        for _batch in reader:
            pass


@mew.parametrize(_PARAMS, ids=_IDS, tags=(*_TAGS, "ml"), min_time=0.2)
def bench_select_to_torch(state: mew.State, n: int) -> None:
    """torch(): native DLPack tensor dict."""
    con = _populate(n)
    for _ in state:
        con.execute("SELECT * FROM t").torch()


@mew.parametrize(_BIND_PARAMS, ids=_BIND_IDS, tags=(*_TAGS, "insert"), min_time=0.2)
def bench_executemany(state: mew.State, n: int) -> None:
    rows = row_tuples(n)
    for _ in state:
        with state.pause():
            con = duckdb.connect()
            con.execute("CREATE TABLE t (a INT, b BIGINT, c DOUBLE)")
        con.executemany("INSERT INTO t VALUES (?, ?, ?)", rows)


@mew.parametrize(_PARAMS, ids=_IDS, tags=(*_TAGS, "insert"), min_time=0.2)
def bench_bulk_append_arrow(state: mew.State, n: int) -> None:
    tbl = arrow_table(n)
    for _ in state:
        with state.pause():
            con = duckdb.connect()
            con.execute("CREATE TABLE t (a INT, b BIGINT, c DOUBLE)")
        con.from_arrow(tbl).insert_into("t")
