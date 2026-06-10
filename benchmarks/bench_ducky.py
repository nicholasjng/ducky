"""ducky-side workloads. Run with `mew run benchmarks/bench_ducky.py`.

Pair with bench_duckdb.py (run in a separate process) for cross-engine
comparison — see benchmarks/run_vs_duckdb.py.
"""

from __future__ import annotations

import mew
from _bench_fixtures import BIND_SIZES, SIZES, arrow_table, row_tuples

import ducky

_PARAMS = [{"n": n} for n in SIZES]
_IDS = [f"n={n}" for n in SIZES]
_BIND_PARAMS = [{"n": n} for n in BIND_SIZES]
_BIND_IDS = [f"n={n}" for n in BIND_SIZES]
_TAGS = ("ducky",)


def _populate(n: int):
    con = ducky.connect()
    con.execute("CREATE TABLE t (a INT, b BIGINT, c DOUBLE)")
    app = con.appender("t")
    app.append_arrow(arrow_table(n))
    app.close()
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
        con.execute("SELECT * FROM t").arrow()


@mew.parametrize(_PARAMS, ids=_IDS, tags=(*_TAGS, "select"), min_time=0.2)
def bench_select_numpy(state: mew.State, n: int) -> None:
    """fetchnumpy: arrow-mediated dict[str, ndarray]."""
    con = _populate(n)
    for _ in state:
        con.execute("SELECT * FROM t").fetchnumpy()


@mew.parametrize(_PARAMS, ids=_IDS, tags=(*_TAGS, "ml"), min_time=0.2)
def bench_select_to_numpy(state: mew.State, n: int) -> None:
    """to_numpy: chunk-direct zero-copy concat path (ducky-specific)."""
    con = _populate(n)
    for _ in state:
        con.execute("SELECT * FROM t").to_numpy()


@mew.parametrize(_PARAMS, ids=_IDS, tags=(*_TAGS, "ml"), min_time=0.2)
def bench_select_iter_batches(state: mew.State, n: int) -> None:
    """iter_batches: streaming per-chunk dict[str, ndarray] (no concat)."""
    con = _populate(n)
    for _ in state:
        for _batch in con.execute("SELECT * FROM t").iter_batches():
            pass


@mew.parametrize(_PARAMS, ids=_IDS, tags=(*_TAGS, "ml"), min_time=0.2)
def bench_select_to_torch(state: mew.State, n: int) -> None:
    """to_torch: DLPack into torch tensor dict."""
    con = _populate(n)
    for _ in state:
        con.execute("SELECT * FROM t").to_torch()


@mew.parametrize(_BIND_PARAMS, ids=_BIND_IDS, tags=(*_TAGS, "insert"), min_time=0.2)
def bench_executemany(state: mew.State, n: int) -> None:
    rows = row_tuples(n)
    for _ in state:
        with state.pause():
            con = ducky.connect()
            con.execute("CREATE TABLE t (a INT, b BIGINT, c DOUBLE)")
            stmt = con.prepare("INSERT INTO t VALUES (?, ?, ?)")
        stmt.executemany(rows)


@mew.parametrize(_PARAMS, ids=_IDS, tags=(*_TAGS, "insert"), min_time=0.2)
def bench_bulk_append_arrow(state: mew.State, n: int) -> None:
    tbl = arrow_table(n)
    for _ in state:
        with state.pause():
            con = ducky.connect()
            con.execute("CREATE TABLE t (a INT, b BIGINT, c DOUBLE)")
            app = con.appender("t")
        app.append_arrow(tbl)
        app.close()
