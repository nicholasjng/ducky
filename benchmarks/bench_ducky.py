"""ducky-side workloads. Run with `mew run benchmarks/bench_ducky.py`.

Pair with bench_duckdb.py (run in a separate process) for cross-engine
comparison — see benchmarks/run_vs_duckdb.py.
"""

from __future__ import annotations

import mew
from _bench_fixtures import BIND_SIZES, SIZES, arrow_table, row_tuples

import ducky

mew.set_context("engine", f"ducky {ducky.__version__} (DuckDB {ducky.__duckdb_version__})")

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
    """fetchnumpy: chunk-direct dict[str, ndarray] (delegates to to_numpy)."""
    con = _populate(n)
    for _ in state:
        con.execute("SELECT * FROM t").fetchnumpy()


@mew.parametrize(_PARAMS, ids=_IDS, tags=(*_TAGS, "select"), min_time=0.2)
def bench_select_df(state: mew.State, n: int) -> None:
    """df(): pandas DataFrame via Arrow."""
    import pandas  # noqa: F401  # keep the one-time import out of the timed loop

    con = _populate(n)
    for _ in state:
        con.execute("SELECT * FROM t").df()


@mew.parametrize(_PARAMS, ids=_IDS, tags=(*_TAGS, "select"), min_time=0.2)
def bench_select_pl(state: mew.State, n: int) -> None:
    """pl(): polars DataFrame via Arrow."""
    import polars  # noqa: F401  # keep the one-time import out of the timed loop

    con = _populate(n)
    for _ in state:
        con.execute("SELECT * FROM t").pl()


@mew.parametrize(_BIND_PARAMS, ids=_BIND_IDS, tags=(*_TAGS, "select"), min_time=0.2)
def bench_select_fetchone(state: mew.State, n: int) -> None:
    """fetchone loop: PEP 249 row-at-a-time consumption (per-call bound)."""
    con = _populate(n)
    for _ in state:
        con.execute("SELECT * FROM t")
        while con.fetchone() is not None:
            pass


@mew.benchmark(tags=(*_TAGS, "select"), min_time=0.2)
def bench_execute_bound(state: mew.State) -> None:
    """Parameter bind + execute + fetch roundtrip on a constant query."""
    con = ducky.connect()
    for _ in state:
        con.execute("SELECT ?::BIGINT + ?::BIGINT", (1, 2)).fetchone()


@mew.parametrize(_PARAMS, ids=_IDS, tags=(*_TAGS, "select"), min_time=0.2)
def bench_arrow_scan_agg(state: mew.State, n: int) -> None:
    """Aggregate over a registered Arrow table (replacement scan, no copy-in)."""
    con = ducky.connect()
    con.register_arrow("src", arrow_table(n))
    for _ in state:
        con.execute("SELECT sum(a), sum(b), sum(c) FROM src").fetchall()


@mew.parametrize(_PARAMS, ids=_IDS, tags=(*_TAGS, "udf"), min_time=0.2)
def bench_udf_scalar(state: mew.State, n: int) -> None:
    """Vectorized scalar UDF: each engine's fastest flavor.

    ducky passes zero-copy ndarrays to create_function; the duckdb counterpart
    uses type='arrow' (its vectorized path — the default 'native' type is
    row-at-a-time and not comparable).
    """
    con = _populate(n)
    con.create_function("add_one", lambda b: b + 1, ["BIGINT"], "BIGINT")
    for _ in state:
        con.execute("SELECT sum(add_one(b)) FROM t").fetchone()


@mew.benchmark(tags=(*_TAGS, "connect"), min_time=0.2)
def bench_connect(state: mew.State) -> None:
    """Open and close an in-memory database."""
    for _ in state:
        ducky.connect().close()


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
    import torch  # noqa: F401  # keep the one-time import out of the timed loop

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
def bench_bulk_insert_arrow(state: mew.State, n: int) -> None:
    """INSERT ... SELECT from a registered Arrow source (parallel pipeline).

    The like-for-like counterpart of duckdb's from_arrow().insert_into().
    """
    tbl = arrow_table(n)
    for _ in state:
        with state.pause():
            con = ducky.connect()
            con.execute("CREATE TABLE t (a INT, b BIGINT, c DOUBLE)")
            con.register_arrow("src", tbl)
        con.execute("INSERT INTO t SELECT * FROM src")


@mew.parametrize(_PARAMS, ids=_IDS, tags=(*_TAGS, "insert"), min_time=0.2)
def bench_bulk_append_arrow(state: mew.State, n: int) -> None:
    """Appender Arrow ingest (single-threaded arrow→chunk→table).

    ducky-only: duckdb-python exposes no appender, so the side-by-side table
    shows no duckdb column for this workload. Compare against
    bulk_insert_arrow to see the appender's overhead.
    """
    tbl = arrow_table(n)
    for _ in state:
        with state.pause():
            con = ducky.connect()
            con.execute("CREATE TABLE t (a INT, b BIGINT, c DOUBLE)")
            app = con.appender("t")
        app.append_arrow(tbl)
        app.close()
