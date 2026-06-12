"""duckdb-side workloads. Run with `mew run benchmarks/bench_duckdb.py`.

Pair with bench_ducky.py (run in a separate process) for cross-engine
comparison — see benchmarks/run_vs_duckdb.py.
"""

from __future__ import annotations

import duckdb
import mew
from _bench_fixtures import BIND_SIZES, SIZES, arrow_table, row_tuples

mew.set_context("engine", f"duckdb {duckdb.__version__}")

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
        con.execute("SELECT * FROM t").to_arrow_table()


@mew.parametrize(_PARAMS, ids=_IDS, tags=(*_TAGS, "select"), min_time=0.2)
def bench_select_numpy(state: mew.State, n: int) -> None:
    """fetchnumpy: native dict[str, ndarray] (duckdb-python's own path)."""
    con = _populate(n)
    for _ in state:
        con.execute("SELECT * FROM t").fetchnumpy()


@mew.parametrize(_PARAMS, ids=_IDS, tags=(*_TAGS, "select"), min_time=0.2)
def bench_select_df(state: mew.State, n: int) -> None:
    """df(): pandas DataFrame."""
    import pandas  # noqa: F401  # keep the one-time import out of the timed loop

    con = _populate(n)
    for _ in state:
        con.execute("SELECT * FROM t").df()


@mew.parametrize(_PARAMS, ids=_IDS, tags=(*_TAGS, "select"), min_time=0.2)
def bench_select_pl(state: mew.State, n: int) -> None:
    """pl(): polars DataFrame."""
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
    con = duckdb.connect()
    for _ in state:
        con.execute("SELECT ?::BIGINT + ?::BIGINT", (1, 2)).fetchone()


@mew.parametrize(_PARAMS, ids=_IDS, tags=(*_TAGS, "select"), min_time=0.2)
def bench_arrow_scan_agg(state: mew.State, n: int) -> None:
    """Aggregate over a registered Arrow table (replacement scan, no copy-in)."""
    con = duckdb.connect()
    con.register("src", arrow_table(n))
    for _ in state:
        con.execute("SELECT sum(a), sum(b), sum(c) FROM src").fetchall()


@mew.parametrize(_PARAMS, ids=_IDS, tags=(*_TAGS, "udf"), min_time=0.2)
def bench_udf_scalar(state: mew.State, n: int) -> None:
    """Vectorized scalar UDF: each engine's fastest flavor.

    type='arrow' is duckdb's vectorized path (pyarrow arrays in/out); the
    default 'native' type is row-at-a-time and not comparable to ducky's
    ndarray-vectorized create_function.
    """
    import pyarrow.compute as pc
    from duckdb.sqltypes import BIGINT

    con = _populate(n)
    con.create_function("add_one", lambda b: pc.add(b, 1), [BIGINT], BIGINT, type="arrow")
    for _ in state:
        con.execute("SELECT sum(add_one(b)) FROM t").fetchone()


@mew.benchmark(tags=(*_TAGS, "connect"), min_time=0.2)
def bench_connect(state: mew.State) -> None:
    """Open and close an in-memory database."""
    for _ in state:
        duckdb.connect().close()


# Map the ducky-specific `to_numpy` (chunk-direct concat) to duckdb's fetchnumpy
# so the side-by-side table shows both engines' fastest column-major path.
@mew.parametrize(_PARAMS, ids=_IDS, tags=(*_TAGS, "ml"), min_time=0.2)
def bench_select_to_numpy(state: mew.State, n: int) -> None:
    con = _populate(n)
    for _ in state:
        con.execute("SELECT * FROM t").fetchnumpy()


@mew.parametrize(_PARAMS, ids=_IDS, tags=(*_TAGS, "ml"), min_time=0.2)
def bench_select_iter_batches(state: mew.State, n: int) -> None:
    """duckdb's chunked path: to_arrow_reader at 2048 rows per batch.

    2048 matches DuckDB's vector size, which is the granularity ducky's
    iter_batches streams at — the default (1M rows/batch) would compare one
    giant batch against ducky's ~489 chunks.
    """
    con = _populate(n)
    for _ in state:
        reader = con.execute("SELECT * FROM t").to_arrow_reader(2048)
        for _batch in reader:
            pass


@mew.parametrize(_PARAMS, ids=_IDS, tags=(*_TAGS, "ml"), min_time=0.2)
def bench_select_to_torch(state: mew.State, n: int) -> None:
    """torch(): native DLPack tensor dict."""
    import torch  # noqa: F401  # keep the one-time import out of the timed loop

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
def bench_bulk_insert_arrow(state: mew.State, n: int) -> None:
    """INSERT from an Arrow relation (parallel pipeline).

    Pairs with ducky's bulk_insert_arrow (register_arrow + INSERT ... SELECT).
    """
    tbl = arrow_table(n)
    for _ in state:
        with state.pause():
            con = duckdb.connect()
            con.execute("CREATE TABLE t (a INT, b BIGINT, c DOUBLE)")
        con.from_arrow(tbl).insert_into("t")
