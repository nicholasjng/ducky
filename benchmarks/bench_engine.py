"""ducky and duckdb-python benchmarks, run per engine by ``benchmarks/run_vs_duckdb.py``.

``DUCKY_BENCH_ENGINE`` picks the engine, ``ducky`` (the default) or ``duckdb``.
Both runs share these benchmark names, so ``mew compare`` matches them; each
process imports only its own engine.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from typing import Any

import mew
import pyarrow as pa
from _bench_fixtures import BIND_SIZES, SIZES, arrow_table, row_tuples

ENGINE = os.environ.get("DUCKY_BENCH_ENGINE", "ducky")
_SCHEMA = "CREATE TABLE t (a INT, b BIGINT, c DOUBLE)"


class _Ducky:
    """ducky's spelling of the operations the two engines name differently."""

    def __init__(self) -> None:
        import ducky

        self.mod = ducky

    def connect(self) -> Any:
        return self.mod.connect()

    def populate(self, con: Any, tbl: pa.Table) -> None:
        app = con.appender("t")
        app.append_arrow(tbl)
        app.close()

    def register(self, con: Any, name: str, tbl: pa.Table) -> None:
        con.register_arrow(name, tbl)

    def create_add_one(self, con: Any) -> None:
        """Vectorized NumPy scalar UDF."""
        con.create_function("add_one", lambda b: b + 1, ["BIGINT"], "BIGINT")

    def to_arrow(self, con: Any, query: str) -> pa.Table:
        """Execute straight into Arrow batches, as duckdb-python's relation does."""
        return con.arrow(query)

    def to_polars(self, con: Any, query: str) -> Any:
        return con.execute(query).pl()

    def to_numpy(self, con: Any, query: str) -> Any:
        """Chunk-direct concatenation into NumPy arrays."""
        return con.execute(query).to_numpy()

    def to_torch(self, con: Any, query: str) -> Any:
        return con.execute(query).to_torch()

    def iter_batches(self, con: Any, query: str) -> Iterator[Any]:
        """Chunk-sized NumPy batches."""
        return con.execute(query).iter_batches()

    def arrow_reader(self, con: Any, query: str) -> Iterator[Any]:
        return con.arrow_reader(query, rows_per_batch=2048)

    def insert_rows(self, con: Any) -> Callable[[list], None]:
        """Prepare a 3-column INSERT; the returned call inserts its rows."""
        return con.prepare("INSERT INTO t VALUES (?, ?, ?)").executemany

    def insert_arrow(self, con: Any, tbl: pa.Table) -> Callable[[], None]:
        """INSERT ... SELECT from a registered Arrow source."""
        con.register_arrow("src", tbl)
        return lambda: con.execute("INSERT INTO t SELECT * FROM src")


class _DuckDB:
    """duckdb-python's spelling of the same operations.

    duckdb-python 2 produces Arrow and polars output from relations
    (``con.sql``); from a DB-API result it copies the rows once more, or, for
    polars, does not offer it.
    """

    def __init__(self) -> None:
        import duckdb

        self.mod = duckdb

    def connect(self) -> Any:
        return self.mod.connect()

    def populate(self, con: Any, tbl: pa.Table) -> None:
        con.from_arrow(tbl).insert_into("t")

    def register(self, con: Any, name: str, tbl: pa.Table) -> None:
        con.register(name, tbl)

    def create_add_one(self, con: Any) -> None:
        """Vectorized Arrow scalar UDF."""
        import pyarrow.compute as pc
        from duckdb.sqltypes import BIGINT

        con.create_function("add_one", lambda b: pc.add(b, 1), [BIGINT], BIGINT, type="arrow")

    def to_arrow(self, con: Any, query: str) -> pa.Table:
        return con.sql(query).to_arrow_table()

    def to_polars(self, con: Any, query: str) -> Any:
        return con.sql(query).pl()

    def to_numpy(self, con: Any, query: str) -> Any:
        """duckdb-python's native fetchnumpy."""
        return con.execute(query).fetchnumpy()

    def to_torch(self, con: Any, query: str) -> Any:
        """Native DLPack tensor dict."""
        return con.execute(query).torch()

    def iter_batches(self, con: Any, query: str) -> Iterator[Any]:
        """2048-row Arrow batches, the closest counterpart of iter_batches."""
        return con.sql(query).to_arrow_reader(2048)

    def arrow_reader(self, con: Any, query: str) -> Iterator[Any]:
        return con.sql(query).to_arrow_reader(2048)

    def insert_rows(self, con: Any) -> Callable[[list], None]:
        return lambda rows: con.executemany("INSERT INTO t VALUES (?, ?, ?)", rows)

    def insert_arrow(self, con: Any, tbl: pa.Table) -> Callable[[], None]:
        """INSERT from an Arrow relation."""
        return lambda: con.from_arrow(tbl).insert_into("t")


_ENGINES: dict[str, type[_Ducky | _DuckDB]] = {"ducky": _Ducky, "duckdb": _DuckDB}
if ENGINE not in _ENGINES:
    raise ValueError(f"DUCKY_BENCH_ENGINE must be one of {sorted(_ENGINES)}, got {ENGINE!r}")
engine = _ENGINES[ENGINE]()

_PARAMS = [{"n": n} for n in SIZES]
_IDS = [f"n={n}" for n in SIZES]
_BIND_PARAMS = [{"n": n} for n in BIND_SIZES]
_BIND_IDS = [f"n={n}" for n in BIND_SIZES]


def _populate(n: int) -> Any:
    con = engine.connect()
    con.execute(_SCHEMA)
    engine.populate(con, arrow_table(n))
    return con


@mew.parametrize(_PARAMS, ids=_IDS, tags="select", min_time=0.2)
def bench_select_fetchall(state: mew.State, n: int) -> None:
    con = _populate(n)
    for _ in state:
        con.execute("SELECT * FROM t").fetchall()


@mew.parametrize(_PARAMS, ids=_IDS, tags="select", min_time=0.2)
def bench_select_arrow(state: mew.State, n: int) -> None:
    con = _populate(n)
    for _ in state:
        engine.to_arrow(con, "SELECT * FROM t")


@mew.parametrize(_PARAMS, ids=_IDS, tags="select", min_time=0.2)
def bench_select_numpy(state: mew.State, n: int) -> None:
    """fetchnumpy: dict[str, ndarray]."""
    con = _populate(n)
    for _ in state:
        con.execute("SELECT * FROM t").fetchnumpy()


@mew.parametrize(_PARAMS, ids=_IDS, tags="select", min_time=0.2)
def bench_select_df(state: mew.State, n: int) -> None:
    """df(): pandas DataFrame."""
    import pandas  # noqa: F401  # keep the one-time import out of the timed loop

    con = _populate(n)
    for _ in state:
        con.execute("SELECT * FROM t").df()


@mew.parametrize(_PARAMS, ids=_IDS, tags="select", min_time=0.2)
def bench_select_pl(state: mew.State, n: int) -> None:
    """pl(): polars DataFrame."""
    import polars  # noqa: F401  # keep the one-time import out of the timed loop

    con = _populate(n)
    for _ in state:
        engine.to_polars(con, "SELECT * FROM t")


@mew.parametrize(_BIND_PARAMS, ids=_BIND_IDS, tags="select", min_time=0.2)
def bench_select_fetchone(state: mew.State, n: int) -> None:
    """fetchone loop: PEP 249 row-at-a-time consumption (per-call bound)."""
    con = _populate(n)
    for _ in state:
        con.execute("SELECT * FROM t")
        while con.fetchone() is not None:
            pass


@mew.benchmark(tags="select", min_time=0.2)
def bench_execute_bound(state: mew.State) -> None:
    """Parameter bind + execute + fetch roundtrip on a constant query."""
    con = engine.connect()
    for _ in state:
        con.execute("SELECT ?::BIGINT + ?::BIGINT", (1, 2)).fetchone()


@mew.parametrize(_PARAMS, ids=_IDS, tags="select", min_time=0.2)
def bench_arrow_scan_agg(state: mew.State, n: int) -> None:
    """Aggregate over a registered Arrow table (replacement scan, no copy-in)."""
    con = engine.connect()
    engine.register(con, "src", arrow_table(n))
    for _ in state:
        con.execute("SELECT sum(a), sum(b), sum(c) FROM src").fetchall()


@mew.parametrize(_PARAMS, ids=_IDS, tags="udf", min_time=0.2)
def bench_udf_scalar(state: mew.State, n: int) -> None:
    """Vectorized scalar UDF: NumPy for ducky, Arrow for duckdb-python."""
    con = _populate(n)
    engine.create_add_one(con)
    for _ in state:
        con.execute("SELECT sum(add_one(b)) FROM t").fetchone()


@mew.benchmark(tags="connect", min_time=0.2)
def bench_connect(state: mew.State) -> None:
    """Open and close an in-memory database."""
    for _ in state:
        engine.connect().close()


@mew.parametrize(_PARAMS, ids=_IDS, tags="ml", min_time=0.2)
def bench_select_to_numpy(state: mew.State, n: int) -> None:
    """to_numpy for ducky, fetchnumpy for duckdb-python."""
    con = _populate(n)
    for _ in state:
        engine.to_numpy(con, "SELECT * FROM t")


@mew.parametrize(_PARAMS, ids=_IDS, tags="ml", min_time=0.2)
def bench_select_iter_batches(state: mew.State, n: int) -> None:
    """Stream chunk-sized batches."""
    con = _populate(n)
    for _ in state:
        for _batch in engine.iter_batches(con, "SELECT * FROM t"):
            pass


@mew.parametrize(_PARAMS, ids=_IDS, tags="ml", min_time=0.2)
def bench_select_arrow_reader(state: mew.State, n: int) -> None:
    """Stream collected 2048-row Arrow batches."""
    con = _populate(n)
    for _ in state:
        for _batch in engine.arrow_reader(con, "SELECT * FROM t"):
            pass


@mew.parametrize(_PARAMS, ids=_IDS, tags="ml", min_time=0.2)
def bench_select_to_torch(state: mew.State, n: int) -> None:
    """DLPack into a torch tensor dict."""
    import torch  # noqa: F401  # keep the one-time import out of the timed loop

    con = _populate(n)
    for _ in state:
        engine.to_torch(con, "SELECT * FROM t")


@mew.parametrize(_BIND_PARAMS, ids=_BIND_IDS, tags="insert", min_time=0.2)
def bench_executemany(state: mew.State, n: int) -> None:
    rows = row_tuples(n)
    for _ in state:
        with state.pause():
            con = engine.connect()
            con.execute(_SCHEMA)
            insert = engine.insert_rows(con)
        insert(rows)


@mew.parametrize(_PARAMS, ids=_IDS, tags="insert", min_time=0.2)
def bench_bulk_insert_arrow(state: mew.State, n: int) -> None:
    """Bulk INSERT from an Arrow table (parallel pipeline)."""
    tbl = arrow_table(n)
    for _ in state:
        with state.pause():
            con = engine.connect()
            con.execute(_SCHEMA)
            insert = engine.insert_arrow(con, tbl)
        insert()


# duckdb-python has no counterpart to these, so only ducky's runs report them.
if ENGINE == "ducky":

    @mew.parametrize(_PARAMS, ids=_IDS, tags="ml", min_time=0.2)
    def bench_select_arrow_batches(state: mew.State, n: int) -> None:
        """Convert and stream one Arrow batch per DuckDB chunk."""
        con = _populate(n)
        for _ in state:
            for _batch in con.execute("SELECT * FROM t").arrow_batches():
                pass

    @mew.parametrize(_PARAMS, ids=_IDS, tags="insert", min_time=0.2)
    def bench_bulk_append_arrow(state: mew.State, n: int) -> None:
        """Append an Arrow table."""
        tbl = arrow_table(n)
        for _ in state:
            with state.pause():
                con = engine.connect()
                con.execute(_SCHEMA)
                app = con.appender("t")
            app.append_arrow(tbl)
            app.close()
