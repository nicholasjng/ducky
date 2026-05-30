"""Tests for Connection.register_arrow."""

import gc

import pyarrow as pa
import pytest

import ducky


def test_pyarrow_table() -> None:
    con = ducky.connect()
    tbl = pa.Table.from_pydict({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    con.register_arrow("t", tbl)
    assert con.execute("SELECT count(*) FROM t").fetchall() == [(3,)]
    assert con.execute("SELECT y, sum(x) FROM t GROUP BY y ORDER BY y").fetchall() == [
        ("a", 1),
        ("b", 2),
        ("c", 3),
    ]


def test_multi_shot() -> None:
    con = ducky.connect()
    tbl = pa.Table.from_pydict({"x": list(range(10))})
    con.register_arrow("t", tbl)
    # Same query twice; materialized into a real table so the second SELECT
    # sees the same rows rather than an exhausted stream.
    first = con.execute("SELECT sum(x) FROM t").fetchall()
    second = con.execute("SELECT sum(x) FROM t").fetchall()
    assert first == second == [(45,)]


def test_source_can_be_dropped() -> None:
    con = ducky.connect()
    tbl = pa.Table.from_pydict({"x": [7, 8, 9]})
    con.register_arrow("t", tbl)
    del tbl
    gc.collect()
    assert con.execute("SELECT sum(x) FROM t").fetchall() == [(24,)]


def test_replace_on_reregister() -> None:
    con = ducky.connect()
    con.register_arrow("t", pa.Table.from_pydict({"x": [1]}))
    con.register_arrow("t", pa.Table.from_pydict({"x": [99]}))
    assert con.execute("SELECT * FROM t").fetchall() == [(99,)]


def test_polars_dataframe() -> None:
    pl = pytest.importorskip("polars")
    con = ducky.connect()
    df = pl.DataFrame({"x": [10, 20, 30, 40], "g": ["a", "a", "b", "b"]})
    con.register_arrow("p", df)
    assert con.execute("SELECT g, avg(x) FROM p GROUP BY g ORDER BY g").fetchall() == [
        ("a", 15.0),
        ("b", 35.0),
    ]


def test_rejects_non_arrow() -> None:
    con = ducky.connect()
    with pytest.raises(Exception, match="__arrow_c_stream__"):
        con.register_arrow("z", 42)


@pytest.mark.parametrize(
    "name",
    ["", "1bad", "has space", "with-dash", "drop;table"],
)
def test_rejects_bad_name(name: str) -> None:
    con = ducky.connect()
    with pytest.raises(Exception, match="(?i)(invalid|empty) table name"):
        con.register_arrow(name, pa.Table.from_pydict({"x": [1]}))
