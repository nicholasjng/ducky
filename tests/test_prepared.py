"""Tests for prepared statements: Connection.prepare() / PreparedStatement."""

from __future__ import annotations

import pytest

import ducky


def test_prepare_execute_positional():
    con = ducky.connect()
    stmt = con.prepare("SELECT ? + ? AS s")
    assert stmt.num_parameters == 2
    assert stmt.execute([2, 3]).fetchone() == (5,)
    # Reusable: a second execute with different params re-binds cleanly.
    assert stmt.execute([10, 20]).fetchone() == (30,)


def test_prepare_execute_named():
    con = ducky.connect()
    stmt = con.prepare("SELECT $x AS x, $y AS y")
    assert stmt.num_parameters == 2
    assert {stmt.parameter_name(1), stmt.parameter_name(2)} == {"x", "y"}
    assert stmt.execute({"x": 1, "y": 2}).fetchone() == (1, 2)


def test_parameter_name_out_of_range_is_none():
    con = ducky.connect()
    stmt = con.prepare("SELECT ? AS x")
    assert stmt.parameter_name(99) is None


def test_executemany_batched_insert():
    con = ducky.connect()
    con.execute("CREATE TABLE t(id INTEGER, name VARCHAR)")
    ins = con.prepare("INSERT INTO t VALUES (?, ?)")
    ins.executemany([(1, "a"), (2, "b"), (3, "c")])
    assert con.execute("SELECT count(*) FROM t").fetchone() == (3,)
    assert con.execute("SELECT id, name FROM t ORDER BY id").fetchall() == [
        (1, "a"),
        (2, "b"),
        (3, "c"),
    ]


def test_result_schema_and_statement_type_before_execution():
    con = ducky.connect()
    con.execute("CREATE TABLE t(id INTEGER, name VARCHAR)")
    sel = con.prepare("SELECT id, name FROM t WHERE id > ?")
    # Schema is known without running the query.
    assert sel.columns == ["id", "name"]
    assert sel.types == ["INTEGER", "VARCHAR"]
    assert sel.statement_type == "SELECT"

    ins = con.prepare("INSERT INTO t VALUES (?, ?)")
    assert ins.statement_type == "INSERT"
    # DuckDB's INSERT reports the affected-row count as a single "Count" column.
    assert ins.columns == ["Count"]


def test_execute_no_parameters():
    con = ducky.connect()
    stmt = con.prepare("SELECT 42")
    assert stmt.num_parameters == 0
    assert stmt.execute().fetchone() == (42,)


def test_prepare_invalid_sql_raises():
    con = ducky.connect()
    with pytest.raises(ducky.Error):
        con.prepare("SELECT * FROM no_such_table")


def test_context_manager_closes():
    con = ducky.connect()
    with con.prepare("SELECT 1") as stmt:
        assert stmt.execute().fetchone() == (1,)
    # After the block the statement is closed.
    with pytest.raises(ducky.Error):
        stmt.execute()


def test_use_after_close_raises():
    con = ducky.connect()
    stmt = con.prepare("SELECT 1")
    stmt.close()
    with pytest.raises(ducky.Error):
        stmt.execute()
    # close() is idempotent.
    stmt.close()


def test_result_outlives_connection():
    # The Result from a prepared statement shares the DuckDBHandle, so it stays
    # usable (incl. Arrow export) after the Connection is dropped.
    con = ducky.connect()
    con.execute("CREATE TABLE t AS SELECT range AS i FROM range(3)")
    result = con.prepare("SELECT i FROM t ORDER BY i").execute()
    con.close()
    assert result.fetchall() == [(0,), (1,), (2,)]


def test_dataframe_path_on_prepared_result():
    con = ducky.connect()
    stmt = con.prepare("SELECT ? AS a, ? AS b")
    df = stmt.execute([1, 2]).df()
    assert df.to_dict("list") == {"a": [1], "b": [2]}
