import numpy as np
import pytest

import ducky


def test_appender_row_api():
    con = ducky.connect()
    con.execute("CREATE TABLE t (id INTEGER, name VARCHAR, score DOUBLE)")
    with con.appender("t") as app:
        assert app.columns == ["id", "name", "score"]
        assert app.types == ["INTEGER", "VARCHAR", "DOUBLE"]
        app.append_row(1, "a", 0.5)
        app.append_row(2, "b", 1.5)
        app.append_row(3, None, None)
    rows = con.execute("SELECT * FROM t ORDER BY id").fetchall()
    assert rows == [(1, "a", 0.5), (2, "b", 1.5), (3, None, None)]


def test_appender_row_arity_mismatch():
    con = ducky.connect()
    con.execute("CREATE TABLE t (a INTEGER, b INTEGER)")
    with con.appender("t") as app, pytest.raises(ducky.Error, match="expected 2 values"):
        app.append_row(1)


def test_appender_columns_numeric():
    con = ducky.connect()
    con.execute("CREATE TABLE t (id BIGINT, score DOUBLE)")
    n = 5000  # larger than STANDARD_VECTOR_SIZE to exercise chunk splitting
    ids = np.arange(n, dtype=np.int64)
    scores = np.linspace(0.0, 1.0, n, dtype=np.float64)
    with con.appender("t") as app:
        app.append_columns({"id": ids, "score": scores})
    got = con.execute("SELECT COUNT(*), SUM(id), AVG(score) FROM t").fetchall()[0]
    assert got[0] == n
    assert got[1] == int(ids.sum())
    assert got[2] == pytest.approx(scores.mean())


def test_appender_columns_partial_fills_null():
    con = ducky.connect()
    con.execute("CREATE TABLE t (id BIGINT, score DOUBLE)")
    with con.appender("t") as app:
        app.append_columns({"id": np.arange(3, dtype=np.int64)})
    rows = con.execute("SELECT * FROM t ORDER BY id").fetchall()
    assert rows == [(0, None), (1, None), (2, None)]


def test_appender_columns_masks():
    con = ducky.connect()
    con.execute("CREATE TABLE t (score DOUBLE)")
    scores = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    mask = np.array([True, False, True, False])
    with con.appender("t") as app:
        app.append_columns({"score": scores}, masks={"score": mask})
    rows = con.execute("SELECT * FROM t").fetchall()
    assert rows == [(0.1,), (None,), (0.3,), (None,)]


def test_appender_columns_dtype_mismatch():
    con = ducky.connect()
    con.execute("CREATE TABLE t (id BIGINT)")
    with con.appender("t") as app, pytest.raises(ducky.Error, match="dtype mismatch"):
        app.append_columns({"id": np.arange(3, dtype=np.int32)})


def test_appender_unknown_table():
    con = ducky.connect()
    with pytest.raises(ducky.Error):
        con.appender("does_not_exist")


def test_appender_schema_qualified():
    con = ducky.connect()
    con.execute("CREATE SCHEMA s")
    con.execute("CREATE TABLE s.t (id INTEGER)")
    with con.appender("t", schema="s") as app:
        app.append_row(1)
        app.append_row(2)
    assert con.execute("SELECT SUM(id) FROM s.t").fetchall() == [(3,)]
