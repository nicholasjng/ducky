import numpy as np
import pytest

import ducky


def test_appender_row_api(con):
    con.execute("CREATE TABLE t (id INTEGER, name VARCHAR, score DOUBLE)")
    with con.appender("t") as app:
        assert app.columns == ["id", "name", "score"]
        assert app.types == ["INTEGER", "VARCHAR", "DOUBLE"]
        app.append_row(1, "a", 0.5)
        app.append_row(2, "b", 1.5)
        app.append_row(3, None, None)
    rows = con.execute("SELECT * FROM t ORDER BY id").fetchall()
    assert rows == [(1, "a", 0.5), (2, "b", 1.5), (3, None, None)]


def test_appender_row_arity_mismatch(con):
    con.execute("CREATE TABLE t (a INTEGER, b INTEGER)")
    with con.appender("t") as app, pytest.raises(ducky.Error, match="expected 2 values"):
        app.append_row(1)


def test_appender_columns_numeric(con):
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


def test_appender_columns_partial_fills_null(con):
    con.execute("CREATE TABLE t (id BIGINT, score DOUBLE)")
    with con.appender("t") as app:
        app.append_columns({"id": np.arange(3, dtype=np.int64)})
    rows = con.execute("SELECT * FROM t ORDER BY id").fetchall()
    assert rows == [(0, None), (1, None), (2, None)]


def test_appender_columns_masks(con):
    con.execute("CREATE TABLE t (score DOUBLE)")
    scores = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    mask = np.array([True, False, True, False])
    with con.appender("t") as app:
        app.append_columns({"score": scores}, masks={"score": mask})
    rows = con.execute("SELECT * FROM t").fetchall()
    assert rows == [(0.1,), (None,), (0.3,), (None,)]


def test_appender_columns_dtype_mismatch(con):
    con.execute("CREATE TABLE t (id BIGINT)")
    with con.appender("t") as app, pytest.raises(ducky.Error, match="dtype mismatch"):
        app.append_columns({"id": np.arange(3, dtype=np.int32)})


def test_appender_mask_wide_dtype_rejected(con):
    # uint16/32/64 share the uint8 DLPack type code; they must not pass as masks.
    con.execute("CREATE TABLE t (score DOUBLE)")
    scores = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    with con.appender("t") as app, pytest.raises(ducky.Error, match="bool/uint8"):
        app.append_columns({"score": scores}, masks={"score": np.ones(3, dtype=np.uint64)})


def test_appender_mask_without_column_rejected(con):
    con.execute("CREATE TABLE t (id BIGINT, score DOUBLE)")
    mask = np.ones(3, dtype=np.uint8)
    with con.appender("t") as app, pytest.raises(ducky.Error, match="no matching entry"):
        app.append_columns({"id": np.arange(3, dtype=np.int64)}, masks={"score": mask})


def test_appender_unknown_table(con):
    with pytest.raises(ducky.Error):
        con.appender("does_not_exist")


def test_appender_schema_qualified(con):
    con.execute("CREATE SCHEMA s")
    con.execute("CREATE TABLE s.t (id INTEGER)")
    with con.appender("t", schema="s") as app:
        app.append_row(1)
        app.append_row(2)
    assert con.execute("SELECT SUM(id) FROM s.t").fetchall() == [(3,)]


# ── DEFAULT values (append_row + ducky.DEFAULT) ──────────────────────────────


def test_appender_default_constant(con):
    con.execute("CREATE TABLE t (id INTEGER DEFAULT 42, name VARCHAR)")
    with con.appender("t") as app:
        app.append_row(ducky.DEFAULT, "a")
        app.append_row(7, "b")
    assert con.execute("SELECT * FROM t ORDER BY name").fetchall() == [(42, "a"), (7, "b")]


def test_appender_default_foldable_now(con):
    # now() is a foldable default — duckdb_append_default resolves it.
    con.execute("CREATE TABLE t (id INTEGER, ts TIMESTAMP DEFAULT now())")
    with con.appender("t") as app:
        app.append_row(1, ducky.DEFAULT)
    assert con.execute("SELECT id, ts IS NOT NULL FROM t").fetchall() == [(1, True)]


def test_appender_default_on_column_without_default_raises(con):
    con.execute("CREATE TABLE t (a INTEGER, b INTEGER)")
    with con.appender("t") as app, pytest.raises(ducky.Error, match="no DEFAULT value"):
        app.append_row(ducky.DEFAULT, 5)


def test_appender_default_non_foldable_surfaces_duckdb_error(con):
    # Sequence defaults aren't foldable; DuckDB's appender rejects them and we
    # surface its message rather than a generic failure.
    con.execute("CREATE SEQUENCE seq START 100")
    con.execute("CREATE TABLE t (id INTEGER DEFAULT nextval('seq'), v INTEGER)")
    with con.appender("t") as app, pytest.raises(ducky.Error, match="not supported"):
        app.append_row(ducky.DEFAULT, 1)


def test_default_sentinel_is_singleton_and_not_constructible():
    assert repr(ducky.DEFAULT) == "ducky.DEFAULT"
    with pytest.raises(TypeError):
        type(ducky.DEFAULT)()


# ── Arrow ingest (append_arrow) ──────────────────────────────────────────────


def test_append_arrow_table_stream(con):
    # __arrow_c_stream__ path, exercising VARCHAR + LIST + NULLs — types the
    # ndarray columnar path can't handle.
    pa = pytest.importorskip("pyarrow")
    con.execute("CREATE TABLE t (id BIGINT, name VARCHAR, tags VARCHAR[])")
    tbl = pa.table(
        {
            "id": pa.array([1, 2, 3], pa.int64()),
            "name": pa.array(["a", "b", None], pa.string()),
            "tags": pa.array([["x", "y"], [], ["z"]], pa.list_(pa.string())),
        }
    )
    with con.appender("t") as app:
        app.append_arrow(tbl)
    rows = con.sql("SELECT id, name, tags FROM t ORDER BY id").fetchall()
    assert rows == [(1, "a", ["x", "y"]), (2, "b", []), (3, None, ["z"])]


def test_append_arrow_record_batch(con):
    # __arrow_c_array__ path (a single pyarrow RecordBatch).
    pa = pytest.importorskip("pyarrow")
    con.execute("CREATE TABLE t (id BIGINT, name VARCHAR)")
    rb = pa.record_batch({"id": pa.array([4, 5], pa.int64()), "name": pa.array(["d", "e"])})
    with con.appender("t") as app:
        app.append_arrow(rb)
    assert con.sql("SELECT id, name FROM t ORDER BY id").fetchall() == [(4, "d"), (5, "e")]


def test_append_arrow_multiple_batches(con):
    # A table built from several record batches yields several stream batches;
    # all must be appended.
    pa = pytest.importorskip("pyarrow")
    con.execute("CREATE TABLE t (id BIGINT)")
    b1 = pa.record_batch({"id": pa.array([1, 2], pa.int64())})
    b2 = pa.record_batch({"id": pa.array([3, 4, 5], pa.int64())})
    with con.appender("t") as app:
        app.append_arrow(pa.Table.from_batches([b1, b2]))
    assert con.sql("SELECT count(*), sum(id) FROM t").fetchall() == [(5, 15)]


def test_append_arrow_struct(con):
    pa = pytest.importorskip("pyarrow")
    con.execute("CREATE TABLE t (p STRUCT(x INTEGER, y INTEGER))")
    tbl = pa.table({"p": pa.array([{"x": 1, "y": 2}, {"x": 3, "y": 4}])})
    with con.appender("t") as app:
        app.append_arrow(tbl)
    assert con.sql("SELECT p.x, p.y FROM t ORDER BY p.x").fetchall() == [(1, 2), (3, 4)]


def test_append_arrow_from_ducky_result(con):
    # A ducky Result is itself an Arrow stream, so it can feed an appender —
    # an in-database copy with no Python materialization.
    con.execute("CREATE TABLE t (i BIGINT)")
    with con.appender("t") as app:
        app.append_arrow(con.sql("SELECT i FROM range(100) t(i)"))
    assert con.sql("SELECT count(*) FROM t").fetchitem() == 100
    assert con.sql("SELECT sum(i) FROM t").fetchitem() == 4950


def test_append_arrow_rejects_non_arrow(con):
    con.execute("CREATE TABLE t (id BIGINT)")
    with con.appender("t") as app, pytest.raises(ducky.Error, match="Arrow PyCapsule"):
        app.append_arrow(object())


def test_appender_row_temporals(con):
    import datetime as dt

    con.execute("CREATE TABLE t (d DATE, tm TIME, ts TIMESTAMP, iv INTERVAL)")
    with con.appender("t") as app:
        app.append_row(
            dt.date(2024, 1, 2),
            dt.time(3, 4, 5, 6),
            dt.datetime(2024, 1, 2, 3, 4, 5, 6),
            dt.timedelta(days=1, seconds=2, microseconds=3),
        )
        app.append_row(None, None, None, None)
    rows = con.execute("SELECT * FROM t").fetchall()
    assert rows == [
        (
            dt.date(2024, 1, 2),
            dt.time(3, 4, 5, 6),
            dt.datetime(2024, 1, 2, 3, 4, 5, 6),
            dt.timedelta(days=1, seconds=2, microseconds=3),
        ),
        (None, None, None, None),
    ]


def test_appender_row_aware_datetime_lands_in_utc(con):
    import datetime as dt

    tz = dt.timezone(dt.timedelta(hours=2))
    aware = dt.datetime(2024, 6, 1, 12, 30, tzinfo=tz)
    con.execute("CREATE TABLE t (ts TIMESTAMPTZ)")
    with con.appender("t") as app:
        app.append_row(aware)
    (row,) = con.execute("SELECT ts FROM t").fetchall()
    assert row[0] == aware


def test_appender_row_date_rejects_datetime(con):
    import datetime as dt

    con.execute("CREATE TABLE t (d DATE)")
    with con.appender("t") as app, pytest.raises(ducky.Error, match="expected datetime.date"):
        app.append_row(dt.datetime(2024, 1, 2, 3, 4))
