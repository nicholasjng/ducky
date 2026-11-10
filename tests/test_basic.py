import datetime
import uuid
from decimal import Decimal

import pytest

import ducky


def test_version_attrs():
    assert isinstance(ducky.__duckdb_version__, str)
    assert ducky.__duckdb_version__


def test_scalar_select(con):
    rows = con.execute(
        "SELECT 42, 'hello', CAST(3.5 AS DOUBLE), TRUE, CAST(NULL AS INTEGER)"
    ).fetchall()
    assert rows == [(42, "hello", 3.5, True, None)]


def test_fetchone_and_fetchmany(con):
    con.execute("SELECT * FROM range(5) t(i)")
    assert con.fetchone() == (0,)
    assert con.fetchmany(2) == [(1,), (2,)]
    assert con.fetchall() == [(3,), (4,)]
    assert con.fetchone() is None


def test_fetchall_spans_chunks_and_repeats(con):
    # > STANDARD_VECTOR_SIZE rows so the pre-sized fetchall list crosses chunk
    # boundaries; a second fetchall on the drained result is empty.
    result = con.sql("SELECT * FROM range(5000) t(i)")
    rows = result.fetchall()
    assert len(rows) == 5000
    assert rows[0] == (0,)
    assert rows[-1] == (4999,)
    assert result.fetchall() == []


def test_fetchall_streaming_result(con):
    # Streaming results don't know their row count up front (the pre-size
    # path must fall back to appending).
    rows = con.sql("SELECT * FROM range(5000) t(i)", streaming=True).fetchall()
    assert len(rows) == 5000
    assert rows[-1] == (4999,)


def test_fetchitem(con):
    # The 1x1 happy path, on Result and on Connection (via current_result).
    assert con.sql("SELECT count(*) FROM range(42)").fetchitem() == 42
    assert con.execute("SELECT 3.5").fetchitem() == 3.5
    # Scalar decoding goes through the same path as fetchone: strings, NULL.
    assert con.sql("SELECT 'hi'").fetchitem() == "hi"
    assert con.sql("SELECT NULL").fetchitem() is None


@pytest.mark.parametrize(
    ("sql", "match"),
    [
        ("SELECT 1, 2", "exactly 1 column"),
        ("SELECT * FROM range(3)", "more"),
        ("SELECT 1 WHERE false", "empty"),
    ],
)
def test_fetchitem_requires_one_by_one(sql, match, con):
    with pytest.raises(ducky.Error, match=match):
        con.sql(sql).fetchitem()


def test_description_and_columns(con):
    res = con.sql("SELECT 1 AS a, 'x' AS b")
    assert res.columns == ["a", "b"]
    assert res.types == ["INTEGER", "VARCHAR"]
    description = res.description
    assert description is not None
    assert [d[0] for d in description] == ["a", "b"]


def test_positional_parameters(con):
    rows = con.execute("SELECT ? + ?, ?", [10, 5, "hi"]).fetchall()
    assert rows == [(15, "hi")]


def test_table_roundtrip(con):
    con.execute("CREATE TABLE t (id INTEGER, name VARCHAR)")
    con.execute("INSERT INTO t VALUES (1, 'a'), (2, 'b')")
    assert con.execute("SELECT count(*) FROM t").fetchall() == [(2,)]
    assert con.sql("SELECT name FROM t ORDER BY id").fetchall() == [("a",), ("b",)]


def test_result_is_iterable(con):
    res = con.sql("SELECT * FROM range(3) t(i)")
    assert [row[0] for row in res] == [0, 1, 2]


def test_error_is_raised(con):
    with pytest.raises(ducky.Error):
        con.execute("SELECT * FROM does_not_exist")


def test_context_manager():
    with ducky.connect() as con:
        assert con.execute("SELECT 1").fetchall() == [(1,)]


def test_decimal_decoding(con):
    rows = con.execute(
        "SELECT 3.5, CAST('123456789.123456789' AS DECIMAL(38, 9)), 0.00::DECIMAL(4, 2)"
    ).fetchall()
    assert rows == [(Decimal("3.5"), Decimal("123456789.123456789"), Decimal("0.00"))]


def test_hugeint_decoding(con):
    (row,) = con.execute(
        "SELECT 170141183460469231731687303715884105727::HUGEINT, (-12345678901234567890)::HUGEINT"
    ).fetchall()
    assert row == (170141183460469231731687303715884105727, -12345678901234567890)


def test_temporal_decoding(con):
    (row,) = con.execute(
        "SELECT DATE '2026-05-27', TIME '13:45:30.5', "
        "TIMESTAMP '2026-05-27 13:45:30.123456', "
        "TIMESTAMPTZ '2026-05-27 13:45:30+00'"
    ).fetchall()
    assert row[0] == datetime.date(2026, 5, 27)
    assert row[1] == datetime.time(13, 45, 30, 500000)
    assert row[2] == datetime.datetime(2026, 5, 27, 13, 45, 30, 123456)
    assert row[3] == datetime.datetime(2026, 5, 27, 13, 45, 30, tzinfo=datetime.UTC)


def test_blob_and_uuid_decoding(con):
    (row,) = con.execute(
        "SELECT '\\xAA\\xBB'::BLOB, '6f9619ff-8b86-d011-b42d-00c04fc964ff'::UUID"
    ).fetchall()
    assert row[0] == b"\xaa\xbb"
    assert row[1] == uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


def test_arrow_c_stream_via_pyarrow(con):
    pa = pytest.importorskip("pyarrow")
    res = con.sql("SELECT i, i * 2 AS d FROM range(3) t(i)")
    table = pa.table(res)  # consumes __arrow_c_stream__
    assert table.column_names == ["i", "d"]
    assert table.to_pydict() == {"i": [0, 1, 2], "d": [0, 2, 4]}


def test_arrow_method(con):
    pytest.importorskip("pyarrow")
    table = con.sql("SELECT 7 AS x, 'hi' AS s").arrow()
    assert table.to_pydict() == {"x": [7], "s": ["hi"]}


def test_pandas_df(con):
    pytest.importorskip("pyarrow")
    pytest.importorskip("pandas")
    frame = con.sql("SELECT 1 AS a, 'x' AS b").df()
    assert list(frame.columns) == ["a", "b"]
    assert frame.iloc[0]["a"] == 1
    assert frame.iloc[0]["b"] == "x"


def test_polars_df(con):
    pytest.importorskip("polars")
    frame = con.sql("SELECT i FROM range(3) t(i)").pl()
    assert frame.columns == ["i"]
    assert frame["i"].to_list() == [0, 1, 2]


def test_fetchnumpy(con):
    pytest.importorskip("pyarrow")
    arrays = con.sql("SELECT i FROM range(4) t(i)").fetchnumpy()
    assert sorted(arrays) == ["i"]
    assert list(arrays["i"]) == [0, 1, 2, 3]


def test_connection_arrow_delegation(con):
    pytest.importorskip("pyarrow")
    con.execute("SELECT 5 AS v")
    assert con.arrow().to_pydict() == {"v": [5]}


def test_result_outlives_closed_connection(con):
    pytest.importorskip("pyarrow")
    res = con.sql("SELECT i FROM range(3) t(i)")
    con.close()  # connection closed, but res shares the underlying handle
    assert res.arrow().to_pydict() == {"i": [0, 1, 2]}


def test_arrow_stream_outlives_connection(con):
    import gc

    pa = pytest.importorskip("pyarrow")
    reader = pa.RecordBatchReader.from_stream(con.sql("SELECT i FROM range(4) t(i)"))
    con.close()
    del con
    gc.collect()  # nothing but the reader's stream now keeps the handle alive
    assert reader.read_all().to_pydict() == {"i": [0, 1, 2, 3]}


def test_list_decoding(con):
    (row,) = con.execute("SELECT [1, 2, 3], [[1], [], [2, 3]]").fetchall()
    assert row == ([1, 2, 3], [[1], [], [2, 3]])


def test_struct_and_map_decoding(con):
    (row,) = con.execute("SELECT {'a': 1, 'b': 'x'}, MAP {'k1': 10, 'k2': 20}").fetchall()
    assert row[0] == {"a": 1, "b": "x"}
    assert row[1] == {"k1": 10, "k2": 20}


def test_array_decoding(con):
    (row,) = con.execute("SELECT [1, 2, 3]::INTEGER[3]").fetchall()
    assert row == ([1, 2, 3],)


def test_enum_decoding(con):
    con.execute("CREATE TYPE mood AS ENUM ('sad', 'ok', 'happy')")
    rows = con.execute("SELECT 'happy'::mood UNION ALL SELECT 'sad'::mood").fetchall()
    assert rows == [("happy",), ("sad",)]


def test_interval_decoding(con):
    (row,) = con.execute("SELECT INTERVAL 90 SECONDS").fetchall()
    assert row == (datetime.timedelta(seconds=90),)


def test_nested_null_decoding(con):
    (row,) = con.execute("SELECT [1, NULL, 3], {'a': NULL}").fetchall()
    assert row == ([1, None, 3], {"a": None})


def test_connection_executemany(con):
    con.execute("CREATE TABLE t (a INTEGER, b VARCHAR)")
    con.executemany("INSERT INTO t VALUES (?, ?)", [(1, "x"), (2, "y"), (3, None)])
    assert con.execute("SELECT * FROM t ORDER BY a").fetchall() == [
        (1, "x"),
        (2, "y"),
        (3, None),
    ]


def test_connection_executemany_error_propagates(con):
    con.execute("CREATE TABLE t (a INTEGER PRIMARY KEY)")
    with pytest.raises(ducky.Error):
        con.executemany("INSERT INTO t VALUES (?)", [(1,), (1,)])


def test_rows_changed(con):
    con.execute("CREATE TABLE t (a INTEGER)")
    assert con.execute("INSERT INTO t VALUES (1), (2), (3)").rows_changed == 3
    assert con.execute("UPDATE t SET a = a + 1 WHERE a > 1").rows_changed == 2
    assert con.execute("DELETE FROM t").rows_changed == 3
    # SELECTs don't report changed rows.
    assert con.execute("SELECT 1").rows_changed == 0
