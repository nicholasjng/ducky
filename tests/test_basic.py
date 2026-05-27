import datetime
import uuid
from decimal import Decimal

import pytest

import ducky


def test_version_attrs():
    assert isinstance(ducky.__duckdb_version__, str)
    assert ducky.__duckdb_version__


def test_scalar_select():
    con = ducky.connect()
    # Note: a bare `3.5` literal is DECIMAL in DuckDB, which ducky does not
    # decode yet, so cast to DOUBLE explicitly.
    rows = con.execute(
        "SELECT 42, 'hello', CAST(3.5 AS DOUBLE), TRUE, CAST(NULL AS INTEGER)"
    ).fetchall()
    assert rows == [(42, "hello", 3.5, True, None)]


def test_fetchone_and_fetchmany():
    con = ducky.connect()
    con.execute("SELECT * FROM range(5) t(i)")
    assert con.fetchone() == (0,)
    assert con.fetchmany(2) == [(1,), (2,)]
    assert con.fetchall() == [(3,), (4,)]
    assert con.fetchone() is None


def test_description_and_columns():
    res = ducky.connect().sql("SELECT 1 AS a, 'x' AS b")
    assert res.columns == ["a", "b"]
    assert res.types == ["INTEGER", "VARCHAR"]
    assert [d[0] for d in res.description] == ["a", "b"]


def test_positional_parameters():
    con = ducky.connect()
    rows = con.execute("SELECT ? + ?, ?", [10, 5, "hi"]).fetchall()
    assert rows == [(15, "hi")]


def test_table_roundtrip():
    con = ducky.connect()
    con.execute("CREATE TABLE t (id INTEGER, name VARCHAR)")
    con.execute("INSERT INTO t VALUES (1, 'a'), (2, 'b')")
    assert con.execute("SELECT count(*) FROM t").fetchall() == [(2,)]
    assert con.sql("SELECT name FROM t ORDER BY id").fetchall() == [("a",), ("b",)]


def test_result_is_iterable():
    res = ducky.connect().sql("SELECT * FROM range(3) t(i)")
    assert [row[0] for row in res] == [0, 1, 2]


def test_error_is_raised():
    con = ducky.connect()
    with pytest.raises(ducky.Error):
        con.execute("SELECT * FROM does_not_exist")


def test_context_manager():
    with ducky.connect() as con:
        assert con.execute("SELECT 1").fetchall() == [(1,)]


def test_decimal_decoding():
    con = ducky.connect()
    rows = con.execute(
        "SELECT 3.5, CAST('123456789.123456789' AS DECIMAL(38, 9)), 0.00::DECIMAL(4, 2)"
    ).fetchall()
    assert rows == [(Decimal("3.5"), Decimal("123456789.123456789"), Decimal("0.00"))]


def test_hugeint_decoding():
    con = ducky.connect()
    (row,) = con.execute(
        "SELECT 170141183460469231731687303715884105727::HUGEINT, (-12345678901234567890)::HUGEINT"
    ).fetchall()
    assert row == (170141183460469231731687303715884105727, -12345678901234567890)


def test_temporal_decoding():
    con = ducky.connect()
    (row,) = con.execute(
        "SELECT DATE '2026-05-27', TIME '13:45:30.5', "
        "TIMESTAMP '2026-05-27 13:45:30.123456', "
        "TIMESTAMPTZ '2026-05-27 13:45:30+00'"
    ).fetchall()
    assert row[0] == datetime.date(2026, 5, 27)
    assert row[1] == datetime.time(13, 45, 30, 500000)
    assert row[2] == datetime.datetime(2026, 5, 27, 13, 45, 30, 123456)
    assert row[3] == datetime.datetime(
        2026, 5, 27, 13, 45, 30, tzinfo=datetime.timezone.utc
    )


def test_blob_and_uuid_decoding():
    con = ducky.connect()
    (row,) = con.execute(
        "SELECT '\\xAA\\xBB'::BLOB, '6f9619ff-8b86-d011-b42d-00c04fc964ff'::UUID"
    ).fetchall()
    assert row[0] == b"\xaa\xbb"
    assert row[1] == uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


def test_arrow_c_stream_via_pyarrow():
    pa = pytest.importorskip("pyarrow")
    res = ducky.connect().sql("SELECT i, i * 2 AS d FROM range(3) t(i)")
    table = pa.table(res)  # consumes __arrow_c_stream__
    assert table.column_names == ["i", "d"]
    assert table.to_pydict() == {"i": [0, 1, 2], "d": [0, 2, 4]}


def test_arrow_method():
    pytest.importorskip("pyarrow")
    table = ducky.connect().sql("SELECT 7 AS x, 'hi' AS s").arrow()
    assert table.to_pydict() == {"x": [7], "s": ["hi"]}


def test_pandas_df():
    pytest.importorskip("pyarrow")
    pytest.importorskip("pandas")
    frame = ducky.connect().sql("SELECT 1 AS a, 'x' AS b").df()
    assert list(frame.columns) == ["a", "b"]
    assert frame.iloc[0]["a"] == 1
    assert frame.iloc[0]["b"] == "x"


def test_polars_df():
    pytest.importorskip("polars")
    frame = ducky.connect().sql("SELECT i FROM range(3) t(i)").pl()
    assert frame.columns == ["i"]
    assert frame["i"].to_list() == [0, 1, 2]


def test_fetchnumpy():
    pytest.importorskip("pyarrow")
    arrays = ducky.connect().sql("SELECT i FROM range(4) t(i)").fetchnumpy()
    assert sorted(arrays) == ["i"]
    assert list(arrays["i"]) == [0, 1, 2, 3]


def test_connection_arrow_delegation():
    pytest.importorskip("pyarrow")
    con = ducky.connect()
    con.execute("SELECT 5 AS v")
    assert con.arrow().to_pydict() == {"v": [5]}


def test_result_outlives_closed_connection():
    pytest.importorskip("pyarrow")
    con = ducky.connect()
    res = con.sql("SELECT i FROM range(3) t(i)")
    con.close()  # connection closed, but res shares the underlying handle
    assert res.arrow().to_pydict() == {"i": [0, 1, 2]}


def test_arrow_stream_outlives_connection():
    import gc

    pa = pytest.importorskip("pyarrow")
    con = ducky.connect()
    reader = pa.RecordBatchReader.from_stream(con.sql("SELECT i FROM range(4) t(i)"))
    con.close()
    del con
    gc.collect()  # nothing but the reader's stream now keeps the handle alive
    assert reader.read_all().to_pydict() == {"i": [0, 1, 2, 3]}


def test_list_decoding():
    con = ducky.connect()
    (row,) = con.execute("SELECT [1, 2, 3], [[1], [], [2, 3]]").fetchall()
    assert row == ([1, 2, 3], [[1], [], [2, 3]])


def test_struct_and_map_decoding():
    con = ducky.connect()
    (row,) = con.execute(
        "SELECT {'a': 1, 'b': 'x'}, MAP {'k1': 10, 'k2': 20}"
    ).fetchall()
    assert row[0] == {"a": 1, "b": "x"}
    assert row[1] == {"k1": 10, "k2": 20}


def test_array_decoding():
    con = ducky.connect()
    (row,) = con.execute("SELECT [1, 2, 3]::INTEGER[3]").fetchall()
    assert row == ([1, 2, 3],)


def test_enum_decoding():
    con = ducky.connect()
    con.execute("CREATE TYPE mood AS ENUM ('sad', 'ok', 'happy')")
    rows = con.execute("SELECT 'happy'::mood UNION ALL SELECT 'sad'::mood").fetchall()
    assert rows == [("happy",), ("sad",)]


def test_interval_decoding():
    con = ducky.connect()
    (row,) = con.execute("SELECT INTERVAL 90 SECONDS").fetchall()
    assert row == (datetime.timedelta(seconds=90),)


def test_nested_null_decoding():
    con = ducky.connect()
    (row,) = con.execute("SELECT [1, NULL, 3], {'a': NULL}").fetchall()
    assert row == ([1, None, 3], {"a": None})


def test_large_result_spans_chunks():
    # More than one STANDARD_VECTOR_SIZE (2048) chunk.
    con = ducky.connect()
    rows = con.execute("SELECT * FROM range(5000) t(i)").fetchall()
    assert len(rows) == 5000
    assert rows[0] == (0,)
    assert rows[-1] == (4999,)
