import numpy as np
import pytest

import ducky


def test_fetch_chunk_basic():
    con = ducky.connect()
    con.execute("CREATE TABLE t (a INTEGER, b DOUBLE)")
    con.execute("INSERT INTO t VALUES (1, 1.5), (2, 2.5), (3, 3.5)")
    chunk = con.sql("SELECT * FROM t ORDER BY a").fetch_chunk()
    assert chunk is not None
    assert len(chunk) == 3
    assert chunk.columns == ["a", "b"]
    assert chunk.types == ["INTEGER", "DOUBLE"]
    np.testing.assert_array_equal(chunk.column("a"), np.array([1, 2, 3], dtype=np.int32))
    np.testing.assert_array_equal(chunk.column(1), np.array([1.5, 2.5, 3.5]))


def test_fetch_chunk_end_of_stream():
    res = ducky.connect().sql("SELECT 1 AS x")
    assert res.fetch_chunk() is not None
    assert res.fetch_chunk() is None


def test_column_dtypes():
    con = ducky.connect()
    chunk = con.sql(
        "SELECT CAST(1 AS TINYINT) ti, CAST(1 AS SMALLINT) si, "
        "CAST(1 AS INTEGER) i, CAST(1 AS BIGINT) bi, "
        "CAST(1 AS UTINYINT) uti, CAST(1 AS USMALLINT) usi, "
        "CAST(1 AS UINTEGER) ui, CAST(1 AS UBIGINT) ubi, "
        "CAST(1 AS FLOAT) f, CAST(1 AS DOUBLE) d, TRUE b"
    ).fetch_chunk()
    assert chunk is not None
    expected = {
        "ti": np.int8,
        "si": np.int16,
        "i": np.int32,
        "bi": np.int64,
        "uti": np.uint8,
        "usi": np.uint16,
        "ui": np.uint32,
        "ubi": np.uint64,
        "f": np.float32,
        "d": np.float64,
        "b": np.bool_,
    }
    for name, dtype in expected.items():
        assert chunk.column(name).dtype == np.dtype(dtype), name


def test_validity_mask():
    con = ducky.connect()
    chunk = con.sql("SELECT * FROM (VALUES (1, 10), (2, NULL), (3, 30)) AS t(a, b)").fetch_chunk()
    assert chunk is not None
    assert chunk.validity("a") is None  # no nulls
    mask = chunk.validity("b")
    np.testing.assert_array_equal(mask, np.array([1, 0, 1], dtype=np.uint8))


def test_unsupported_type_raises():
    con = ducky.connect()
    chunk = con.sql("SELECT 'hello' AS s").fetch_chunk()
    assert chunk is not None
    with pytest.raises(ducky.Error, match="VARCHAR"):
        chunk.column("s")


def test_column_view_outlives_chunk():
    res = ducky.connect().sql("SELECT i FROM range(5) t(i)")
    chunk = res.fetch_chunk()
    assert chunk is not None
    arr = chunk.column("i")
    del chunk  # chunk dropped; arr should keep the buffer alive via nb owner
    np.testing.assert_array_equal(arr, np.arange(5, dtype=np.int64))


def test_bad_key_errors():
    chunk = ducky.connect().sql("SELECT 1 AS a").fetch_chunk()
    assert chunk is not None
    with pytest.raises(ducky.Error, match="no such column"):
        chunk.column("nope")
    with pytest.raises(ducky.Error, match="out of range"):
        chunk.column(5)
