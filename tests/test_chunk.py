import numpy as np
import pytest

import ducky


def test_fetch_chunk_basic(con):
    con.execute("CREATE TABLE t (a INTEGER, b DOUBLE)")
    con.execute("INSERT INTO t VALUES (1, 1.5), (2, 2.5), (3, 3.5)")
    chunk = con.sql("SELECT * FROM t ORDER BY a").fetch_chunk()
    assert chunk is not None
    assert len(chunk) == 3
    assert chunk.columns == ["a", "b"]
    assert chunk.types == ["INTEGER", "DOUBLE"]
    np.testing.assert_array_equal(chunk.column("a"), np.array([1, 2, 3], dtype=np.int32))
    np.testing.assert_array_equal(chunk.column(1), np.array([1.5, 2.5, 3.5]))


def test_fetch_chunk_end_of_stream(con):
    res = con.sql("SELECT 1 AS x")
    assert res.fetch_chunk() is not None
    assert res.fetch_chunk() is None


def test_multi_chunk_scan_reuses_pool_without_corrupting_data(con):
    # Regression test for fetch_chunk()'s buffer pool: each chunk is dropped
    # before the next is fetched (the common streaming pattern), so the pool
    # reuses the same underlying duckdb_data_chunk buffer across calls. Every
    # chunk's values must still be independently correct -- a reuse bug would
    # show up as stale/aliased data from a previous chunk bleeding into the
    # next.
    n = 5000  # > 2 * STANDARD_VECTOR_SIZE (2048), so several chunks get pooled and reused
    res = con.sql(f"SELECT i FROM range({n}) t(i)")
    seen = []
    while (chunk := res.fetch_chunk()) is not None:
        seen.extend(int(v) for v in chunk.column("i"))
    assert seen == list(range(n))


def test_two_live_chunks_fall_back_without_pool_and_stay_independent(con):
    # If the pool is empty (a caller holding more than one live batch at
    # once), fetch_chunk() must fall back to a fresh allocation rather than
    # reuse a buffer that's still referenced -- otherwise chunk_a's values
    # would get silently overwritten once chunk_b is fetched.
    n = 3 * 2048  # three full STANDARD_VECTOR_SIZE chunks
    res = con.sql(f"SELECT i FROM range({n}) t(i)")
    chunk_a = res.fetch_chunk()
    chunk_b = res.fetch_chunk()  # pool is empty (chunk_a still live) -> fresh allocation
    assert chunk_a is not None
    assert chunk_b is not None
    a_values = chunk_a.column("i").copy()
    b_values = chunk_b.column("i").copy()
    np.testing.assert_array_equal(a_values, np.arange(0, 2048, dtype=np.int64))
    np.testing.assert_array_equal(b_values, np.arange(2048, 4096, dtype=np.int64))
    # chunk_a's buffer must not have been mutated by fetching chunk_b.
    np.testing.assert_array_equal(chunk_a.column("i"), a_values)

    # Drop both, so their buffers return to the pool; fetching the third
    # chunk should still be correct whether or not it reused one of them.
    del chunk_a, chunk_b
    chunk_c = res.fetch_chunk()
    assert chunk_c is not None
    np.testing.assert_array_equal(chunk_c.column("i"), np.arange(4096, 6144, dtype=np.int64))
    assert res.fetch_chunk() is None


def test_column_dtypes(con):
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
        "b": np.bool,
    }
    for name, dtype in expected.items():
        assert chunk.column(name).dtype == np.dtype(dtype), name


def test_validity_mask(con):
    chunk = con.sql("SELECT * FROM (VALUES (1, 10), (2, NULL), (3, 30)) AS t(a, b)").fetch_chunk()
    assert chunk is not None
    assert chunk.validity("a") is None  # no nulls
    mask = chunk.validity("b")
    np.testing.assert_array_equal(mask, np.array([1, 0, 1], dtype=np.uint8))


def test_unsupported_type_raises(con):
    chunk = con.sql("SELECT 'hello' AS s").fetch_chunk()
    assert chunk is not None
    with pytest.raises(ducky.Error, match="VARCHAR"):
        chunk.column("s")


def test_column_view_outlives_chunk(con):
    res = con.sql("SELECT i FROM range(5) t(i)")
    chunk = res.fetch_chunk()
    assert chunk is not None
    arr = chunk.column("i")
    del chunk  # chunk dropped; arr should keep the buffer alive via nb owner
    np.testing.assert_array_equal(arr, np.arange(5, dtype=np.int64))


def test_bad_key_errors(con):
    chunk = con.sql("SELECT 1 AS a").fetch_chunk()
    assert chunk is not None
    with pytest.raises(ducky.Error, match="no such column"):
        chunk.column("nope")
    with pytest.raises(ducky.Error, match="out of range"):
        chunk.column(5)


# ── Structured / wide-integer types ───────────────────────────────────────


def test_hugeint_column(con):
    chunk = con.sql("SELECT CAST(1 AS HUGEINT) h").fetch_chunk()
    assert chunk is not None
    col = chunk.column("h")
    assert col.dtype == np.dtype([("lower", "<u8"), ("upper", "<i8")])
    assert col[0]["lower"] == 1
    assert col[0]["upper"] == 0


def test_uhugeint_column(con):
    chunk = con.sql("SELECT CAST(1 AS UHUGEINT) h").fetch_chunk()
    assert chunk is not None
    col = chunk.column("h")
    assert col.dtype == np.dtype([("lower", "<u8"), ("upper", "<u8")])
    assert col[0]["lower"] == 1
    assert col[0]["upper"] == 0


def test_interval_column(con):
    chunk = con.sql("SELECT INTERVAL '2' MONTH AS iv").fetch_chunk()
    assert chunk is not None
    col = chunk.column("iv")
    assert col.dtype == np.dtype([("months", "<i4"), ("days", "<i4"), ("micros", "<i8")])
    assert col[0]["months"] == 2
    assert col[0]["days"] == 0
    assert col[0]["micros"] == 0


def test_decimal_column_small(con):
    chunk = con.sql("SELECT CAST(1.5 AS DECIMAL(4,2)) d").fetch_chunk()
    assert chunk is not None
    col = chunk.column("d")
    # DECIMAL(width<=4) is DuckDB's smallest internal storage class, INT16 —
    # raw integer value should be 150 (1.5 * 10^2).
    assert col.dtype == np.dtype("int16")
    assert col[0] == 150


def test_decimal_scale(con):
    chunk = con.sql("SELECT CAST(1.5 AS DECIMAL(5,2)) d").fetch_chunk()
    assert chunk is not None
    assert chunk.decimal_scale("d") == 2
    assert chunk.decimal_scale(0) == 2


def test_decimal_scale_non_decimal_raises(con):
    chunk = con.sql("SELECT 42 AS x").fetch_chunk()
    assert chunk is not None
    with pytest.raises(ducky.Error, match="non-DECIMAL"):
        chunk.decimal_scale("x")


# ── DLPack export ──────────────────────────────────────────────────────────


def test_dlpack_protocol_returns_valid_capsule(con):
    chunk = con.sql("SELECT i FROM range(3) t(i)").fetch_chunk()
    assert chunk is not None
    obj = chunk.dlpack("i")
    device_type, device_id = obj.__dlpack_device__()
    assert (device_type, device_id) == (1, 0)  # kDLCPU, device 0
    capsule = obj.__dlpack__()
    assert type(capsule).__name__ == "PyCapsule"


def test_dlpack_torch(con):
    torch = pytest.importorskip("torch")
    chunk = con.sql("SELECT i, CAST(i * 1.5 AS DOUBLE) d FROM range(4) t(i)").fetch_chunk()
    assert chunk is not None
    ti = torch.from_dlpack(chunk.dlpack("i"))
    td = torch.from_dlpack(chunk.dlpack("d"))
    assert ti.dtype == torch.int64
    assert td.dtype == torch.float64
    assert ti.tolist() == [0, 1, 2, 3]
    assert td.tolist() == [0.0, 1.5, 3.0, 4.5]


def test_dlpack_jax(con):
    pytest.importorskip("jax")
    import jax.numpy as jnp

    chunk = con.sql("SELECT CAST(i AS FLOAT) f FROM range(3) t(i)").fetch_chunk()
    assert chunk is not None
    arr = jnp.asarray(chunk.dlpack("f"))
    np.testing.assert_allclose(np.asarray(arr), [0.0, 1.0, 2.0])


def test_dlpack_outlives_chunk(con):
    torch = pytest.importorskip("torch")
    res = con.sql("SELECT i FROM range(5) t(i)")
    chunk = res.fetch_chunk()
    assert chunk is not None
    t = torch.from_dlpack(chunk.dlpack("i"))
    del chunk
    assert t.tolist() == [0, 1, 2, 3, 4]


def test_dlpack_structured_type_raises(con):
    chunk = con.sql("SELECT CAST(1 AS HUGEINT) h").fetch_chunk()
    assert chunk is not None
    with pytest.raises(ducky.Error, match="structured"):
        chunk.dlpack("h")


def test_dlpack_varchar_raises(con):
    chunk = con.sql("SELECT 'hello' AS s").fetch_chunk()
    assert chunk is not None
    with pytest.raises(ducky.Error, match="VARCHAR"):
        chunk.dlpack("s")
