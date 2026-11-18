import numpy as np
import pytest


def test_chunks_iterates_all(con):
    # range() with > STANDARD_VECTOR_SIZE (2048) forces multiple chunks.
    n = 5000
    res = con.sql(f"SELECT i FROM range({n}) t(i)")
    total = 0
    for chunk in res.chunks():
        total += len(chunk)
    assert total == n


def test_chunks_terminates(con):
    res = con.sql("SELECT 1 AS x")
    assert list(res.chunks())  # at least one chunk
    assert next(res.chunks(), None) is None  # drained


def test_iter_batches_returns_ndarrays(con):
    res = con.sql("SELECT i, CAST(i * 2 AS DOUBLE) AS d FROM range(3) t(i)")
    batches = list(res.iter_batches())
    assert len(batches) == 1
    batch = batches[0]
    assert set(batch) == {"i", "d"}
    np.testing.assert_array_equal(batch["i"], np.array([0, 1, 2], dtype=np.int64))
    np.testing.assert_array_equal(batch["d"], np.array([0.0, 2.0, 4.0]))


def test_iter_batches_column_subset(con):
    res = con.sql("SELECT 1 AS a, 2 AS b, 3 AS c")
    [batch] = list(res.iter_batches(columns=["a", "c"]))
    assert list(batch) == ["a", "c"]


def test_iter_batches_unknown_column_raises(con):
    res = con.sql("SELECT 1 AS a")
    with pytest.raises(KeyError, match="nope"):
        list(res.iter_batches(columns=["nope"]))


def test_iter_batches_with_validity(con):
    res = con.sql("SELECT * FROM (VALUES (1, 10), (2, NULL), (3, 30)) AS t(a, b)")
    [batch] = list(res.iter_batches(with_validity=True))
    a_vals, a_mask = batch["a"]
    _, b_mask = batch["b"]
    assert a_mask is None
    np.testing.assert_array_equal(a_vals, [1, 2, 3])
    np.testing.assert_array_equal(b_mask, [1, 0, 1])


def test_to_numpy_concatenates_chunks(con):
    n = 5000
    arrs = con.sql(f"SELECT i FROM range({n}) t(i)").to_numpy()
    np.testing.assert_array_equal(arrs["i"], np.arange(n, dtype=np.int64))


def test_to_numpy_via_connection(con):
    con.execute("CREATE TABLE t (i INTEGER)")
    con.execute("INSERT INTO t SELECT i FROM range(5) t(i)")
    con.execute("SELECT * FROM t ORDER BY i")
    arrs = con.to_numpy()
    np.testing.assert_array_equal(arrs["i"], np.arange(5, dtype=np.int32))


def test_to_numpy_filter_nulls_in_sql(con):
    # NULLs in an integer column raise, so filter them in SQL.
    res = con.sql("SELECT b FROM (VALUES (1), (NULL), (3)) AS t(b) WHERE b IS NOT NULL")
    arrs = res.to_numpy()
    np.testing.assert_array_equal(arrs["b"], [1, 3])


def test_to_torch(con):
    torch = pytest.importorskip("torch")
    res = con.sql("SELECT i, CAST(i * 2.5 AS DOUBLE) AS d FROM range(4) t(i)")
    tensors = res.to_torch()
    assert isinstance(tensors["i"], torch.Tensor)
    assert tensors["i"].dtype == torch.int64
    assert tensors["d"].dtype == torch.float64
    assert tensors["i"].tolist() == [0, 1, 2, 3]


def test_to_jax(con):
    pytest.importorskip("jax")
    res = con.sql("SELECT i FROM range(4) t(i)")
    arrs = res.to_jax()
    assert arrs["i"].shape == (4,)
    np.testing.assert_array_equal(np.asarray(arrs["i"]), [0, 1, 2, 3])


def test_to_jax_device(con):
    jax = pytest.importorskip("jax")
    cpu = jax.devices("cpu")[0]
    res = con.sql("SELECT i FROM range(4) t(i)")
    arrs = res.to_jax(device=cpu)
    # JAX places the array on the requested device.
    assert list(arrs["i"].devices()) == [cpu]


# ── iter_batches_torch / iter_batches_jax ─────────────────────────────────


def test_iter_batches_torch_yields_tensors(con):
    torch = pytest.importorskip("torch")
    n = 5000
    res = con.sql(f"SELECT i FROM range({n}) t(i)")
    tensors = []
    for batch in res.iter_batches_torch():
        assert isinstance(batch["i"], torch.Tensor)
        tensors.append(batch["i"])
    combined = torch.cat(tensors)
    assert combined.tolist() == list(range(n))


def test_iter_batches_torch_column_subset(con):
    torch = pytest.importorskip("torch")
    res = con.sql("SELECT 1 AS a, 2 AS b, 3 AS c")
    [batch] = list(res.iter_batches_torch(columns=["a", "c"]))
    assert set(batch) == {"a", "c"}
    assert isinstance(batch["a"], torch.Tensor)


def test_iter_batches_torch_via_connection(con):
    pytest.importorskip("torch")
    con.execute("SELECT i FROM range(4) t(i)")
    [batch] = list(con.iter_batches_torch())
    assert batch["i"].tolist() == [0, 1, 2, 3]


def test_to_torch_goes_through_to_numpy(con, monkeypatch):
    # to_numpy is measurably faster than accumulating iter_batches_torch chunks,
    # and a structural check catches a regression to the slow path.
    torch = pytest.importorskip("torch")
    from ducky._core import Result

    calls = []
    orig = Result.to_numpy
    monkeypatch.setattr(
        Result, "to_numpy", lambda self, columns=None: (calls.append(1), orig(self, columns))[1]
    )
    n = 5000
    arrs = con.sql(f"SELECT i FROM range({n}) t(i)").to_torch()
    assert calls == [1]
    assert isinstance(arrs["i"], torch.Tensor)
    assert arrs["i"].tolist() == list(range(n))


def test_iter_batches_jax_yields_arrays(con):
    pytest.importorskip("jax")
    import jax.numpy as jnp

    n = 5000
    res = con.sql(f"SELECT i FROM range({n}) t(i)")
    arrays = [batch["i"] for batch in res.iter_batches_jax()]
    combined = jnp.concatenate(arrays)
    np.testing.assert_array_equal(np.asarray(combined), np.arange(n))


def test_iter_batches_jax_via_connection(con):
    pytest.importorskip("jax")
    con.execute("SELECT i FROM range(4) t(i)")
    [batch] = list(con.iter_batches_jax())
    np.testing.assert_array_equal(np.asarray(batch["i"]), [0, 1, 2, 3])


def test_to_jax_goes_through_to_numpy(con, monkeypatch):
    # See test_to_torch_goes_through_to_numpy: same reasoning, same guard.
    pytest.importorskip("jax")
    from ducky._core import Result

    calls = []
    orig = Result.to_numpy
    monkeypatch.setattr(
        Result, "to_numpy", lambda self, columns=None: (calls.append(1), orig(self, columns))[1]
    )
    n = 5000
    arrs = con.sql(f"SELECT i FROM range({n}) t(i)").to_jax()
    assert calls == [1]
    np.testing.assert_array_equal(np.asarray(arrs["i"]), np.arange(n))


# ── to_mlx / iter_batches_mlx ──────────────────────────────────────────────


def test_iter_batches_mlx_yields_arrays(con):
    mx = pytest.importorskip("mlx.core")
    n = 5000
    res = con.sql(f"SELECT i FROM range({n}) t(i)")
    arrays = [batch["i"] for batch in res.iter_batches_mlx()]
    combined = mx.concatenate(arrays)
    np.testing.assert_array_equal(np.asarray(combined), np.arange(n))


def test_iter_batches_mlx_via_connection(con):
    pytest.importorskip("mlx.core")
    con.execute("SELECT i FROM range(4) t(i)")
    [batch] = list(con.iter_batches_mlx())
    np.testing.assert_array_equal(np.asarray(batch["i"]), [0, 1, 2, 3])


def test_to_mlx_goes_through_to_numpy(con, monkeypatch):
    # See test_to_torch_goes_through_to_numpy: same reasoning, same guard.
    pytest.importorskip("mlx.core")
    from ducky._core import Result

    calls = []
    orig = Result.to_numpy
    monkeypatch.setattr(
        Result, "to_numpy", lambda self, columns=None: (calls.append(1), orig(self, columns))[1]
    )
    n = 5000
    arrs = con.sql(f"SELECT i FROM range({n}) t(i)").to_mlx()
    assert calls == [1]
    np.testing.assert_array_equal(np.asarray(arrs["i"]), np.arange(n))


def test_to_numpy_fixed_size_array(con):
    n = 5000  # spans chunks
    res = con.sql(
        f"SELECT [i, i + 1, i + 2]::FLOAT[3] e, [[i, 0], [0, i]]::INT[2][2] m FROM range({n}) t(i)"
    )
    arrays = res.to_numpy()
    i = np.arange(n, dtype=np.float32)
    assert arrays["e"].shape == (n, 3)
    assert arrays["e"].dtype == np.float32
    np.testing.assert_array_equal(arrays["e"], np.stack([i, i + 1, i + 2], axis=1))
    assert arrays["m"].shape == (n, 2, 2)
    np.testing.assert_array_equal(arrays["m"][7], [[7, 0], [0, 7]])


def test_to_numpy_array_nulls_become_nan(con):
    res = con.sql(
        "SELECT CASE i WHEN 1 THEN NULL WHEN 2 THEN [NULL, 1, 2]::DOUBLE[3] "
        "ELSE [i, i, i]::DOUBLE[3] END e FROM range(4) t(i)"
    )
    np.testing.assert_array_equal(
        res.to_numpy()["e"],
        [[0, 0, 0], [np.nan, np.nan, np.nan], [np.nan, 1, 2], [3, 3, 3]],
    )


def test_to_numpy_integer_array_nulls_raise(con):
    with pytest.raises(Exception, match="contains NULLs"):
        con.sql("SELECT [1, NULL]::INT[2] e").to_numpy()
    with pytest.raises(Exception, match="contains NULLs"):
        con.sql("SELECT NULL::INT[2] e").to_numpy()


def test_to_numpy_array_of_unsupported_type_raises(con):
    with pytest.raises(Exception, match="ARRAY of a type"):
        con.sql("SELECT ['a']::VARCHAR[1] e").to_numpy()


def test_iter_batches_fixed_size_array(con):
    res = con.sql("SELECT [i, -i]::BIGINT[2] e FROM range(3000) t(i)")
    batches = [batch["e"] for batch in res.iter_batches()]
    assert [b.shape for b in batches] == [(2048, 2), (952, 2)]
    np.testing.assert_array_equal(np.concatenate(batches)[2999], [2999, -2999])


def test_chunk_dlpack_fixed_size_array(con):
    torch = pytest.importorskip("torch")
    [chunk] = con.sql("SELECT [i, i * 2]::FLOAT[2] e FROM range(4) t(i)").chunks()
    tensor = torch.from_dlpack(chunk.dlpack("e"))
    assert tensor.shape == (4, 2)
    np.testing.assert_array_equal(tensor.numpy(), chunk.column("e"))
    np.testing.assert_array_equal(tensor[3].numpy(), [3, 6])
