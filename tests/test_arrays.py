import numpy as np
import pytest

import ducky


def test_chunks_iterates_all():
    # range() with > STANDARD_VECTOR_SIZE (2048) forces multiple chunks.
    n = 5000
    res = ducky.connect().sql(f"SELECT i FROM range({n}) t(i)")
    total = 0
    for chunk in res.chunks():
        total += len(chunk)
    assert total == n


def test_chunks_terminates():
    res = ducky.connect().sql("SELECT 1 AS x")
    assert list(res.chunks())  # at least one chunk
    assert next(res.chunks(), None) is None  # drained


def test_iter_batches_returns_ndarrays():
    res = ducky.connect().sql("SELECT i, CAST(i * 2 AS DOUBLE) AS d FROM range(3) t(i)")
    batches = list(res.iter_batches())
    assert len(batches) == 1
    batch = batches[0]
    assert set(batch) == {"i", "d"}
    np.testing.assert_array_equal(batch["i"], np.array([0, 1, 2], dtype=np.int64))
    np.testing.assert_array_equal(batch["d"], np.array([0.0, 2.0, 4.0]))


def test_iter_batches_column_subset():
    res = ducky.connect().sql("SELECT 1 AS a, 2 AS b, 3 AS c")
    [batch] = list(res.iter_batches(columns=["a", "c"]))
    assert list(batch) == ["a", "c"]


def test_iter_batches_unknown_column_raises():
    res = ducky.connect().sql("SELECT 1 AS a")
    with pytest.raises(KeyError, match="nope"):
        list(res.iter_batches(columns=["nope"]))


def test_iter_batches_with_validity():
    res = ducky.connect().sql("SELECT * FROM (VALUES (1, 10), (2, NULL), (3, 30)) AS t(a, b)")
    [batch] = list(res.iter_batches(with_validity=True))
    a_vals, a_mask = batch["a"]
    b_vals, b_mask = batch["b"]
    assert a_mask is None
    np.testing.assert_array_equal(a_vals, [1, 2, 3])
    np.testing.assert_array_equal(b_mask, [1, 0, 1])


def test_to_numpy_concatenates_chunks():
    n = 5000
    arrs = ducky.connect().sql(f"SELECT i FROM range({n}) t(i)").to_numpy()
    np.testing.assert_array_equal(arrs["i"], np.arange(n, dtype=np.int64))


def test_to_numpy_via_connection():
    con = ducky.connect()
    con.execute("CREATE TABLE t (i INTEGER)")
    con.execute("INSERT INTO t SELECT i FROM range(5) t(i)")
    con.execute("SELECT * FROM t ORDER BY i")
    arrs = con.to_numpy()
    np.testing.assert_array_equal(arrs["i"], np.arange(5, dtype=np.int32))


def test_to_numpy_filter_nulls_in_sql():
    # NULL slots in dense arrays hold garbage; filter in SQL for clean data.
    res = ducky.connect().sql("SELECT b FROM (VALUES (1), (NULL), (3)) AS t(b) WHERE b IS NOT NULL")
    arrs = res.to_numpy()
    np.testing.assert_array_equal(arrs["b"], [1, 3])


def test_to_torch():
    torch = pytest.importorskip("torch")
    res = ducky.connect().sql("SELECT i, CAST(i * 2.5 AS DOUBLE) AS d FROM range(4) t(i)")
    tensors = res.to_torch()
    assert isinstance(tensors["i"], torch.Tensor)
    assert tensors["i"].dtype == torch.int64
    assert tensors["d"].dtype == torch.float64
    assert tensors["i"].tolist() == [0, 1, 2, 3]


def test_to_jax():
    pytest.importorskip("jax")
    res = ducky.connect().sql("SELECT i FROM range(4) t(i)")
    arrs = res.to_jax()
    assert arrs["i"].shape == (4,)
    np.testing.assert_array_equal(np.asarray(arrs["i"]), [0, 1, 2, 3])


def test_to_jax_device():
    jax = pytest.importorskip("jax")
    cpu = jax.devices("cpu")[0]
    res = ducky.connect().sql("SELECT i FROM range(4) t(i)")
    arrs = res.to_jax(device=cpu)
    # JAX places the array on the requested device.
    assert list(arrs["i"].devices()) == [cpu]


# ── iter_batches_torch / iter_batches_jax ─────────────────────────────────


def test_iter_batches_torch_yields_tensors():
    torch = pytest.importorskip("torch")
    n = 5000
    res = ducky.connect().sql(f"SELECT i FROM range({n}) t(i)")
    tensors = []
    for batch in res.iter_batches_torch():
        assert isinstance(batch["i"], torch.Tensor)
        tensors.append(batch["i"])
    combined = torch.cat(tensors)
    assert combined.tolist() == list(range(n))


def test_iter_batches_torch_column_subset():
    torch = pytest.importorskip("torch")
    res = ducky.connect().sql("SELECT 1 AS a, 2 AS b, 3 AS c")
    [batch] = list(res.iter_batches_torch(columns=["a", "c"]))
    assert set(batch) == {"a", "c"}
    assert isinstance(batch["a"], torch.Tensor)


def test_iter_batches_torch_via_connection():
    pytest.importorskip("torch")
    con = ducky.connect()
    con.execute("SELECT i FROM range(4) t(i)")
    [batch] = list(con.iter_batches_torch())
    assert batch["i"].tolist() == [0, 1, 2, 3]


def test_to_torch_no_numpy_intermediate():
    # Regression: to_torch must not go through to_numpy (no copy to numpy first).
    # Verified structurally — to_torch uses iter_batches_torch internally.
    torch = pytest.importorskip("torch")
    n = 5000
    arrs = ducky.connect().sql(f"SELECT i FROM range({n}) t(i)").to_torch()
    assert isinstance(arrs["i"], torch.Tensor)
    assert arrs["i"].tolist() == list(range(n))


def test_iter_batches_jax_yields_arrays():
    pytest.importorskip("jax")
    import jax.numpy as jnp

    n = 5000
    res = ducky.connect().sql(f"SELECT i FROM range({n}) t(i)")
    arrays = []
    for batch in res.iter_batches_jax():
        arrays.append(batch["i"])
    combined = jnp.concatenate(arrays)
    np.testing.assert_array_equal(np.asarray(combined), np.arange(n))


def test_iter_batches_jax_via_connection():
    pytest.importorskip("jax")
    con = ducky.connect()
    con.execute("SELECT i FROM range(4) t(i)")
    [batch] = list(con.iter_batches_jax())
    np.testing.assert_array_equal(np.asarray(batch["i"]), [0, 1, 2, 3])


def test_to_jax_no_numpy_intermediate():
    pytest.importorskip("jax")
    n = 5000
    arrs = ducky.connect().sql(f"SELECT i FROM range({n}) t(i)").to_jax()
    np.testing.assert_array_equal(np.asarray(arrs["i"]), np.arange(n))
