from typing import Any

import numpy as np
import pytest

import ducky

N = 10_000
SPLIT = ducky.split(0.8, seed=1)
FIELDS = {
    "X": ducky.tensor("emb", 2),
    "x": ducky.vector("x", standardize=True),
    "y": ducky.vector("y", dtype="i64"),
}


@pytest.fixture(scope="module")
def parquet_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("stream") / "data.parquet"
    ducky.connect().execute(
        f"COPY (SELECT i AS id, [i, -i]::FLOAT[2] AS emb, i::DOUBLE AS x, i % 3 AS y "
        f"FROM range({N}) t(i)) TO '{p}'"
    )
    return str(p)


@pytest.fixture(scope="module")
def parquet_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("stream_files")
    con = ducky.connect()
    for f in range(4):
        con.execute(
            f"COPY (SELECT {f} * 1000 + i AS id, ({f} * 1000 + i)::DOUBLE AS x "
            f"FROM range(1000) t(i)) TO '{d}/part{f}.parquet'"
        )
    return str(d / "*.parquet")


def ids(stream, field="X"):
    """The first X component, which is the row id, of every streamed row."""
    batches = [b[field] for b in stream]
    return np.concatenate([b[:, 0] if b.ndim == 2 else b for b in batches])


def test_sequential_matches_dataset_fold(parquet_path):
    ds = ducky.dataset(parquet_path, fields=FIELDS, split=SPLIT)
    stream = ducky.stream(parquet_path, fields=FIELDS, split=SPLIT, batch_size=256)
    batches = list(stream)
    for name in FIELDS:
        got = np.concatenate([b[name] for b in batches])
        np.testing.assert_allclose(got, ds["train"][name], rtol=1e-6)
    assert all(len(b["X"]) == 256 for b in batches[:-1])


def test_val_fold_and_drop_last(parquet_path):
    ds = ducky.dataset(parquet_path, fields=FIELDS, split=SPLIT)
    stream = ducky.stream(
        parquet_path, fields=FIELDS, split=SPLIT, fold="val", batch_size=300, drop_last=True
    )
    batches = list(stream)
    assert all(len(b["y"]) == 300 for b in batches)
    assert len(batches) == ds["val"].n_rows // 300


def test_unknown_fold_rejected(parquet_path):
    with pytest.raises(ValueError, match="unknown fold"):
        ducky.stream(parquet_path, fields=FIELDS, split=SPLIT, fold="test", batch_size=8)


@pytest.mark.parametrize("shuffle", [ducky.shuffle(4, seed=3), ducky.shuffle("global", seed=3)])
def test_shuffle_permutes_and_reshuffles_per_epoch(parquet_path, shuffle):
    plain = ids(ducky.stream(parquet_path, fields=FIELDS, split=SPLIT, batch_size=128))
    stream = ducky.stream(parquet_path, fields=FIELDS, split=SPLIT, batch_size=128, shuffle=shuffle)
    first, second = ids(stream), ids(stream)
    np.testing.assert_array_equal(np.sort(first), np.sort(plain))
    assert not np.array_equal(first, plain)
    assert not np.array_equal(first, second)
    assert stream.epoch == 2
    stream.set_epoch(0)
    np.testing.assert_array_equal(ids(stream), first)


def test_buffer_shuffle_with_buffer_smaller_than_input(parquet_path):
    # A one-batch buffer still yields every row exactly once.
    stream = ducky.stream(
        parquet_path, fields=FIELDS, batch_size=1000, shuffle=ducky.shuffle(1, seed=0)
    )
    np.testing.assert_array_equal(np.sort(ids(stream)), np.arange(N))


def test_resume_from_state_dict(parquet_path):
    kwargs: dict[str, Any] = {
        "fields": FIELDS,
        "batch_size": 256,
        "shuffle": ducky.shuffle(4, seed=5),
    }
    stream = ducky.stream(parquet_path, **kwargs)
    full = [b["X"] for b in stream]
    stream.set_epoch(0)
    it = iter(stream)
    for _ in range(3):
        next(it)
    state = stream.state_dict()
    assert state == {"epoch": 0, "batch": 3}
    it.close()

    resumed = ducky.stream(parquet_path, **kwargs)
    resumed.load_state_dict(state)
    rest = [b["X"] for b in resumed]
    assert len(rest) == len(full) - 3
    for got, want in zip(rest, full[3:], strict=True):
        np.testing.assert_array_equal(got, want)


def test_iteration_cut_short_is_not_resumed(parquet_path):
    stream = ducky.stream(parquet_path, fields=FIELDS, batch_size=256, split=SPLIT)
    full = ids(stream)
    stream.set_epoch(0)
    next(iter(stream))  # a peek, say at the batch shapes
    np.testing.assert_array_equal(ids(stream), full)
    assert stream.state_dict() == {"epoch": 1, "batch": 0}


@pytest.mark.parametrize("key", [None, "id"])
def test_shards_partition_the_fold(parquet_path, key):
    whole = ids(ducky.stream(parquet_path, fields=FIELDS, split=SPLIT, key=key, batch_size=256))
    parts = [
        ids(
            ducky.stream(
                parquet_path,
                fields=FIELDS,
                split=SPLIT,
                key=key,
                batch_size=256,
                shard=(r, 3),
                even_shards=False,
            )
        )
        for r in range(3)
    ]
    assert all(len(p) for p in parts)
    np.testing.assert_array_equal(np.sort(np.concatenate(parts)), np.sort(whole))


def test_file_shards_partition_the_fold(parquet_dir):
    fields = {"x": ducky.vector("x", dtype="f64")}
    split = ducky.split(0.5, seed=0)
    whole = ids(ducky.stream(parquet_dir, fields=fields, split=split, key="id", batch_size=64), "x")
    parts = [
        ids(
            ducky.stream(
                parquet_dir,
                fields=fields,
                split=split,
                key="id",
                batch_size=64,
                shard=(r, 2),
                even_shards=False,
            ),
            "x",
        )
        for r in range(2)
    ]
    # Each shard reads two of the four files, so the ids split by thousands.
    assert {int(v) // 1000 for v in parts[0]} == {0, 2}
    assert {int(v) // 1000 for v in parts[1]} == {1, 3}
    np.testing.assert_array_equal(np.sort(np.concatenate(parts)), np.sort(whole))


@pytest.mark.parametrize("shuffle", [None, ducky.shuffle(2), ducky.shuffle("global")])
@pytest.mark.parametrize("files", [False, True])
def test_even_shards_take_the_same_steps(parquet_path, parquet_dir, files, shuffle):
    source, key = (parquet_dir, "id") if files else (parquet_path, None)
    fields = {"x": ducky.vector("x", dtype="f64")}
    whole = ids(ducky.stream(source, fields=fields, split=SPLIT, key=key, batch_size=64), "x")
    parts = []
    for r in range(3):
        stream = ducky.stream(
            source,
            fields=fields,
            split=SPLIT,
            key=key,
            batch_size=64,
            shard=(r, 3),
            shuffle=shuffle,
        )
        parts.append(ids(stream, "x"))
        parts.append(ids(stream, "x"))  # the count is cached for later epochs
    sizes = {len(p) for p in parts}
    assert len(sizes) == 1
    assert len(whole) - 3 * sizes.pop() < 3 * 64  # trimmed, but only a little
    together = np.concatenate(parts[::2])
    assert len(np.unique(together)) == len(together)
    assert set(together) <= set(whole)


def test_even_shards_keep_balanced_files(parquet_dir):
    fields = {"x": ducky.vector("x", dtype="f64")}
    parts = [
        ids(ducky.stream(parquet_dir, fields=fields, key="id", batch_size=64, shard=(r, 2)), "x")
        for r in range(2)
    ]
    assert {int(v) // 1000 for v in parts[0]} == {0, 2}
    assert len(parts[0]) == len(parts[1]) == 2000


def test_key_assigns_the_same_folds_as_dataset(parquet_dir):
    fields = {"x": ducky.vector("x", dtype="f64")}
    split = ducky.split(0.5, seed=0)
    ds = ducky.dataset(parquet_dir, fields=fields, split=split, key="id")
    for shard in [None, (0, 2)]:
        got = ids(
            ducky.stream(
                parquet_dir, fields=fields, split=split, key="id", batch_size=64, shard=shard
            ),
            "x",
        )
        assert set(got) <= set(ds["train"]["x"])
    whole = ids(ducky.stream(parquet_dir, fields=fields, split=split, key="id", batch_size=64), "x")
    np.testing.assert_array_equal(np.sort(whole), np.sort(ds["train"]["x"]))


def test_category_codes_match_dataset(parquet_path):
    fields = {"y": ducky.category("y::VARCHAR"), "id": ducky.vector("id", dtype="i64")}
    ds = ducky.dataset(parquet_path, fields=fields, split=SPLIT)
    stream = ducky.stream(parquet_path, fields=fields, split=SPLIT, fold="val", batch_size=500)
    assert stream.categories == ds.categories == {"y": ["0", "1", "2"]}
    got = np.concatenate([b["y"] for b in stream])
    np.testing.assert_array_equal(got, ds["val"]["y"])


def test_prefetch_matches_inline(parquet_path):
    kwargs: dict[str, Any] = {
        "fields": FIELDS,
        "batch_size": 512,
        "shuffle": ducky.shuffle(2, seed=0),
    }
    a = ids(ducky.stream(parquet_path, prefetch=0, **kwargs))
    b = ids(ducky.stream(parquet_path, prefetch=3, **kwargs))
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("shuffle", [None, ducky.shuffle("global")])
def test_abandoning_an_iteration_stops_the_producer(parquet_path, shuffle):
    import threading

    stream = ducky.stream(parquet_path, fields=FIELDS, batch_size=16, shuffle=shuffle)
    it = iter(stream)
    next(it)
    it.close()
    assert not [t for t in threading.enumerate() if t.name == "ducky-stream"]


def test_nulls_fill_floats_and_reject_integers(tmp_path):
    p = tmp_path / "nulls.parquet"
    ducky.connect().execute(
        f"COPY (SELECT CASE WHEN i = 3 THEN NULL ELSE i END::DOUBLE AS x, "
        f"CASE WHEN i = 3 THEN NULL ELSE i END AS n FROM range(10) t(i)) TO '{p}'"
    )
    x = ids(ducky.stream(str(p), fields={"x": ducky.vector("x", dtype="f64")}, batch_size=4), "x")
    assert np.isnan(x[3])
    assert np.nansum(x) == 45 - 3
    with pytest.raises(ducky.Error, match="contains NULLs"):
        list(ducky.stream(str(p), fields={"n": ducky.vector("n", dtype="i64")}, batch_size=4))


def test_errors_propagate_from_the_producer(parquet_path):
    stream = ducky.stream(parquet_path, fields={"z": ducky.vector("no_such_column")}, batch_size=8)
    with pytest.raises(ducky.Error, match="no_such_column"):
        list(stream)


@pytest.mark.parametrize("backend", ["torch", "jax", "mlx"])
def test_backends(parquet_path, backend):
    pytest.importorskip("mlx.core" if backend == "mlx" else backend)
    stream = ducky.stream(parquet_path, fields=FIELDS, batch_size=1000, backend=backend)
    batch = next(iter(stream))
    assert tuple(batch["X"].shape) == (1000, 2)
    np.testing.assert_array_equal(
        np.asarray(batch["X"]),
        next(iter(ducky.stream(parquet_path, fields=FIELDS, batch_size=1000)))["X"],
    )


def test_dataloader_workers_shard_automatically(parquet_path):
    torch = pytest.importorskip("torch")
    stream = ducky.stream(parquet_path, fields=FIELDS, batch_size=500, backend="torch")
    loader = torch.utils.data.DataLoader(stream, batch_size=None, num_workers=2)
    got = np.concatenate([b["X"][:, 0].numpy() for b in loader])
    np.testing.assert_array_equal(np.sort(got), np.arange(N))


def test_validation():
    with pytest.raises(ValueError, match="positive int or 'global'"):
        ducky.shuffle(0)
    with pytest.raises(ValueError, match="batch_size"):
        ducky.stream("x.parquet", fields=FIELDS, batch_size=0)
    with pytest.raises(ValueError, match="shard"):
        ducky.stream("x.parquet", fields={"y": ducky.vector("y")}, batch_size=1, shard=(2, 2))
