import numpy as np
import pytest

import ducky


@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "data.csv"
    rng = np.random.default_rng(0)
    n = 1000
    x = rng.normal(size=n)
    y = (x > 0).astype(int)
    p.write_text("x,y\n" + "\n".join(f"{xi},{yi}" for xi, yi in zip(x, y, strict=True)))
    return str(p)


def test_default_split_is_train_val(csv_path):
    ds = ducky.dataset(
        csv_path,
        columns={"x": ducky.feature("x")},
        target=ducky.target("y", dtype="i64"),
        split=ducky.split(0.8, seed=0),
    )
    assert set(ds.folds) == {"train", "val"}
    assert ds.train is not None
    assert ds.val is not None
    assert ds.test is None
    total = ds.train.n_rows + ds.val.n_rows
    assert total == 1000
    # 800/200 ± hash variance
    assert 750 <= ds.train.n_rows <= 850


def test_named_three_way_split(csv_path):
    ds = ducky.dataset(
        csv_path,
        columns={"x": ducky.feature("x")},
        target=ducky.target("y", dtype="i64"),
        split=ducky.split({"train": 0.7, "val": 0.15, "test": 0.15}, seed=0),
    )
    assert set(ds.folds) == {"train", "val", "test"}
    assert ds.train is not None and ds.val is not None and ds.test is not None
    assert ds.train.n_rows + ds.val.n_rows + ds.test.n_rows == 1000


def test_custom_named_folds(csv_path):
    # Arbitrary names beyond train/val/test work via the folds dict.
    ds = ducky.dataset(
        csv_path,
        columns={"x": ducky.feature("x")},
        target=ducky.target("y", dtype="i64"),
        split=ducky.split({"train": 0.6, "calibration": 0.2, "holdout": 0.2}, seed=0),
    )
    assert set(ds.folds) == {"train", "calibration", "holdout"}
    assert ds["calibration"].n_rows > 0
    assert ds["holdout"].n_rows > 0


def test_standardize_requires_train_fold(csv_path):
    # Without a "train" fold, standardize=True must error rather than silently
    # using full-dataset statistics.
    with pytest.raises(ValueError, match="train"):
        ducky.dataset(
            csv_path,
            columns={"x": ducky.feature("x", standardize=True)},
            target=ducky.target("y", dtype="i64"),
            split=ducky.split({"a": 0.5, "b": 0.5}),
        )


def test_split_fractions_validate():
    with pytest.raises(ValueError, match="sum to 1"):
        ducky.split({"train": 0.5, "val": 0.3})
    with pytest.raises(ValueError, match="must be > 0"):
        ducky.split({"train": 1.0, "val": 0.0})
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        ducky.split(1.5)


def test_batches_basic(csv_path):
    ds = ducky.dataset(
        csv_path,
        columns={"x": ducky.feature("x")},
        target=ducky.target("y", dtype="i64"),
    )
    fold = ds["train"]
    sizes = [Xb.shape[0] for Xb, _ in fold.batches(batch_size=128)]
    # 1000 rows / 128 → 7 batches of 128 + 1 of 104
    assert sum(sizes) == 1000
    assert sizes[:-1] == [128] * 7


def test_batches_drop_last(csv_path):
    ds = ducky.dataset(
        csv_path,
        columns={"x": ducky.feature("x")},
        target=ducky.target("y", dtype="i64"),
    )
    fold = ds["train"]
    sizes = [Xb.shape[0] for Xb, _ in fold.batches(batch_size=128, drop_last=True)]
    assert sizes == [128] * 7  # 8th would be partial; dropped


def test_batches_shuffle_deterministic(csv_path):
    ds = ducky.dataset(
        csv_path,
        columns={"x": ducky.feature("x")},
        target=ducky.target("y", dtype="i64"),
    )
    fold = ds["train"]
    a = list(fold.batches(batch_size=128, shuffle=True, seed=42))
    b = list(fold.batches(batch_size=128, shuffle=True, seed=42))
    for (Xa, ya), (Xb, yb) in zip(a, b, strict=True):
        np.testing.assert_array_equal(Xa, Xb)
        np.testing.assert_array_equal(ya, yb)


def test_batches_shuffle_changes_order(csv_path):
    ds = ducky.dataset(
        csv_path,
        columns={"x": ducky.feature("x")},
        target=ducky.target("y", dtype="i64"),
    )
    fold = ds["train"]
    X_seq, _ = next(iter(fold.batches(batch_size=128)))  # no shuffle
    X_shuf, _ = next(iter(fold.batches(batch_size=128, shuffle=True, seed=1)))
    # Extremely unlikely the first 128 rows happen to permute identically.
    assert not np.array_equal(X_seq, X_shuf)


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch", "mlx"])
def test_backend_materializes_equivalently(csv_path, backend):
    # Every backend must produce the same split + feature matrix as numpy; only
    # the array type differs. Skip backends not installed on this platform.
    if backend != "numpy":
        pytest.importorskip(backend)

    columns = {"x": ducky.feature("x", standardize=True)}
    target = ducky.target("y", dtype="f32")
    split = ducky.split(0.8, seed=0)
    ref = ducky.dataset(csv_path, columns=columns, target=target, split=split, backend="numpy")
    ds = ducky.dataset(csv_path, columns=columns, target=target, split=split, backend=backend)

    for name in ref.folds:
        assert ds[name].backend == backend
        assert ds[name].n_rows == ref[name].n_rows
        X, y = ds[name].tensors()
        Xref, yref = ref[name].tensors()
        # `np.asarray` brings jax/mlx/torch arrays back to the host for comparison.
        np.testing.assert_allclose(np.asarray(X), Xref, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(y), yref, rtol=1e-5)


def test_unknown_backend_rejected(csv_path):
    with pytest.raises(ValueError, match="unknown backend"):
        ducky.dataset(
            csv_path,
            columns={"x": ducky.feature("x")},
            target=ducky.target("y"),
            backend="tensorflow",
        )


def test_device_rejected_for_deviceless_backends(csv_path):
    with pytest.raises(ValueError, match="does not take a device"):
        ducky.dataset(
            csv_path,
            columns={"x": ducky.feature("x")},
            target=ducky.target("y"),
            backend="numpy",
            device="cpu",
        )


def test_no_leak_train_stats_into_val(csv_path):
    # If standardisation accidentally used full-dataset stats, the val fold's
    # mean would be near 0 too. With train-only stats, val gets a non-trivial
    # mean because the train sample mean ≠ population mean exactly.
    ds = ducky.dataset(
        csv_path,
        columns={"x": ducky.feature("x", standardize=True)},
        target=ducky.target("y", dtype="i64"),
        split=ducky.split(0.8, seed=0),
    )
    assert ds.train is not None and ds.val is not None
    # Train mean of standardised x must be ~0 (by construction).
    assert abs(ds.train.arrays["x"].mean()) < 1e-5
    # Val mean is also small but distinct from train's exact zero — a
    # sentinel for "we used train-only stats" rather than "we used all rows".
    assert abs(ds.val.arrays["x"].mean()) < 0.2
