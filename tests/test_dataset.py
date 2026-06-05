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
    assert abs(ds.train["X"][:, 0].mean()) < 1e-5
    # Val mean is also small but distinct from train's exact zero — a
    # sentinel for "we used train-only stats" rather than "we used all rows".
    assert abs(ds.val["X"][:, 0].mean()) < 0.2


# ── Named fields API ──────────────────────────────────────────────────────


@pytest.fixture
def weighted_csv_path(tmp_path):
    """Like csv_path but with a `w` sample-weight column and a `gid` group id."""
    p = tmp_path / "weighted.csv"
    rng = np.random.default_rng(0)
    n = 1000
    x = rng.normal(size=n)
    y = (x > 0).astype(int)
    w = rng.uniform(0.5, 1.5, size=n)
    gid = rng.integers(0, 10, size=n)
    lines = [f"{xi},{yi},{wi},{gi}" for xi, yi, wi, gi in zip(x, y, w, gid, strict=True)]
    p.write_text("x,y,w,gid\n" + "\n".join(lines))
    return str(p)


def test_fields_named_outputs(weighted_csv_path):
    # The full fields= API: an X matrix, y label vector, sample-weight vector,
    # and a passthrough id vector — each materialised as its own field.
    ds = ducky.dataset(
        weighted_csv_path,
        fields={
            "X": ducky.matrix({"x": ducky.feature("x", standardize=True)}),
            "y": ducky.vector("y", dtype="i64"),
            "w": ducky.vector("w"),
            "ids": ducky.vector("gid", dtype="i64"),
        },
        split=ducky.split(0.8, seed=0),
    )
    train = ds.train
    assert train is not None
    assert train["X"].shape == (train.n_rows, 1)
    assert train["y"].shape == (train.n_rows,)
    assert train["w"].shape == (train.n_rows,)
    assert train["ids"].shape == (train.n_rows,)
    # tensors() sugar still works because X and y are present.
    Xtr, ytr = train.tensors()
    np.testing.assert_array_equal(np.asarray(Xtr), np.asarray(train["X"]))
    np.testing.assert_array_equal(np.asarray(ytr), np.asarray(train["y"]))


def test_fields_attr_access_and_swizzle(weighted_csv_path):
    ds = ducky.dataset(
        weighted_csv_path,
        fields={
            "X": ducky.matrix({"x": ducky.feature("x")}),
            "y": ducky.vector("y", dtype="i64"),
            "w": ducky.vector("w"),
        },
        split=ducky.split(0.8, seed=0),
    )
    train = ds.train
    assert train is not None
    # Attribute access: single-char field name → array.
    np.testing.assert_array_equal(train.X, train["X"])
    np.testing.assert_array_equal(train.y, train["y"])
    # Swizzle: multi-char attribute is split into one field per char.
    X, y = train.Xy
    np.testing.assert_array_equal(X, train["X"])
    np.testing.assert_array_equal(y, train["y"])
    X, y, w = train.Xyw
    np.testing.assert_array_equal(w, train["w"])
    # Unknown field is an AttributeError, not a silent miss.
    with pytest.raises(AttributeError):
        _ = train.Z


def test_multi_column_matrix_stack_order(weighted_csv_path):
    ds = ducky.dataset(
        weighted_csv_path,
        fields={
            "X": ducky.matrix(
                {
                    "a": ducky.feature("x"),
                    "b": ducky.feature("w"),
                    "c": ducky.feature("gid", dtype="f32"),
                }
            ),
            "y": ducky.vector("y", dtype="i64"),
        },
        split=ducky.split(0.8, seed=0),
    )
    train = ds.train
    assert train is not None
    X = train["X"]
    assert X.shape == (train.n_rows, 3)
    # Column order in X matches declaration order in the matrix spec.
    ref = ducky.dataset(
        weighted_csv_path,
        fields={
            "a": ducky.vector("x"),
            "b": ducky.vector("w"),
            "c": ducky.vector("gid", dtype="f32"),
            "y": ducky.vector("y", dtype="i64"),
        },
        split=ducky.split(0.8, seed=0),
    )
    rtrain = ref.train
    assert rtrain is not None
    np.testing.assert_allclose(X[:, 0], rtrain["a"])
    np.testing.assert_allclose(X[:, 1], rtrain["b"])
    np.testing.assert_allclose(X[:, 2], rtrain["c"])


def test_standardize_on_vector_field(weighted_csv_path):
    ds = ducky.dataset(
        weighted_csv_path,
        fields={
            "X": ducky.matrix({"x": ducky.feature("x")}),
            "y": ducky.vector("y", dtype="f32", standardize=True),
        },
        split=ducky.split(0.8, seed=0),
    )
    assert ds.train is not None
    # Standardised y over the train fold has mean ~0.
    assert abs(np.asarray(ds.train["y"]).mean()) < 1e-5


def test_fields_and_columns_mutually_exclusive(csv_path):
    with pytest.raises(ValueError, match="either fields=|not both"):
        ducky.dataset(
            csv_path,
            fields={"X": ducky.matrix({"x": ducky.feature("x")}), "y": ducky.vector("y")},
            columns={"x": ducky.feature("x")},
            target=ducky.target("y"),
        )


def test_dataset_requires_some_spec(csv_path):
    with pytest.raises(ValueError, match="fields=|columns="):
        ducky.dataset(csv_path)


def test_empty_matrix_rejected():
    with pytest.raises(ValueError, match="at least one column"):
        ducky.matrix({})


def test_tensors_requires_X_and_y(weighted_csv_path):
    ds = ducky.dataset(
        weighted_csv_path,
        fields={"a": ducky.vector("x"), "b": ducky.vector("y", dtype="i64")},
    )
    assert ds.train is not None
    with pytest.raises(AttributeError, match="tensors\\(\\) requires"):
        ds.train.tensors()
