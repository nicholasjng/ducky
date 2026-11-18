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
    assert ds.train is not None
    assert ds.val is not None
    assert ds.test is not None
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


def test_standardize_empty_train_fold_rejected(csv_path):
    # The "train" fraction rounds to an empty bucket range, which must error
    # rather than standardize against NULL statistics.
    with pytest.raises(ValueError, match="empty"):
        ducky.dataset(
            csv_path,
            columns={"x": ducky.feature("x", standardize=True)},
            target=ducky.target("y", dtype="i64"),
            split=ducky.split({"train": 0.0001, "val": 0.9999}),
        )


def test_reserved_bucket_alias_rejected(csv_path):
    # "_bucket" is ducky's internal column, so a same-named field must be
    # rejected up front rather than surface as a confusing KeyError.
    with pytest.raises(ValueError, match="_bucket"):
        ducky.dataset(
            csv_path,
            fields={"_bucket": ducky.vector("x"), "y": ducky.vector("y", dtype="i64")},
        )


def test_drop_nulls_quotes_source_column_names(tmp_path):
    path = tmp_path / "quoted.csv"
    path.write_text('"odd name","a""b",y\n1,2,3\n,4,5\n6,,7\n')
    ds = ducky.dataset(
        str(path),
        columns={"x": ducky.feature('"odd name"')},
        target=ducky.target("y"),
        drop_nulls=["odd name", 'a"b'],
    )
    assert ds.train is not None
    assert ds.train.X.tolist() == [[1.0]]
    assert ds.train.y.tolist() == [3.0]


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
    assert sizes == [128] * 7  # 8th would be partial, so dropped


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
    # Every backend must produce numpy's split and feature matrix, only the
    # array type differs.
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


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch", "mlx"])
def test_dtype_objects_match_dtype_names(csv_path, backend):
    if backend == "numpy":
        dtypes = {"i32": np.int32, "f64": np.dtype("float64"), "u8": np.uint8, "bool": np.bool}
    elif backend == "jax":
        jnp = pytest.importorskip("jax.numpy")
        dtypes = {"i32": jnp.int32, "f64": jnp.dtype("float64"), "u8": jnp.uint8, "bool": jnp.bool}
    elif backend == "torch":
        torch = pytest.importorskip("torch")
        dtypes = {"i32": torch.int32, "f64": torch.float64, "u8": torch.uint8, "bool": torch.bool}
    else:
        mx = pytest.importorskip("mlx.core")
        dtypes = {"i32": mx.int32, "f64": mx.float64, "u8": mx.uint8, "bool": mx.bool_}
    for name, dtype in dtypes.items():
        assert ducky.vector("x", dtype=dtype).dtype == name
    assert ducky.category("x", dtype=dtypes["i32"]).dtype == "i32"

    fields = {
        "x": ducky.vector("x", dtype=dtypes["f64"]),
        "y": ducky.vector("y", dtype=dtypes["u8"]),
    }
    ds = ducky.dataset(csv_path, fields=fields)
    assert ds["train"]["x"].dtype == np.float64
    assert ds["train"]["y"].dtype == np.uint8


def test_dtype_names_and_builtins():
    for given, name in [
        ("float32", "f32"),
        ("int16", "i16"),
        (float, "f64"),
        (int, "i64"),
        (bool, "bool"),
    ]:
        assert ducky.feature("x", dtype=given).dtype == name
    assert ducky.Tensor("x", (2,), dtype="int8").dtype == "i8"  # direct construction too
    for bad in ["f16", "float16", np.float16, np.complex64, object]:
        with pytest.raises(ValueError, match="unknown dtype"):
            ducky.feature("x", dtype=bad)
    with pytest.raises(ValueError, match="signed integer dtype"):
        ducky.category("x", dtype=np.uint8)


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
    # A loose check such as "val mean is small" can't tell train-only from
    # full-dataset statistics, which both land near 0 here. So the raw folds
    # (same split and seed) are standardized in numpy and compared exactly.
    split = ducky.split(0.8, seed=0)
    raw = ducky.dataset(
        csv_path,
        columns={"x": ducky.feature("x", dtype="f64")},
        target=ducky.target("y", dtype="i64"),
        split=split,
    )
    assert raw.train is not None
    assert raw.val is not None
    train_x = raw.train["X"][:, 0]
    val_x = raw.val["X"][:, 0]
    mean = train_x.mean()
    std = train_x.std(ddof=0)  # ducky standardizes with stddev_pop (ddof=0)

    ds = ducky.dataset(
        csv_path,
        columns={"x": ducky.feature("x", standardize=True)},
        target=ducky.target("y", dtype="i64"),
        split=split,
    )
    assert ds.train is not None
    assert ds.val is not None
    np.testing.assert_allclose(ds.train["X"][:, 0], (train_x - mean) / std, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(ds.val["X"][:, 0], (val_x - mean) / std, rtol=1e-4, atol=1e-4)


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
    # An X matrix, y labels, sample weights and a passthrough id, each its own
    # field.
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
    with pytest.raises(ValueError, match=r"either fields=|not both"):
        ducky.dataset(
            csv_path,
            fields={"X": ducky.matrix({"x": ducky.feature("x")}), "y": ducky.vector("y")},
            columns={"x": ducky.feature("x")},
            target=ducky.target("y"),
        )


def test_dataset_requires_some_spec(csv_path):
    with pytest.raises(ValueError, match=r"fields=|columns="):
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


def test_source_path_with_single_quote(tmp_path):
    # The source is embedded as a SQL string literal, so quotes must be escaped.
    p = tmp_path / "it's data.csv"
    p.write_text("x,y\n1.0,0\n2.0,1\n")
    ds = ducky.dataset(
        str(p),
        columns={"x": ducky.feature("x")},
        target=ducky.target("y", dtype="i64"),
    )
    assert ds.train is not None
    assert ds.train.n_rows == 2


def test_standardize_constant_column_yields_zeros(tmp_path):
    # std = 0 yields 0 (StandardScaler convention), not a division by zero whose
    # NULL turns into garbage.
    p = tmp_path / "const.csv"
    p.write_text("x,y\n" + "\n".join("3.0,1" for _ in range(50)))
    ds = ducky.dataset(
        p.as_posix(),
        columns={"x": ducky.feature("x", standardize=True)},
        target=ducky.target("y", dtype="i64"),
    )
    assert ds.train is not None
    X, _ = ds.train.tensors()
    assert np.all(X == 0.0)


def test_tensor_field(tmp_path):
    p = tmp_path / "emb.parquet"
    con = ducky.connect()
    con.execute(
        f"COPY (SELECT [i, i * 2, i * 3]::DOUBLE[3] emb, i % 2 AS y FROM range(100) t(i)) TO '{p}'"
    )
    ds = ducky.dataset(
        str(p),
        fields={"X": ducky.tensor("emb", 3), "y": ducky.vector("y", dtype="i64")},
        split=ducky.split(0.8, seed=0),
    )
    X, y = ds["train"].tensors()
    assert X.shape == (ds["train"].n_rows, 3)
    assert X.dtype == np.float32
    np.testing.assert_array_equal(X[:, 1], 2 * X[:, 0])
    np.testing.assert_array_equal(y, X[:, 0].astype(np.int64) % 2)


def test_tensor_shape_validates():
    with pytest.raises(ValueError, match="positive ints"):
        ducky.tensor("emb", ())
    with pytest.raises(ValueError, match="positive ints"):
        ducky.tensor("emb", (3, 0))


@pytest.fixture
def labels_path(tmp_path):
    p = tmp_path / "labels.parquet"
    ducky.connect().execute(
        f"COPY (SELECT i AS id, (['b', 'a', 'c'])[1 + i % 3] AS s, "
        f"CASE WHEN i = 5 THEN NULL ELSE i % 4 END AS n FROM range(300) t(i)) TO '{p}'"
    )
    return str(p)


def test_category_learns_sorted_train_values(labels_path):
    ds = ducky.dataset(
        labels_path,
        fields={"s": ducky.category("s"), "n": ducky.category("n"), "id": ducky.vector("id")},
        split=ducky.split(0.8, seed=0),
    )
    assert ds.categories == {"s": ["a", "b", "c"], "n": [0, 1, 2, 3]}
    for fold in ds.folds.values():
        ids = fold["id"].astype(int)
        assert fold["s"].dtype == np.int64
        np.testing.assert_array_equal(fold["s"], ["abc".index("bac"[i % 3]) for i in ids])
        np.testing.assert_array_equal(fold["n"], [-1 if i == 5 else i % 4 for i in ids])


def test_category_with_given_categories(labels_path):
    ds = ducky.dataset(labels_path, fields={"s": ducky.category("s", ["c", "b"], dtype="i8")})
    assert ds.categories == {"s": ["c", "b"]}
    codes = ds["train"]["s"]
    assert codes.dtype == np.int8
    np.testing.assert_array_equal(codes[:3], [1, -1, 0])  # unknown "a" is -1


def test_category_matches_given_categories_as_duckdb_writes_them(labels_path):
    ds = ducky.dataset(
        labels_path,
        fields={
            "b": ducky.category("n = 1", [True, False]),  # DuckDB writes true, Python True
            "d": ducky.category("n::DECIMAL(4,1)", [3, 2, 1, 0]),  # 3.0, not 3
            "n": ducky.vector("n"),
        },
    )
    n = ds["train"]["n"]
    np.testing.assert_array_equal(ds["train"]["b"], np.where(np.isnan(n), -1, n != 1))
    np.testing.assert_array_equal(ds["train"]["d"], np.where(np.isnan(n), -1, 3 - n))


def test_category_uses_an_enum_dictionary(labels_path):
    con = ducky.connect()
    con.execute("CREATE TYPE letter AS ENUM ('c', 'b', 'a')")
    ds = ducky.dataset(labels_path, fields={"s": ducky.category("s::letter")}, con=con)
    assert ds.categories == {"s": ["c", "b", "a"]}
    np.testing.assert_array_equal(ds["train"]["s"][:3], [1, 2, 0])


def test_category_validation(labels_path):
    with pytest.raises(ValueError, match="signed integer dtype"):
        ducky.category("s", dtype="u8")
    for expr, given in [("s", ["a", "a"]), ("n", [1.0, 1])]:  # 1.0 and 1 are both BIGINT 1
        with pytest.raises(ValueError, match="distinct"):
            ducky.dataset(labels_path, fields={"s": ducky.category(expr, given)})
    with pytest.raises(ducky.Error, match="Could not convert"):
        ducky.dataset(labels_path, fields={"n": ducky.category("n", ["x"])})
    with pytest.raises(ValueError, match="more than i8 codes"):
        ducky.dataset(labels_path, fields={"id": ducky.category("id", dtype="i8")})
    with pytest.raises(ValueError, match="fold named 'train'"):
        ducky.dataset(
            labels_path,
            fields={"s": ducky.category("s")},
            split=ducky.split({"fit": 0.5, "val": 0.5}),
        )
    with pytest.raises(ValueError, match="no values in the 'train' fold"):
        ducky.dataset(labels_path, fields={"s": ducky.category("NULL::VARCHAR")})
