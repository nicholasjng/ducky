import numpy as np
import pytest

import ducky


def test_positional_single_arg():
    con = ducky.connect()
    con.create_function(
        "doublef",
        lambda x: x * 2.0,
        parameters=["DOUBLE"],
        return_type="DOUBLE",
    )
    arr = con.sql("SELECT doublef(CAST(i AS DOUBLE)) AS v FROM range(4) t(i)").to_numpy()["v"]
    np.testing.assert_array_equal(arr, [0.0, 2.0, 4.0, 6.0])


def test_positional_multi_arg():
    con = ducky.connect()

    def add3(a, b, c):
        return a.astype(np.int64) + b.astype(np.int64) + c.astype(np.int64)

    con.create_function(
        "add3",
        add3,
        parameters=["INTEGER", "INTEGER", "INTEGER"],
        return_type="BIGINT",
    )
    rows = con.sql(
        "SELECT add3(CAST(i AS INTEGER), CAST(i*2 AS INTEGER), CAST(i*3 AS INTEGER)) AS v "
        "FROM range(4) t(i)"
    ).to_numpy()
    np.testing.assert_array_equal(rows["v"], [0, 6, 12, 18])


def test_dict_style():
    con = ducky.connect()

    def score(args):
        return np.sqrt(args["x"] ** 2 + args["y"] ** 2)

    con.create_function(
        "score",
        score,
        parameters={"x": "DOUBLE", "y": "DOUBLE"},
        return_type="DOUBLE",
    )
    [(v,)] = con.sql("SELECT score(CAST(3 AS DOUBLE), CAST(4 AS DOUBLE))").fetchall()
    assert v == 5.0


def test_dtype_round_trip():
    # FLOAT in, FLOAT out — verify the dtype check is strict.
    con = ducky.connect()
    con.create_function(
        "neg",
        lambda x: -x,
        parameters=["FLOAT"],
        return_type="FLOAT",
    )
    [(v,)] = con.sql("SELECT neg(CAST(1.5 AS FLOAT))").fetchall()
    assert v == pytest.approx(-1.5)


def test_wrong_return_dtype_errors():
    con = ducky.connect()
    con.create_function(
        "bad",
        lambda x: x.astype(np.int32),  # declared DOUBLE
        parameters=["DOUBLE"],
        return_type="DOUBLE",
    )
    with pytest.raises(ducky.Error, match="wrong dtype"):
        con.sql("SELECT bad(CAST(1.0 AS DOUBLE))").fetchall()


def test_wrong_return_shape_errors():
    con = ducky.connect()
    con.create_function(
        "shrink",
        lambda x: x[:1],
        parameters=["DOUBLE"],
        return_type="DOUBLE",
    )
    with pytest.raises(ducky.Error, match="shape"):
        con.sql("SELECT shrink(CAST(i AS DOUBLE)) FROM range(3) t(i)").fetchall()


def test_user_exception_propagates():
    con = ducky.connect()

    def boom(x):
        raise ValueError("nope")

    con.create_function(
        "boom",
        boom,
        parameters=["DOUBLE"],
        return_type="DOUBLE",
    )
    with pytest.raises(ducky.Error, match="nope"):
        con.sql("SELECT boom(CAST(1.0 AS DOUBLE))").fetchall()


def test_unknown_type_string():
    con = ducky.connect()
    with pytest.raises(ducky.Error, match="unknown UDF type"):
        con.create_function("f", lambda x: x, parameters=["WAT"], return_type="DOUBLE")


def test_no_parameters_errors():
    con = ducky.connect()
    with pytest.raises(ducky.Error, match="at least one parameter"):
        con.create_function("f", lambda: None, parameters=[], return_type="DOUBLE")


def test_nullable_output_uint8_mask():
    con = ducky.connect()

    def half(x):
        values = x // 2
        mask = (x % 2 == 0).astype(np.uint8)  # 1=valid, 0=null
        return values, mask

    con.create_function(
        "half_even",
        half,
        parameters=["BIGINT"],
        return_type="BIGINT",
    )
    rows = con.sql("SELECT i, half_even(i) AS v FROM range(5) t(i) ORDER BY i").fetchall()
    assert rows == [(0, 0), (1, None), (2, 1), (3, None), (4, 2)]


def test_nullable_output_bool_mask():
    con = ducky.connect()
    con.create_function(
        "maybe",
        lambda x: (x, (x > 0).astype(np.bool_)),
        parameters=["BIGINT"],
        return_type="BIGINT",
    )
    rows = con.sql("SELECT maybe(i) AS v FROM range(-1, 2) t(i) ORDER BY i").fetchall()
    assert rows == [(None,), (None,), (1,)]


def test_bad_mask_shape_errors():
    con = ducky.connect()
    con.create_function(
        "bad",
        lambda x: (x, np.ones(1, dtype=np.uint8)),
        parameters=["BIGINT"],
        return_type="BIGINT",
    )
    with pytest.raises(ducky.Error, match="mask shape"):
        con.sql("SELECT bad(i) FROM range(5) t(i)").fetchall()


def test_bad_mask_dtype_errors():
    con = ducky.connect()
    con.create_function(
        "bad",
        lambda x: (x, x.astype(np.int32)),
        parameters=["BIGINT"],
        return_type="BIGINT",
    )
    with pytest.raises(ducky.Error, match="mask must be uint8 or bool"):
        con.sql("SELECT bad(i) FROM range(3) t(i)").fetchall()


def test_bad_tuple_length_errors():
    con = ducky.connect()
    con.create_function(
        "bad",
        lambda x: (x, x, x),
        parameters=["BIGINT"],
        return_type="BIGINT",
    )
    with pytest.raises(ducky.Error, match=r"\(values, mask\)"):
        con.sql("SELECT bad(i) FROM range(3) t(i)").fetchall()


def test_infer_types_from_annotations():
    con = ducky.connect()

    def add(x: int, y: float) -> float:
        return x + y

    con.create_function("add", add)
    [(v,)] = con.sql("SELECT add(3, CAST(0.5 AS DOUBLE))").fetchall()
    assert v == pytest.approx(3.5)


def test_infer_only_return_type():
    con = ducky.connect()

    def neg(x: float) -> float:
        return -x

    # Explicit parameters, inferred return_type.
    con.create_function("neg", neg, parameters=["DOUBLE"])
    [(v,)] = con.sql("SELECT neg(CAST(2.0 AS DOUBLE))").fetchall()
    assert v == -2.0


def test_infer_only_parameters():
    con = ducky.connect()

    def neg(x: float) -> float:
        return -x

    # Inferred parameters, explicit return_type.
    con.create_function("neg", neg, return_type="DOUBLE")
    [(v,)] = con.sql("SELECT neg(CAST(2.0 AS DOUBLE))").fetchall()
    assert v == -2.0


def test_infer_bool():
    con = ducky.connect()

    def isodd(x: int) -> bool:
        return (x % 2 == 1).astype(np.bool_)  # type: ignore[union-attr]

    con.create_function("isodd", isodd)
    rows = con.sql("SELECT isodd(i) FROM range(4) t(i) ORDER BY i").fetchall()
    assert rows == [(False,), (True,), (False,), (True,)]


def test_missing_annotation_errors():
    con = ducky.connect()

    def f(x, y: float) -> float:
        return y

    with pytest.raises(ducky.Error, match="no type annotation"):
        con.create_function("f", f)


def test_missing_return_annotation_errors():
    con = ducky.connect()

    def f(x: float):
        return x

    with pytest.raises(ducky.Error, match="no return type annotation"):
        con.create_function("f", f)


def test_unsupported_annotation_errors():
    con = ducky.connect()

    def f(x: str) -> float:
        return 0.0

    with pytest.raises(ducky.Error, match="cannot infer"):
        con.create_function("f", f)


def test_works_across_chunk_boundaries():
    # > STANDARD_VECTOR_SIZE rows forces multiple chunks.
    n = 5000
    con = ducky.connect()
    con.create_function(
        "sq",
        lambda x: x * x,
        parameters=["BIGINT"],
        return_type="BIGINT",
    )
    arr = con.sql(f"SELECT sq(i) AS v FROM range({n}) t(i)").to_numpy()["v"]
    np.testing.assert_array_equal(arr, np.arange(n, dtype=np.int64) ** 2)
