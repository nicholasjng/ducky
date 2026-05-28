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
