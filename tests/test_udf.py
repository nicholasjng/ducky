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
        return (x % 2 == 1).astype(np.bool_)  # ty: ignore[unresolved-attribute]

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


def test_varargs_basic():
    con = ducky.connect()

    def add_all(*cols):
        return sum(c.astype(np.int64) for c in cols)

    con.create_function("add_all", add_all, varargs="INTEGER", return_type="BIGINT")
    [(v2,)] = con.sql("SELECT add_all(CAST(1 AS INTEGER), CAST(2 AS INTEGER))").fetchall()
    [(v4,)] = con.sql(
        "SELECT add_all(CAST(1 AS INTEGER), CAST(2 AS INTEGER), "
        "CAST(3 AS INTEGER), CAST(4 AS INTEGER))"
    ).fetchall()
    assert (v2, v4) == (3, 10)


def test_varargs_zero_args():
    con = ducky.connect()
    con.create_function(
        "vempty", lambda *xs: np.zeros(1, dtype=np.int64), varargs="BIGINT", return_type="BIGINT"
    )
    [(v,)] = con.sql("SELECT vempty()").fetchall()
    assert v == 0


def test_varargs_conflicts_with_parameters():
    con = ducky.connect()
    with pytest.raises(ducky.Error, match="cannot set both"):
        con.create_function(
            "bad",
            lambda *xs: xs[0],
            parameters=["DOUBLE"],
            varargs="DOUBLE",
            return_type="DOUBLE",
        )


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


# ── volatile / special_handling / init=state ─────────────────────────────────


def test_volatile_function_runs():
    # Smoke test: volatile=True registers and a query against the function runs.
    con = ducky.connect()
    con.create_function(
        "vmul",
        lambda x: x * 2.0,
        parameters=["DOUBLE"],
        return_type="DOUBLE",
        volatile=True,
    )
    [(v,)] = con.sql("SELECT vmul(CAST(3.0 AS DOUBLE))").fetchall()
    assert v == 6.0


def test_volatile_not_constant_folded():
    # Two calls with the same literal argument should both invoke the UDF when
    # marked volatile — without volatile the optimizer is free to evaluate it
    # once and reuse the result (common subexpression elimination).
    con = ducky.connect()
    calls = [0]

    def stamp(x):
        calls[0] += 1
        return np.full_like(x, calls[0], dtype=np.int64)

    con.create_function(
        "stamp",
        stamp,
        parameters=["BIGINT"],
        return_type="BIGINT",
        volatile=True,
    )
    [(a, b)] = con.sql(
        "SELECT stamp(CAST(1 AS BIGINT)) AS a, stamp(CAST(1 AS BIGINT)) AS b"
    ).fetchall()
    assert a != b  # would be equal if the optimizer had folded the two calls


def test_special_handling_observes_null_input():
    # With special_handling=True, the UDF sees the input mask and can produce
    # a non-NULL output even where the input is NULL — here, coalesce to 0.
    con = ducky.connect()
    con.execute("CREATE TABLE t (x BIGINT)")
    con.execute("INSERT INTO t VALUES (1), (NULL), (3)")

    seen_masks = []

    def coalesce_zero(x):
        values, mask = x
        seen_masks.append(mask.copy())
        out = values.copy()
        out[mask == 0] = 0
        return out

    con.create_function(
        "coalesce0",
        coalesce_zero,
        parameters=["BIGINT"],
        return_type="BIGINT",
        special_handling=True,
    )
    rows = con.sql("SELECT coalesce0(x) FROM t ORDER BY x NULLS LAST").fetchall()
    assert rows == [(1,), (3,), (0,)]
    # The UDF actually saw at least one NULL slot via the mask.
    assert any((m == 0).any() for m in seen_masks)


def test_special_handling_all_valid_mask():
    # With no NULL inputs, the mask passed in must be all-1.
    con = ducky.connect()
    seen = []

    def echo(x):
        values, mask = x
        seen.append(mask.copy())
        return values

    con.create_function(
        "echo",
        echo,
        parameters=["BIGINT"],
        return_type="BIGINT",
        special_handling=True,
    )
    con.sql("SELECT echo(i) FROM range(8) t(i)").fetchall()
    assert seen and (seen[0] == 1).all()


def test_special_handling_dict_style():
    # Dict-style input under special_handling: each value in the kwargs dict is
    # a (values, mask) tuple keyed by the parameter name.
    con = ducky.connect()
    con.execute("CREATE TABLE t (a BIGINT, b BIGINT)")
    con.execute("INSERT INTO t VALUES (1, 10), (NULL, 20), (3, NULL)")

    def add_or_zero(kwargs):
        av, am = kwargs["a"]
        bv, bm = kwargs["b"]
        out = np.where(am == 0, 0, av) + np.where(bm == 0, 0, bv)
        return out.astype(np.int64)

    con.create_function(
        "add_or_zero",
        add_or_zero,
        parameters={"a": "BIGINT", "b": "BIGINT"},
        return_type="BIGINT",
        special_handling=True,
    )
    rows = con.sql("SELECT add_or_zero(a, b) FROM t ORDER BY a NULLS LAST, b NULLS LAST").fetchall()
    assert rows == [(11,), (3,), (20,)]


def test_init_state_threads_through_call():
    # The init= callable runs once per worker thread; its return value is
    # passed as the first positional arg of the UDF on every chunk.
    con = ducky.connect()

    def factory():
        return {"calls": 0}

    def stamp(state, x):
        state["calls"] += 1
        return np.full_like(x, state["calls"], dtype=np.int64)

    con.create_function(
        "stamp",
        stamp,
        parameters=["BIGINT"],
        return_type="BIGINT",
        init=factory,
        volatile=True,
    )
    rows = con.sql("SELECT stamp(i) FROM range(8) t(i)").fetchall()
    # Single chunk in a small range; state increments to 1 for every row.
    assert {r[0] for r in rows} == {1}


def test_init_state_persists_across_chunks():
    # With >STANDARD_VECTOR_SIZE rows DuckDB emits multiple chunks; per-thread
    # state must persist so the counter keeps incrementing.
    n = 5000
    con = ducky.connect("", threads=1)

    def factory():
        return [0]

    def stamp(state, x):
        state[0] += 1
        return np.full_like(x, state[0], dtype=np.int64)

    con.create_function(
        "stamp",
        stamp,
        parameters=["BIGINT"],
        return_type="BIGINT",
        init=factory,
        volatile=True,
    )
    arr = con.sql(f"SELECT stamp(i) AS v FROM range({n}) t(i)").to_numpy()["v"]
    # The counter must reach > 1 — at least one chunk boundary was crossed.
    assert arr.max() > 1


def test_init_state_dict_style():
    # init= + dict-style parameters: fn is called as fn(state, kwargs).
    con = ducky.connect()

    def factory():
        return {"mult": 10}

    def mult(state, kwargs):
        return kwargs["x"] * state["mult"]

    con.create_function(
        "mult",
        mult,
        parameters={"x": "BIGINT"},
        return_type="BIGINT",
        init=factory,
    )
    rows = con.sql("SELECT mult(i) FROM range(3) t(i)").fetchall()
    assert rows == [(0,), (10,), (20,)]


def test_init_state_with_varargs():
    # init= + varargs: fn(state, *cols).
    con = ducky.connect()

    def factory():
        return 100

    def add_all(state, *cols):
        return sum(c.astype(np.int64) for c in cols) + state

    con.create_function(
        "add_all",
        add_all,
        varargs="INTEGER",
        return_type="BIGINT",
        init=factory,
    )
    [(v,)] = con.sql("SELECT add_all(CAST(1 AS INTEGER), CAST(2 AS INTEGER))").fetchall()
    assert v == 103


def test_init_error_propagates():
    # A failure in init= must surface as a query error, not a silent crash.
    con = ducky.connect()

    def bad_factory():
        raise RuntimeError("init failure")

    def use_state(state, x):
        return x

    con.create_function(
        "use_state",
        use_state,
        parameters=["BIGINT"],
        return_type="BIGINT",
        init=bad_factory,
    )
    with pytest.raises(ducky.Error, match="init failure"):
        con.sql("SELECT use_state(i) FROM range(3) t(i)").fetchall()


def test_init_not_callable_rejected():
    con = ducky.connect()
    with pytest.raises(ducky.Error, match="init= must be a callable"):
        con.create_function(
            "f",
            lambda x: x,
            parameters=["BIGINT"],
            return_type="BIGINT",
            init=42,  # ty: ignore[invalid-argument-type]
        )


def test_special_handling_state_and_mask_combine():
    # All three features together: a stateful UDF that observes NULL inputs.
    con = ducky.connect()
    con.execute("CREATE TABLE t (x BIGINT)")
    con.execute("INSERT INTO t VALUES (1), (NULL), (3), (NULL), (5)")

    def factory():
        return [0]  # count of NULLs seen across this thread's lifetime

    def fill_with_null_count(state, x):
        values, mask = x
        state[0] += int((mask == 0).sum())
        out = values.copy()
        out[mask == 0] = state[0]
        return out

    con.create_function(
        "fillnc",
        fill_with_null_count,
        parameters=["BIGINT"],
        return_type="BIGINT",
        init=factory,
        special_handling=True,
        volatile=True,
    )
    rows = sorted(r[0] for r in con.sql("SELECT fillnc(x) FROM t").fetchall())
    # Non-null values pass through (1, 3, 5); the two NULL slots get the
    # running NULL count (the exact replacement depends on chunk ordering,
    # but at least one of the NULL-replaced rows must reflect a count >= 1).
    assert {1, 3, 5}.issubset(set(rows))
    assert any(r >= 1 and r not in (1, 3, 5) for r in rows)
