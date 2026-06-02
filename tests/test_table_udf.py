import pytest

import ducky


def test_fibonacci():
    def fibonacci(n: int):
        a, b = 0, 1
        for _ in range(n):
            yield (a,)
            a, b = b, a + b

    con = ducky.connect()
    con.create_table_function(
        "fibonacci", fibonacci, parameters=["BIGINT"], columns={"value": "BIGINT"}
    )
    rows = con.sql("SELECT value FROM fibonacci(8)").fetchall()
    expected = [0, 1, 1, 2, 3, 5, 8, 13]
    assert [r[0] for r in rows] == expected


def test_multi_column():
    def pairs(n):
        for i in range(n):
            yield (i, str(i))

    con = ducky.connect()
    con.create_table_function(
        "pairs", pairs, parameters=["BIGINT"], columns={"num": "BIGINT", "label": "VARCHAR"}
    )
    rows = con.sql("SELECT num, label FROM pairs(3)").fetchall()
    assert rows == [(0, "0"), (1, "1"), (2, "2")]


def test_with_parameter():
    def counter(start, stop):
        for i in range(start, stop):
            yield (i,)

    con = ducky.connect()
    con.create_table_function(
        "counter", counter, parameters=["BIGINT", "BIGINT"], columns={"v": "BIGINT"}
    )
    rows = con.sql("SELECT v FROM counter(3, 6)").fetchall()
    assert [r[0] for r in rows] == [3, 4, 5]


def test_empty_generator():
    def empty(n):
        return
        yield

    con = ducky.connect()
    con.create_table_function("empty_fn", empty, parameters=["BIGINT"], columns={"v": "BIGINT"})
    rows = con.sql("SELECT v FROM empty_fn(5)").fetchall()
    assert rows == []


def test_null_values():
    def with_nulls():
        yield (None, 1)
        yield (2, None)

    con = ducky.connect()
    con.create_table_function(
        "with_nulls", with_nulls, parameters=[], columns={"a": "BIGINT", "b": "BIGINT"}
    )
    rows = con.sql("SELECT a, b FROM with_nulls()").fetchall()
    assert rows == [(None, 1), (2, None)]


def test_multi_chunk_boundary():
    n = 5000

    def big(n):
        for i in range(n):
            yield (i,)

    con = ducky.connect()
    con.create_table_function("big_gen", big, parameters=["BIGINT"], columns={"v": "BIGINT"})
    rows = con.sql(f"SELECT v FROM big_gen({n})").fetchall()
    assert len(rows) == n
    assert [r[0] for r in rows] == list(range(n))


def test_user_exception_propagates():
    def bad_gen(n):
        yield (1,)
        raise RuntimeError("intentional generator error")

    con = ducky.connect()
    con.create_table_function("bad_gen", bad_gen, parameters=["BIGINT"], columns={"v": "BIGINT"})
    with pytest.raises(ducky.Error, match="intentional generator error"):
        con.sql("SELECT v FROM bad_gen(5)").fetchall()


def test_varchar_output():
    def words():
        for w in ["hello", "world", "ducky"]:
            yield (w,)

    con = ducky.connect()
    con.create_table_function("words", words, parameters=[], columns={"word": "VARCHAR"})
    rows = con.sql("SELECT word FROM words()").fetchall()
    assert [r[0] for r in rows] == ["hello", "world", "ducky"]


def test_invalid_column_type_raises():
    def gen():
        yield (1,)

    con = ducky.connect()
    with pytest.raises(ducky.Error):
        con.create_table_function("bad_type", gen, parameters=[], columns={"v": "NOT_A_REAL_TYPE"})


def test_single_value_row():
    # Generator yielding bare scalars (not tuples) should also work.
    def scalars(n):
        yield from range(n)

    con = ducky.connect()
    con.create_table_function("scalars", scalars, parameters=["BIGINT"], columns={"v": "BIGINT"})
    rows = con.sql("SELECT v FROM scalars(3)").fetchall()
    assert [r[0] for r in rows] == [0, 1, 2]
