import datetime as dt

import pytest

import ducky


def test_fibonacci(con):
    def fibonacci(n: int):
        a, b = 0, 1
        for _ in range(n):
            yield (a,)
            a, b = b, a + b

    con.create_table_function(
        "fibonacci", fibonacci, parameters=["BIGINT"], columns={"value": "BIGINT"}
    )
    rows = con.sql("SELECT value FROM fibonacci(8)").fetchall()
    expected = [0, 1, 1, 2, 3, 5, 8, 13]
    assert [r[0] for r in rows] == expected


def test_multi_column(con):
    def pairs(n):
        for i in range(n):
            yield (i, str(i))

    con.create_table_function(
        "pairs", pairs, parameters=["BIGINT"], columns={"num": "BIGINT", "label": "VARCHAR"}
    )
    rows = con.sql("SELECT num, label FROM pairs(3)").fetchall()
    assert rows == [(0, "0"), (1, "1"), (2, "2")]


def test_with_parameter(con):
    def counter(start, stop):
        for i in range(start, stop):
            yield (i,)

    con.create_table_function(
        "counter", counter, parameters=["BIGINT", "BIGINT"], columns={"v": "BIGINT"}
    )
    rows = con.sql("SELECT v FROM counter(3, 6)").fetchall()
    assert [r[0] for r in rows] == [3, 4, 5]


def test_empty_generator(con):
    def empty(n):
        return
        yield

    con.create_table_function("empty_fn", empty, parameters=["BIGINT"], columns={"v": "BIGINT"})
    rows = con.sql("SELECT v FROM empty_fn(5)").fetchall()
    assert rows == []


def test_null_values(con):
    def with_nulls():
        yield (None, 1)
        yield (2, None)

    con.create_table_function(
        "with_nulls", with_nulls, parameters=[], columns={"a": "BIGINT", "b": "BIGINT"}
    )
    rows = con.sql("SELECT a, b FROM with_nulls()").fetchall()
    assert rows == [(None, 1), (2, None)]


def test_multi_chunk_boundary(con):
    n = 5000

    def big(n):
        for i in range(n):
            yield (i,)

    con.create_table_function("big_gen", big, parameters=["BIGINT"], columns={"v": "BIGINT"})
    rows = con.sql(f"SELECT v FROM big_gen({n})").fetchall()
    assert len(rows) == n
    assert [r[0] for r in rows] == list(range(n))


def test_user_exception_propagates(con):
    def bad_gen(n):
        yield (1,)
        raise RuntimeError("intentional generator error")

    con.create_table_function("bad_gen", bad_gen, parameters=["BIGINT"], columns={"v": "BIGINT"})
    with pytest.raises(ducky.Error, match="intentional generator error"):
        con.sql("SELECT v FROM bad_gen(5)").fetchall()


def test_varchar_output(con):
    def words():
        for w in ["hello", "world", "ducky"]:
            yield (w,)

    con.create_table_function("words", words, parameters=[], columns={"word": "VARCHAR"})
    rows = con.sql("SELECT word FROM words()").fetchall()
    assert [r[0] for r in rows] == ["hello", "world", "ducky"]


def test_invalid_column_type_raises(con):
    def gen():
        yield (1,)

    with pytest.raises(ducky.Error):
        con.create_table_function("bad_type", gen, parameters=[], columns={"v": "NOT_A_REAL_TYPE"})


def test_single_value_row(con):
    # Generator yielding bare scalars (not tuples) should also work.
    def scalars(n):
        yield from range(n)

    con.create_table_function("scalars", scalars, parameters=["BIGINT"], columns={"v": "BIGINT"})
    rows = con.sql("SELECT v FROM scalars(3)").fetchall()
    assert [r[0] for r in rows] == [0, 1, 2]


def test_temporal_columns(con):
    def rows():
        yield (
            dt.date(2024, 1, 2),
            dt.time(3, 4, 5, 6),
            dt.datetime(2024, 1, 2, 3, 4, 5, 6),
        )
        # Raw integer storage values are accepted alongside the objects.
        yield (0, 0, 0)

    con.create_table_function(
        "temporals",
        rows,
        parameters=[],
        columns={"d": "DATE", "t": "TIME", "ts": "TIMESTAMP"},
    )
    got = con.sql("SELECT d, t, ts FROM temporals()").fetchall()
    assert got == [
        (dt.date(2024, 1, 2), dt.time(3, 4, 5, 6), dt.datetime(2024, 1, 2, 3, 4, 5, 6)),
        (dt.date(1970, 1, 1), dt.time(0, 0), dt.datetime(1970, 1, 1)),
    ]


def test_timestamp_tz_column_shifts_aware_to_utc(con):
    tz = dt.timezone(dt.timedelta(hours=2))
    aware = dt.datetime(2024, 6, 1, 12, 30, 0, tzinfo=tz)

    def rows():
        yield (aware,)

    con.create_table_function("tz_rows", rows, parameters=[], columns={"ts": "TIMESTAMPTZ"})
    (row,) = con.sql("SELECT ts FROM tz_rows()").fetchall()
    # aware datetime comparison normalizes across offsets.
    assert row[0] == aware


def test_unsupported_column_type_rejected_at_registration(con):
    def gen():
        yield (1,)

    # Unsupported types fail during registration.
    with pytest.raises(ducky.Error, match="unsupported type DECIMAL"):
        con.create_table_function("dec_fn", gen, parameters=[], columns={"v": "DECIMAL(9,2)"})
    with pytest.raises(ducky.Error, match="unsupported type LIST"):
        con.create_table_function("list_fn", gen, parameters=[], columns={"v": "INTEGER[]"})


def test_wrong_arity_row_raises(con):
    def short_rows():
        yield (1,)

    def long_rows():
        yield (1, 2, 3)

    con.create_table_function(
        "short_rows", short_rows, parameters=[], columns={"a": "BIGINT", "b": "BIGINT"}
    )
    con.create_table_function(
        "long_rows", long_rows, parameters=[], columns={"a": "BIGINT", "b": "BIGINT"}
    )
    with pytest.raises(ducky.Error, match="has 1 values; expected 2"):
        con.sql("SELECT * FROM short_rows()").fetchall()
    with pytest.raises(ducky.Error, match="has 3 values; expected 2"):
        con.sql("SELECT * FROM long_rows()").fetchall()


def test_out_of_range_int_raises(con):
    def gen():
        yield (300,)

    con.create_table_function("overflow_gen", gen, parameters=[], columns={"v": "TINYINT"})
    with pytest.raises(ducky.Error, match="out of range for TINYINT"):
        con.sql("SELECT v FROM overflow_gen()").fetchall()
