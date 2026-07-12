"""Round-trip coverage for parameter binding in Connection.execute."""

import datetime as dt
import uuid
from decimal import Decimal

import numpy as np
import pytest

import ducky


def roundtrip(value, cast: str | None = None):
    con = ducky.connect()
    sql = f"SELECT CAST(? AS {cast})" if cast else "SELECT ?"
    return con.execute(sql, [value]).fetchitem()


def test_bind_bytes():
    assert roundtrip(b"\x00\x01\x02hello") == b"\x00\x01\x02hello"


def test_bind_date():
    assert roundtrip(dt.date(2024, 11, 5)) == dt.date(2024, 11, 5)


def test_bind_time():
    assert roundtrip(dt.time(12, 34, 56, 789012)) == dt.time(12, 34, 56, 789012)


def test_bind_datetime_naive():
    naive = dt.datetime(2024, 11, 5, 12, 34, 56, 789012)
    assert roundtrip(naive) == naive


def test_bind_datetime_aware():
    aware = dt.datetime(2024, 11, 5, 12, 34, 56, 789012, tzinfo=dt.UTC)
    got = roundtrip(aware)
    assert got == aware


def test_bind_timedelta():
    td = dt.timedelta(days=3, seconds=42, microseconds=123)
    got = roundtrip(td, cast="INTERVAL")
    assert got == td


def test_bind_decimal():
    got = roundtrip(Decimal("3.14159"), cast="DECIMAL(10, 5)")
    assert got == Decimal("3.14159")


def test_bind_uuid():
    u = uuid.UUID("12345678-1234-5678-1234-567812345678")
    got = roundtrip(u, cast="UUID")
    assert got == u


def test_bind_hugeint_positive():
    big = 2**100 + 7
    got = roundtrip(big, cast="HUGEINT")
    assert got == big


def test_bind_hugeint_negative():
    big = -(2**100 + 7)
    got = roundtrip(big, cast="HUGEINT")
    assert got == big


def test_bind_hugeint_overflow(con):
    with pytest.raises(ducky.Error, match="overflows"):
        con.execute("SELECT ?", [2**200])


def test_bind_hugeint_boundary(con):
    # -2**127 fits a signed int128 even though bit_length is 128.
    assert roundtrip(-(2**127), cast="HUGEINT") == -(2**127)
    assert roundtrip(2**127 - 1, cast="HUGEINT") == 2**127 - 1
    # 2**128 - 1 fits a UHUGEINT.
    assert roundtrip(2**128 - 1, cast="UHUGEINT") == 2**128 - 1
    # Just past UHUGEINT must trigger our binding-side overflow.
    with pytest.raises(ducky.Error, match="overflows"):
        con.execute("SELECT ?", [2**128])
    # And the negative side, just past INT128_MIN.
    with pytest.raises(ducky.Error, match="overflows"):
        con.execute("SELECT ?", [-(2**127) - 1])


def test_bind_numpy_scalar_int():
    assert roundtrip(np.int32(7)) == 7
    assert roundtrip(np.int64(9_999_999_999)) == 9_999_999_999


def test_bind_numpy_scalar_float():
    assert roundtrip(np.float32(0.5)) == pytest.approx(0.5)
    assert roundtrip(np.float64(2.25)) == 2.25


def test_bind_named_parameters(con):
    got = con.execute(
        "SELECT $a + $b, $a * $b",
        {"a": 3, "b": 4},
    ).fetchone()
    assert got == (7, 12)


def test_bind_named_unknown_key(con):
    with pytest.raises(ducky.Error, match="unknown named parameter"):
        con.execute("SELECT $a", {"missing": 1})


def test_bind_unsupported_type(con):
    with pytest.raises(ducky.Error, match="unsupported parameter type"):
        con.execute("SELECT ?", [object()])


# Integer parameters bind at the width the statement expects (see
# bind_int64_as_param_type in connection.cpp): a BIGINT value bound into e.g.
# an INTEGER parameter would otherwise force DuckDB to re-bind and re-plan the
# statement on every execute.


def test_bind_int_widths_roundtrip(con):
    con.execute(
        "CREATE TABLE widths (t TINYINT, s SMALLINT, i INTEGER, b BIGINT, "
        "ut UTINYINT, us USMALLINT, ui UINTEGER, ub UBIGINT, h HUGEINT)"
    )
    row = (
        -128,
        -32768,
        -2147483648,
        -(2**63),
        255,
        65535,
        4294967295,
        2**64 - 1,  # exceeds int64: takes the uhugeint path, cast on execute
        -7,  # int64-fitting value into a HUGEINT parameter
    )
    con.execute("INSERT INTO widths VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", list(row))
    assert con.sql("SELECT * FROM widths").fetchone() == row


def test_bind_float_param_roundtrip():
    assert roundtrip(1.5, "FLOAT") == 1.5
    assert roundtrip(1.5, "DOUBLE") == 1.5


def test_bind_small_int_into_hugeint_param():
    assert roundtrip(-5, "HUGEINT") == -5
    assert roundtrip(5, "HUGEINT") == 5


def test_bind_int_overflowing_narrow_column_errors(con):
    con.execute("CREATE TABLE narrow (t TINYINT)")
    with pytest.raises(ducky.Error):
        con.execute("INSERT INTO narrow VALUES (?)", [300])


def test_bind_negative_into_unsigned_column_errors(con):
    con.execute("CREATE TABLE u (x UINTEGER)")
    with pytest.raises(ducky.Error):
        con.execute("INSERT INTO u VALUES (?)", [-1])


def test_executemany_typed_widths_roundtrip(con):
    con.execute("CREATE TABLE typed (a INT, b BIGINT, c DOUBLE, f FLOAT)")
    rows = [(i, i * 2, float(i), float(i)) for i in range(-50, 50)]
    stmt = con.prepare("INSERT INTO typed VALUES (?, ?, ?, ?)")
    stmt.executemany(rows)
    assert con.sql("SELECT * FROM typed ORDER BY a").fetchall() == rows
