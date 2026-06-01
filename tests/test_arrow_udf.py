# FIX: pyarrow.compute's stub is apparently not very good.
# ty: ignore[unresolved-attribute]
from decimal import Decimal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import ducky

# ── Default calling convention: positional pa.Array args ──────────────────


def test_varchar_in_out():
    con = ducky.connect()
    con.create_arrow_function("up", pc.utf8_upper, parameters=["VARCHAR"], return_type="VARCHAR")
    rows = con.sql("SELECT up(x) FROM (VALUES ('hi'), ('there'), (NULL)) t(x)").fetchall()
    assert rows == [("HI",), ("THERE",), (None,)]


def test_multi_arg_positional():
    con = ducky.connect()

    def concat(a, b):
        return pc.binary_join_element_wise(a, b, "-")

    con.create_arrow_function(
        "join_dash", concat, parameters=["VARCHAR", "VARCHAR"], return_type="VARCHAR"
    )
    [(v,)] = con.sql("SELECT join_dash('foo', 'bar')").fetchall()
    assert v == "foo-bar"


def test_list_output():
    con = ducky.connect()

    def to_pair(x):
        return pa.ListArray.from_arrays(
            pa.array([0, 2, 4, 6], type=pa.int32()),
            pa.compute.cast(
                pa.concat_arrays([pa.array([v, v + 1]) for v in x.to_pylist()]),
                pa.int64(),
            ),
        )

    con.create_arrow_function("pair", to_pair, parameters=["BIGINT"], return_type="BIGINT[]")
    rows = con.sql("SELECT pair(i) FROM range(3) t(i) ORDER BY i").fetchall()
    assert rows == [([0, 1],), ([1, 2],), ([2, 3],)]


def test_struct_output():
    con = ducky.connect()

    def split(x):
        lo = pc.bit_wise_and(x, pa.scalar(0xFFFF, type=pa.int64()))
        hi = pc.shift_right(x, pa.scalar(16, type=pa.int64()))
        return pa.StructArray.from_arrays([lo, hi], ["lo", "hi"])

    con.create_arrow_function(
        "split", split, parameters=["BIGINT"], return_type="STRUCT(lo BIGINT, hi BIGINT)"
    )
    [(v,)] = con.sql("SELECT split(CAST(70000 AS BIGINT))").fetchall()
    assert v == {"lo": 70000 & 0xFFFF, "hi": 70000 >> 16}


def test_decimal_round_trip():
    con = ducky.connect()

    def add_tax(x):
        return pc.multiply(x, pa.scalar(Decimal("1.10"), type=pa.decimal128(5, 2)))

    con.create_arrow_function(
        "tax", add_tax, parameters=["DECIMAL(5,2)"], return_type="DECIMAL(7,4)"
    )
    [(v,)] = con.sql("SELECT tax(CAST(10.00 AS DECIMAL(5,2)))").fetchall()
    assert v == Decimal("11.0000")


def test_chunk_boundary():
    n = 5000
    con = ducky.connect()
    con.create_arrow_function(
        "len_py", pc.utf8_length, parameters=["VARCHAR"], return_type="INTEGER"
    )
    arr = con.sql(f"SELECT len_py(CAST(i AS VARCHAR)) AS v FROM range({n}) t(i)").to_numpy()["v"]
    expected = np.array([len(str(i)) for i in range(n)], dtype=np.int32)
    np.testing.assert_array_equal(arr, expected)


# ── Dict-style calling convention ─────────────────────────────────────────


def test_dict_named_params():
    con = ducky.connect()

    def concat(cols):
        assert list(cols.keys()) == ["a", "b"]
        return pc.binary_join_element_wise(cols["a"], cols["b"], "-")

    con.create_arrow_function(
        "join_dash", concat, parameters={"a": "VARCHAR", "b": "VARCHAR"}, return_type="VARCHAR"
    )
    [(v,)] = con.sql("SELECT join_dash('foo', 'bar')").fetchall()
    assert v == "foo-bar"


# ── record_batch=True calling convention ──────────────────────────────────


def test_record_batch_mode():
    con = ducky.connect()

    def upper(rb):
        assert isinstance(rb, pa.RecordBatch)
        return pc.utf8_upper(rb.column(0))

    con.create_arrow_function(
        "up_rb", upper, parameters=["VARCHAR"], return_type="VARCHAR", record_batch=True
    )
    rows = con.sql("SELECT up_rb(x) FROM (VALUES ('hi'), (NULL)) t(x)").fetchall()
    assert rows == [("HI",), (None,)]


def test_record_batch_schema_names_with_dict_params():
    con = ducky.connect()

    def concat(rb):
        assert rb.schema.names == ["a", "b"]
        return pc.binary_join_element_wise(rb.column("a"), rb.column("b"), "-")

    con.create_arrow_function(
        "join_dash_rb",
        concat,
        parameters={"a": "VARCHAR", "b": "VARCHAR"},
        return_type="VARCHAR",
        record_batch=True,
    )
    [(v,)] = con.sql("SELECT join_dash_rb('foo', 'bar')").fetchall()
    assert v == "foo-bar"


# ── Error cases ────────────────────────────────────────────────────────────


def test_empty_parameters_errors():
    con = ducky.connect()
    with pytest.raises(ducky.Error, match="at least one parameter"):
        con.create_arrow_function("f", lambda x: x, parameters=[], return_type="VARCHAR")


def test_invalid_type_errors():
    con = ducky.connect()
    with pytest.raises(ducky.Error, match="invalid"):
        con.create_arrow_function(
            "f", lambda x: x, parameters=["NOT_A_TYPE"], return_type="VARCHAR"
        )


def test_user_exception_propagates():
    con = ducky.connect()

    def boom(x):
        raise ValueError("nope")

    con.create_arrow_function("boom", boom, parameters=["VARCHAR"], return_type="VARCHAR")
    with pytest.raises(ducky.Error, match="nope"):
        con.sql("SELECT boom('hi')").fetchall()


def test_wrong_row_count_errors():
    con = ducky.connect()

    def shrink(x):
        return x.slice(0, 1)

    con.create_arrow_function("shrink", shrink, parameters=["VARCHAR"], return_type="VARCHAR")
    with pytest.raises(ducky.Error, match="rows; expected"):
        con.sql("SELECT shrink(x) FROM (VALUES ('a'), ('b'), ('c')) t(x)").fetchall()
