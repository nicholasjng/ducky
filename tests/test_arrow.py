"""Arrow export (`.arrow()` / the Arrow C stream).

The stream has a combined fast path for results whose columns are all
fixed-width types with a DuckDB layout bit-identical to Arrow's
(`fast_arrow_width` in result.cpp); everything else falls back to one Arrow
array per DuckDB data chunk. These tests pin that the fast path is byte-for-byte
equal to the fallback (DuckDB's own `duckdb_data_chunk_to_arrow`), so extending
the eligible type set can't silently corrupt data.
"""

from __future__ import annotations

import pytest

import ducky

pytest.importorskip("pyarrow")

# Expressions over `range(n) s(i)`, one per fast-path-eligible type. Each must
# stay byte-equal to the fallback when widened into a combined batch.
_FAST_EXPRS = {
    "TINYINT": "(i % 100)::TINYINT",
    "SMALLINT": "i::SMALLINT",
    "INTEGER": "i::INTEGER",
    "BIGINT": "i::BIGINT",
    "UTINYINT": "(i % 200)::UTINYINT",
    "USMALLINT": "i::USMALLINT",
    "UINTEGER": "i::UINTEGER",
    "UBIGINT": "i::UBIGINT",
    "FLOAT": "i::FLOAT",
    "DOUBLE": "(i * 1.5)::DOUBLE",
    "DATE": "(DATE '2000-01-01' + (i || ' days')::INTERVAL)::DATE",
    "TIME": "(TIME '00:00:00' + (i * 1000 || ' microseconds')::INTERVAL)::TIME",
    "TIMESTAMP": "(TIMESTAMP '2000-01-01' + (i || ' seconds')::INTERVAL)",
    "TIMESTAMP_TZ": "(TIMESTAMP '2000-01-01' + (i || ' seconds')::INTERVAL)::TIMESTAMPTZ",
    "TIMESTAMP_S": "(TIMESTAMP '2000-01-01' + (i || ' seconds')::INTERVAL)::TIMESTAMP_S",
    "TIMESTAMP_MS": "(TIMESTAMP '2000-01-01' + (i || ' seconds')::INTERVAL)::TIMESTAMP_MS",
    "TIMESTAMP_NS": "(TIMESTAMP '2000-01-01' + (i || ' seconds')::INTERVAL)::TIMESTAMP_NS",
}

# n > STANDARD_VECTOR_SIZE (2048) so the fallback yields many chunks and the
# fast path must concatenate.
_N = 20_000


@pytest.mark.parametrize("expr", _FAST_EXPRS.values(), ids=list(_FAST_EXPRS))
def test_fast_path_equals_fallback(expr: str) -> None:
    con = ducky.connect()
    # All-eligible result -> combined fast path (a single chunk).
    fast = con.execute(f"SELECT {expr} AS v FROM range({_N}) s(i)").arrow()
    # A VARCHAR column makes the result ineligible -> per-chunk fallback.
    fallback = con.execute(f"SELECT {expr} AS v, i::VARCHAR x FROM range({_N}) s(i)").arrow()

    assert fast.num_rows == _N
    assert fast.column("v").num_chunks == 1
    assert fallback.column("v").num_chunks > 1
    assert fast.column("v").combine_chunks().equals(fallback.column("v").combine_chunks())


def test_fast_path_nulls_across_chunks() -> None:
    # Nulls scattered across chunk boundaries must land in the right bitmap bits.
    con = ducky.connect()
    n = 50_000
    q = f"SELECT CASE WHEN i % 11 = 0 THEN NULL ELSE i END::INT AS v FROM range({n}) s(i)"
    tbl = con.execute(q).arrow()
    assert tbl.column("v").num_chunks == 1
    assert tbl.column("v").null_count == sum(1 for i in range(n) if i % 11 == 0)
    assert tbl.column("v").to_pylist() == [None if i % 11 == 0 else i for i in range(n)]


def test_fast_path_multi_column_combined() -> None:
    con = ducky.connect()
    tbl = con.execute(
        "SELECT i::INT a, (i * 2)::BIGINT b, (i * 1.5)::DOUBLE c FROM range(10000) s(i)"
    ).arrow()
    assert tbl.column("a").num_chunks == 1
    assert tbl.column("a").to_pylist()[:3] == [0, 1, 2]
    assert tbl.column("c").to_pylist()[:3] == [0.0, 1.5, 3.0]


def test_fallback_handles_strings_and_nested() -> None:
    # Ineligible types still export correctly via the per-chunk fallback.
    con = ducky.connect()
    tbl = con.execute("SELECT i::VARCHAR s, [i, i + 1] AS lst FROM range(5000) s(i)").arrow()
    assert tbl.num_rows == 5000
    assert tbl.column("s").to_pylist()[:3] == ["0", "1", "2"]
    assert tbl.column("lst").to_pylist()[0] == [0, 1]


def test_empty_result_emits_no_rows() -> None:
    con = ducky.connect()
    tbl = con.execute("SELECT i::INT v FROM range(10) s(i) WHERE i < 0").arrow()
    assert tbl.num_rows == 0
    assert tbl.column_names == ["v"]
