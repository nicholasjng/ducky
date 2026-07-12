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


def test_arrow_batches_streams_per_chunk(con):
    import pyarrow as pa

    con.execute("CREATE TABLE t AS SELECT range AS a, range * 2 AS b FROM range(5000)")
    # Result.arrow_batches() yields a lazy RecordBatchReader, one batch per chunk.
    reader = con.sql("SELECT * FROM t").arrow_batches()
    assert isinstance(reader, pa.RecordBatchReader)
    batches = list(reader)
    assert len(batches) > 1  # 5000 rows > STANDARD_VECTOR_SIZE → multiple chunks
    assert sum(b.num_rows for b in batches) == 5000
    # Same data as the eager table.
    assert pa.Table.from_batches(batches).equals(con.sql("SELECT * FROM t").arrow())
    # Also reachable via the Connection (delegates to the current result).
    rows = sum(b.num_rows for b in con.execute("SELECT * FROM t").arrow_batches())
    assert rows == 5000


def test_arrow_reader_born_as_arrow(con):
    import pyarrow as pa

    con.execute("CREATE TABLE t AS SELECT range AS a, range * 2 AS b FROM range(20000)")
    # Connection.arrow_reader(query): DuckDB's collector assembles batches during execution.
    reader = con.arrow_reader("SELECT * FROM t", rows_per_batch=2048)
    assert isinstance(reader, pa.RecordBatchReader)
    batches = list(reader)
    assert sum(b.num_rows for b in batches) == 20000
    # Same schema + data as the chunk-converted path.
    table = pa.Table.from_batches(batches)
    assert table.sort_by("a").equals(con.sql("SELECT * FROM t ORDER BY a").arrow())
    # A second execution reuses the cached statement and starts clean.
    again = list(con.arrow_reader("SELECT * FROM t", rows_per_batch=2048))
    assert sum(b.num_rows for b in again) == 20000


def test_arrow_query_overload_born_as_arrow(con):
    import pyarrow as pa

    con.execute("CREATE TABLE t AS SELECT range AS a, range * 2 AS b FROM range(20000)")
    # arrow(query) collects; arrow() (no arg) converts the current result. Both equal.
    collected = con.arrow("SELECT * FROM t")
    assert isinstance(collected, pa.Table)
    converted = con.execute("SELECT * FROM t").arrow()
    assert collected.sort_by("a").equals(converted.sort_by("a"))
    assert collected.num_rows == 20000


def test_arrow_collected_parameter_binding(con):
    con.execute("CREATE TABLE t AS SELECT range AS a, range * 2 AS b FROM range(20000)")
    # Positional params on both born-as-Arrow entry points.
    table = con.arrow("SELECT * FROM t WHERE a >= ? AND a < ?", [100, 200])
    assert table.num_rows == 100
    assert table.column("a").to_pylist() == list(range(100, 200))
    reader = con.arrow_reader("SELECT * FROM t WHERE a < ?", [50], rows_per_batch=16)
    assert sum(b.num_rows for b in reader) == 50
    # Named params (dict) work too.
    named = con.arrow("SELECT * FROM t WHERE a = $x", {"x": 12345})
    assert named.column("b").to_pylist() == [24690]
    # A bound query then an unbound one reuses the cached statement, unbound clean.
    assert con.arrow("SELECT count(*) AS n FROM t").column("n").to_pylist() == [20000]


@pytest.mark.parametrize("expr", _FAST_EXPRS.values(), ids=list(_FAST_EXPRS))
def test_fast_path_equals_fallback(expr: str, con) -> None:
    # All-eligible result -> combined fast path (a single chunk).
    fast = con.execute(f"SELECT {expr} AS v FROM range({_N}) s(i)").arrow()
    # A VARCHAR column makes the result ineligible -> per-chunk fallback.
    fallback = con.execute(f"SELECT {expr} AS v, i::VARCHAR x FROM range({_N}) s(i)").arrow()

    assert fast.num_rows == _N
    assert fast.column("v").num_chunks == 1
    assert fallback.column("v").num_chunks > 1
    assert fast.column("v").combine_chunks().equals(fallback.column("v").combine_chunks())


def test_fast_path_nulls_across_chunks(con) -> None:
    # Nulls scattered across chunk boundaries must land in the right bitmap bits.
    n = 50_000
    q = f"SELECT CASE WHEN i % 11 = 0 THEN NULL ELSE i END::INT AS v FROM range({n}) s(i)"
    tbl = con.execute(q).arrow()
    assert tbl.column("v").num_chunks == 1
    assert tbl.column("v").null_count == sum(1 for i in range(n) if i % 11 == 0)
    assert tbl.column("v").to_pylist() == [None if i % 11 == 0 else i for i in range(n)]


def test_fast_path_multi_column_combined(con) -> None:
    tbl = con.execute(
        "SELECT i::INT a, (i * 2)::BIGINT b, (i * 1.5)::DOUBLE c FROM range(10000) s(i)"
    ).arrow()
    assert tbl.column("a").num_chunks == 1
    assert tbl.column("a").to_pylist()[:3] == [0, 1, 2]
    assert tbl.column("c").to_pylist()[:3] == [0.0, 1.5, 3.0]


def test_fallback_handles_strings_and_nested(con) -> None:
    # Ineligible types still export correctly via the per-chunk fallback.
    tbl = con.execute("SELECT i::VARCHAR s, [i, i + 1] AS lst FROM range(5000) s(i)").arrow()
    assert tbl.num_rows == 5000
    assert tbl.column("s").to_pylist()[:3] == ["0", "1", "2"]
    assert tbl.column("lst").to_pylist()[0] == [0, 1]


def test_empty_result_emits_no_rows(con) -> None:
    tbl = con.execute("SELECT i::INT v FROM range(10) s(i) WHERE i < 0").arrow()
    assert tbl.num_rows == 0
    assert tbl.column_names == ["v"]
