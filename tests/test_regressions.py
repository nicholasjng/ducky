"""Regression tests."""

import datetime
import gc
import subprocess
import sys
import weakref

import pytest

import ducky

# Isolate native failures from the test process.

_FAILED_REGISTRATION_SCRIPT = """
import ducky

con = ducky.connect()

def f(x: int) -> int:
    return x

for register in (
    lambda: con.create_function("", f),
    lambda: con.create_arrow_function("", lambda a: a, ["INTEGER"], "INTEGER"),
    lambda: con.create_table_function("", lambda: iter(()), [], {"x": "INTEGER"}),
):
    try:
        register()
    except ducky.Error:
        pass

class Agg:
    def __init__(self):
        self.total = 0

    def update(self, values):
        self.total += int(values.sum())

    def combine(self, other):
        self.total += other.total

    def finalize(self):
        return self.total

try:
    con.create_aggregate_function("", Agg, ["BIGINT"], "BIGINT")
except ducky.Error:
    pass

# The connection must still be fully usable afterwards.
assert con.sql("SELECT 42").fetchitem() == 42
print("OK")
"""


def test_failed_udf_registration_does_not_crash(subprocess_python):
    executable, env = subprocess_python
    proc = subprocess.run(
        [executable, "-c", _FAILED_REGISTRATION_SCRIPT],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_out_of_range_date_raises_cleanly(con):
    with pytest.raises(ValueError, match="year"):
        con.sql("SELECT (DATE '2000-01-01' + INTERVAL 9000 YEARS)::DATE").fetchone()
    # The connection stays usable.
    assert con.sql("SELECT DATE '2000-01-01'").fetchitem() == datetime.date(2000, 1, 1)


def test_out_of_range_timestamp_raises_cleanly(con):
    with pytest.raises(ValueError, match="year"):
        con.sql("SELECT TIMESTAMP '2000-01-01' + INTERVAL 9000 YEARS").fetchone()


def test_interval_large_seconds_decodes_exactly(con):
    value = con.sql("SELECT INTERVAL '100000000' SECONDS").fetchitem()
    assert value == datetime.timedelta(seconds=100_000_000)


def test_interval_months_beyond_timedelta_range_raises(con):
    with pytest.raises(ducky.Error, match="overflows datetime.timedelta"):
        con.sql("SELECT INTERVAL '200000000' MONTHS").fetchone()


def test_profile_sink_exception_reaches_unraisablehook(con):

    def bad_sink(sql, info):
        raise ValueError("sink boom")

    # Avoid retaining the traceback through UnraisableHookArgs.
    seen = []
    previous_hook = sys.unraisablehook
    sys.unraisablehook = lambda args: seen.append((args.exc_type, str(args.exc_value)))
    try:
        con.set_profile_sink(bad_sink)
        rows = con.execute("SELECT 1").fetchall()
    finally:
        sys.unraisablehook = previous_hook
        con.set_profile_sink(None)

    assert rows == [(1,)]  # the sink error must not break the query
    assert seen == [(ValueError, "sink boom")]


def test_appender_destructor_swallows_flush_error(con):
    con.execute("CREATE TABLE checked(x INTEGER CHECK (x > 0))")
    app = con.appender("checked")
    app.append_row(-1)
    del app  # destructor must not raise / abort
    gc.collect()
    assert con.sql("SELECT count(*) FROM checked").fetchitem() == 0


def test_appender_explicit_close_surfaces_flush_error(con):
    con.execute("CREATE TABLE checked(x INTEGER CHECK (x > 0))")
    app = con.appender("checked")
    app.append_row(-1)
    with pytest.raises(ducky.Error):
        app.close()


def test_table_udf_blob_with_embedded_nul(con):
    payload = b"a\x00b"

    def factory():
        yield (payload,)

    con.create_table_function("blob_rows", factory, [], {"b": "BLOB"})
    assert con.sql("SELECT b FROM blob_rows()").fetchitem() == payload


def test_profile_sink_cycle_is_collectable():
    # `con` must live only inside make() — a fixture-provided one would stay
    # reachable for the whole test, and the cycle below would never collect.
    def make():
        con = ducky.connect()

        def sink(_sql, _info):
            return con.description  # closes over the connection -> reference cycle

        con.set_profile_sink(sink)
        return weakref.ref(sink)

    ref = make()
    gc.collect()
    assert ref() is None


def test_registered_arrow_source_cycle_is_collectable():
    # See test_profile_sink_cycle_is_collectable: `con` must stay local to make().
    pa = pytest.importorskip("pyarrow")

    class Source:
        def __init__(self, con, table):
            self.con = con  # points back at the connection -> reference cycle
            self.table = table

        def __arrow_c_stream__(self, requested_schema=None):
            return self.table.__arrow_c_stream__(requested_schema)

    def make():
        con = ducky.connect()
        src = Source(con, pa.table({"x": [1, 2, 3]}))
        con.register_arrow("t", src)
        assert con.sql("SELECT count(*) FROM t").fetchitem() == 3
        return weakref.ref(src)

    ref = make()
    gc.collect()
    assert ref() is None


def test_close_releases_profile_sink(con):

    def sink(_sql, _info):
        pass

    ref = weakref.ref(sink)
    con.set_profile_sink(sink)
    del sink
    assert ref() is not None
    con.close()
    assert ref() is None


def test_close_releases_registered_arrow_sources(con):
    pa = pytest.importorskip("pyarrow")

    class Source:
        def __init__(self, table):
            self.table = table

        def __arrow_c_stream__(self, requested_schema=None):
            return self.table.__arrow_c_stream__(requested_schema)

    src = Source(pa.table({"x": [1]}))
    ref = weakref.ref(src)
    con.register_arrow("t", src)
    del src
    assert ref() is not None
    con.close()
    assert ref() is None


def test_arrow_udf_invalid_parameter_type_raises(con):
    with pytest.raises(ducky.Error, match="invalid DuckDB type"):
        con.create_arrow_function("bad", lambda a, b: a, ["INTEGER", "NOT_A_TYPE"], "INTEGER")
    assert con.sql("SELECT 1").fetchitem() == 1


def test_table_udf_invalid_column_type_raises(con):
    with pytest.raises(ducky.Error, match="invalid DuckDB type"):
        con.create_table_function("bad", lambda: iter(()), [], {"a": "INTEGER", "b": "NOT_A_TYPE"})
    assert con.sql("SELECT 1").fetchitem() == 1


def test_appender_columns_multiple_slices_with_masks(con):
    np = pytest.importorskip("numpy")

    con.execute("CREATE TABLE big(x INTEGER, y DOUBLE)")
    n = 5000  # spans multiple ~2048-row slices
    x = np.arange(n, dtype=np.int32)
    mask = x % 3 != 0  # NULL every third row
    with con.appender("big") as app:
        app.append_columns({"x": x}, masks={"x": mask})  # y missing -> all NULL
    result = con.sql(
        "SELECT count(*), count(*) - count(x), count(*) - count(y), sum(x) FROM big"
    ).fetchone()
    assert result is not None
    total, x_nulls, y_nulls, x_sum = result
    assert total == n
    assert x_nulls == int((~mask).sum())
    assert y_nulls == n
    assert x_sum == int(x[mask].sum())


# ── Concurrency: blocking DuckDB calls must not hold the GIL ─────────────────
# Each scenario runs in a child process: a regression deadlocks rather than
# fails, and the timeout turns that into a test failure.


def _run_isolated(subprocess_python, script: str) -> None:
    executable, env = subprocess_python
    proc = subprocess.run(
        [executable, "-c", script], capture_output=True, text=True, timeout=120, env=env
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_streaming_fetch_with_udf_does_not_deadlock(subprocess_python):
    _run_isolated(
        subprocess_python,
        """
import ducky
con = ducky.connect(threads=8)
con.create_function("plus1", lambda x: x + 1, ["BIGINT"], "BIGINT")
res = con.sql("SELECT plus1(i) AS y FROM range(800000) t(i) ORDER BY y DESC", streaming=True)
rows = 0
while (chunk := res.fetch_chunk()) is not None:
    rows += len(chunk)
assert rows == 800000, rows
print("OK")
""",
    )


def test_streaming_scan_of_registered_arrow_does_not_deadlock(subprocess_python):
    pytest.importorskip("pyarrow")
    _run_isolated(
        subprocess_python,
        """
import numpy as np, pyarrow as pa, ducky
con = ducky.connect(threads=8)
n = 2_000_000
con.register_arrow("a", pa.table({"k": np.arange(n) % 1000, "v": np.arange(n)}))
con.register_arrow("b", pa.table({"k": np.arange(1000), "w": np.arange(1000)}))
for _ in range(3):
    res = con.sql(
        "SELECT a.k, v, w FROM a JOIN b USING (k) UNION ALL SELECT k, v, v FROM a", streaming=True
    )
    rows = 0
    while (chunk := res.fetch_chunk()) is not None:
        rows += len(chunk)
    assert rows == 2 * n, rows
print("OK")
""",
    )


def test_registering_udf_while_udf_query_runs(subprocess_python):
    _run_isolated(
        subprocess_python,
        """
import threading, time, ducky
con = ducky.connect(threads=4)

def slow(x):
    time.sleep(0.001)
    return x

con.create_function("slow", slow, ["BIGINT"], "BIGINT")
done = threading.Event()

def run():
    try:
        con.sql("SELECT sum(slow(i)) FROM range(2000000) t(i)").fetchall()
    except ducky.Error:
        pass  # interrupted below
    done.set()

thread = threading.Thread(target=run)
thread.start()
time.sleep(0.3)
con.create_function("ident", lambda x: x, ["BIGINT"], "BIGINT")
con.interrupt()
thread.join(60)
assert done.is_set()
print("OK")
""",
    )


def test_close_during_query_interrupts_it(subprocess_python):
    _run_isolated(
        subprocess_python,
        """
import threading, time, ducky
con = ducky.connect(threads=4)
con.set_profile_sink(lambda sql, info: None)
errors = []

def closer():
    time.sleep(0.3)
    con.close()

thread = threading.Thread(target=closer)
thread.start()
try:
    con.sql(
        "SELECT sum(x) FROM (SELECT row_number() OVER (ORDER BY hash(i)) AS x "
        "FROM range(30000000) t(i))"
    ).fetchall()
except ducky.Error as e:
    errors.append(str(e))
thread.join(60)
assert errors, "the query should have been interrupted"
print("OK")
""",
    )


def test_duckdb_python_loaded_after_ducky_works(subprocess_python):
    # ducky must not export DuckDB's symbols for another DuckDB to bind to.
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    _run_isolated(
        subprocess_python,
        """
import ducky
import duckdb
import pyarrow as pa
table = pa.table({"a": [1, 2, 3]})
con = duckdb.connect()
assert con.execute("SELECT 42").fetchall() == [(42,)]
con.execute("CREATE TABLE t (a BIGINT)")
con.from_arrow(table).insert_into("t")
assert con.execute("SELECT sum(a) FROM t").fetchall() == [(6,)]
dcon = ducky.connect()
dcon.register_arrow("s", table)
assert dcon.execute("SELECT sum(a) FROM s").fetchall() == [(6,)]
print("OK")
""",
    )


# ── Results ──────────────────────────────────────────────────────────────────


def test_streaming_runtime_error_raises(con):
    res = con.sql(
        "SELECT CASE WHEN i < 300000 THEN i ELSE error('boom') END AS x FROM range(500000) t(i)",
        streaming=True,
    )
    with pytest.raises(ducky.Error, match="boom"):
        res.fetchall()


def test_failed_stream_keeps_raising_in_bulk_exports(con):
    res = con.sql(
        "SELECT (CASE WHEN i < 3000 THEN i ELSE error('boom') END)::DOUBLE AS x "
        "FROM range(500000) t(i)",
        streaming=True,
    )
    for _ in range(2):  # the error is sticky, never an empty result
        with pytest.raises(ducky.Error, match="boom"):
            res.to_numpy()


def test_open_streaming_result_holds_the_connection(con):
    res = con.sql("SELECT * FROM range(100000)", streaming=True)
    res.fetchone()
    with pytest.raises(ducky.Error, match="open result"):
        con.execute("SELECT 42")
    assert len(res.fetchall()) == 99999  # the refused statement left the stream intact
    assert con.execute("SELECT 42").fetchone() == (42,)
    res = con.sql("SELECT * FROM range(100000)", streaming=True)
    res.fetchone()
    del res  # releasing the result frees the connection
    assert con.sql("SELECT 42").fetchitem() == 42


def test_execute_replaces_open_streaming_result(con):
    con.execute("SELECT * FROM range(100000)", streaming=True)
    assert con.fetchone() == (0,)
    con.execute("SELECT 42")
    assert con.fetchone() == (42,)


def test_streaming_end_is_sticky(con):
    res = con.sql("SELECT * FROM range(10)", streaming=True)
    assert len(res.fetchall()) == 10
    assert res.fetchone() is None
    assert res.fetch_chunk() is None


def test_bulk_exports_continue_after_fetchone(con):
    pytest.importorskip("pyarrow")
    q = "SELECT i::DOUBLE AS x FROM range(9999) t(i)"
    res = con.sql(q)
    assert res.fetchone() == (0.0,)
    assert res.arrow().num_rows == 9998
    res = con.sql(q)
    res.fetchone()
    assert len(res.to_numpy()["x"]) == 9998
    res = con.sql(q)
    res.fetchone()
    rows = 0
    while (chunk := res.fetch_chunk()) is not None:
        rows += len(chunk)
    assert rows == 9998


def test_second_drain_of_result_is_empty(con):
    pytest.importorskip("pyarrow")
    res = con.sql("SELECT i::DOUBLE AS x FROM range(10) t(i)")
    assert len(res.to_numpy()["x"]) == 10
    assert len(res.to_numpy()["x"]) == 0
    assert res.arrow().num_rows == 0
    assert res.fetchall() == []


def test_to_numpy_nulls(con):
    np = pytest.importorskip("numpy")
    out = con.sql("SELECT x FROM (VALUES (1.5::DOUBLE), (NULL)) t(x)").to_numpy()
    assert out["x"][0] == 1.5 and np.isnan(out["x"][1])
    with pytest.raises(ducky.Error, match="NULL"):
        con.sql("SELECT x FROM (VALUES (1), (NULL)) t(x)").to_numpy()


def test_fetch_numpy_chunk_rejects_unsupported_type_before_pulling(con):
    res = con.sql("SELECT i, i::VARCHAR AS s FROM range(5000) t(i)")
    with pytest.raises(ducky.Error, match="VARCHAR"):
        res.fetch_numpy_chunk()
    assert len(res.fetchall()) == 5000


def test_temporal_edge_values(con):
    assert con.sql("SELECT 'infinity'::DATE").fetchitem() == datetime.date.max
    assert con.sql("SELECT '-infinity'::DATE").fetchitem() == datetime.date.min
    naive_max = datetime.datetime(9999, 12, 31, 23, 59, 59, 999999)  # datetime.max, naive
    assert con.sql("SELECT 'infinity'::TIMESTAMP").fetchitem() == naive_max
    assert con.sql("SELECT 'infinity'::TIMESTAMP_S").fetchitem() == naive_max
    utc = datetime.UTC
    assert con.sql("SELECT '-infinity'::TIMESTAMPTZ").fetchitem() == datetime.datetime.min.replace(
        tzinfo=utc
    )
    # Sub-microsecond instants before the epoch round towards the past.
    assert con.sql("SELECT '1969-12-31 23:59:59.999999999'::TIMESTAMP_NS").fetchitem() == (
        datetime.datetime(1969, 12, 31, 23, 59, 59, 999999)
    )
    assert con.sql("SELECT INTERVAL '-1 hour'").fetchitem() == datetime.timedelta(hours=-1)


def test_decoding_newer_types(con):
    assert con.sql("SELECT '12:34:56.123456789'::TIME_NS").fetchitem() == datetime.time(
        12, 34, 56, 123456
    )
    tz = datetime.timezone(datetime.timedelta(hours=2, minutes=30))
    assert con.sql("SELECT '12:34:56+02:30'::TIMETZ").fetchitem() == datetime.time(
        12, 34, 56, tzinfo=tz
    )
    assert con.sql("SELECT '0000000011'::BIT").fetchitem() == "0000000011"
    big = -12345678901234567890123456789
    assert con.sql(f"SELECT '{big}'::BIGNUM").fetchitem() == big
    union = "UNION(num INT, str VARCHAR)"
    assert con.sql(f"SELECT union_value(str := 'x')::{union}").fetchitem() == "x"
    wkb = con.sql("SELECT 'POINT(1 2)'::GEOMETRY").fetchitem()
    assert isinstance(wkb, bytes) and len(wkb) == 21
    assert con.sql("SELECT '1'::BIT, NULL").types == ["BIT", "NULL"]


# ── Parameters and statements ────────────────────────────────────────────────


def test_decimal_and_uuid_parameters(con):
    import decimal
    import uuid

    d = decimal.Decimal("123456789012345678901234567890.12345678")
    assert con.execute("SELECT ?, typeof(?)", [d, d]).fetchone() == (d, "DECIMAL(38,8)")
    assert con.execute("SELECT ? * 2", [decimal.Decimal("1.5")]).fetchitem() == decimal.Decimal(
        "3.0"
    )
    u = uuid.uuid4()
    assert con.execute("SELECT ?, typeof(?)", [u, u]).fetchone() == (u, "UUID")


def test_cached_plans_see_writes(con):
    con.execute("CREATE TABLE w AS SELECT (i % 4)::INTEGER AS x FROM range(1000) t(i)")
    q = "SELECT count(*) FROM w WHERE x > 5"
    assert con.sql(q).fetchitem() == 0
    con.execute("INSERT INTO w VALUES (10)")
    assert con.sql(q).fetchitem() == 1  # served from the statement cache


# ── UDFs ─────────────────────────────────────────────────────────────────────


def test_udf_null_inputs_yield_null(con):
    seen = []

    def plus1(x):
        seen.append(len(x))
        return x + 1

    con.create_function("plus1", plus1, ["BIGINT"], "BIGINT")
    assert con.sql("SELECT plus1(x) FROM (VALUES (1), (NULL), (3)) t(x)").fetchall() == [
        (2,),
        (None,),
        (4,),
    ]
    assert seen == [2]  # the NULL row never reached the function


def test_udf_errors_keep_database_usable(con):
    con.execute("CREATE TABLE keep AS SELECT 1 AS a")
    con.create_function("short", lambda x: x[:1], ["BIGINT"], "BIGINT")
    with pytest.raises(ducky.Error, match="shape"):
        con.sql("SELECT short(i) FROM range(10) t(i)").fetchall()
    assert con.sql("SELECT * FROM keep").fetchall() == [(1,)]


class _SlottedSum:
    __slots__ = ("total",)

    def __init__(self):
        self.total = 0

    def update(self, values):
        self.total += int(values.sum())

    def combine(self, other):
        self.total += other.total

    def finalize(self):
        return self.total


class _Count:
    def __init__(self):
        self.n = 0

    def update(self, values):
        self.n += len(values)

    def combine(self, other):
        self.n += other.n

    def finalize(self):
        return self.n


def test_aggregate_skips_null_rows(con):
    con.create_aggregate_function("cnt", _Count, ["BIGINT"], "BIGINT")
    assert con.sql("SELECT cnt(x), count(x) FROM (VALUES (1), (NULL), (3)) t(x)").fetchone() == (
        2,
        2,
    )


def test_aggregate_slotted_class_window_and_ordered(con):
    con.create_aggregate_function("isum", _SlottedSum, ["BIGINT"], "BIGINT")
    assert con.sql("SELECT isum(i) FROM range(10) t(i)").fetchitem() == 45
    assert con.sql("SELECT isum(i) OVER () FROM range(3) t(i)").fetchall() == [(3,)] * 3
    assert con.sql(
        "SELECT i % 2 AS g, isum(i ORDER BY i) FROM range(10) t(i) GROUP BY g ORDER BY g"
    ).fetchall() == [(0, 20), (1, 25)]
    moving = con.sql(
        "SELECT isum(i) OVER (ORDER BY i ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) "
        "FROM range(100) t(i)"
    ).fetchall()
    assert moving[-1] == (sum(range(79, 100)),)
    con.execute("CREATE TABLE big AS SELECT i FROM range(3000000) t(i)")
    assert con.sql("SELECT isum(i), sum(i) FROM big").fetchone() == (4499998500000,) * 2
    assert con.sql("SELECT isum(DISTINCT i % 7) FROM big").fetchitem() == 21


def test_table_udf_timestamp_ns_overflow_raises(con):
    con.create_table_function(
        "far", lambda: iter([(datetime.datetime(2300, 1, 1),)]), [], {"t": "TIMESTAMP_NS"}
    )
    with pytest.raises(ducky.Error, match="TIMESTAMP_NS"):
        con.sql("SELECT * FROM far()").fetchall()


# ── Appender ─────────────────────────────────────────────────────────────────


def test_appender_skips_generated_columns(con):
    np = pytest.importorskip("numpy")
    con.execute("CREATE TABLE g(a INTEGER, b INTEGER GENERATED ALWAYS AS (a * 10), c INTEGER)")
    with con.appender("g") as app:
        assert app.columns == ["a", "c"]
        app.append_row(1, 7)
        app.append_columns({"a": np.array([2], dtype=np.int32), "c": np.array([8], dtype=np.int32)})
    assert con.sql("SELECT * FROM g ORDER BY a").fetchall() == [(1, 10, 7), (2, 20, 8)]


def test_append_row_failure_keeps_buffered_rows(con):
    con.execute("CREATE TABLE q(a INTEGER, b VARCHAR)")
    with con.appender("q") as app:
        app.append_row(1, "x")
        with pytest.raises(ducky.Error, match="expected str, got int"):
            app.append_row(2, 3)
        with pytest.raises(ducky.Error, match="out of range"):
            app.append_row(2**40, "y")
        app.append_row(4, "z")
    assert con.sql("SELECT * FROM q ORDER BY a").fetchall() == [(1, "x"), (4, "z")]


def test_append_arrow_large_batch_with_cast(con):
    pa = pytest.importorskip("pyarrow")
    con.execute("CREATE TABLE ai(a INTEGER)")
    with con.appender("ai") as app:
        app.append_arrow(pa.table({"a": pa.array(range(100_000), pa.int64())}))
    assert con.sql("SELECT count(*), sum(a) FROM ai").fetchone() == (100_000, 4999950000)


# ── Arrow sources ────────────────────────────────────────────────────────────


def test_one_shot_reader_scans_once(con):
    pa = pytest.importorskip("pyarrow")
    schema = pa.schema([("a", pa.int64())])
    batches = [pa.record_batch([pa.array([1, 2])], schema=schema)]
    batches.append(pa.record_batch([pa.array([3])], schema=schema))
    con.register_arrow("r", pa.RecordBatchReader.from_batches(schema, batches))
    assert con.sql("SELECT * FROM r").fetchall() == [(1,), (2,), (3,)]
    with pytest.raises(ducky.Error, match="one-shot"):
        con.sql("SELECT * FROM r").fetchall()


def test_lazy_arrow_scan_stops_at_limit(con):
    pa = pytest.importorskip("pyarrow")
    pulled = []

    def batches():
        for i in range(100):
            pulled.append(i)
            yield pa.record_batch([pa.array(range(i * 10_000, (i + 1) * 10_000))], names=["x"])

    schema = pa.schema([("x", pa.int64())])
    con.register_arrow("g", pa.RecordBatchReader.from_batches(schema, batches()))
    assert con.sql("SELECT x FROM g LIMIT 1").fetchall() == [(0,)]
    assert len(pulled) == 1


def test_empty_arrow_table(con):
    pa = pytest.importorskip("pyarrow")
    con.register_arrow("e", pa.table({"a": pa.array([], pa.int64())}))
    assert con.sql("SELECT count(*) FROM e").fetchitem() == 0
    assert con.sql("SELECT * FROM e").types == ["BIGINT"]


def test_arrow_source_names_are_case_insensitive(con):
    pa = pytest.importorskip("pyarrow")
    con.register_arrow("MyTable", pa.table({"a": [1]}))
    assert con.sql("SELECT * FROM mytable").fetchall() == [(1,)]
    assert con.sql("SELECT * FROM MYTABLE").fetchall() == [(1,)]


def test_dictionary_and_nested_arrow_columns(con):
    pa = pytest.importorskip("pyarrow")
    n = 10_000
    table = pa.table(
        {
            "i": pa.array(range(n)),
            "d": pa.array([str(k % 5) for k in range(n)]).dictionary_encode(),
            "l": pa.array([[k, k + 1] for k in range(n)]),
            "s": pa.array([{"a": k} for k in range(n)]),
        }
    )
    con.register_arrow("t", table)
    assert con.sql("SELECT count(*), count(d) FROM t").fetchone() == (n, n)
    assert con.sql("SELECT d, l, s FROM t WHERE i = 4097").fetchone() == (
        "2",
        [4097, 4098],
        {"a": 4097},
    )
    assert [r[0] for r in con.sql("SELECT i FROM t").fetchall()] == list(range(n))


def test_windows_of_one_large_arrow_batch(con):
    # Every vector-size window of a single batch reads the right rows, through
    # operators that consume the scan's vectors as is and ones that copy them.
    pa = pytest.importorskip("pyarrow")
    import datetime
    import decimal

    n = 10_000

    def value(k, v):
        return None if k % 7 == 3 else v

    table = pa.table(
        {
            "i": pa.array(range(n)),
            "f": pa.array([value(k, k / 4) for k in range(n)]),
            "b": pa.array([value(k, k % 3 == 0) for k in range(n)]),
            "s": pa.array([value(k, "x" * (k % 20)) for k in range(n)]),
            "d": pa.array([value(k, str(k % 5)) for k in range(n)]).dictionary_encode(),
            "l": pa.array([value(k, [k, None]) for k in range(n)]),
            "st": pa.array([value(k, {"a": k}) for k in range(n)]),
            "dec": pa.array(
                [value(k, decimal.Decimal(k) / 100) for k in range(n)], pa.decimal128(9, 2)
            ),
            "ts": pa.array(
                [
                    value(k, datetime.datetime(2024, 1, 1) + datetime.timedelta(seconds=k))
                    for k in range(n)
                ]
            ),
        }
    )
    for source in (table, table.slice(3, 9000)):
        expected = [tuple(row.values()) for row in source.to_pylist()]
        con.register_arrow("src", source)
        assert con.sql("SELECT * FROM src ORDER BY i").fetchall() == expected
        con.execute("CREATE OR REPLACE TABLE copy AS SELECT * FROM src")
        assert con.sql("SELECT * FROM copy ORDER BY i").fetchall() == expected
        assert con.sql("SELECT count(f), count(s), sum(i) FROM src WHERE i % 5 = 1").fetchone() == (
            sum(1 for r in expected if r[0] % 5 == 1 and r[1] is not None),
            sum(1 for r in expected if r[0] % 5 == 1 and r[3] is not None),
            sum(r[0] for r in expected if r[0] % 5 == 1),
        )


def test_reregistered_arrow_source_is_rebound(con):
    pa = pytest.importorskip("pyarrow")
    con.register_arrow("p", pa.table({"a": [1, 2]}))
    stmt = con.prepare("SELECT * FROM p")
    con.register_arrow("p", pa.table({"a": ["x", "y"]}))
    assert stmt.execute().fetchall() == [("x",), ("y",)]


def test_arrow_reader_holds_the_connection_until_drained(con):
    pytest.importorskip("pyarrow")
    reader = con.arrow_reader("SELECT * FROM range(100000)")
    with pytest.raises(ducky.Error, match="open result"):
        con.sql("SELECT 42").fetchall()
    assert reader.read_all().num_rows == 100000
    assert con.sql("SELECT 42").fetchitem() == 42


def test_arrow_reader_keeps_current_result(con):
    pytest.importorskip("pyarrow")
    con.execute("SELECT 7")
    reader = con.arrow_reader("SELECT * FROM range(10)")
    assert reader.read_all().num_rows == 10
    assert con.fetchall() == [(7,)]


# ── Python helpers ───────────────────────────────────────────────────────────


def test_progress_bar_and_profile_keep_current_result(con):
    import io

    with ducky.progress_bar(con, use_tqdm=False, out=io.StringIO()):
        con.execute("SELECT 42")
    assert con.fetchall() == [(42,)]
    with ducky.profile(con):
        con.execute("SELECT 43")
    assert con.fetchall() == [(43,)]


async def test_concurrent_asql_on_one_connection(con):
    import asyncio

    results = await asyncio.gather(
        con.asql("SELECT sum(i) FROM range(20000000) t(i)"), con.asql("SELECT 2")
    )
    assert [r.fetchitem() for r in results] == [199999990000000, 2]


def test_nested_and_enum_rows_decode(con):
    con.execute("CREATE TYPE mood AS ENUM ('sad', 'ok', 'happy')")
    rows = con.sql(
        """
        SELECT
            ['happy', 'sad']::mood[] AS moods,
            {'a': [1, NULL], 'b': {'c': 'x'}} AS s,
            MAP {'k': [1.5]} AS m,
            union_value(num := 2)::UNION(num INT, str VARCHAR) AS u,
            [[1, 2], [3, 4]]::INT[2][2] AS arr,
            ('12:00:00+02:00'::TIMETZ + to_minutes(i)) AS t
        FROM range(3) t(i)
        """
    ).fetchall()
    assert len(rows) == 3
    moods, s, m, u, arr, t1 = rows[0]
    t2 = rows[1][-1]
    assert moods == ["happy", "sad"]
    assert s == {"a": [1, None], "b": {"c": "x"}}
    assert m == {"k": [1.5]}
    assert u == 2
    assert arr == [[1, 2], [3, 4]]
    assert t1.utcoffset() == t2.utcoffset() == datetime.timedelta(hours=2)
    assert t1 != t2
    assert t1.tzinfo is t2.tzinfo  # one timezone object per offset


def test_arrow_fast_path_validity_across_misaligned_chunks(con):
    pytest.importorskip("pyarrow")
    con.execute("SET threads=1")
    parts = [2050, 7, 3000]
    query = " UNION ALL ".join(
        f"SELECT CASE WHEN i % 3 = 0 THEN NULL ELSE i END AS v FROM range({n}) t(i)" for n in parts
    )
    got = con.sql(query).arrow().column("v").to_pylist()
    expected = [None if i % 3 == 0 else i for n in parts for i in range(n)]
    assert got == expected
