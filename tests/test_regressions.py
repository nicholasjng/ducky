"""Regression tests for bugs found in the 2026-07 codebase review.

Each test pins a specific fixed defect; see the test docstrings for the
original failure mode.
"""

import datetime
import gc
import subprocess
import sys
import weakref

import pytest

import ducky

# ---------------------------------------------------------------------------
# Failed UDF registration used to double-free the context (SIGABRT): DuckDB's
# function object owns extra_info via its delete callback even when
# registration fails, and ducky freed it a second time. Run in a subprocess so
# a regression aborts the child, not the test session.
# ---------------------------------------------------------------------------

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


def test_failed_udf_registration_does_not_crash():
    proc = subprocess.run(
        [sys.executable, "-c", _FAILED_REGISTRATION_SCRIPT],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "OK" in proc.stdout


# ---------------------------------------------------------------------------
# Temporal values outside Python's datetime range used to leak a NULL object
# out of the decoder ("SystemError: ... returned a result with an exception
# set"). They must surface the constructor's ValueError instead.
# ---------------------------------------------------------------------------


def test_out_of_range_date_raises_cleanly():
    con = ducky.connect()
    with pytest.raises(ValueError, match="year"):
        con.sql("SELECT (DATE '2000-01-01' + INTERVAL 9000 YEARS)::DATE").fetchone()
    # The connection stays usable.
    assert con.sql("SELECT DATE '2000-01-01'").fetchitem() == datetime.date(2000, 1, 1)


def test_out_of_range_timestamp_raises_cleanly():
    con = ducky.connect()
    with pytest.raises(ValueError, match="year"):
        con.sql("SELECT TIMESTAMP '2000-01-01' + INTERVAL 9000 YEARS").fetchone()


# ---------------------------------------------------------------------------
# INTERVAL decoding used to compute months*30 and pass raw microseconds
# through `int` arguments, silently overflowing for large values.
# ---------------------------------------------------------------------------


def test_interval_large_seconds_decodes_exactly():
    con = ducky.connect()
    value = con.sql("SELECT INTERVAL '100000000' SECONDS").fetchitem()
    assert value == datetime.timedelta(seconds=100_000_000)


def test_interval_months_beyond_timedelta_range_raises():
    con = ducky.connect()
    with pytest.raises(ducky.Error, match="overflows datetime.timedelta"):
        con.sql("SELECT INTERVAL '200000000' MONTHS").fetchone()


# ---------------------------------------------------------------------------
# A raising profile sink used to call PyErr_WriteUnraisable after the error
# indicator was already cleared, so the exception vanished without reaching
# sys.unraisablehook.
# ---------------------------------------------------------------------------


def test_profile_sink_exception_reaches_unraisablehook():
    con = ducky.connect()

    def bad_sink(sql, info):
        raise ValueError("sink boom")

    seen = []
    previous_hook = sys.unraisablehook
    sys.unraisablehook = seen.append
    try:
        con.set_profile_sink(bad_sink)
        rows = con.execute("SELECT 1").fetchall()
    finally:
        sys.unraisablehook = previous_hook
        con.set_profile_sink(None)

    assert rows == [(1,)]  # the sink error must not break the query
    assert len(seen) == 1
    assert seen[0].exc_type is ValueError
    assert "sink boom" in str(seen[0].exc_value)


# ---------------------------------------------------------------------------
# Dropping an Appender whose final flush fails (e.g. a constraint violation)
# used to throw out of the destructor and terminate the process. The error is
# still observable via an explicit close().
# ---------------------------------------------------------------------------


def test_appender_destructor_swallows_flush_error():
    con = ducky.connect()
    con.execute("CREATE TABLE checked(x INTEGER CHECK (x > 0))")
    app = con.appender("checked")
    app.append_row(-1)
    del app  # destructor must not raise / abort
    gc.collect()
    assert con.sql("SELECT count(*) FROM checked").fetchitem() == 0


def test_appender_explicit_close_surfaces_flush_error():
    con = ducky.connect()
    con.execute("CREATE TABLE checked(x INTEGER CHECK (x > 0))")
    app = con.appender("checked")
    app.append_row(-1)
    with pytest.raises(ducky.Error):
        app.close()


# ---------------------------------------------------------------------------
# Table-UDF BLOB output went through the strlen-based string assignment,
# truncating at the first embedded NUL byte (and rejecting bytes values).
# ---------------------------------------------------------------------------


def test_table_udf_blob_with_embedded_nul():
    con = ducky.connect()
    payload = b"a\x00b"

    def factory():
        yield (payload,)

    con.create_table_function("blob_rows", factory, [], {"b": "BLOB"})
    assert con.sql("SELECT b FROM blob_rows()").fetchitem() == payload


# ---------------------------------------------------------------------------
# A profile sink / registered Arrow source usually closes over its connection.
# Connection lacked tp_traverse/tp_clear, so those reference cycles were
# invisible to the garbage collector and leaked.
# ---------------------------------------------------------------------------


def test_profile_sink_cycle_is_collectable():
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


# ---------------------------------------------------------------------------
# close() used to keep the profile sink and registered Arrow sources pinned
# until the Connection wrapper itself was collected.
# ---------------------------------------------------------------------------


def test_close_releases_profile_sink():
    con = ducky.connect()

    def sink(_sql, _info):
        pass

    ref = weakref.ref(sink)
    con.set_profile_sink(sink)
    del sink
    assert ref() is not None
    con.close()
    assert ref() is None


def test_close_releases_registered_arrow_sources():
    pa = pytest.importorskip("pyarrow")

    class Source:
        def __init__(self, table):
            self.table = table

        def __arrow_c_stream__(self, requested_schema=None):
            return self.table.__arrow_c_stream__(requested_schema)

    con = ducky.connect()
    src = Source(pa.table({"x": [1]}))
    ref = weakref.ref(src)
    con.register_arrow("t", src)
    del src
    assert ref() is not None
    con.close()
    assert ref() is None


# ---------------------------------------------------------------------------
# A type-string parse failure partway through Arrow/table UDF registration
# used to leak the logical types (and Arrow options) parsed before it. The
# contexts now own those resources; these tests exercise the throwing path.
# ---------------------------------------------------------------------------


def test_arrow_udf_invalid_parameter_type_raises():
    con = ducky.connect()
    with pytest.raises(ducky.Error, match="invalid DuckDB type"):
        con.create_arrow_function("bad", lambda a, b: a, ["INTEGER", "NOT_A_TYPE"], "INTEGER")
    assert con.sql("SELECT 1").fetchitem() == 1


def test_table_udf_invalid_column_type_raises():
    con = ducky.connect()
    with pytest.raises(ducky.Error, match="invalid DuckDB type"):
        con.create_table_function("bad", lambda: iter(()), [], {"a": "INTEGER", "b": "NOT_A_TYPE"})
    assert con.sql("SELECT 1").fetchitem() == 1


# ---------------------------------------------------------------------------
# append_columns now reuses one data chunk across vector-size slices
# (duckdb_data_chunk_reset must restore all-valid validity between slices).
# ---------------------------------------------------------------------------


def test_appender_columns_multiple_slices_with_masks():
    np = pytest.importorskip("numpy")

    con = ducky.connect()
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
