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
