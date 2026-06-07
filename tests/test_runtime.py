"""Tests for the runtime-control surface: config, config_options,
Connection.interrupt(), Connection.progress()."""

from __future__ import annotations

import os
import signal
import sys
import threading
import time

import pytest

import ducky


def test_config_round_trip():
    con = ducky.connect(memory_limit="1GB", threads=2)
    row = con.execute(
        "SELECT current_setting('memory_limit'), current_setting('threads')"
    ).fetchone()
    assert row is not None
    mem, threads = row
    # DuckDB normalises memory_limit; just make sure it didn't ignore us.
    assert "MiB" in mem or "GB" in mem.upper()
    assert threads == 2


def test_config_native_types_coerced():
    # int and bool are coerced to DuckDB's string form; the round-trip confirms
    # threads=3 (not "3") and the flag arrives as a real boolean (not "1"/"0").
    con = ducky.connect(threads=3, preserve_insertion_order=False)
    row = con.execute(
        "SELECT current_setting('threads'), current_setting('preserve_insertion_order')"
    ).fetchone()
    assert row is not None
    threads, preserve = row
    assert threads == 3
    assert preserve is False


def test_config_bool_true_coerced():
    con = ducky.connect(preserve_insertion_order=True)
    assert con.execute("SELECT current_setting('preserve_insertion_order')").fetchitem() is True


def test_config_str_value_passes_through():
    # str values (sizes, enums) are forwarded untouched.
    con = ducky.connect(memory_limit="512MB", access_mode="READ_WRITE")
    assert con.execute("SELECT 1").fetchone() == (1,)


def test_config_bad_value_still_raises():
    # Coercion forwards an unparseable string unchanged; DuckDB rejects it,
    # surfacing as ducky.Error rather than a Python coercion error.
    with pytest.raises(ducky.Error):
        ducky.connect(**{"threads": "not_a_number"})  # ty: ignore[invalid-argument-type]


def test_config_typed_dict_unpacked():
    cfg: ducky.DuckDBConfig = {"access_mode": "READ_WRITE"}
    con = ducky.connect(**cfg)
    assert con.execute("SELECT 1").fetchone() == (1,)


def test_config_bad_key_raises():
    # DuckDB validates unknown options at open time — surface the underlying
    # error verbatim. The kwargs splat deliberately bypasses DuckDBConfig's
    # static key check, which is exactly the runtime scenario we want.
    with pytest.raises(ducky.Error, match="not_a_real_setting"):
        ducky.connect(**{"not_a_real_setting": "x"})  # ty: ignore[invalid-argument-type]


def test_config_options_returns_known_keys():
    options = ducky.config_options()
    assert isinstance(options, list)
    assert options, "expected non-empty list of config options"
    names = {name for name, _desc in options}
    for known in ("memory_limit", "threads", "access_mode"):
        assert known in names, f"expected {known!r} in config_options()"


def test_progress_on_idle_connection():
    con = ducky.connect()
    # No query in flight: DuckDB reports percentage = -1 sentinel.
    pct, rows, total = con.progress()
    assert pct == pytest.approx(-1.0) or pct == 0.0
    assert rows == 0
    assert total == 0


def test_interrupt_cancels_long_query():
    # Force DuckDB to think this query is worth parallelizing — without
    # threads >= 2 it tends to short-circuit before the interrupt arrives.
    con = ducky.connect(threads=2)

    err: list[BaseException] = []

    def runner():
        try:
            con.execute(
                "SELECT count(*) FROM range(10_000_000_000) t(i) WHERE i % 7 = 0"
            ).fetchall()
        except Exception as e:
            err.append(e)

    t = threading.Thread(target=runner)
    t.start()
    time.sleep(0.05)  # give the query a chance to actually start
    con.interrupt()
    t.join(timeout=10)
    assert not t.is_alive(), "query did not respond to interrupt within 10s"
    assert err, "expected the interrupted query to raise"
    assert isinstance(err[0], ducky.Error)


@pytest.mark.skipif(sys.platform == "win32", reason="SIGINT-on-self is unreliable on Windows")
def test_keyboard_interrupt_lands_mid_query():
    # The pending-execute loop checks PyErr_CheckSignals() between ticks, so
    # SIGINT delivered to the process raises KeyboardInterrupt out of execute()
    # rather than parking the main thread until the query finishes. Without the
    # pending substrate the old gil_scoped_release path could only surface
    # signals *after* duckdb_execute_prepared returned.
    con = ducky.connect(threads=2)

    prior = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:

        def kicker():
            time.sleep(0.05)
            os.kill(os.getpid(), signal.SIGINT)

        t = threading.Thread(target=kicker)
        t.start()
        try:
            with pytest.raises(KeyboardInterrupt):
                con.execute(
                    "SELECT count(*) FROM range(10_000_000_000) t(i) WHERE i % 7 = 0"
                ).fetchall()
        finally:
            t.join()
    finally:
        signal.signal(signal.SIGINT, prior)

    # The connection survives an interrupted query — a fresh query still runs.
    assert con.execute("SELECT 1").fetchone() == (1,)


def test_streaming_result_iterates_all_chunks():
    # Smoke test: streaming=True returns a Result whose chunks are pulled
    # lazily via duckdb_fetch_chunk. We can't directly observe peak memory in
    # a unit test, but we can verify the streaming result iterates correctly
    # and yields every row.
    con = ducky.connect()
    n = 50_000  # several chunks at STANDARD_VECTOR_SIZE=2048
    res = con.execute("SELECT * FROM range(?) t(i)", [n], streaming=True).current_result
    chunks = 0
    seen = 0
    while True:
        chunk = res.fetch_chunk()
        if chunk is None:
            break
        chunks += 1
        seen += len(chunk.column("i"))
    assert seen == n
    assert chunks > 1, "expected the streaming result to span multiple chunks"
    # Exhausted streaming result returns None on further fetches.
    assert res.fetch_chunk() is None


def test_streaming_via_prepared_statement():
    con = ducky.connect()
    with con.prepare("SELECT * FROM range(?) t(i)") as stmt:
        res = stmt.execute([20_000], streaming=True)
        rows = res.fetchall()
    assert len(rows) == 20_000
    assert rows[0] == (0,)
    assert rows[-1] == (19_999,)


def test_progress_during_long_query():
    con = ducky.connect(threads=2)

    samples: list[tuple[float, int, int]] = []
    done = threading.Event()

    def runner():
        try:
            con.execute("SELECT count(*) FROM range(50_000_000) t(i) WHERE i % 13 = 0").fetchall()
        finally:
            done.set()

    t = threading.Thread(target=runner)
    t.start()
    # Sample a few times while the query runs.
    deadline = time.monotonic() + 2.0
    while not done.is_set() and time.monotonic() < deadline:
        samples.append(con.progress())
        time.sleep(0.01)
    t.join(timeout=10)
    assert not t.is_alive()
    # At least one sample should have been collected; values are well-formed.
    assert samples
    for pct, rows, total in samples:
        assert pct == -1.0 or 0.0 <= pct <= 100.0
        assert rows >= 0
        assert total >= 0


def test_profiling_info_none_when_disabled():
    # Without `SET enable_profiling=...`, DuckDB returns no profiling tree.
    con = ducky.connect()
    con.execute("SELECT 1 + 2").fetchall()
    assert con.get_profiling_info() is None


def test_profiling_info_walks_tree():
    con = ducky.connect()
    con.execute("SET enable_profiling='no_output'")
    con.execute("SELECT i, i * 2 FROM range(1000) t(i) WHERE i % 3 = 0").fetchall()

    root = con.get_profiling_info()
    assert isinstance(root, dict)
    assert set(root.keys()) == {"metrics", "children"}
    assert isinstance(root["metrics"], dict)
    assert isinstance(root["children"], list)
    # The root is the QUERY_ROOT node with summary metrics; the operator tree
    # hangs off the first child.
    assert root["children"], "expected an operator subtree under the root"

    # Walk the tree and confirm we see at least one operator with a timing metric.
    def all_nodes(node):
        yield node
        for c in node["children"]:
            yield from all_nodes(c)

    timing_seen = False
    for node in all_nodes(root):
        for k, v in node["metrics"].items():
            assert isinstance(k, str)
            assert isinstance(v, str)
        if "operator_timing" in node["metrics"] or "timing" in node["metrics"]:
            timing_seen = True
    assert timing_seen, "expected at least one node to expose a timing metric"


def test_profiling_info_detailed_mode_extra_metrics():
    # detailed mode unlocks additional per-operator counters (cardinality,
    # cpu_time, etc.); verify the dict grows accordingly.
    con = ducky.connect()
    con.execute("SET enable_profiling='no_output'")
    con.execute("SET profiling_mode='detailed'")
    con.execute("SELECT count(*) FROM range(10_000) t(i)").fetchall()

    root = con.get_profiling_info()
    assert root is not None

    def metric_keys(node):
        keys = set(node["metrics"].keys())
        for c in node["children"]:
            keys |= metric_keys(c)
        return keys

    keys = metric_keys(root)
    # Detailed mode reliably surfaces these across the operator tree.
    assert "cpu_time" in keys
    assert "intermediate_rows" in keys


def test_profiling_info_after_close_raises():
    con = ducky.connect()
    con.close()
    with pytest.raises(ducky.Error, match="closed"):
        con.get_profiling_info()
