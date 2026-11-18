"""Runtime controls."""

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
    # DuckDB normalises memory_limit, so only check it took effect.
    assert "MiB" in mem or "GB" in mem.upper()
    assert threads == 2


def test_config_native_types_coerced():
    # int and bool are coerced to DuckDB's string form, so threads=3 arrives as
    # a number and the flag as a boolean, not "1" or "0".
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
    # An unparseable string passes through unchanged, so DuckDB's error surfaces
    # rather than a Python coercion error.
    with pytest.raises(ducky.Error):
        ducky.connect(threads="not_a_number")  # ty: ignore[invalid-argument-type]


def test_config_typed_dict_unpacked():
    cfg: ducky.DuckDBConfig = {"access_mode": "READ_WRITE"}
    con = ducky.connect(**cfg)
    assert con.execute("SELECT 1").fetchone() == (1,)


def test_config_bad_key_raises():
    # DuckDB validates options when opening, and its error surfaces verbatim. The
    # splat bypasses DuckDBConfig's static key check on purpose.
    with pytest.raises(ducky.Error, match="not_a_real_setting"):
        ducky.connect(not_a_real_setting="x")


def test_file_database_spills_next_to_its_file(tmp_path):
    path = str(tmp_path / "spill.db")
    con = ducky.connect(path)
    assert con.execute("SELECT current_setting('temp_directory')").fetchone() == (path + ".tmp",)
    con.close()
    con = ducky.connect(path, temp_directory=str(tmp_path / "elsewhere"))
    assert con.execute("SELECT current_setting('temp_directory')").fetchone() == (
        str(tmp_path / "elsewhere"),
    )


def test_opening_an_open_file_again_raises(tmp_path):
    path = str(tmp_path / "open.db")
    con = ducky.connect(path)
    with pytest.raises(ducky.Error, match="conflict"):
        ducky.connect(path)
    con.close()
    ducky.connect(path).close()


def test_config_options_returns_known_keys():
    options = ducky.config_options()
    assert isinstance(options, list)
    assert options, "expected non-empty list of config options"
    names = {name for name, _desc in options}
    for known in ("memory_limit", "threads", "access_mode"):
        assert known in names, f"expected {known!r} in config_options()"


def test_progress_on_idle_connection(con):
    # No query in flight: DuckDB reports percentage = -1 sentinel.
    pct, rows, total = con.progress()
    assert pct == pytest.approx(-1.0)
    assert rows == 0
    assert total == 0


def test_interrupt_cancels_long_query():
    # With fewer than 2 threads the query tends to finish before the interrupt
    # arrives.
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


def test_interrupt_cancels_born_as_arrow_query():
    # Connection.arrow runs the same loop, so it must be just as interruptible.
    con = ducky.connect(threads=2)

    err: list[BaseException] = []

    def runner():
        try:
            # A long scan with a one-row result, so the time is all interruptible scan.
            con.arrow("SELECT count(*) FROM range(10_000_000_000) t(i) WHERE i % 7 = 0")
        except Exception as e:
            err.append(e)

    t = threading.Thread(target=runner)
    t.start()
    # Longer than execute()'s test, since the Arrow path takes extra Python hops
    # before the executor starts, and an earlier interrupt would be a no-op.
    time.sleep(0.2)
    con.interrupt()
    t.join(timeout=10)
    assert not t.is_alive(), "born-as-Arrow query did not respond to interrupt within 10s"
    assert err
    assert isinstance(err[0], ducky.Error)
    # The connection survives.
    assert con.arrow("SELECT 1 AS x").column("x").to_pylist() == [1]


@pytest.mark.skipif(sys.platform == "win32", reason="SIGINT-on-self is unreliable on Windows")
def test_keyboard_interrupt_lands_mid_query():
    # Signal checks must interrupt the query before it completes.
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

    # The connection survives an interrupted query.
    assert con.execute("SELECT 1").fetchone() == (1,)


def test_streaming_result_iterates_all_chunks(con):
    # Peak memory isn't observable here, so this only checks a streaming result
    # yields every row.
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


def test_streaming_via_prepared_statement(con):
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
    # At least one well-formed sample.
    assert samples
    for pct, rows, total in samples:
        assert pct == -1.0 or 0.0 <= pct <= 100.0
        assert rows >= 0
        assert total >= 0


def test_profiling_info_none_when_disabled(con):
    # Without `SET enable_profiling=...`, DuckDB returns no profiling tree.
    con.execute("SELECT 1 + 2").fetchall()
    assert con.get_profiling_info() is None


def test_profiling_info_walks_tree(con):
    con.execute("SET enable_profiling='no_output'")
    con.execute("SELECT i, i * 2 FROM range(1000) t(i) WHERE i % 3 = 0").fetchall()

    root = con.get_profiling_info()
    assert isinstance(root, dict)
    assert set(root.keys()) == {"metrics", "children"}
    assert isinstance(root["metrics"], dict)
    assert isinstance(root["children"], list)
    # The root holds the summary metrics, and the operator tree hangs off the
    # first child.
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


def test_profiling_info_detailed_mode_extra_metrics(con):
    # detailed mode adds per-operator counters such as cardinality and cpu_time.
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


def test_profiling_info_after_close_raises(con):
    con.close()
    with pytest.raises(ducky.Error, match="closed"):
        con.get_profiling_info()
