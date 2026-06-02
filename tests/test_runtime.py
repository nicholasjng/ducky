"""Tests for the runtime-control surface: config, config_options,
Connection.interrupt(), Connection.progress()."""

from __future__ import annotations

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
    row = con.execute("SELECT current_setting('preserve_insertion_order')").fetchone()
    assert row is not None
    assert row[0] is True


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
