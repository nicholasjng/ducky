"""Tests for the `ducky.profile` context manager and
`ducky.format_profiling_info` pretty-printer."""

from __future__ import annotations

import sys

import pytest

import ducky


def test_profile_snapshots_info_on_exit():
    con = ducky.connect()
    with ducky.profile(con) as p:
        con.execute("SELECT count(*) FROM range(1000)").fetchall()
    # Snapshot was taken before settings reset, so .info survives the with-block.
    assert p.info is not None
    assert "metrics" in p.info and "children" in p.info


def test_profile_restores_settings_after_block():
    con = ducky.connect()
    # Capture defaults.
    assert con.execute("SELECT current_setting('enable_profiling')").fetchitem() is None
    assert con.execute("SELECT current_setting('profiling_mode')").fetchitem() is None

    with ducky.profile(con):
        assert con.execute("SELECT current_setting('enable_profiling')").fetchitem() == "no_output"

    # Both settings are reset to defaults.
    assert con.execute("SELECT current_setting('enable_profiling')").fetchitem() is None
    assert con.execute("SELECT current_setting('profiling_mode')").fetchitem() is None


def test_profile_preserves_prior_user_setting():
    # If the user already had enable_profiling set, exiting profile() restores
    # it (rather than blindly unsetting it).
    con = ducky.connect()
    con.execute("SET enable_profiling='query_tree'")
    with ducky.profile(con):
        # Inside the block, the helper has overridden it.
        assert con.execute("SELECT current_setting('enable_profiling')").fetchitem() == "no_output"
    # Prior value is back.
    assert con.execute("SELECT current_setting('enable_profiling')").fetchitem() == "query_tree"
    # Tidy up so this test doesn't pollute global state for parallel runs.
    con.execute("RESET enable_profiling")


def test_profile_detailed_mode_unlocks_extra_metrics():
    con = ducky.connect()
    with ducky.profile(con, mode="detailed") as p:
        con.execute("SELECT count(*) FROM range(10_000)").fetchall()

    def all_keys(node):
        keys = set(node["metrics"].keys())
        for c in node["children"]:
            keys |= all_keys(c)
        return keys

    keys = all_keys(p.info)
    # Detailed mode reliably surfaces these (the standard set lacks cpu_time
    # on operators below the summary).
    assert "cpu_time" in keys
    assert "intermediate_rows" in keys


def test_profile_restores_on_exception():
    con = ducky.connect()
    with pytest.raises(RuntimeError, match="boom"), ducky.profile(con):
        raise RuntimeError("boom")
    assert con.execute("SELECT current_setting('enable_profiling')").fetchitem() is None


def test_profile_result_does_not_pin_connection():
    # A long-lived ProfileResult must not keep a transient connection open —
    # _con is dropped on context exit. Connection is a nanobind type without
    # weakref support, so verify via the refcount delta instead.
    con = ducky.connect()
    before = sys.getrefcount(con)
    with ducky.profile(con) as p:
        con.execute("SELECT 1").fetchall()
    assert p._con is None
    after = sys.getrefcount(con)
    # The only added refs across the block come from the `with` machinery's
    # locals, which are gone by the time we check. ProfileResult must not
    # contribute one.
    assert after == before, f"profile leaked a reference (before={before}, after={after})"


def test_profile_result_refresh_after_exit_raises():
    con = ducky.connect()
    with ducky.profile(con) as p:
        con.execute("SELECT 1").fetchall()
    with pytest.raises(RuntimeError, match="inside the"):
        p.refresh()


def test_profile_refresh_inside_block_overwrites_info():
    con = ducky.connect()
    with ducky.profile(con) as p:
        con.execute("SELECT 1").fetchall()
        first = p.refresh()
        assert first is not None
        con.execute("SELECT 2").fetchall()
        second = p.refresh()
        assert second is not None
        # Different queries produce distinct trees (SQL text differs in summary).
        first_summary = next((c for c in first["children"] if "sql" in c["metrics"]), None)
        second_summary = next((c for c in second["children"] if "sql" in c["metrics"]), None)
        assert first_summary and second_summary
        assert first_summary["metrics"]["sql"] != second_summary["metrics"]["sql"]


def test_format_profiling_info_none_message():
    out = ducky.format_profiling_info(None)
    assert "profiling not enabled" in out
    assert "ducky.profile(con)" in out


def test_format_profiling_info_renders_operator_tree():
    con = ducky.connect()
    with ducky.profile(con) as p:
        con.execute("SELECT count(*) FROM range(1000) t(i) WHERE i % 3 = 0").fetchall()

    rendered = str(p)
    # Header with SQL + total / cpu summary appears.
    assert rendered.startswith("SQL: ")
    assert "total=" in rendered
    # Operator names show up, indented per depth.
    assert "RESULT_COLLECTOR" in rendered
    # range() expands to a TABLE_SCAN-like node.
    assert "SCAN" in rendered or "TABLE_FUNCTION" in rendered or "GENERATE" in rendered


def test_format_profiling_info_str_equals_function():
    con = ducky.connect()
    with ducky.profile(con) as p:
        con.execute("SELECT 42").fetchall()
    assert str(p) == ducky.format_profiling_info(p.info)
