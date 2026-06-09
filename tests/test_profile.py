"""Tests for the `ducky.profile` context manager and
`ducky.format_profiling_info` pretty-printer."""

from __future__ import annotations

import os
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


# ── always-on sink ───────────────────────────────────────────────────────────


def test_set_profile_sink_fires_per_query():
    con = ducky.connect()
    captured: list[tuple[str, dict]] = []
    con.set_profile_sink(lambda sql, info: captured.append((sql, info)))
    con.execute("SELECT 1").fetchall()
    con.execute("SELECT count(*) FROM range(100) t(i)").fetchall()
    assert len(captured) == 2
    sqls = [c[0] for c in captured]
    assert sqls == ["SELECT 1", "SELECT count(*) FROM range(100) t(i)"]
    # Each record carries a real profiling tree.
    for _, info in captured:
        assert "metrics" in info and "children" in info


def test_set_profile_sink_sample_n():
    con = ducky.connect()
    sqls: list[str] = []
    con.set_profile_sink(lambda sql, _info: sqls.append(sql), sample=3)
    for k in range(7):
        con.execute(f"SELECT {k}").fetchall()
    # Fires on 1st, 4th, 7th.
    assert sqls == ["SELECT 0", "SELECT 3", "SELECT 6"]


def test_set_profile_sink_none_disables():
    con = ducky.connect()
    sqls: list[str] = []
    con.set_profile_sink(lambda sql, _info: sqls.append(sql))
    con.execute("SELECT 1").fetchall()
    assert sqls == ["SELECT 1"]
    con.set_profile_sink(None)
    con.execute("SELECT 2").fetchall()
    assert sqls == ["SELECT 1"]


def test_set_profile_sink_detailed_mode():
    con = ducky.connect()
    captured: list[dict] = []
    con.set_profile_sink(lambda _sql, info: captured.append(info), mode="detailed")
    con.execute("SELECT count(*) FROM range(1000) t(i)").fetchall()
    assert captured

    def all_keys(node):
        keys = set(node["metrics"].keys())
        for c in node["children"]:
            keys |= all_keys(c)
        return keys

    keys = all_keys(captured[0])
    assert "cpu_time" in keys
    assert "intermediate_rows" in keys


def test_set_profile_sink_rejects_non_callable():
    con = ducky.connect()
    with pytest.raises(ducky.Error, match="callable or None"):
        con.set_profile_sink(42)  # ty: ignore[invalid-argument-type]


def test_set_profile_sink_rejects_bad_mode():
    con = ducky.connect()
    with pytest.raises(ducky.Error, match="standard.*detailed"):
        con.set_profile_sink(lambda _s, _i: None, mode="bogus")


def test_set_profile_sink_skips_streaming():
    # Streaming results profile the *previous* materialized query (the chunks
    # for the streaming one haven't been pulled yet), so we deliberately skip
    # emission. Verify by interleaving and checking the sink doesn't see the
    # streaming query.
    con = ducky.connect()
    sqls: list[str] = []
    con.set_profile_sink(lambda sql, _info: sqls.append(sql))
    con.execute("SELECT 1").fetchall()
    con.execute("SELECT 2 FROM range(10) t(i)", streaming=True).fetchall()
    con.execute("SELECT 3").fetchall()
    assert "SELECT 1" in sqls and "SELECT 3" in sqls
    assert "SELECT 2 FROM range(10) t(i)" not in sqls


def test_set_profile_sink_swallows_sink_errors(capsys):
    # A buggy sink must not break the query; the exception is forwarded to
    # PyErr_WriteUnraisable (which lands on stderr).
    con = ducky.connect()

    def bad(_sql, _info):
        raise RuntimeError("sink boom")

    con.set_profile_sink(bad)
    # Query still succeeds.
    assert con.execute("SELECT 1").fetchall() == [(1,)]


def test_jsonl_profile_sink_writes_lines(tmp_path):
    import json

    path = tmp_path / "p.jsonl"
    con = ducky.connect()
    con.set_profile_sink(ducky.jsonl_profile_sink(path))
    con.execute("SELECT count(*) FROM range(50) t(i)").fetchall()
    con.execute("SELECT 7").fetchall()
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert set(rec) == {"ts", "sql", "info"}
    assert rec["sql"] == "SELECT count(*) FROM range(50) t(i)"
    assert "metrics" in rec["info"]


def test_jsonl_profile_sink_with_info_false(tmp_path):
    import json

    path = tmp_path / "p.jsonl"
    con = ducky.connect()
    con.set_profile_sink(ducky.jsonl_profile_sink(path, with_info=False))
    con.execute("SELECT 1").fetchall()
    rec = json.loads(path.read_text().splitlines()[0])
    assert "info" not in rec
    assert "summary" in rec
    # Summary block carries top-line numbers as strings.
    assert "total_time" in rec["summary"]


def test_profile_config_from_env_off_without_dir(monkeypatch):
    monkeypatch.delenv("DUCKY_PROFILE_DIR", raising=False)
    assert ducky.ProfileConfig.from_env() is None


def test_profile_config_from_env_reads_all_knobs(monkeypatch, tmp_path):
    monkeypatch.setenv("DUCKY_PROFILE_DIR", str(tmp_path))
    monkeypatch.setenv("DUCKY_PROFILE_SAMPLE", "4")
    monkeypatch.setenv("DUCKY_PROFILE_MODE", "detailed")
    from ducky import _profile

    _profile._env_sink_cache.clear()

    cfg = ducky.ProfileConfig.from_env()
    assert cfg is not None
    assert callable(cfg.sink)
    assert cfg.sample == 4
    assert cfg.mode == "detailed"


def test_profile_config_constructor_defaults():
    # The class is general-purpose, not tied to env — defaults match
    # set_profile_sink's defaults.
    sink = lambda _s, _i: None  # noqa: E731
    cfg = ducky.ProfileConfig(sink)
    assert cfg.sink is sink
    assert cfg.sample == 1
    assert cfg.mode == "standard"


def test_connect_installs_env_sink(tmp_path, monkeypatch):
    import json

    monkeypatch.setenv("DUCKY_PROFILE_DIR", str(tmp_path))
    monkeypatch.setenv("DUCKY_PROFILE_SAMPLE", "2")
    # Force a fresh cache entry so a prior test's sink doesn't leak in.
    from ducky import _profile

    _profile._env_sink_cache.clear()

    con = ducky.connect()
    for k in range(5):
        con.execute(f"SELECT {k}").fetchall()

    path = tmp_path / f"profile-{os.getpid()}.jsonl"
    assert path.exists()
    lines = path.read_text().splitlines()
    # sample=2 over 5 queries → records on 1st, 3rd, 5th.
    assert len(lines) == 3
    sqls = [json.loads(line)["sql"] for line in lines]
    assert sqls == ["SELECT 0", "SELECT 2", "SELECT 4"]


def test_connect_env_sink_compact(tmp_path, monkeypatch):
    import json

    monkeypatch.setenv("DUCKY_PROFILE_DIR", str(tmp_path))
    monkeypatch.setenv("DUCKY_PROFILE_NO_INFO", "1")
    from ducky import _profile

    _profile._env_sink_cache.clear()

    con = ducky.connect()
    con.execute("SELECT 1").fetchall()
    line = (tmp_path / f"profile-{os.getpid()}.jsonl").read_text().splitlines()[0]
    rec = json.loads(line)
    assert "info" not in rec
    assert "summary" in rec
