"""Tests for the built-in progress-bar helper, ``ducky.progress_bar``."""

from __future__ import annotations

import io

import pytest

import ducky
from ducky import _progress


def test_make_bar_prefers_tqdm_when_available():
    pytest.importorskip("tqdm")
    bar = _progress._make_bar("t", io.StringIO(), use_tqdm=True)
    assert isinstance(bar, _progress._TqdmBar)
    bar.close()


def test_make_bar_falls_back_when_tqdm_disabled():
    bar = _progress._make_bar("t", io.StringIO(), use_tqdm=False)
    assert isinstance(bar, _progress._AsciiBar)
    bar.close()


def test_progress_bar_runs_query_and_finalizes():
    con = ducky.connect(threads=2)
    out = io.StringIO()
    with ducky.progress_bar(con, desc="t", out=out, use_tqdm=False, interval=0.01) as c:
        assert c is con
        count: int = con.execute(
            "SELECT count(*) FROM range(20_000_000) t(i) WHERE i % 13 = 0"
        ).fetchitem()
    assert count > 0
    rendered = out.getvalue()
    # The fallback bar always finalizes to 100% with a trailing newline.
    assert "100.0%" in rendered
    assert rendered.endswith("\n")


def test_progress_bar_restores_settings():
    con = ducky.connect()

    def setting(name: str) -> bool:
        return bool(con.execute(f"SELECT current_setting('{name}')").fetchitem())

    assert setting("enable_progress_bar") is False
    with ducky.progress_bar(con, out=io.StringIO(), use_tqdm=False):
        con.execute("SELECT 1").fetchall()
    # Settings touched by the helper are restored to their prior values.
    assert setting("enable_progress_bar") is False
    assert setting("enable_progress_bar_print") is True


def test_progress_bar_does_not_print_duckdb_native_bar(capfd):
    # enable_progress_bar_print is forced off, so DuckDB never writes its own
    # bar to stdout — only our handle (here a StringIO) sees output.
    con = ducky.connect(threads=2)
    out = io.StringIO()
    with ducky.progress_bar(con, out=out, use_tqdm=False, interval=0.01):
        con.execute("SELECT count(*) FROM range(20_000_000) t(i) WHERE i % 13 = 0").fetchall()
    captured = capfd.readouterr()
    assert captured.out == ""


def test_progress_bar_propagates_query_errors():
    con = ducky.connect()
    with pytest.raises(ducky.Error), ducky.progress_bar(con, out=io.StringIO(), use_tqdm=False):
        con.execute("SELECT * FROM no_such_table").fetchall()
    # Settings are still restored even though the body raised.
    assert bool(con.execute("SELECT current_setting('enable_progress_bar')").fetchitem()) is False
