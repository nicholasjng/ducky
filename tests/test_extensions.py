"""Extension loading and configuration."""

from __future__ import annotations

import os

import pytest

import ducky


def test_load_statically_linked_extension_succeeds(con):
    # json is statically linked, so LOAD is a successful no-op.
    con.load_extension("json")
    assert con.execute("SELECT json('[1,2]')").fetchone() == ("[1,2]",)


def test_load_missing_path_raises(con):
    with pytest.raises(ducky.Error, match="failed to load extension"):
        con.load_extension("/nope/does_not_exist.duckdb_extension")


def test_load_after_close_raises(con):
    con.close()
    with pytest.raises(ducky.Error, match="closed"):
        con.load_extension("json")


def test_single_quote_in_path_is_escaped(con):
    # A single quote must not break the `LOAD '<path>'` literal, so the failure
    # is an IO error, never a parser error.
    with pytest.raises(ducky.Error) as exc:
        con.load_extension("/tmp/we'ird.duckdb_extension")
    assert "Parser" not in str(exc.value)
    assert "syntax" not in str(exc.value).lower()


def test_allow_unsigned_extensions_reaches_open_config():
    # allow_unsigned_extensions can only be set when opening, so it must reach
    # the config before open.
    con = ducky.connect(allow_unsigned_extensions=True)
    assert con.execute("SELECT current_setting('allow_unsigned_extensions')").fetchone() == (True,)


def test_default_connect_is_signed_only(con):
    assert con.execute("SELECT current_setting('allow_unsigned_extensions')").fetchone() == (False,)


def test_icu_statically_linked(con):
    # icu is linked statically, as in duckdb-python, so the TimeZone setting
    # works without loading anything.
    loaded = {
        name
        for (name,) in con.execute(
            "SELECT extension_name FROM duckdb_extensions() WHERE loaded"
        ).fetchall()
    }
    assert "icu" in loaded
    con.execute("SET TimeZone='America/New_York'")
    assert con.execute("SELECT (TIMESTAMPTZ '2026-06-19 12:00:00+00')::VARCHAR").fetchone() == (
        "2026-06-19 08:00:00-04",
    )


@pytest.mark.skipif(
    not os.environ.get("DUCKY_EXT_BINARY"),
    reason=(
        "set DUCKY_EXT_BINARY to a loadable .duckdb_extension built against ducky's pinned DuckDB"
    ),
)
def test_load_external_unsigned_extension(con):
    # DuckDB's build links every extension it builds statically, so a loadable
    # one has to be built separately against the pin and passed in
    # DUCKY_EXT_BINARY.
    binary = os.environ["DUCKY_EXT_BINARY"]

    # Self-built binaries are unsigned: the default (signed-only) connection must
    # reject this one before it ever dlopen's.
    signed_only = con
    with pytest.raises(ducky.Error):
        signed_only.load_extension(binary)

    # With allow_unsigned_extensions it loads, since its metadata was stamped
    # from the same build.
    con = ducky.connect(allow_unsigned_extensions=True)
    con.load_extension(binary)
