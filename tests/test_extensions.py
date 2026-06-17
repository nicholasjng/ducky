"""Tests for Connection.load_extension() and connect(allow_unsigned_extensions=...).

ducky pins a development DuckDB commit, so we can't pull a real prebuilt
extension from the official repository (the C++-ABI version check won't match).
The default tests therefore exercise the binding's mechanics — the
statically-linked path, error surfacing, SQL-literal escaping, the icu parity
add, and the open-only allow_unsigned_extensions setting. A genuine
not-statically-linked load is covered by test_load_external_unsigned_extension,
gated on DUCKY_EXT_BINARY (a loadable binary built out-of-band against our pin).
"""

from __future__ import annotations

import os

import pytest

import ducky


def test_load_statically_linked_extension_succeeds():
    # json is statically linked and already registered, so LOAD is a no-op that
    # returns success — a clean positive path through load_extension.
    con = ducky.connect()
    con.load_extension("json")
    assert con.execute("SELECT json('[1,2]')").fetchone() == ("[1,2]",)


def test_load_missing_path_raises():
    con = ducky.connect()
    with pytest.raises(ducky.Error, match="failed to load extension"):
        con.load_extension("/nope/does_not_exist.duckdb_extension")


def test_load_after_close_raises():
    con = ducky.connect()
    con.close()
    with pytest.raises(ducky.Error, match="closed"):
        con.load_extension("json")


def test_single_quote_in_path_is_escaped():
    # A path with a single quote must not break the `LOAD '<path>'` literal:
    # the failure should be a file/IO error, never a SQL parser error.
    con = ducky.connect()
    with pytest.raises(ducky.Error) as exc:
        con.load_extension("/tmp/we'ird.duckdb_extension")
    assert "Parser" not in str(exc.value)
    assert "syntax" not in str(exc.value).lower()


def test_allow_unsigned_extensions_reaches_open_config():
    # allow_unsigned_extensions is an open-only DuckDB setting exposed as a typed
    # DuckDBConfig key; it must reach duckdb_set_config before open, observable
    # via current_setting().
    con = ducky.connect(allow_unsigned_extensions=True)
    assert con.execute("SELECT current_setting('allow_unsigned_extensions')").fetchone() == (True,)


def test_default_connect_is_signed_only():
    con = ducky.connect()
    assert con.execute("SELECT current_setting('allow_unsigned_extensions')").fetchone() == (False,)


def test_icu_statically_linked():
    # icu is in ducky's static set (parity with duckdb-python), so ICU-only
    # features like the TimeZone setting work out of the box — no extension load.
    con = ducky.connect()
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
    reason="set DUCKY_EXT_BINARY to a loadable .duckdb_extension built against ducky's pinned DuckDB",
)
def test_load_external_unsigned_extension():
    # Integration test for a genuinely not-statically-linked extension. DuckDB's
    # embedded BUILD_EXTENSIONS model force-links every extension it builds, so
    # ducky's own build can't emit a loadable-only binary — CI builds one
    # out-of-band (against our pinned commit) and points DUCKY_EXT_BINARY at it.
    binary = os.environ["DUCKY_EXT_BINARY"]

    # Self-built binaries are unsigned: the default (signed-only) connection must
    # reject this one before it ever dlopen's.
    signed_only = ducky.connect()
    with pytest.raises(ducky.Error):
        signed_only.load_extension(binary)

    # With allow_unsigned_extensions, the same binary loads (its platform / ABI /
    # version were stamped from our exact build, so the metadata check passes).
    con = ducky.connect(allow_unsigned_extensions=True)
    con.load_extension(binary)
