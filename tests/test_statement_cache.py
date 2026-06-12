"""The per-connection prepared-statement cache used by execute()/sql().

The cache is invisible by design — these tests pin down the behaviors that
must match a fresh prepare on every call: no parameter leakage, catalog
changes picked up, errors identical, and correctness past the LRU capacity.
"""

import pytest

import ducky


def test_repeated_execute_returns_fresh_results():
    con = ducky.connect()
    con.execute("CREATE TABLE t (a INT)")
    con.execute("INSERT INTO t VALUES (1), (2)")
    for expected in (2, 3, 4):
        assert con.execute("SELECT count(*) FROM t").fetchitem() == expected
        con.execute("INSERT INTO t VALUES (99)")


def test_parameters_do_not_leak_between_calls():
    con = ducky.connect()
    assert con.execute("SELECT ?::INT", [1]).fetchone() == (1,)
    # A fresh prepare starts unbound; the cached statement must behave the same.
    with pytest.raises(ducky.Error, match="[Pp]arameter"):
        con.execute("SELECT ?::INT")


def test_rebind_after_schema_change():
    con = ducky.connect()
    con.execute("CREATE TABLE t (a INT)")
    con.execute("INSERT INTO t VALUES (1)")
    assert con.execute("SELECT * FROM t").fetchall() == [(1,)]
    con.execute("DROP TABLE t")
    con.execute("CREATE TABLE t (a VARCHAR, b INT)")
    con.execute("INSERT INTO t VALUES ('x', 2)")
    # Same query text, new plan: both the rows and the schema must reflect
    # the recreated table.
    result = con.sql("SELECT * FROM t")
    assert result.columns == ["a", "b"]
    assert result.fetchall() == [("x", 2)]


def test_set_invalidates_settings_dependent_plans():
    # The optimizer folds current_setting() into the plan, and DuckDB does not
    # rebind prepared statements on SET — so SET must flush the cache.
    con = ducky.connect()
    query = "SELECT current_setting('enable_profiling')"
    assert con.execute(query).fetchitem() is None
    con.execute("SET enable_profiling='no_output'")
    assert con.execute(query).fetchitem() == "no_output"
    con.execute("RESET enable_profiling")
    assert con.execute(query).fetchitem() is None


def test_error_after_table_dropped():
    con = ducky.connect()
    con.execute("CREATE TABLE t (a INT)")
    con.execute("SELECT * FROM t")
    con.execute("DROP TABLE t")
    with pytest.raises(ducky.Error, match="t"):
        con.execute("SELECT * FROM t")


def test_failed_execution_does_not_poison_the_cache():
    con = ducky.connect()
    con.execute("CREATE TABLE t (a INT PRIMARY KEY)")
    con.execute("INSERT INTO t VALUES (?)", [1])
    with pytest.raises(ducky.Error, match="[Cc]onstraint"):
        con.execute("INSERT INTO t VALUES (?)", [1])
    con.execute("INSERT INTO t VALUES (?)", [2])
    assert con.execute("SELECT count(*) FROM t").fetchitem() == 2


def test_streaming_and_materialized_share_cached_statement():
    con = ducky.connect()
    con.execute("CREATE TABLE t AS SELECT range AS a FROM range(10000)")
    query = "SELECT * FROM t"
    assert len(con.sql(query).fetchall()) == 10000
    streamed = sum(len(batch["a"]) for batch in con.sql(query, streaming=True).iter_batches())
    assert streamed == 10000
    assert len(con.sql(query).fetchall()) == 10000


def test_correct_past_lru_capacity():
    con = ducky.connect()
    # More distinct queries than the cache holds; eviction must be invisible.
    for i in range(300):
        assert con.execute(f"SELECT {i}").fetchone() == (i,)
    assert con.execute("SELECT 0").fetchone() == (0,)


def test_close_with_populated_cache():
    con = ducky.connect()
    for i in range(10):
        con.execute(f"SELECT {i}")
    con.close()
    with pytest.raises(ducky.Error, match="closed"):
        con.execute("SELECT 1")
