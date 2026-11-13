import numpy as np
import pytest

import ducky

LABELS = ["sad", "ok", "happy"]


@pytest.fixture
def moods(con):
    con.execute(f"CREATE TYPE mood AS ENUM ({', '.join(repr(m) for m in LABELS)})")
    con.execute(
        "CREATE TABLE t AS SELECT (['sad', 'ok', 'happy'])[1 + i % 3]::mood m FROM range(5000) t(i)"
    )
    return np.arange(5000) % 3


def test_numpy_outputs_hold_codes(con, moods):
    for out in [con.sql("SELECT m FROM t").to_numpy(), con.sql("SELECT m FROM t").fetchnumpy()]:
        assert out["m"].dtype == np.uint8
        np.testing.assert_array_equal(out["m"], moods)
    res = con.sql("SELECT m FROM t")
    assert res.categories("m") == res.categories(0) == LABELS
    got = np.concatenate([b["m"] for b in res.iter_batches()])
    np.testing.assert_array_equal(got, moods)


def test_chunk_views_hold_codes(con, moods):
    chunk = con.sql("SELECT m FROM t").fetch_chunk()
    assert chunk is not None
    assert chunk.categories("m") == LABELS
    np.testing.assert_array_equal(chunk.column("m"), moods[: len(chunk)])
    np.testing.assert_array_equal(np.from_dlpack(chunk.dlpack("m")), moods[: len(chunk)])


def test_wide_dictionaries_widen_the_codes(con):
    labels = ", ".join(f"'v{i}'" for i in range(300))
    out = con.sql(f"SELECT 'v299'::ENUM({labels}) AS e").to_numpy()
    assert out["e"].dtype == np.uint16
    assert out["e"].tolist() == [299]


def test_null_codes_raise(con, moods):
    with pytest.raises(ducky.Error, match=r"'m' \(ENUM\) contains NULLs"):
        con.sql("SELECT NULL::mood AS m").to_numpy()


def test_categories_rejects_other_types(con):
    with pytest.raises(ducky.Error, match="'i' is INTEGER, not ENUM"):
        con.sql("SELECT 1 AS i").categories("i")
    with pytest.raises(ducky.Error, match="no such column"):
        con.sql("SELECT 1 AS i").categories("j")
