import numpy as np
import pytest

import ducky

# ── helpers ──────────────────────────────────────────────────────────────────


class SumAgg:
    def __init__(self):
        self.total = 0.0

    def update(self, values: np.ndarray):
        self.total += float(values.sum())

    def combine(self, other: "SumAgg"):
        self.total += other.total

    def finalize(self) -> float:
        return self.total


class RMSEAgg:
    def __init__(self):
        self.sse = 0.0
        self.n = 0

    def update(self, pred: np.ndarray, actual: np.ndarray):
        diff = pred.astype(float) - actual.astype(float)
        self.sse += float((diff * diff).sum())
        self.n += len(pred)

    def combine(self, other: "RMSEAgg"):
        self.sse += other.sse
        self.n += other.n

    def finalize(self) -> float:
        if self.n == 0:
            return 0.0
        return (self.sse / self.n) ** 0.5


class NoCombineSum:
    def __init__(self):
        self.total = 0.0

    def update(self, values: np.ndarray):
        self.total += float(values.sum())

    def finalize(self) -> float:
        return self.total


class NullFinalize:
    def __init__(self):
        pass

    def update(self, values: np.ndarray):
        pass

    def finalize(self):
        return None


class ErrorOnUpdate:
    def __init__(self):
        pass

    def update(self, values: np.ndarray):
        raise ValueError("intentional update error")

    def finalize(self) -> float:
        return 0.0


# ── tests ─────────────────────────────────────────────────────────────────────


def test_sum_no_group_by():
    con = ducky.connect()
    con.create_aggregate_function("my_sum", SumAgg, parameters=["DOUBLE"], return_type="DOUBLE")
    result = con.sql("SELECT my_sum(v) FROM (VALUES (1.0), (2.0), (3.0)) t(v)").fetchone()
    assert result is not None
    assert abs(result[0] - 6.0) < 1e-9


def test_sum_group_by():
    con = ducky.connect()
    con.create_aggregate_function("grp_sum", SumAgg, parameters=["DOUBLE"], return_type="DOUBLE")
    rows = con.sql(
        "SELECT g, grp_sum(v) FROM (VALUES (1, 10.0), (1, 20.0), (2, 5.0), (2, 5.0)) t(g, v) "
        "GROUP BY g ORDER BY g"
    ).fetchall()
    assert len(rows) == 2
    assert abs(rows[0][1] - 30.0) < 1e-9
    assert abs(rows[1][1] - 10.0) < 1e-9


def test_rmse():
    con = ducky.connect()
    con.create_aggregate_function(
        "rmse", RMSEAgg, parameters=["DOUBLE", "DOUBLE"], return_type="DOUBLE"
    )
    # pred=[1,2,3], actual=[1,2,4] -> errors=[0,0,1] -> rmse=sqrt(1/3)
    result = con.sql(
        "SELECT rmse(pred, actual) FROM (VALUES (1.0, 1.0), (2.0, 2.0), (3.0, 4.0)) t(pred, actual)"
    ).fetchone()
    assert result is not None
    expected = (1.0 / 3.0) ** 0.5
    assert abs(result[0] - expected) < 1e-9


def test_combine_optional():
    con = ducky.connect()
    con.create_aggregate_function(
        "no_comb_sum", NoCombineSum, parameters=["DOUBLE"], return_type="DOUBLE"
    )
    result = con.sql("SELECT no_comb_sum(v) FROM (VALUES (4.0), (6.0)) t(v)").fetchone()
    assert result is not None
    assert abs(result[0] - 10.0) < 1e-9


def test_finalize_returns_none_produces_null():
    con = ducky.connect()
    con.create_aggregate_function(
        "null_agg", NullFinalize, parameters=["DOUBLE"], return_type="DOUBLE"
    )
    result = con.sql("SELECT null_agg(v) FROM (VALUES (1.0)) t(v)").fetchone()
    assert result is not None
    assert result[0] is None


def test_user_exception_propagates():
    con = ducky.connect()
    con.create_aggregate_function(
        "err_agg", ErrorOnUpdate, parameters=["DOUBLE"], return_type="DOUBLE"
    )
    with pytest.raises(ducky.Error, match="intentional update error"):
        con.sql("SELECT err_agg(v) FROM (VALUES (1.0)) t(v)").fetchone()


def test_multi_chunk_boundary():
    n = 5000
    con = ducky.connect()
    con.create_aggregate_function("big_sum", SumAgg, parameters=["DOUBLE"], return_type="DOUBLE")
    result = con.sql(f"SELECT big_sum(CAST(i AS DOUBLE)) FROM range({n}) t(i)").fetchone()
    assert result is not None
    expected = n * (n - 1) / 2.0
    assert abs(result[0] - expected) < 1e-6


def test_integer_return_type():
    class IntSum:
        def __init__(self):
            self.total = 0

        def update(self, values: np.ndarray):
            self.total += int(values.sum())

        def finalize(self) -> int:
            return self.total

    con = ducky.connect()
    con.create_aggregate_function("int_sum", IntSum, parameters=["BIGINT"], return_type="BIGINT")
    result = con.sql("SELECT int_sum(i) FROM (VALUES (1), (2), (3)) t(i)").fetchone()
    assert result is not None
    assert result[0] == 6
