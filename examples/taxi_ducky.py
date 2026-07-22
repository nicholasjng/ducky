# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "ducky",
#     "jax>=0.10.1",
# ]
#
# [tool.uv.sources]
# ducky = { path = "../", editable = true }
# ///
"""Real-world counterpart to large_scale_ducky.py: NYC Yellow Taxi trip
records (https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
instead of data synthesized in SQL.

Run standalone:

    uv run examples/taxi_ducky.py [month ...]

Each argument is a "YYYY-MM" TLC trip-data month (default: 2023-01, ~3M rows
after filtering); multiple months union into one table, so this is the scale knob.

Or run both halves via the paired driver, which times this and taxi_duckdb.py
in separate subprocesses (ducky and the official duckdb package both
statically link DuckDB and can't share a process — see run_taxi_vs_duckdb.py):

    uv run examples/run_taxi_vs_duckdb.py [month ...]

Real feature engineering on top of large_scale_*.py's synthetic benchmark:
cyclical (sin/cos) hour/day-of-week encodings, outlier filtering for
meter-error trips (_taxi_common.prep_sql), and a regression target (trip
duration) instead of a synthetic binary label.

API friction found while writing this: ``ducky.dataset()``'s ``source=`` always
compiles to ``FROM '<source>'`` (see ``ducky._dataset._compile_sql``), so it can't
take an inline ``read_parquet([...])`` call for the multi-month union below —
same workaround as large_scale_ducky.py's ``data`` table: build it as a
``CREATE TABLE`` first, then hand ``ducky.dataset()`` the table name.
"""

import sys
import time

import jax
from _taxi_common import (
    DEFAULT_MONTHS,
    EPOCHS,
    FEATURE_EXPRS,
    STANDARDIZE,
    TARGET_EXPR,
    init_params,
    prep_sql,
    report,
    train,
)

import ducky

TABLE = "trips"


def main() -> None:
    months = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_MONTHS

    con = ducky.connect()
    for stmt in prep_sql(TABLE, months):
        con.execute(stmt)
    n = con.execute(f"SELECT count(*) FROM {TABLE}").fetchitem()

    t0 = time.perf_counter()
    ds = ducky.dataset(
        TABLE,
        fields={
            "X": ducky.matrix(
                {
                    name: ducky.feature(expr, standardize=name in STANDARDIZE)
                    for name, expr in FEATURE_EXPRS.items()
                }
            ),
            "y": ducky.vector(TARGET_EXPR),
        },
        split=ducky.split(0.8, seed=0),
        con=con,
        backend="jax",
    )
    assert ds.train is not None and ds.val is not None
    Xtr, ytr = ds.train.tensors()
    Xval, yval = ds.val.tensors()
    jax.block_until_ready((Xtr, ytr, Xval, yval))
    load_s = time.perf_counter() - t0

    params = init_params(jax.random.PRNGKey(0), len(FEATURE_EXPRS))

    t0 = time.perf_counter()
    params, val_mse, val_mae = train(params, Xtr, ytr, Xval, yval, epochs=EPOCHS)
    train_s = time.perf_counter() - t0

    report("ducky", n, load_s, train_s, val_mse, val_mae)


if __name__ == "__main__":
    main()
