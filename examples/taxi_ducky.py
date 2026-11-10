# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "ducky",
#     "jax>=0.10.1",
# ]
#
# [tool.uv.sources]
# ducky = { path = "../", editable = true }
# ///
"""NYC Yellow Taxi loader using ducky.dataset and JAX."""

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
    assert ds.train is not None
    assert ds.val is not None
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
