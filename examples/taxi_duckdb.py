# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "duckdb>=1.5",
#     "jax>=0.10.1",
#     "numpy>=2.4.6",
# ]
# ///
"""NYC Yellow Taxi loader using duckdb-python and JAX."""

import sys
import time

import duckdb
import jax
import jax.numpy as jnp
import numpy as np
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

TABLE = "trips"
N_BUCKETS = 1000  # matches ducky._dataset's hash-bucket resolution
TRAIN_FRACTION = 0.8


def main() -> None:
    months = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_MONTHS

    con = duckdb.connect()
    for stmt in prep_sql(TABLE, months):
        con.execute(stmt)
    n = con.execute(f"SELECT count(*) FROM {TABLE}").fetchone()[0]

    t0 = time.perf_counter()
    cols = list(FEATURE_EXPRS)
    select_cols = ", ".join(f"{expr} AS {name}" for name, expr in FEATURE_EXPRS.items())
    # Same hash-bucket split as ducky._dataset._compile_sql (seed=0), so both
    # scripts partition rows the same way.
    bucket_expr = f"(hash(row_number() OVER () + 0) % {N_BUCKETS})"
    raw = con.execute(
        f"SELECT {select_cols}, {TARGET_EXPR} AS y, {bucket_expr} AS _bucket FROM {TABLE}"
    ).fetchnumpy()

    bucket = raw["_bucket"]
    train_mask = bucket < int(TRAIN_FRACTION * N_BUCKETS)
    val_mask = ~train_mask

    X = np.stack([raw[c] for c in cols], axis=1).astype(np.float32)
    y = raw["y"].astype(np.float32)

    standardize_mask = np.array([c in STANDARDIZE for c in cols])
    mean = X[train_mask][:, standardize_mask].mean(axis=0)
    std = X[train_mask][:, standardize_mask].std(axis=0)
    std[std == 0] = 1.0  # constant column -> untouched, matching ducky's convention
    X[:, standardize_mask] = (X[:, standardize_mask] - mean) / std

    Xtr = jnp.asarray(X[train_mask])
    ytr = jnp.asarray(y[train_mask])
    Xval = jnp.asarray(X[val_mask])
    yval = jnp.asarray(y[val_mask])
    jax.block_until_ready((Xtr, ytr, Xval, yval))
    load_s = time.perf_counter() - t0

    params = init_params(jax.random.PRNGKey(0), len(FEATURE_EXPRS))

    t0 = time.perf_counter()
    params, val_mse, val_mae = train(params, Xtr, ytr, Xval, yval, epochs=EPOCHS)
    train_s = time.perf_counter() - t0

    report("duckdb-python", n, load_s, train_s, val_mse, val_mae)


if __name__ == "__main__":
    main()
