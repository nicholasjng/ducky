"""Shared NYC Yellow Taxi data preparation and training code."""

from __future__ import annotations

import shutil
import sys
import urllib.request
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

URL_TEMPLATE = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet"
DEFAULT_MONTHS = ["2023-01"]

CACHE_DIR = Path(__file__).parent / ".cache"


def _cached_path(month: str) -> Path:
    """Return a local path for `month`'s Parquet file, downloading it once if absent."""
    dest = CACHE_DIR / f"yellow_tripdata_{month}.parquet"
    if dest.exists():
        return dest
    CACHE_DIR.mkdir(exist_ok=True)
    url = URL_TEMPLATE.format(month=month)
    print(f"caching {url} -> {dest} (one-time download)...", file=sys.stderr)
    tmp = dest.with_suffix(".parquet.tmp")
    try:
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f)
        tmp.rename(dest)
    finally:
        tmp.unlink(missing_ok=True)  # no-op once renamed; cleans up on failure
    return dest


# Feature name to SQL expression.
FEATURE_EXPRS: dict[str, str] = {
    "distance": "trip_distance",
    "passengers": "passenger_count",
    "hour_sin": "sin(2 * pi() * pickup_hour / 24.0)",
    "hour_cos": "cos(2 * pi() * pickup_hour / 24.0)",
    "dow_sin": "sin(2 * pi() * pickup_dow / 7.0)",
    "dow_cos": "cos(2 * pi() * pickup_dow / 7.0)",
    "pu_loc": "PULocationID",
    "do_loc": "DOLocationID",
    "payment_type": "payment_type",
}
STANDARDIZE = {"distance", "pu_loc", "do_loc"}
TARGET_EXPR = "duration_min"
N_FEATURES = len(FEATURE_EXPRS)

HIDDEN = (32, 16)
BATCH_SIZE = 4096
EPOCHS = 5
LR = 0.001


def prep_sql(table: str, months: list[str]) -> list[str]:
    """Build a filtered table from locally cached Yellow Taxi data."""
    paths = ", ".join(f"'{_cached_path(m)}'" for m in months)
    return [
        f"""
        CREATE OR REPLACE TABLE {table} AS
        SELECT
            trip_distance,
            passenger_count,
            PULocationID,
            DOLocationID,
            payment_type,
            hour(tpep_pickup_datetime) AS pickup_hour,
            dayofweek(tpep_pickup_datetime) AS pickup_dow,
            date_diff('minute', tpep_pickup_datetime, tpep_dropoff_datetime) AS duration_min
        FROM read_parquet([{paths}], union_by_name = true)
        WHERE trip_distance BETWEEN 0.1 AND 100
          AND passenger_count BETWEEN 1 AND 6
          AND fare_amount > 0
          AND date_diff('minute', tpep_pickup_datetime, tpep_dropoff_datetime) BETWEEN 1 AND 180
        """
    ]


class MLPParams(NamedTuple):
    w1: jax.Array
    b1: jax.Array
    w2: jax.Array
    b2: jax.Array
    w3: jax.Array
    b3: jax.Array


def init_params(key: jax.Array, n_features: int) -> MLPParams:
    h1, h2 = HIDDEN
    k1, k2, k3 = jax.random.split(key, 3)

    def layer(k: jax.Array, fan_in: int, fan_out: int) -> jax.Array:
        return jax.random.normal(k, (fan_in, fan_out)) * (2.0 / fan_in) ** 0.5

    return MLPParams(
        w1=layer(k1, n_features, h1),
        b1=jnp.zeros((h1,)),
        w2=layer(k2, h1, h2),
        b2=jnp.zeros((h2,)),
        w3=layer(k3, h2, 1),
        b3=jnp.zeros((1,)),
    )


def forward(params: MLPParams, X: jax.Array) -> jax.Array:
    h1 = jax.nn.relu(X @ params.w1 + params.b1)
    h2 = jax.nn.relu(h1 @ params.w2 + params.b2)
    return (h2 @ params.w3 + params.b3)[:, 0]


def mse(params: MLPParams, X: jax.Array, y: jax.Array) -> jax.Array:
    return jnp.mean((forward(params, X) - y) ** 2)


@jax.jit
def _sgd_step(params: MLPParams, X: jax.Array, y: jax.Array, lr: float) -> MLPParams:
    grads = jax.grad(mse)(params, X, y)
    return jax.tree.map(lambda p, g: p - lr * g, params, grads)


def _batches(X: jax.Array, y: jax.Array, batch_size: int, seed: int):
    idx = np.arange(X.shape[0])
    np.random.default_rng(seed).shuffle(idx)
    for start in range(0, len(idx), batch_size):
        sl = idx[start : start + batch_size]
        yield X[sl], y[sl]


def train(
    params: MLPParams,
    Xtr: jax.Array,
    ytr: jax.Array,
    Xval: jax.Array,
    yval: jax.Array,
    *,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
) -> tuple[MLPParams, float, float]:
    """Minibatch SGD; returns ``(params, final val_mse, final val_mae_minutes)``."""
    for epoch in range(epochs):
        for Xb, yb in _batches(Xtr, ytr, batch_size, seed=epoch):
            params = _sgd_step(params, Xb, yb, lr)
    val_mse = float(mse(params, Xval, yval))
    val_mae = float(jnp.mean(jnp.abs(forward(params, Xval) - yval)))
    return params, val_mse, val_mae


def report(
    engine: str, n: int, load_s: float, train_s: float, val_mse: float, val_mae: float
) -> None:
    """Print a machine-parseable result line; read by run_taxi_vs_duckdb.py."""
    print(
        f"RESULT engine={engine} n={n} load_s={load_s:.4f} train_s={train_s:.4f} "
        f"val_mse={val_mse:.4f} val_mae={val_mae:.4f}",
        flush=True,
    )
