"""End-to-end demo: train a Titanic survival classifier in JAX, with all the
feature engineering and the train/val split running inside DuckDB.

Run with:

    uv run --group dev python examples/titanic_jax.py

What this shows:
  1. Reading a remote CSV directly into the query engine (httpfs).
  2. Doing the feature engineering in SQL (encoding, null filtering, splitting,
     standardisation against the train-set statistics).
  3. Zero-copy hand-off into JAX via Result.to_jax().
  4. A tiny logistic-regression training loop in plain JAX.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import ducky

URL = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"

# Stable 80/20 split: hash the row position into one of 10 buckets and reserve
# buckets {0, 1} for validation. Deterministic across runs; no leakage between
# train and val statistics because we only standardise against the train set.
FEATURE_SQL = """
WITH raw AS (
    SELECT
        Pclass,
        Sex,
        Age,
        "Siblings/Spouses Aboard" AS sibsp,
        "Parents/Children Aboard" AS parch,
        Fare,
        Survived,
        (hash(row_number() OVER ()) % 10) AS bucket
    FROM read_csv_auto($url)
    WHERE Age IS NOT NULL
),
stats AS (
    SELECT
        avg(Age)        AS age_mean,
        stddev_pop(Age) AS age_std,
        avg(Fare)        AS fare_mean,
        stddev_pop(Fare) AS fare_std
    FROM raw WHERE bucket >= 2
)
SELECT
    CAST(Pclass AS DOUBLE)                                  AS pclass,
    CAST(CASE WHEN Sex = 'male' THEN 1.0 ELSE 0.0 END
         AS DOUBLE)                                          AS sex_male,
    CAST((Age  - stats.age_mean)  / stats.age_std  AS DOUBLE) AS age_z,
    CAST(sibsp AS DOUBLE)                                    AS sibsp,
    CAST(parch AS DOUBLE)                                    AS parch,
    CAST((Fare - stats.fare_mean) / stats.fare_std AS DOUBLE) AS fare_z,
    CAST(Survived AS DOUBLE)                                  AS y,
    CAST(bucket AS BIGINT)                                    AS bucket
FROM raw, stats
"""


def load() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    con = ducky.connect()
    cols = con.execute(FEATURE_SQL, [URL]).to_jax()
    feature_keys = ["pclass", "sex_male", "age_z", "sibsp", "parch", "fare_z"]
    X = jnp.stack([cols[k] for k in feature_keys], axis=1)
    y = cols["y"]
    is_val = cols["bucket"] < 2  # ~20% of rows
    return X[~is_val], y[~is_val], X[is_val], y[is_val]


def model(params: tuple[jax.Array, jax.Array], X: jax.Array) -> jax.Array:
    w, b = params
    return jax.nn.sigmoid(X @ w + b)


def bce(params: tuple[jax.Array, jax.Array], X: jax.Array, y: jax.Array) -> jax.Array:
    p = model(params, X)
    eps = 1e-7
    return -jnp.mean(y * jnp.log(p + eps) + (1.0 - y) * jnp.log(1.0 - p + eps))


@jax.jit
def step(params, X, y, lr):
    g = jax.grad(bce)(params, X, y)
    return jax.tree.map(lambda p, gi: p - lr * gi, params, g)


def main() -> None:
    Xtr, ytr, Xval, yval = load()
    print(f"train: {Xtr.shape}, val: {Xval.shape}")

    key = jax.random.PRNGKey(0)
    params = (jax.random.normal(key, (Xtr.shape[1],)) * 0.01, jnp.zeros(()))

    lr = 0.1
    for epoch in range(401):
        params = step(params, Xtr, ytr, lr)
        if epoch % 50 == 0:
            tr_loss = bce(params, Xtr, ytr)
            val_loss = bce(params, Xval, yval)
            val_acc = jnp.mean((model(params, Xval) > 0.5) == (yval > 0.5))
            print(
                f"epoch {epoch:>3} | train loss {tr_loss:.4f} | "
                f"val loss {val_loss:.4f} | val acc {val_acc:.3f}"
            )

    w, b = params
    feature_names = ["pclass", "sex_male", "age_z", "sibsp", "parch", "fare_z"]
    print("\nlearned weights:")
    for name, wi in zip(feature_names, w.tolist(), strict=True):
        print(f"  {name:>10}: {wi:+.3f}")
    print(f"  {'bias':>10}: {float(b):+.3f}")


if __name__ == "__main__":
    main()
