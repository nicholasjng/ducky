"""End-to-end demo: train a Titanic survival classifier in JAX.

Run with:

    uv run --group dev python examples/titanic_jax.py

What this shows:
  1. Reading a remote CSV directly into DuckDB (httpfs).
  2. The high-level ``ducky.dataset`` API — column-to-feature mapping, dtype,
     standardisation (with train-fold-only stats), and split — declared once.
  3. Zero-copy hand-off into JAX via ``Fold.to_jax()``.
  4. A tiny logistic-regression training loop in plain JAX.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import ducky

URL = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"


def load() -> ducky.Dataset:
    return ducky.dataset(
        URL,
        columns={
            "pclass": ducky.feature("Pclass"),
            "sex_male": ducky.feature("Sex = 'male'"),
            "age": ducky.feature("Age", standardize=True),
            "sibsp": ducky.feature('"Siblings/Spouses Aboard"'),
            "parch": ducky.feature('"Parents/Children Aboard"'),
            "fare": ducky.feature("Fare", standardize=True),
        },
        target=ducky.target("Survived"),
        drop_nulls=["Age"],
        split=ducky.split(0.8, seed=0),
    )


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
    ds = load()
    Xtr, ytr = ds.train.to_jax()
    assert ds.val is not None
    Xval, yval = ds.val.to_jax()

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
    print("\nlearned weights:")
    for name, wi in zip(ds.feature_names, w.tolist(), strict=True):
        print(f"  {name:>10}: {wi:+.3f}")
    print(f"  {'bias':>10}: {float(b):+.3f}")


if __name__ == "__main__":
    main()
