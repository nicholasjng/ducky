# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "ducky",
#     "jax>=0.10.1",
#     "mlx>=0.31.2; sys_platform == 'darwin'",
# ]
#
# [tool.uv.sources]
# ducky = { path = "../", editable = true }
# ///
"""End-to-end demo: training a Titanic survival classifier in JAX or MLX.

Run with:

    uv run examples/titanic_jax.py                   # JAX (default; any platform)
    uv run examples/titanic_jax.py --backend mlx      # MLX (Apple Silicon only)

What this shows:
  1. Reading a remote CSV directly into DuckDB (httpfs).
  2. The high-level ``ducky.dataset`` API — column-to-feature mapping, dtype,
     standardisation (with train-fold-only stats), and split — declared once
     and identical for both backends.
  3. ``backend="jax"`` / ``backend="mlx"`` — folds are streamed straight into
     the array library (zero host copy via DLPack), so ``Fold.tensors()``
     hands back native arrays directly.
  4. A tiny logistic-regression training loop, one flavor per backend (JAX's
     ``jax.grad``/``jax.jit`` and MLX's ``mx.value_and_grad``/``mx.eval``
     differ enough that forcing one shared loop would obscure more than it'd
     save — everything *except* the loop, including ``load()``, is shared).
"""

import argparse
from typing import Any

import ducky

URL = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"

COLUMNS = {
    "pclass": ducky.feature("Pclass"),
    "sex_male": ducky.feature("Sex = 'male'"),
    "age": ducky.feature("Age", standardize=True),
    "sibsp": ducky.feature('"Siblings/Spouses Aboard"'),
    "parch": ducky.feature('"Parents/Children Aboard"'),
    "fare": ducky.feature("Fare", standardize=True),
}


def load(backend: str) -> ducky.Dataset[Any]:
    return ducky.dataset(
        URL,
        columns=COLUMNS,
        target=ducky.target("Survived"),
        drop_nulls=["Age"],
        split=ducky.split(0.8, seed=0),
        backend=backend,
    )


def run_jax() -> None:
    import jax
    import jax.numpy as jnp

    Params = tuple[jax.Array, jax.Array]

    def model(params: Params, X: jax.Array) -> jax.Array:
        w, b = params
        return jax.nn.sigmoid(X @ w + b)

    def bce(params: Params, X: jax.Array, y: jax.Array) -> jax.Array:
        p = model(params, X)
        eps = 1e-7
        return -jnp.mean(y * jnp.log(p + eps) + (1.0 - y) * jnp.log(1.0 - p + eps))

    @jax.jit
    def step(params: Params, X: jax.Array, y: jax.Array, lr: float) -> Params:
        g = jax.grad(bce)(params, X, y)
        return jax.tree.map(lambda p, gi: p - lr * gi, params, g)

    ds = load("jax")
    Xtr, ytr = ds["train"].tensors()
    Xval, yval = ds["val"].tensors()

    key = jax.random.PRNGKey(0)
    params: Params = (jax.random.normal(key, (Xtr.shape[1],)) * 0.01, jnp.zeros(()))

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


def run_mlx() -> None:
    import mlx.core as mx  # ty: ignore[unresolved-import]

    Params = tuple[mx.array, mx.array]

    def model(params: Params, X: mx.array) -> mx.array:
        w, b = params
        return mx.sigmoid(X @ w + b)

    def bce(params: Params, X: mx.array, y: mx.array) -> mx.array:
        p = model(params, X)
        eps = 1e-7
        return -mx.mean(y * mx.log(p + eps) + (1.0 - y) * mx.log(1.0 - p + eps))

    ds = load("mlx")
    Xtr, ytr = ds["train"].tensors()
    Xval, yval = ds["val"].tensors()

    key = mx.random.key(0)
    params: Params = (mx.random.normal((Xtr.shape[1],), key=key) * 0.01, mx.zeros(()))

    loss_and_grad = mx.value_and_grad(bce)

    lr = 0.1
    for epoch in range(401):
        _, grads = loss_and_grad(params, Xtr, ytr)
        params = tuple(p - lr * g for p, g in zip(params, grads, strict=True))
        mx.eval(params)
        if epoch % 50 == 0:
            tr_loss = bce(params, Xtr, ytr)
            val_loss = bce(params, Xval, yval)
            # `array == array` is typed `array | bool`; it's always an array here.
            correct = (model(params, Xval) > 0.5) == (yval > 0.5)
            assert isinstance(correct, mx.array)
            val_acc = mx.mean(correct)
            print(
                f"epoch {epoch:>3} | train loss {tr_loss.item():.4f} | "
                f"val loss {val_loss.item():.4f} | val acc {val_acc.item():.3f}"
            )

    w, b = params
    weights = w.tolist()  # 1-D array -> list[float]
    assert isinstance(weights, list)
    print("\nlearned weights:")
    for name, wi in zip(ds.feature_names, weights, strict=True):
        print(f"  {name:>10}: {wi:+.3f}")
    print(f"  {'bias':>10}: {b.item():+.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("jax", "mlx"), default="jax")
    args = parser.parse_args()
    (run_jax if args.backend == "jax" else run_mlx)()


if __name__ == "__main__":
    main()
