# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "ducky",
#     "jax>=0.10.1",
#     "mlx>=0.31.2; sys_platform == 'darwin'",
# ]
#
# [tool.uv.sources]
# ducky = { path = "../", editable = true }
# ///
"""Train an MLP on millions of rows that DuckDB generates and prepares.

The step up from ``titanic_jax.py``: the table has millions of rows, the
features are engineered in SQL, and the network has three hidden layers. DuckDB
does everything up to the arrays:

- ``CREATE TABLE ... FROM range(n)`` builds the raw event table in parallel;
- ``ducky.dataset`` compiles the feature expressions, the train-fold
  standardization and the split into one query, and copies its result once,
  into one ``(rows, features)`` array per fold.

Run with ``--backend jax`` or ``--backend mlx``, and ``--rows`` to scale.
"""

from __future__ import annotations

import argparse
import itertools
import time
from typing import Any

import ducky

HIDDEN = (256, 128, 64)
BATCH_SIZE = 8192
LEARNING_RATE = 2e-3

# Raw events: sixteen sensor readings, a device, an hour, and a label that
# depends on them non-linearly. `hash(i, k)` is a reproducible random source.
RAW_SQL = """
CREATE TABLE events AS
WITH raw AS (
    SELECT
        i,
        {readings},
        ['ios', 'android', 'web', 'tv', 'watch'][1 + (hash(i, 100) % 5)::INTEGER] AS device,
        (hash(i, 101) % 24)::INTEGER AS hour
    FROM range({rows}) AS t(i)
)
SELECT
    *,
    (
        sin(3 * s0) * s1 + s2 * s3 - 0.5 * s4 ^ 2
        + CASE device WHEN 'ios' THEN 0.4 WHEN 'tv' THEN -0.4 ELSE 0 END
        + 0.3 * cos(2 * pi() * hour / 24)
        + 0.25 * ((hash(i, 102) % 2000) / 1000.0 - 1)
    ) > 0 AS converted
FROM raw
"""


def columns() -> dict[str, ducky.Feature]:
    """The model's inputs, as SQL expressions over the raw table."""
    cols = {f"s{k}": ducky.feature(f"s{k}", standardize=True) for k in range(16)}
    # Interactions and cyclical encodings cost one expression each.
    cols["s0_x_s1"] = ducky.feature("s0 * s1", standardize=True)
    cols["s2_x_s3"] = ducky.feature("s2 * s3", standardize=True)
    cols["hour_sin"] = ducky.feature("sin(2 * pi() * hour / 24)")
    cols["hour_cos"] = ducky.feature("cos(2 * pi() * hour / 24)")
    for device in ("ios", "android", "web", "tv"):  # one-hot, 'watch' is the baseline
        cols[f"is_{device}"] = ducky.feature(f"device = '{device}'")
    return cols


def load(rows: int, backend: str) -> ducky.Dataset[Any]:
    con = ducky.connect()
    readings = ", ".join(f"(hash(i, {k}) % 20000) / 10000.0 - 1 AS s{k}" for k in range(16))
    start = time.perf_counter()
    con.execute(RAW_SQL.format(readings=readings, rows=rows))
    made = time.perf_counter()
    ds = ducky.dataset(
        "events",
        columns=columns(),
        target=ducky.target("converted"),
        split=ducky.split({"train": 0.8, "val": 0.1, "test": 0.1}, seed=0),
        con=con,
        backend=backend,
    )
    loaded = time.perf_counter()
    con.close()
    print(f"generated {rows:,} rows in {made - start:.2f} s; dataset in {loaded - made:.2f} s")
    print(ds)
    return ds


def run_jax(ds: ducky.Dataset[Any], epochs: int) -> None:
    import jax
    import jax.numpy as jnp

    Xtr, ytr = ds["train"].tensors()
    Xval, yval = ds["val"].tensors()
    sizes = (Xtr.shape[1], *HIDDEN, 1)

    def init(key: jax.Array) -> list[tuple[jax.Array, jax.Array]]:
        keys = jax.random.split(key, len(sizes) - 1)
        return [
            (jax.random.normal(k, (m, n)) * jnp.sqrt(2.0 / m), jnp.zeros(n))
            for k, m, n in zip(keys, sizes[:-1], sizes[1:], strict=True)
        ]

    def logits(params: Any, X: jax.Array) -> jax.Array:
        for w, b in params[:-1]:
            X = jax.nn.relu(X @ w + b)
        w, b = params[-1]
        return (X @ w + b)[:, 0]

    def loss(params: Any, X: jax.Array, y: jax.Array) -> jax.Array:
        z = logits(params, X)
        return jnp.mean(jnp.logaddexp(0.0, z) - y * z)

    def adam(state: Any, grads: Any) -> Any:
        params, m, v, t = state
        t = t + 1
        m = jax.tree.map(lambda a, g: 0.9 * a + 0.1 * g, m, grads)
        v = jax.tree.map(lambda a, g: 0.999 * a + 0.001 * g * g, v, grads)
        scale = LEARNING_RATE * jnp.sqrt(1 - 0.999**t) / (1 - 0.9**t)
        params = jax.tree.map(lambda p, a, b: p - scale * a / (jnp.sqrt(b) + 1e-8), params, m, v)
        return params, m, v, t

    n_batches = Xtr.shape[0] // BATCH_SIZE

    # One compiled epoch: shuffle on the device and scan over the batches, so
    # Python is entered once per epoch rather than once per batch.
    @jax.jit
    def epoch(state: Any, key: jax.Array) -> tuple[Any, jax.Array]:
        order = jax.random.permutation(key, Xtr.shape[0])[: n_batches * BATCH_SIZE]

        def step(state: Any, idx: jax.Array) -> tuple[Any, jax.Array]:
            value, grads = jax.value_and_grad(loss)(state[0], Xtr[idx], ytr[idx])
            return adam(state, grads), value

        state, losses = jax.lax.scan(step, state, order.reshape(n_batches, BATCH_SIZE))
        return state, jnp.mean(losses)

    @jax.jit
    def evaluate(params: Any, X: jax.Array, y: jax.Array) -> tuple[jax.Array, jax.Array]:
        return loss(params, X, y), jnp.mean((logits(params, X) > 0) == (y > 0.5))

    key = jax.random.PRNGKey(0)
    params = init(key)
    zeros = jax.tree.map(jnp.zeros_like, params)
    state = (params, zeros, zeros, jnp.zeros((), jnp.int32))
    for e in range(epochs):
        key, sub = jax.random.split(key)
        start = time.perf_counter()
        state, train_loss = epoch(state, sub)
        val_loss, val_acc = evaluate(state[0], Xval, yval)
        print(
            f"epoch {e + 1:>2} | train loss {float(train_loss):.4f} | val loss "
            f"{float(val_loss):.4f} | val acc {float(val_acc):.3f} | "
            f"{time.perf_counter() - start:.2f} s"
        )
    Xte, yte = ds["test"].tensors()
    print(f"test acc {float(evaluate(state[0], Xte, yte)[1]):.3f}")


def run_mlx(ds: ducky.Dataset[Any], epochs: int) -> None:
    import mlx.core as mx  # ty: ignore[unresolved-import]
    import mlx.optimizers as optim  # ty: ignore[unresolved-import]
    from mlx import nn  # ty: ignore[unresolved-import]

    Xtr, ytr = ds["train"].tensors()
    Xval, yval = ds["val"].tensors()

    layers: list[Any] = []
    sizes = (Xtr.shape[1], *HIDDEN)
    for m, n in itertools.pairwise(sizes):
        layers += [nn.Linear(m, n), nn.ReLU()]
    model = nn.Sequential(*layers, nn.Linear(sizes[-1], 1))
    mx.eval(model.parameters())
    optimizer = optim.Adam(learning_rate=LEARNING_RATE)

    def loss(model: Any, X: Any, y: Any) -> Any:
        return nn.losses.binary_cross_entropy(model(X)[:, 0], y, with_logits=True)

    step_fn = nn.value_and_grad(model, loss)
    state = [model.state, optimizer.state]

    def train_step(X: Any, y: Any) -> Any:
        value, grads = step_fn(model, X, y)
        optimizer.update(model, grads)
        return value

    step = mx.compile(train_step, inputs=state, outputs=state)

    def evaluate(X: Any, y: Any) -> tuple[float, float]:
        z = model(X)[:, 0]
        correct = (z > 0) == (y > 0.5)
        return loss(model, X, y).item(), mx.mean(correct).item()  # ty: ignore[invalid-argument-type]

    n_batches = Xtr.shape[0] // BATCH_SIZE
    for e in range(epochs):
        start = time.perf_counter()
        order = mx.random.permutation(Xtr.shape[0])
        total = mx.zeros(())
        for b in range(n_batches):
            idx = order[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            total = total + step(Xtr[idx], ytr[idx])
            mx.eval(state, total)
        val_loss, val_acc = evaluate(Xval, yval)
        print(
            f"epoch {e + 1:>2} | train loss {total.item() / n_batches:.4f} | val loss "
            f"{val_loss:.4f} | val acc {val_acc:.3f} | {time.perf_counter() - start:.2f} s"
        )
    Xte, yte = ds["test"].tensors()
    print(f"test acc {evaluate(Xte, yte)[1]:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=("jax", "mlx"), default="jax")
    parser.add_argument("--rows", type=int, default=2_000_000)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    ds = load(args.rows, args.backend)
    (run_jax if args.backend == "jax" else run_mlx)(ds, args.epochs)


if __name__ == "__main__":
    main()
