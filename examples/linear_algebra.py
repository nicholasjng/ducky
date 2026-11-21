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
"""Least squares and PCA where DuckDB reduces the data and JAX or MLX solves.

Many matrix problems need the rows only through small sums. For ``d`` columns,
the Gram matrix ``XᵀX``, the moments ``Xᵀy`` and the covariance are ``d x d``
numbers, whatever the row count. DuckDB computes them in one parallel scan,
out of core if it must, and the array library solves a ``d x d`` system:

1. ridge regression from the normal equations ``(XᵀX + λI) w = Xᵀy``;
2. PCA from the eigenvectors of the covariance matrix.

Both are checked against the same computation on the full matrix, which
``ducky.dataset`` loads as one array.

Run with ``--backend jax`` or ``--backend mlx``, and ``--rows`` to scale.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

import ducky

D = 12
RIDGE = 1e-3
TRUE_W = [round(0.5 * (k - D / 2) + 0.25, 2) for k in range(D)]


def make_table(con: ducky.Connection, rows: int) -> None:
    """Correlated features (each mixes in its left neighbour) and a noisy linear target."""
    base = ", ".join(f"(hash(i, {k}) % 20000) / 10000.0 - 1 AS u{k}" for k in range(D + 1))
    feats = ", ".join(f"u{k} + 0.5 * u{k + 1} AS x{k}" for k in range(D))
    target = " + ".join(f"{w} * x{k}" for k, w in enumerate(TRUE_W))
    con.execute(
        f"CREATE TABLE points AS WITH base AS (SELECT i, {base} FROM range({rows}) t(i)), "
        f"feats AS (SELECT i, {feats} FROM base) "
        f"SELECT *, {target} + 0.1 * ((hash(i, 99) % 2000) / 1000.0 - 1) AS y FROM feats"
    )


def gram_in_sql(con: ducky.Connection) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return ``XᵀX``, ``Xᵀy``, the column sums and the row count from one scan."""
    pairs = [(a, b) for a in range(D) for b in range(a, D)]
    sums = [f"sum(x{a} * x{b})" for a, b in pairs]
    sums += [f"sum(x{a} * y)" for a in range(D)]
    sums += [f"sum(x{a})" for a in range(D)]
    (row,) = con.execute(f"SELECT {', '.join(sums)}, count(*) FROM points").fetchall()
    gram = np.zeros((D, D))
    for (a, b), value in zip(pairs, row, strict=False):
        gram[a, b] = gram[b, a] = value
    moments = np.array(row[len(pairs) : len(pairs) + D])
    totals = np.array(row[len(pairs) + D : len(pairs) + 2 * D])
    return gram, moments, totals, row[-1]


def solvers(backend: str) -> tuple[Any, Any]:
    """Return ``solve(A, b)`` and ``eigh(A)`` for the backend, taking and returning NumPy."""
    if backend == "jax":
        import jax.numpy as jnp

        def solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return np.asarray(jnp.linalg.solve(jnp.asarray(a), jnp.asarray(b)))

        def eigh(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            values, vectors = jnp.linalg.eigh(jnp.asarray(a))
            return np.asarray(values), np.asarray(vectors)

        return solve, eigh

    import mlx.core as mx  # ty: ignore[unresolved-import]

    def solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        x = mx.linalg.solve(mx.array(a, mx.float32), mx.array(b, mx.float32), stream=mx.cpu)
        return np.asarray(x)

    def eigh(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values, vectors = mx.linalg.eigh(mx.array(a, mx.float32), stream=mx.cpu)
        return np.asarray(values), np.asarray(vectors)

    return solve, eigh


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=("jax", "mlx"), default="jax")
    parser.add_argument("--rows", type=int, default=5_000_000)
    args = parser.parse_args()
    solve, eigh = solvers(args.backend)

    con = ducky.connect()
    make_table(con, args.rows)

    start = time.perf_counter()
    gram, moments, totals, n = gram_in_sql(con)
    reduced = time.perf_counter() - start
    print(f"{n:,} rows reduced to a {D} x {D} system by DuckDB in {reduced:.2f} s")

    # 1. Ridge regression from the normal equations.
    w = solve(gram / n + RIDGE * np.eye(D), moments / n)
    print(f"ridge: max |w - w_true| = {np.abs(w - np.array(TRUE_W)).max():.4f}")

    # 2. PCA from the covariance matrix: cov = XᵀX / n - mean meanᵀ.
    mean = totals / n
    cov = gram / n - np.outer(mean, mean)
    values, vectors = eigh(cov)
    order = np.argsort(values)[::-1]
    explained = values[order] / values.sum()
    print("pca: variance explained by the first 4 components:", np.round(explained[:4], 3))

    # The same from the full matrix, loaded as one (rows, D) float32 array.
    start = time.perf_counter()
    full = ducky.dataset(
        "points",
        columns={f"x{k}": ducky.feature(f"x{k}") for k in range(D)},
        target=ducky.target("y"),
        con=con,
    )["train"]
    X, y = full["X"].astype(np.float64), full["y"].astype(np.float64)
    loaded = time.perf_counter() - start
    w_full = np.linalg.solve(X.T @ X / n + RIDGE * np.eye(D), X.T @ y / n)
    values_full = np.linalg.eigvalsh(np.cov(X, rowvar=False, bias=True))[::-1]
    print(
        f"full matrix ({full['X'].nbytes / 2**20:.0f} MB) loaded in {loaded:.2f} s: "
        f"ridge differs by {np.abs(w - w_full).max():.1e}, "
        f"eigenvalues by {np.abs(values[order] - values_full).max():.1e}"
    )

    # Projecting onto the components is SQL again, as two feature expressions:
    # nothing but the (rows, 2) result leaves DuckDB.
    top = vectors[:, order[:2]]
    components = {
        f"pc{c}": ducky.feature(
            " + ".join(f"{float(top[k, c])!r} * (x{k} - {float(mean[k])!r})" for k in range(D))
        )
        for c in range(2)
    }
    scores = ducky.dataset("points", fields={"Z": ducky.matrix(components)}, con=con)["train"]["Z"]
    scores = scores.astype(np.float64)  # float32 sums drift over millions of rows
    print(
        f"projected {scores.shape[0]:,} rows onto 2 components in SQL; "
        f"their variances {np.round(scores.var(axis=0), 3)} match the eigenvalues "
        f"{np.round(values[order][:2], 3)}"
    )
    con.close()


if __name__ == "__main__":
    main()
