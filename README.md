# ducky

Tiny, fast [nanobind](https://nanobind.readthedocs.io) bindings for
[DuckDB](https://duckdb.org), aimed at building **low-overhead data-to-model
training pipelines**.

`ducky` binds DuckDB's stable C API rather than its C++ API: a small,
dependency-light extension that compiles quickly and leans on nanobind's
zero-copy Python interop. On top of that sits a thin dataset API that turns an
SQL query into ready-to-train tensors for NumPy, PyTorch, or JAX.

## Installation

`ducky` is not yet on PyPI. Add it as a git dependency with `[tool.uv.sources]`:

```toml
[project]
dependencies = ["ducky"]

[tool.uv.sources]
ducky = { git = "https://github.com/nicholasjunge/ducky" }
```

Then simply `uv sync`. The first build compiles DuckDB from source via the
`ext/duckdb` submodule, which takes a few minutes; subsequent builds are
incremental.

To hack on ducky itself, see [DEVELOPMENT.md](DEVELOPMENT.md) for a local build flow.

## An example

```python
import ducky

con = ducky.connect()
con.execute("CREATE TABLE t (id INTEGER, name VARCHAR)")
con.execute("INSERT INTO t VALUES (1, 'a'), (2, 'b')")

print(con.execute("SELECT * FROM t WHERE id = ?", [2]).fetchall())
# [(2, 'b')]

# Zero-copy hand-off into NumPy / PyTorch / JAX
arr = con.sql("SELECT id FROM t").to_numpy()

# DataFrame / Arrow output (pyarrow / pandas / polars imported lazily)
table = con.sql("SELECT * FROM t").arrow()   # pyarrow.Table
frame = con.sql("SELECT * FROM t").df()      # pandas.DataFrame
```

## End-to-end: data to a training loop

The main point of ducky is to create a path from a remote dataset to a training
step, short and (mostly) copy-free.
See [`examples/titanic_jax.py`](examples/titanic_jax.py) for a complete walkthrough:
a CSV is read directly from HTTPS via DuckDB's `httpfs`, columns are declared as features
(with on-the-fly standardisation against train-fold-only stats), the dataset is split,
and each fold is handed to JAX as a tuple of `jax.Array`s — ready for a `jax.jit`'d training step.

```python
ds = ducky.dataset(
    URL,
    columns={
        "age": ducky.feature("Age", standardize=True),
        "fare": ducky.feature("Fare", standardize=True),
        ...
    },
    target=ducky.target("Survived"),
    split=ducky.split(0.8, seed=0),
)

Xtr, ytr = ds["train"].to_jax()
Xval, yval = ds["val"].to_jax()
```

Run it with:

```sh
uv run examples/titanic_jax.py
```
