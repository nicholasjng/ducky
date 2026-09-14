# ducky

Small [nanobind](https://nanobind.readthedocs.io) bindings for
[DuckDB](https://duckdb.org), focused on data loading for machine learning.

`ducky` returns query results as Arrow, NumPy, PyTorch, JAX, or MLX arrays and
includes a dataset API for feature transforms and reproducible splits.

## Installation

`ducky` is not yet on PyPI. Install it with uv:

```toml
[project]
dependencies = ["ducky"]

[tool.uv.sources]
ducky = { git = "https://github.com/nicholasjunge/ducky" }
```

Run `uv sync`. See [DEVELOPMENT.md](DEVELOPMENT.md) to build the repository.

## Querying

```python
import ducky

con = ducky.connect()
con.execute("CREATE TABLE t (id INTEGER, name VARCHAR)")
con.execute("INSERT INTO t VALUES (1, 'a'), (2, 'b')")

rows = con.execute("SELECT * FROM t WHERE id = ?", [2]).fetchall()
arrays = con.sql("SELECT id FROM t").to_numpy()
table = con.sql("SELECT * FROM t").arrow()
frame = con.sql("SELECT * FROM t").df()
```

## Datasets

```python
ds = ducky.dataset(
    URL,
    columns={
        "age": ducky.feature("Age", standardize=True),
        "fare": ducky.feature("Fare", standardize=True),
    },
    target=ducky.target("Survived"),
    split=ducky.split(0.8, seed=0),
)

X_train, y_train = ds.train.tensors()
X_val, y_val = ds.val.tensors()
```

See [examples/titanic_jax.py](examples/titanic_jax.py) for a complete example.
