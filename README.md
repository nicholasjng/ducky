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

`ducky.category` encodes labels as integer codes. The categories come from
an ENUM's labels, or else from the train fold's sorted distinct values, unless
you pass them. NULLs and unseen values become -1.

```python
ds = ducky.dataset(URL, fields={"sex": ducky.category("Sex")})
ds.categories["sex"]  # ['female', 'male']
```

`ducky.split()` takes `group=` to keep rows with the same value in one fold and
`stratify=` to keep each value's share of rows in every fold:

```python
split = ducky.split({"train": 0.8, "val": 0.2}, group="PassengerId", stratify="Survived")
```

### Examples

Each runs with `--backend jax` or `--backend mlx`.

| Example | Shows |
|---|---|
| [titanic_jax.py](examples/titanic_jax.py) | Logistic regression on a CSV file read over HTTP. |
| [tabular_mlp.py](examples/tabular_mlp.py) | An MLP on millions of generated rows, with features, standardization and a three-way split compiled into one query. |
| [linear_algebra.py](examples/linear_algebra.py) | Ridge regression and PCA, where DuckDB reduces the rows to a small matrix and projects them in SQL. |
| [text_lm.py](examples/text_lm.py) | A character-level transformer: tokenization and training windows in SQL, and nearest-neighbour search over embeddings stored as `FLOAT[d]`. |

## Streaming

`ducky.stream` reads one fold per epoch instead of holding it in memory, with
the same fields and splits as `ducky.dataset`. Fixed-size `ARRAY` columns such
as embeddings become `(n, *shape)` arrays through `ducky.tensor`.

```python
train = ducky.stream(
    "data/train/*.parquet",
    fields={"X": ducky.tensor("emb", 768), "y": ducky.vector("label", dtype="i64")},
    split=ducky.split(0.9, seed=0),
    key="id",
    batch_size=256,
    shuffle=ducky.shuffle(64),  # a 64-batch buffer, or ducky.shuffle("global")
    backend="torch",
)
loader = torch.utils.data.DataLoader(train, batch_size=None, num_workers=4)
for epoch in range(n_epochs):
    train.set_epoch(epoch)
    for batch in loader:
        ...
```

DataLoader workers and `torch.distributed` ranks each read their own share,
and ranks get equally many rows, so they take the same number of steps.
With a `key`, rows keep their fold however the data is sharded, and a
multi-file Parquet source is sharded by file. `state_dict()` and
`load_state_dict()` resume an epoch part-way through.
