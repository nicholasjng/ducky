# ducky

Tiny, fast [nanobind](https://nanobind.readthedocs.io) bindings for
[DuckDB](https://duckdb.org).

`ducky` binds DuckDB's **stable C API** (`duckdb.h`) rather than its C++ API. The
result is a small, dependency-light extension that compiles quickly and leans on
nanobind's low-overhead Python interop. DuckDB itself is built from source via the
`ext/duckdb` git submodule (a shallow pin of [duckdb/duckdb](https://github.com/duckdb/duckdb));
initialise it with `git submodule update --init ext/duckdb`.

## Status

This is a sketch. It implements the core round-trip — connect, execute, fetch —
and is meant to grow toward DuckDB's Python API surface.

Implemented so far:

- `connect(database=":memory:")` → `Connection`
- `Connection.execute(query, parameters=None)` (returns self, PEP 249 style)
- `Connection.sql(query)` / `Connection.query(query)` → `Result`
- `Connection.fetchone()` / `fetchmany(size=1)` / `fetchall()` / `description`
- `Result` is iterable and exposes `columns`, `types`, `description`, and the
  `fetch*` methods
- Positional query parameters (`bool`, `int`, `float`, `str`, `None`)
- Errors surface as `ducky.Error`
- DataFrame / Arrow output via the Arrow C stream interface: `Result` (and
  `Connection`) implement `__arrow_c_stream__` and offer `.arrow()` (pyarrow),
  `.df()` (pandas), `.pl(lazy=False)` (polars), and `.fetchnumpy()`. These import
  their target library lazily — none are hard dependencies. Consuming the stream
  drains the result, so call one of these (or the `fetch*` methods) per query.

Value decoding currently covers:

- booleans, all integer widths (incl. `HUGEINT`/`UHUGEINT` → Python `int`)
- `FLOAT`/`DOUBLE` and `DECIMAL` (→ `decimal.Decimal`, exact)
- `VARCHAR` (→ `str`) and `BLOB` (→ `bytes`)
- `DATE`, `TIME`, `TIMESTAMP` (+ `_S`/`_MS`/`_NS`, and `_TZ` → tz-aware), `UUID`
- `INTERVAL` (→ `datetime.timedelta`; months approximated as 30 days)
- `ENUM` (→ `str`)
- nested types, arbitrarily deep: `LIST`/`ARRAY` (→ `list`), `STRUCT`/`MAP`
  (→ `dict`)
- SQL `NULL`, including inside nested values (→ `None`)

A few remaining types (e.g. `UNION`, `BIT`, `TIME_TZ`) are not decoded yet and
raise a clear `ducky.Error`; `convert_vector` in `src/cpp/result.cpp` is the
recursive extension point. The `.arrow()`/`.df()`/`.pl()` path already handles
every type via DuckDB's own Arrow converter.

## Layout

```
src/cpp/        nanobind + DuckDB C API sources
  ducky.cpp       module entry point (NB_MODULE), class registration
  connection.*    Connection: open/execute/prepare/close
  result.*        Result: lazy, chunk-based row decoding
src/ducky/      the Python package (re-exports from the _core extension)
tests/          pytest suite
CMakeLists.txt  builds DuckDB from the submodule and links the C API statically
```

## Building

Requires CMake ≥ 3.29 and Ninja. The first build compiles DuckDB from source and
takes a while; subsequent builds reuse `build/`.

Build is done **without build isolation** so nanobind's headers (already in the
environment) are visible to CMake:

```sh
uv pip install nanobind scikit-build-core   # build-time deps
uv pip install -e . --no-build-isolation    # editable install
```

To bundle more DuckDB extensions, pass them through at configure time:

```sh
uv pip install -e . --no-build-isolation \
  -C cmake.define.BUILD_EXTENSIONS="core_functions;parquet;json;icu"
```

## Testing

```sh
uv run pytest
```

## Linting

```sh
uvx prek run --all-files --show-diff-on-failure
```

## Example

```python
import ducky

con = ducky.connect()
con.execute("CREATE TABLE t (id INTEGER, name VARCHAR)")
con.execute("INSERT INTO t VALUES (1, 'a'), (2, 'b')")

print(con.execute("SELECT * FROM t WHERE id = ?", [2]).fetchall())
# [(2, 'b')]

for row in con.sql("SELECT * FROM t ORDER BY id"):
    print(row)

# DataFrame / Arrow output (pyarrow / pandas / polars imported lazily)
table = con.sql("SELECT * FROM t").arrow()      # pyarrow.Table
frame = con.sql("SELECT * FROM t").df()         # pandas.DataFrame
```
