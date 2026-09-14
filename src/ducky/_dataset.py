"""Build ML-ready arrays from DuckDB-readable data."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Literal, Protocol, TypeVar, overload

from ._core import Connection, Result, connect

if TYPE_CHECKING:
    import jax
    import mlx.core as mx  # ty: ignore[unresolved-import]
    import numpy as np
    import torch


@dataclass(frozen=True)
class Feature:
    """A SQL expression, dtype, and optional train-fold standardization."""

    expr: str
    dtype: str = "f32"
    standardize: bool = False


@dataclass(frozen=True)
class Target:
    """Single-column label shorthand."""

    expr: str
    dtype: str = "f32"


@dataclass(frozen=True)
class Vector:
    """A one-column field, materialised as a 1-D ``(n,)`` array."""

    expr: str
    dtype: str = "f32"
    standardize: bool = False


@dataclass(frozen=True)
class Matrix:
    """Ordered feature columns materialized as an ``(n, d)`` array."""

    columns: dict[str, Feature] = field(default_factory=dict)


Field = Vector | Matrix


@dataclass(frozen=True)
class Split:
    """Reproducible named-fold fractions keyed by row-position hashes."""

    fractions: dict[str, float] = field(default_factory=lambda: {"train": 0.8, "val": 0.2})
    seed: int = 0


def feature(expr: str, *, dtype: str = "f32", standardize: bool = False) -> Feature:
    """Shorthand for :class:`Feature`."""
    return Feature(expr=expr, dtype=dtype, standardize=standardize)


def target(expr: str, *, dtype: str = "f32") -> Target:
    """Shorthand for :class:`Target`."""
    return Target(expr=expr, dtype=dtype)


def vector(expr: str, *, dtype: str = "f32", standardize: bool = False) -> Vector:
    """Shorthand for :class:`Vector`."""
    return Vector(expr=expr, dtype=dtype, standardize=standardize)


def matrix(columns: dict[str, Feature]) -> Matrix:
    """Shorthand for :class:`Matrix`."""
    if not columns:
        raise ValueError("matrix() requires at least one column")
    return Matrix(columns=dict(columns))


def split(
    fractions: float | dict[str, float] = 0.8,
    *,
    seed: int = 0,
) -> Split:
    """Build a split from a train fraction or named fractions summing to 1."""
    if isinstance(fractions, int | float):
        f = float(fractions)
        if not 0.0 < f < 1.0:
            raise ValueError(f"train fraction must be in (0, 1); got {f}")
        return Split(fractions={"train": f, "val": 1.0 - f}, seed=seed)
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"fold fractions must sum to 1.0; got {total}")
    if not all(v > 0 for v in fractions.values()):
        raise ValueError("each fold fraction must be > 0")
    return Split(fractions=dict(fractions), seed=seed)


_DUCKDB_DTYPES: dict[str, str] = {
    "f32": "FLOAT",
    "f64": "DOUBLE",
    "i8": "TINYINT",
    "i16": "SMALLINT",
    "i32": "INTEGER",
    "i64": "BIGINT",
    "u8": "UTINYINT",
    "u16": "USMALLINT",
    "u32": "UINTEGER",
    "u64": "UBIGINT",
    "bool": "BOOLEAN",
}


def _to_duckdb_dtype(name: str) -> str:
    try:
        return _DUCKDB_DTYPES[name]
    except KeyError as exc:
        raise ValueError(f"unknown dtype {name!r}; pick one of {sorted(_DUCKDB_DTYPES)}") from exc


Backend = Literal["numpy", "jax", "torch", "mlx"]
_BACKENDS: tuple[Backend, ...] = ("numpy", "jax", "torch", "mlx")
_NO_DEVICE: tuple[Backend, ...] = ("numpy", "mlx")


class AbstractArray(Protocol):
    """Array with a shape."""

    @property
    def shape(self) -> tuple[int, ...]: ...


ArrayT = TypeVar("ArrayT", bound=AbstractArray)


def _materialize(result: Result, backend: Backend, device: Any) -> dict[str, Any]:
    """Drain ``result`` into backend arrays."""
    from . import _conversions as conv

    if backend == "numpy":
        return conv.to_numpy(result)
    if backend == "mlx":
        return conv.to_mlx(result)

    # Keep the fold bucket on the host.
    arrays = conv.to_numpy(result)
    bucket = arrays.pop("_bucket")
    out: dict[str, Any]
    if backend == "jax":
        import jax.numpy as jnp

        out = {name: jnp.asarray(arr, device=device) for name, arr in arrays.items()}
    else:
        import torch

        out = (
            {name: torch.from_numpy(arr) for name, arr in arrays.items()}
            if device is None
            else {name: torch.from_numpy(arr).to(device) for name, arr in arrays.items()}
        )
    out["_bucket"] = bucket
    return out


def _bucket_to_host(arr: Any) -> Any:
    """Return the bucket column as a NumPy array."""
    import numpy as np

    return np.asarray(arr)


def _gather(arr: Any, idx: Any, backend: Backend) -> Any:
    """Gather rows `idx` (a 1-D numpy int array) from `arr` along axis 0."""
    if backend == "torch":
        import torch

        return arr[torch.as_tensor(idx, device=arr.device)]
    if backend == "mlx":
        import mlx.core as mx  # ty: ignore[unresolved-import]

        return arr[mx.array(idx)]
    return arr[idx]  # numpy and jax both index with a host ndarray


def _stack(cols: list[Any], backend: Backend) -> Any:
    """Stack 1-D feature columns into an ``(n, len(cols))`` matrix."""
    if backend == "jax":
        import jax.numpy as jnp

        return jnp.stack(cols, axis=1)
    if backend == "torch":
        import torch

        return torch.stack(cols, dim=1)
    if backend == "mlx":
        import mlx.core as mx  # ty: ignore[unresolved-import]

        return mx.stack(cols, axis=1)
    import numpy as np

    return np.stack(cols, axis=1)


@dataclass
class Fold(Generic[ArrayT]):
    """Named arrays for one split; ``fold.Xy`` returns ``(fold.X, fold.y)``."""

    backend: Backend
    _fields: dict[str, ArrayT]

    @property
    def fields(self) -> dict[str, ArrayT]:
        return dict(self._fields)

    @property
    def n_rows(self) -> int:
        return int(next(iter(self._fields.values())).shape[0])

    def __getitem__(self, name: str) -> ArrayT:
        return self._fields[name]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        fields = self.__dict__.get("_fields")
        if not fields:
            raise AttributeError(name)
        if name in fields:
            return fields[name]
        if len(name) > 1 and all(ch in fields for ch in name):
            return tuple(fields[ch] for ch in name)
        raise AttributeError(name)

    def tensors(self) -> tuple[ArrayT, ArrayT]:
        """Return ``(self["X"], self["y"])``. Raises if either field is missing."""
        try:
            return self._fields["X"], self._fields["y"]
        except KeyError as exc:
            missing = exc.args[0]
            raise AttributeError(
                f"tensors() requires fields 'X' and 'y'; field {missing!r} not found "
                f"(available: {sorted(self._fields)})"
            ) from None

    def batches(
        self,
        batch_size: int,
        *,
        shuffle: bool = False,
        seed: int | None = None,
        drop_last: bool = False,
    ) -> Iterator[tuple[ArrayT, ArrayT]]:
        """Yield ``(X, y)`` batches, optionally shuffled or dropping the tail."""
        import numpy as np

        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0; got {batch_size}")
        X, y = self.tensors()
        n = self.n_rows
        indices = np.arange(n)
        if shuffle:
            np.random.default_rng(seed).shuffle(indices)
        end = (n // batch_size) * batch_size if drop_last else n
        for start in range(0, end, batch_size):
            idx = indices[start : start + batch_size]
            yield _gather(X, idx, self.backend), _gather(y, idx, self.backend)

    def __repr__(self) -> str:
        shapes = ", ".join(f"{n}={tuple(a.shape)}" for n, a in self._fields.items())
        return f"Fold(n_rows={self.n_rows}, backend={self.backend!r}, {shapes})"


@dataclass
class Dataset(Generic[ArrayT]):
    """Backend arrays grouped into named folds."""

    folds: dict[str, Fold[ArrayT]]
    field_specs: dict[str, Field] = field(default_factory=dict)

    @property
    def train(self) -> Fold[ArrayT] | None:
        return self.folds.get("train")

    @property
    def val(self) -> Fold[ArrayT] | None:
        return self.folds.get("val")

    @property
    def test(self) -> Fold[ArrayT] | None:
        return self.folds.get("test")

    @property
    def field_names(self) -> list[str]:
        return list(self.field_specs)

    @property
    def feature_names(self) -> list[str]:
        """Column names of the ``X`` matrix."""
        spec = self.field_specs.get("X")
        if isinstance(spec, Matrix):
            return list(spec.columns)
        raise AttributeError("feature_names is only defined when field 'X' is a Matrix")

    @property
    def target_name(self) -> str:
        """SQL expression of the ``y`` vector."""
        spec = self.field_specs.get("y")
        if isinstance(spec, Vector):
            return spec.expr
        raise AttributeError("target_name is only defined when field 'y' is a Vector")

    def __getitem__(self, name: str) -> Fold[ArrayT]:
        return self.folds[name]

    def __repr__(self) -> str:
        parts = ", ".join(f"{name}={f.n_rows}" for name, f in self.folds.items())
        return f"Dataset({parts} rows, fields={list(self.field_specs)!r})"


# 0.1% resolution on fold fractions.
_N_BUCKETS = 1000


def _quote_literal(s: str) -> str:
    """Quote a string for embedding as a SQL string literal ('' escapes ')."""
    return "'" + s.replace("'", "''") + "'"


def _quote_ident(s: str) -> str:
    """Quote an identifier for embedding in SQL ("" escapes ")."""
    return '"' + s.replace('"', '""') + '"'


def _fold_ranges(fractions: dict[str, float]) -> list[tuple[str, int, int]]:
    """Convert fractions to contiguous bucket ranges."""
    ranges: list[tuple[str, int, int]] = []
    cum = 0.0
    items = list(fractions.items())
    for i, (name, frac) in enumerate(items):
        lo = round(cum * _N_BUCKETS)
        cum += frac
        hi = _N_BUCKETS if i == len(items) - 1 else round(cum * _N_BUCKETS)
        ranges.append((name, lo, hi))
    return ranges


def _iter_field_columns(
    fields: dict[str, Field],
) -> Iterator[tuple[str, str, Feature | Vector]]:
    """Yield each materialized column's alias, field name, and specification."""
    for field_name, spec in fields.items():
        if isinstance(spec, Vector):
            yield field_name, field_name, spec
        else:
            for col_name, feat in spec.columns.items():
                yield f"{field_name}__{col_name}", field_name, feat


def _compile_sql(
    source: str,
    fields: dict[str, Field],
    drop_nulls: list[str] | None,
    split_spec: Split | None,
) -> tuple[str, list[tuple[str, int, int]], str | None]:
    """Compile fields, filtering, splitting, and standardization into SQL."""
    if not fields:
        raise ValueError("at least one field is required")

    columns = list(_iter_field_columns(fields))
    if not columns:
        raise ValueError("fields must contain at least one column")

    aliases = [a for a, _, _ in columns]
    if len(set(aliases)) != len(aliases):
        dupes = sorted({a for a in aliases if aliases.count(a) > 1})
        raise ValueError(f"duplicate field/column aliases: {dupes}")
    if "_bucket" in aliases:
        raise ValueError(
            "field/column alias '_bucket' is reserved for ducky's internal fold "
            "bookkeeping; rename this field or matrix column"
        )

    from_clause = _quote_literal(source)

    if split_spec is not None:
        ranges = _fold_ranges(split_spec.fractions)
        bucket_expr = f"(hash(row_number() OVER () + {split_spec.seed}) % {_N_BUCKETS})"
    else:
        ranges = [("train", 0, _N_BUCKETS)]
        bucket_expr = "0::BIGINT"

    null_filter = (
        " WHERE " + " AND ".join(f"{_quote_ident(c)} IS NOT NULL" for c in drop_nulls)
        if drop_nulls
        else ""
    )
    raw_sql = f"SELECT *, {bucket_expr} AS _bucket FROM {from_clause}{null_filter}"

    needs_stats = any(spec.standardize for _, _, spec in columns)
    train_check_sql: str | None = None
    if needs_stats:
        train_range = next((r for r in ranges if r[0] == "train"), None)
        if train_range is None:
            raise ValueError(
                "standardize=True requires a fold named 'train'; either rename "
                "your training fold or compute statistics yourself in SQL"
            )
        _, train_lo, train_hi = train_range
        is_train = f"_bucket >= {train_lo} AND _bucket < {train_hi}"
        stats_cols: list[str] = []
        for alias, _, spec in columns:
            if spec.standardize:
                stats_cols.append(f"avg({spec.expr}) AS {_quote_ident(f'_{alias}_mean')}")
                stats_cols.append(f"stddev_pop({spec.expr}) AS {_quote_ident(f'_{alias}_std')}")
        stats_sql = f"SELECT {', '.join(stats_cols)} FROM raw WHERE {is_train}"
        ctes = f"WITH raw AS ({raw_sql}), stats AS ({stats_sql})"
        from_outer = "raw, stats"
        # avg()/stddev_pop() over zero rows return NULL, which coalesce(nullif(std, 0), 1)
        # below would treat as a constant column instead of an empty fold — silently
        # NULL-ing every standardized value. Catch it explicitly instead.
        train_check_sql = f"WITH raw AS ({raw_sql}) SELECT count(*) FROM raw WHERE {is_train}"
    else:
        ctes = f"WITH raw AS ({raw_sql})"
        from_outer = "raw"

    select_items: list[str] = []
    for alias, _, spec in columns:
        if spec.standardize:
            # coalesce(nullif(std, 0), 1): a constant column (std = 0) yields 0
            # (sklearn's StandardScaler convention) instead of a division by
            # zero, whose NULL result would surface as garbage in the dense
            # ndarrays downstream.
            mean = f"stats.{_quote_ident(f'_{alias}_mean')}"
            std = f"stats.{_quote_ident(f'_{alias}_std')}"
            expr = f"({spec.expr} - {mean}) / coalesce(nullif({std}, 0), 1)"
        else:
            expr = spec.expr
        select_items.append(
            f"CAST({expr} AS {_to_duckdb_dtype(spec.dtype)}) AS {_quote_ident(alias)}"
        )
    select_items.append("CAST(_bucket AS BIGINT) AS _bucket")

    sql = f"{ctes} SELECT {', '.join(select_items)} FROM {from_outer}"
    return sql, ranges, train_check_sql


def _assemble_field(name: str, spec: Field, columns_data: dict[str, Any], backend: Backend) -> Any:
    """Stack a matrix field's columns / return a vector field's lone column."""
    if isinstance(spec, Vector):
        return columns_data[name]
    cols = [columns_data[f"{name}__{c}"] for c in spec.columns]
    return _stack(cols, backend)


def _normalize_fields(
    fields: dict[str, Field] | None,
    columns: dict[str, Feature] | None,
    target_spec: Target | None,
) -> dict[str, Field]:
    """Resolve the ``fields=`` / ``columns=``+``target=`` alternatives."""
    if fields is not None:
        if columns is not None or target_spec is not None:
            raise ValueError("pass either fields= or columns=/target=, not both")
        if not fields:
            raise ValueError("fields must contain at least one entry")
        return dict(fields)
    if columns is None or target_spec is None:
        raise ValueError("either fields= or both columns= and target= are required")
    return {
        "X": matrix(columns),
        "y": Vector(expr=target_spec.expr, dtype=target_spec.dtype),
    }


@overload
def dataset(
    source: str,
    *,
    fields: dict[str, Field] | None = ...,
    columns: dict[str, Feature] | None = ...,
    target: Target | None = ...,
    drop_nulls: list[str] | None = ...,
    split: Split | None = ...,
    con: Connection | None = ...,
    backend: Literal["numpy"] = ...,
    device: Any = ...,
) -> Dataset[np.ndarray]: ...
@overload
def dataset(
    source: str,
    *,
    fields: dict[str, Field] | None = ...,
    columns: dict[str, Feature] | None = ...,
    target: Target | None = ...,
    drop_nulls: list[str] | None = ...,
    split: Split | None = ...,
    con: Connection | None = ...,
    backend: Literal["jax"],
    device: Any = ...,
) -> Dataset[jax.Array]: ...
@overload
def dataset(
    source: str,
    *,
    fields: dict[str, Field] | None = ...,
    columns: dict[str, Feature] | None = ...,
    target: Target | None = ...,
    drop_nulls: list[str] | None = ...,
    split: Split | None = ...,
    con: Connection | None = ...,
    backend: Literal["torch"],
    device: Any = ...,
) -> Dataset[torch.Tensor]: ...
@overload
def dataset(
    source: str,
    *,
    fields: dict[str, Field] | None = ...,
    columns: dict[str, Feature] | None = ...,
    target: Target | None = ...,
    drop_nulls: list[str] | None = ...,
    split: Split | None = ...,
    con: Connection | None = ...,
    backend: Literal["mlx"],
    device: Any = ...,
) -> Dataset[mx.array]: ...
@overload
def dataset(
    source: str,
    *,
    fields: dict[str, Field] | None = ...,
    columns: dict[str, Feature] | None = ...,
    target: Target | None = ...,
    drop_nulls: list[str] | None = ...,
    split: Split | None = ...,
    con: Connection | None = ...,
    backend: str,
    device: Any = ...,
) -> Dataset[Any]: ...
def dataset(
    source: str,
    *,
    fields: dict[str, Field] | None = None,
    columns: dict[str, Feature] | None = None,
    target: Target | None = None,
    drop_nulls: list[str] | None = None,
    split: Split | None = None,
    con: Connection | None = None,
    backend: str = "numpy",
    device: Any = None,
) -> Dataset[Any]:
    """Load a DuckDB-readable source into split NumPy, JAX, Torch, or MLX arrays.

    ``fields`` maps names to vectors or matrices. ``columns`` and ``target``
    provide the ``X``/``y`` shorthand. Standardization uses only the train fold.
    """
    if backend not in _BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; supported backends: {list(_BACKENDS)}")
    if device is not None and backend in _NO_DEVICE:
        raise ValueError(f"backend {backend!r} does not take a device argument")

    field_specs = _normalize_fields(fields, columns, target)

    sql, ranges, train_check_sql = _compile_sql(source, field_specs, drop_nulls, split)
    own_connection = con is None
    if own_connection:
        con = connect()
    try:
        if train_check_sql is not None and con.sql(train_check_sql).fetchitem() == 0:
            raise ValueError(
                "standardize=True requires at least one row in the 'train' fold "
                "after filtering/splitting, but it's empty; widen the split, check "
                "drop_nulls, or drop standardize=True"
            )
        columns_data = _materialize(con.sql(sql), backend, device)
    finally:
        if own_connection:
            con.close()

    import numpy as np

    bucket = _bucket_to_host(columns_data.pop("_bucket"))
    assembled = {
        field_name: _assemble_field(field_name, spec, columns_data, backend)
        for field_name, spec in field_specs.items()
    }
    folds: dict[str, Fold[Any]] = {}
    for name, lo, hi in ranges:
        idx = np.nonzero((bucket >= lo) & (bucket < hi))[0]
        gathered = {field_name: _gather(arr, idx, backend) for field_name, arr in assembled.items()}
        folds[name] = Fold(backend=backend, _fields=gathered)
    return Dataset(folds=folds, field_specs=field_specs)
