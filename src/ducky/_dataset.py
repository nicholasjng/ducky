"""Build ML-ready arrays from DuckDB-readable data."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, cast, overload

from ._core import Connection, connect

if TYPE_CHECKING:
    import jax
    import mlx.core as mx
    import numpy as np
    import torch


type DTypeLike = str | type | np.dtype[Any] | torch.dtype | mx.Dtype
"""A dtype name such as ``"f32"`` or ``"float32"``, or a NumPy, JAX, torch or MLX dtype."""

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
_DTYPE_ALIASES: dict[str, str] = {
    "float32": "f32",
    "float64": "f64",
    "int8": "i8",
    "int16": "i16",
    "int32": "i32",
    "int64": "i64",
    "uint8": "u8",
    "uint16": "u16",
    "uint32": "u32",
    "uint64": "u64",
}
_BUILTIN_DTYPES: dict[type, str] = {bool: "bool", int: "i64", float: "f64"}


def _canonical_dtype(dtype: DTypeLike) -> str:
    """Return the short name (``"f32"``, ``"i64"``, ...) of a dtype or dtype name."""
    if isinstance(dtype, str):
        name = dtype
    elif isinstance(dtype, type) and dtype in _BUILTIN_DTYPES:
        name = _BUILTIN_DTYPES[dtype]
    elif type(dtype).__module__.partition(".")[0] in ("torch", "mlx"):
        # torch.float32 and mlx.core.float32 print as their qualified names
        name = str(dtype).rpartition(".")[2]
    else:
        # NumPy dtypes and scalar types, and JAX's, which NumPy understands
        import numpy as np

        try:
            name = np.dtype(cast(Any, dtype)).name
        except TypeError:
            name = None
    short = _DTYPE_ALIASES.get(name, name) if name is not None else None
    if short not in _DUCKDB_DTYPES:
        raise ValueError(
            f"unknown dtype {dtype!r}; pick one of {sorted(_DUCKDB_DTYPES)}, "
            "or the matching NumPy, JAX, torch or MLX dtype"
        )
    return short


def _normalize_dtype(spec: Any) -> None:
    object.__setattr__(spec, "dtype", _canonical_dtype(spec.dtype))


@dataclass(frozen=True)
class Feature:
    """A SQL expression, dtype, and optional train-fold standardization."""

    expr: str
    dtype: str = "f32"
    standardize: bool = False

    def __post_init__(self) -> None:
        _normalize_dtype(self)


@dataclass(frozen=True)
class Target:
    """Single-column label shorthand."""

    expr: str
    dtype: str = "f32"

    def __post_init__(self) -> None:
        _normalize_dtype(self)


@dataclass(frozen=True)
class Vector:
    """A one-column field, materialised as a 1-D ``(n,)`` array."""

    expr: str
    dtype: str = "f32"
    standardize: bool = False

    def __post_init__(self) -> None:
        _normalize_dtype(self)


@dataclass(frozen=True)
class Tensor:
    """A fixed-size ARRAY field, materialised as an ``(n, *shape)`` array."""

    expr: str
    shape: tuple[int, ...]
    dtype: str = "f32"
    standardize: ClassVar[bool] = False  # per-element statistics aren't supported

    def __post_init__(self) -> None:
        _normalize_dtype(self)


@dataclass(frozen=True)
class Category:
    """A categorical field, materialised as ``(n,)`` integer codes.

    Codes index the field's categories: the given ones, else an ENUM's labels,
    else the train fold's distinct values in sorted order. NULLs and values
    outside the categories become -1.
    """

    expr: str
    categories: tuple[Any, ...] | None = None
    dtype: str = "i64"
    standardize: ClassVar[bool] = False

    def __post_init__(self) -> None:
        _normalize_dtype(self)


@dataclass(frozen=True)
class Matrix:
    """Ordered feature columns materialized as an ``(n, d)`` array."""

    columns: dict[str, Feature] = field(default_factory=dict)


Field = Vector | Matrix | Tensor | Category


@dataclass(frozen=True)
class Split:
    """Reproducible named-fold fractions, optionally grouped or stratified.

    Without ``group`` or ``stratify``, each row lands in a fold by the hash of
    its identity. With either, rows are ranked in a seeded hash order and each
    fold takes its exact share of the ranking: with ``group``, all rows of a
    group stay in one fold and fractions count groups; with ``stratify``, each
    fold keeps every stratum's share of rows (or groups).
    """

    fractions: dict[str, float] = field(default_factory=lambda: {"train": 0.8, "val": 0.2})
    seed: int = 0
    group: str | None = None
    stratify: str | None = None

    @property
    def ranked(self) -> bool:
        """Whether folds come from ranking rather than from each row's own hash."""
        return self.group is not None or self.stratify is not None


def feature(expr: str, *, dtype: DTypeLike = "f32", standardize: bool = False) -> Feature:
    """Shorthand for :class:`Feature`."""
    return Feature(expr=expr, dtype=_canonical_dtype(dtype), standardize=standardize)


def target(expr: str, *, dtype: DTypeLike = "f32") -> Target:
    """Shorthand for :class:`Target`."""
    return Target(expr=expr, dtype=_canonical_dtype(dtype))


def vector(expr: str, *, dtype: DTypeLike = "f32", standardize: bool = False) -> Vector:
    """Shorthand for :class:`Vector`."""
    return Vector(expr=expr, dtype=_canonical_dtype(dtype), standardize=standardize)


def tensor(expr: str, shape: int | tuple[int, ...], *, dtype: DTypeLike = "f32") -> Tensor:
    """Shorthand for :class:`Tensor`. An int ``shape`` is a 1-D embedding size."""
    dims = (shape,) if isinstance(shape, int) else tuple(shape)
    if not dims or not all(isinstance(d, int) and d > 0 for d in dims):
        raise ValueError(f"tensor shape must be one or more positive ints; got {shape!r}")
    return Tensor(expr=expr, shape=dims, dtype=_canonical_dtype(dtype))


_CODE_DTYPES = ("i8", "i16", "i32", "i64")


def category(
    expr: str, categories: Iterable[Any] | None = None, *, dtype: DTypeLike = "i64"
) -> Category:
    """Shorthand for :class:`Category`. ``categories`` fixes the codes' order."""
    code_dtype = _canonical_dtype(dtype)
    if code_dtype not in _CODE_DTYPES:
        raise ValueError(
            f"category codes need a signed integer dtype, one of {list(_CODE_DTYPES)}; "
            f"got {dtype!r}"
        )
    if categories is None:
        return Category(expr=expr, dtype=code_dtype)
    given = tuple(categories)
    if not given:
        raise ValueError("categories must not be empty")
    return Category(expr=expr, categories=given, dtype=code_dtype)


def matrix(columns: dict[str, Feature]) -> Matrix:
    """Shorthand for :class:`Matrix`."""
    if not columns:
        raise ValueError("matrix() requires at least one column")
    return Matrix(columns=dict(columns))


def split(
    fractions: float | dict[str, float] = 0.8,
    *,
    seed: int = 0,
    group: str | None = None,
    stratify: str | None = None,
) -> Split:
    """Build a split from a train fraction or named fractions summing to 1.

    ``group`` is a SQL expression whose value keeps rows together: every row
    with the same value lands in the same fold. ``stratify`` is a SQL
    expression, usually the label, whose value distribution each fold keeps.
    With both, every group must have a single ``stratify`` value. NULL is one
    group or stratum like any other value.
    """
    if group is not None and not group.strip():
        raise ValueError("group must be a non-empty SQL expression")
    if stratify is not None and not stratify.strip():
        raise ValueError("stratify must be a non-empty SQL expression")
    if isinstance(fractions, int | float):
        f = float(fractions)
        if not 0.0 < f < 1.0:
            raise ValueError(f"train fraction must be in (0, 1); got {f}")
        return Split(
            fractions={"train": f, "val": 1.0 - f}, seed=seed, group=group, stratify=stratify
        )
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"fold fractions must sum to 1.0; got {total}")
    if not all(v > 0 for v in fractions.values()):
        raise ValueError("each fold fraction must be > 0")
    return Split(fractions=dict(fractions), seed=seed, group=group, stratify=stratify)


Backend = Literal["numpy", "jax", "torch", "mlx"]
_BACKENDS: tuple[Backend, ...] = ("numpy", "jax", "torch", "mlx")
_NO_DEVICE: tuple[Backend, ...] = ("numpy", "mlx")


class AbstractArray(Protocol):
    """Array with a shape."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    def __getitem__(self, key: Any, /) -> Any: ...


def _to_backend(arr: Any, backend: Backend, device: Any) -> Any:
    """Move one NumPy array to ``backend``."""
    if backend == "numpy":
        return arr
    if backend == "mlx":
        import mlx.core as mx

        return mx.array(arr)
    if backend == "jax":
        import jax.numpy as jnp

        return jnp.asarray(arr, device=device)
    import torch

    tensor = torch.from_numpy(arr)
    return tensor if device is None else tensor.to(device)


def _gather(arr: Any, idx: Any, backend: Backend) -> Any:
    if backend == "torch":
        import torch

        return arr[torch.as_tensor(idx, device=arr.device)]
    if backend == "mlx":
        import mlx.core as mx

        return arr[mx.array(idx)]
    return arr[idx]  # numpy and jax both index with a host ndarray


@dataclass
class Fold[ArrayT: AbstractArray]:
    """Named arrays for one split. ``fold.Xy`` returns ``(fold.X, fold.y)``."""

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
        if not shuffle:
            # Slices are views; gathering by index would copy every batch.
            for start in range(0, end, batch_size):
                yield X[start : start + batch_size], y[start : start + batch_size]
            return
        for start in range(0, end, batch_size):
            idx = indices[start : start + batch_size]
            yield _gather(X, idx, self.backend), _gather(y, idx, self.backend)

    def __repr__(self) -> str:
        shapes = ", ".join(f"{n}={tuple(a.shape)}" for n, a in self._fields.items())
        return f"Fold(n_rows={self.n_rows}, backend={self.backend!r}, {shapes})"


@dataclass
class Dataset[ArrayT: AbstractArray]:
    """Backend arrays grouped into named folds."""

    folds: dict[str, Fold[ArrayT]]
    field_specs: dict[str, Field] = field(default_factory=dict)
    categories: dict[str, list[Any]] = field(default_factory=dict)
    """Each category field's categories, indexed by its codes."""

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
    return "'" + s.replace("'", "''") + "'"


def _quote_ident(s: str) -> str:
    return '"' + s.replace('"', '""') + '"'


def _fold_ranges(fractions: dict[str, float]) -> list[tuple[str, int, int]]:
    ranges: list[tuple[str, int, int]] = []
    cum = 0.0
    items = list(fractions.items())
    for i, (name, frac) in enumerate(items):
        lo = round(cum * _N_BUCKETS)
        cum += frac
        hi = _N_BUCKETS if i == len(items) - 1 else round(cum * _N_BUCKETS)
        if hi <= lo:
            raise ValueError(
                f"fold {name!r} has fraction {frac}, below the split resolution of "
                f"{1 / _N_BUCKETS:.1%}; it would always be empty"
            )
        ranges.append((name, lo, hi))
    return ranges


Column = Feature | Vector | Tensor | Category


def _iter_field_columns(fields: dict[str, Field]) -> Iterator[tuple[str, str, Column]]:
    for field_name, spec in fields.items():
        if isinstance(spec, Vector | Tensor | Category):
            yield field_name, field_name, spec
        else:
            for col_name, feat in spec.columns.items():
                yield f"{field_name}__{col_name}", field_name, feat


def _row_id_expr(key: str | None) -> str:
    return f"({key})" if key is not None else "row_number() OVER ()"


def _bucket_expr(row_id: str, split_spec: Split, key: str | None) -> str:
    """Hash a row's identity (from ``_row_id_expr``) into its fold bucket."""
    # The seed is a second hash input: added to a row number, it would only
    # shift the same split by `seed` rows.
    return f"(hash({row_id}, {split_spec.seed}) % {_N_BUCKETS})"


def _ranked_split_sql(source_sql: str, split_spec: Split, key: str | None) -> str:
    """Select ``source_sql`` with a ``_bucket`` from ranking units within strata.

    A unit is a group, or a row without ``group``. Units are ordered by a seeded
    hash within each stratum, and unit ``r`` of ``n`` gets bucket
    ``floor((r - 0.5) * N / n)``: each fold's bucket range then takes its share
    of every stratum's units, rounded to the nearest unit.
    """
    unit = f"({split_spec.group})" if split_spec.group is not None else _row_id_expr(key)
    stratum = f"({split_spec.stratify})" if split_spec.stratify is not None else "NULL"
    keyed = f"SELECT *, {unit} AS __ducky_unit, {stratum} AS __ducky_stratum FROM ({source_sql})"
    order = f"hash(__ducky_unit, {split_spec.seed}), __ducky_unit"
    rank = f"dense_rank() OVER (PARTITION BY __ducky_stratum ORDER BY {order}) AS __ducky_rank"
    ranked = f"SELECT *, {rank} FROM ({keyed})"
    n_units = "max(__ducky_rank) OVER (PARTITION BY __ducky_stratum)"
    bucket = f"CAST(floor((__ducky_rank - 0.5) * {_N_BUCKETS} / {n_units}) AS BIGINT)"
    if split_spec.group is not None and split_spec.stratify is not None:
        # A group with several strata would be ranked once per stratum and could
        # straddle folds. The check sits in the bucket expression because DuckDB
        # drops columns nothing reads. count(DISTINCT) skips NULL, so add it back.
        w = "OVER (PARTITION BY __ducky_unit)"
        n_strata = f"count(DISTINCT __ducky_stratum) {w} + max((__ducky_stratum IS NULL)::INT) {w}"
        message = _quote_literal(
            f"split(group={split_spec.group!r}, stratify={split_spec.stratify!r}) needs "
            "one stratify value per group, but this group has several: "
        )
        group_text = "coalesce(CAST(__ducky_unit AS VARCHAR), 'NULL')"
        bucket = (
            f"CASE WHEN {n_strata} > 1 THEN CAST(error({message} || {group_text}) AS BIGINT) "
            f"ELSE {bucket} END"
        )
    return f"SELECT *, {bucket} AS _bucket FROM ({ranked})"


def _null_filter(drop_nulls: list[str] | None) -> str:
    if not drop_nulls:
        return ""
    return " WHERE " + " AND ".join(f"{_quote_ident(c)} IS NOT NULL" for c in drop_nulls)


def _train_filter(ranges: list[tuple[str, int, int]], bucket: str) -> str | None:
    """SQL selecting the train fold's rows by their ``bucket``, if there is one."""
    train = next((r for r in ranges if r[0] == "train"), None)
    if train is None:
        return None
    _, lo, hi = train
    return f"{bucket} >= {lo} AND {bucket} < {hi}"


def _resolve_categories(
    con: Connection,
    from_clause: str,
    ctes: str,
    train: str | None,
    fields: dict[str, Field],
) -> tuple[dict[str, Field], dict[str, list[Any]]]:
    """Replace category fields by vectors of their codes, and find their categories.

    ``ctes`` define a table ``raw`` of the source's rows, of which ``train``
    selects the train fold.
    """
    wanted = {name: spec for name, spec in fields.items() if isinstance(spec, Category)}
    if not wanted:
        return fields, {}
    probe = con.sql(
        f"SELECT {', '.join(s.expr for s in wanted.values())} FROM {from_clause} LIMIT 0"
    )
    found: dict[str, tuple[list[Any], list[str] | None]] = {}  # categories, labels
    given, learn = [], []
    for i, (name, spec) in enumerate(wanted.items()):
        if spec.categories is not None:
            given.append(name)
        elif probe.types[i] == "ENUM":
            found[name] = probe.categories(i), None  # its codes already
        else:
            learn.append(name)
    if given:
        # Values match by their text, so DuckDB writes the given categories as it
        # writes the field's values (true, 1.0), and one it can't represent raises.
        exprs = ", ".join(f"{wanted[n].expr} AS _g{i}" for i, n in enumerate(given))
        typeofs = ", ".join(f"typeof(s._g{i})" for i in range(len(given)))
        sql_types = con.sql(
            f"SELECT {typeofs} FROM (SELECT 1) LEFT JOIN "
            f"(SELECT {exprs} FROM {from_clause} LIMIT 0) s ON true"
        ).fetchone()
        assert sql_types is not None
        for name, sql_type in zip(given, sql_types, strict=True):
            categories = list(wanted[name].categories or ())
            casts = ", ".join(f"CAST(CAST(? AS {sql_type}) AS VARCHAR)" for _ in categories)
            labels = con.execute(f"SELECT {casts}", categories).fetchone()
            assert labels is not None
            if len(set(labels)) != len(labels):
                raise ValueError(
                    f"category field {name!r}: categories must be distinct as {sql_type}"
                )
            found[name] = categories, list(labels)
    if learn:
        if train is None:
            raise ValueError(
                "category() learns its categories from a fold named 'train'; either "
                "rename your training fold or pass categories="
            )
        lists = ", ".join(
            f"list(DISTINCT {wanted[n].expr} ORDER BY {wanted[n].expr}) "
            f"FILTER (WHERE {wanted[n].expr} IS NOT NULL) AS _c{i}"
            for i, n in enumerate(learn)
        )
        both = ", ".join(
            f"_c{i}, list_transform(_c{i}, lambda v: v::VARCHAR)" for i in range(len(learn))
        )
        row = con.sql(f"{ctes} SELECT {both} FROM (SELECT {lists} FROM raw WHERE {train})")
        values = row.fetchone()
        assert values is not None
        for i, name in enumerate(learn):
            if not values[2 * i]:
                raise ValueError(
                    f"category field {name!r} has no values in the 'train' fold; "
                    "pass categories= explicitly"
                )
            found[name] = values[2 * i], values[2 * i + 1]

    resolved = dict(fields)
    for name, spec in wanted.items():
        largest = 2 ** (int(spec.dtype[1:]) - 1) - 1
        if len(found[name][0]) - 1 > largest:
            raise ValueError(
                f"category field {name!r} has {len(found[name][0])} categories, more than "
                f"{spec.dtype} codes can number; use a wider dtype"
            )
        labels = found[name][1]
        if labels is None:
            codes = f"coalesce(enum_code({spec.expr}), -1)"
        else:
            enum = ", ".join(_quote_literal(label) for label in labels)
            codes = (
                f"coalesce(enum_code(TRY_CAST(CAST({spec.expr} AS VARCHAR) AS ENUM({enum}))), -1)"
            )
        resolved[name] = Vector(expr=codes, dtype=spec.dtype)
    return resolved, {name: found[name][0] for name in wanted}


def _select_item(alias: str, spec: Column, mean: str | None = None, std: str | None = None) -> str:
    """Cast one field column, standardized with ``mean``/``std`` SQL if given."""
    assert not isinstance(spec, Category), "resolved by _resolve_categories"
    expr = spec.expr
    if mean is not None and std is not None:
        # A constant column (std = 0) yields 0, sklearn's StandardScaler
        # convention, rather than a division by zero whose NULL would surface as
        # garbage in the dense arrays.
        expr = f"({expr} - {mean}) / coalesce(nullif({std}, 0), 1)"
    sql_type = _DUCKDB_DTYPES[spec.dtype]
    if isinstance(spec, Tensor):
        sql_type += "".join(f"[{d}]" for d in spec.shape)
    return f"CAST({expr} AS {sql_type}) AS {_quote_ident(alias)}"


def _check_columns(fields: dict[str, Field]) -> list[tuple[str, str, Column]]:
    """Return the fields' columns, rejecting empty, duplicate and reserved aliases."""
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
    return columns


def _raw_sql(
    source: str, drop_nulls: list[str] | None, split_spec: Split | None, key: str | None
) -> tuple[str, list[tuple[str, int, int]]]:
    """SQL for the source's rows with their fold ``_bucket``, and the folds' ranges."""
    if split_spec is not None:
        ranges = _fold_ranges(split_spec.fractions)
        bucket_expr = _bucket_expr(_row_id_expr(key), split_spec, key)
    else:
        ranges = [("train", 0, _N_BUCKETS)]
        bucket_expr = "0::BIGINT"
    null_filter = _null_filter(drop_nulls)
    if split_spec is not None and split_spec.ranked:
        rows = f"SELECT * FROM {_quote_literal(source)}{null_filter}"
        return _ranked_split_sql(rows, split_spec, key), ranges
    return f"SELECT *, {bucket_expr} AS _bucket FROM {_quote_literal(source)}{null_filter}", ranges


def _compile_sql(
    source: str,
    fields: dict[str, Field],
    drop_nulls: list[str] | None,
    split_spec: Split | None,
    key: str | None = None,
) -> tuple[str, list[tuple[str, int, int]]]:
    """Compile fields, filtering, splitting, and standardization into SQL."""
    columns = _check_columns(fields)

    raw_sql, ranges = _raw_sql(source, drop_nulls, split_spec, key)

    needs_stats = any(spec.standardize for _, _, spec in columns)
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
    else:
        ctes = f"WITH raw AS ({raw_sql})"
        from_outer = "raw"

    select_items: list[str] = []
    for alias, _, spec in columns:
        if spec.standardize:
            mean = f"stats.{_quote_ident(f'_{alias}_mean')}"
            std = f"stats.{_quote_ident(f'_{alias}_std')}"
            select_items.append(_select_item(alias, spec, mean, std))
        else:
            select_items.append(_select_item(alias, spec))
    select_items.append("CAST(_bucket AS BIGINT) AS _bucket")

    sql = f"{ctes} SELECT {', '.join(select_items)} FROM {from_outer}"
    return sql, ranges


def _normalize_fields(
    fields: Mapping[str, Field] | None,
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
    fields: Mapping[str, Field] | None = ...,
    columns: dict[str, Feature] | None = ...,
    target: Target | None = ...,
    drop_nulls: list[str] | None = ...,
    split: Split | None = ...,
    key: str | None = ...,
    con: Connection | None = ...,
    backend: Literal["numpy"] = ...,
    device: Any = ...,
) -> Dataset[np.ndarray]: ...
@overload
def dataset(
    source: str,
    *,
    fields: Mapping[str, Field] | None = ...,
    columns: dict[str, Feature] | None = ...,
    target: Target | None = ...,
    drop_nulls: list[str] | None = ...,
    split: Split | None = ...,
    key: str | None = ...,
    con: Connection | None = ...,
    backend: Literal["jax"],
    device: Any = ...,
) -> Dataset[jax.Array]: ...
@overload
def dataset(
    source: str,
    *,
    fields: Mapping[str, Field] | None = ...,
    columns: dict[str, Feature] | None = ...,
    target: Target | None = ...,
    drop_nulls: list[str] | None = ...,
    split: Split | None = ...,
    key: str | None = ...,
    con: Connection | None = ...,
    backend: Literal["torch"],
    device: Any = ...,
) -> Dataset[torch.Tensor]: ...
@overload
def dataset(
    source: str,
    *,
    fields: Mapping[str, Field] | None = ...,
    columns: dict[str, Feature] | None = ...,
    target: Target | None = ...,
    drop_nulls: list[str] | None = ...,
    split: Split | None = ...,
    key: str | None = ...,
    con: Connection | None = ...,
    backend: Literal["mlx"],
    device: Any = ...,
) -> Dataset[mx.array]: ...
@overload
def dataset(
    source: str,
    *,
    fields: Mapping[str, Field] | None = ...,
    columns: dict[str, Feature] | None = ...,
    target: Target | None = ...,
    drop_nulls: list[str] | None = ...,
    split: Split | None = ...,
    key: str | None = ...,
    con: Connection | None = ...,
    backend: str,
    device: Any = ...,
) -> Dataset[Any]: ...
def dataset(
    source: str,
    *,
    fields: Mapping[str, Field] | None = None,
    columns: dict[str, Feature] | None = None,
    target: Target | None = None,
    drop_nulls: list[str] | None = None,
    split: Split | None = None,
    key: str | None = None,
    con: Connection | None = None,
    backend: str = "numpy",
    device: Any = None,
) -> Dataset[Any]:
    """Load a DuckDB-readable source into split NumPy, JAX, Torch, or MLX arrays.

    ``fields`` maps names to vectors, matrices, or tensors. ``columns`` and
    ``target`` provide the ``X``/``y`` shorthand. Standardization, and the
    categories of category fields that need to learn them, use only the train
    fold. ``key`` is a SQL expression identifying each row, which assigns
    it to its fold. Without one, a row's position in the scan does.
    """
    if backend not in _BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; supported backends: {list(_BACKENDS)}")
    if device is not None and backend in _NO_DEVICE:
        raise ValueError(f"backend {backend!r} does not take a device argument")

    field_specs = _normalize_fields(fields, columns, target)
    _check_columns(field_specs)

    own_connection = con is None
    if own_connection:
        con = connect()
    try:
        raw_sql, ranges = _raw_sql(source, drop_nulls, split, key)
        resolved, categories = _resolve_categories(
            con,
            _quote_literal(source),
            f"WITH raw AS ({raw_sql})",
            _train_filter(ranges, "_bucket"),
            field_specs,
        )
        sql, ranges = _compile_sql(source, resolved, drop_nulls, split, key)
        # Columns of a one-dtype matrix are written side by side into its array.
        groups = {
            name: [f"{name}__{c}" for c in spec.columns]
            for name, spec in resolved.items()
            if isinstance(spec, Matrix) and len({f.dtype for f in spec.columns.values()}) == 1
        }
        bounds = [ranges[0][1], *(hi for _, _, hi in ranges)]
        # Each chunk's rows are copied once, into the arrays of their fold:
        # nothing full-size is stacked or gathered afterwards.
        parts = con.sql(sql).to_numpy_folds("_bucket", bounds, groups)
    finally:
        if own_connection:
            con.close()

    import numpy as np

    folds: dict[str, Fold[Any]] = {}
    for (name, _, _), arrays in zip(ranges, parts, strict=True):
        fold_fields: dict[str, Any] = {}
        for field_name, spec in field_specs.items():
            if isinstance(spec, Matrix) and field_name not in groups:
                # Mixed dtypes: NumPy picks the common one.
                arr = np.stack([arrays.pop(f"{field_name}__{c}") for c in spec.columns], axis=1)
            else:
                arr = arrays.pop(field_name)
            fold_fields[field_name] = _to_backend(arr, backend, device)
        folds[name] = Fold(backend=backend, _fields=fold_fields)
    # avg()/stddev_pop() over an empty train fold return NULL, which the
    # standardization would treat as a constant column instead of an empty fold.
    if (
        any(spec.standardize for _, _, spec in _iter_field_columns(field_specs))
        and folds["train"].n_rows == 0
    ):
        raise ValueError(
            "standardize=True requires at least one row in the 'train' fold "
            "after filtering/splitting, but it's empty; widen the split, check "
            "drop_nulls, or drop standardize=True"
        )
    return Dataset(folds=folds, field_specs=field_specs, categories=categories)
