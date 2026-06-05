"""Higher-level dataset / feature API for ML pipelines on top of DuckDB.

The output is a dict of named **fields**, each either a :class:`Matrix` (several
columns stacked into ``(n, d)``) or a :class:`Vector` (one column → ``(n,)``):

    ds = ducky.dataset(
        "https://…/titanic.csv",
        fields={
            "X": ducky.matrix({
                "pclass":   ducky.feature("Pclass"),
                "age":      ducky.feature("Age", standardize=True, dtype="f32"),
                "fare":     ducky.feature("Fare", standardize=True, dtype="f32"),
            }),
            "y": ducky.vector("Survived", dtype="f32"),
            "w": ducky.vector("sample_weight"),          # sample weights
        },
        drop_nulls=["Age"],
        split=ducky.split(0.8, seed=0),
        backend="jax",          # materialise folds straight into JAX (numpy default)
    )
    Xtr, ytr = ds.train.tensors()       # sugar for ds.train["X"], ds.train["y"]
    Xval, wval = ds.val["X"], ds.val["w"]

For the classic single-features / single-target case, pass ``columns=`` and
``target=`` instead of ``fields=`` — they desugar to a ``{"X": matrix(columns),
"y": vector(target)}`` field spec.

The ``backend`` chooses the array library each fold is materialised into
("numpy", "jax", "torch", "mlx"). The source is scanned once, streamed into the
backend via DLPack (zero host copy for the native backends), then each fold is
split out with an on-device integer gather and stacked into ``(n, d)`` on the
device. A load targets a single backend; for an unsupported framework, use
``backend="numpy"`` and convert from numpy at the cost of one copy.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Literal, Protocol, TypeVar, cast, overload

from ._core import Connection, Result, connect

if TYPE_CHECKING:
    import jax
    import mlx.core as mx
    import numpy as np
    import torch


@dataclass(frozen=True)
class Feature:
    """One column in a :class:`Matrix` field.

    ``expr`` is a SQL expression (not a Python expression) — typically just a
    column name, optionally quoted: ``"Pclass"``, ``'"Siblings/Spouses Aboard"'``,
    or a small expression like ``"Sex = 'male'"`` or
    ``"coalesce(Age, 30.0)"``.

    When ``standardize=True``, the column is z-scored using stats computed
    against the *train* fold only — never the full dataset.
    """

    expr: str
    dtype: str = "f32"
    standardize: bool = False


@dataclass(frozen=True)
class Target:
    """Legacy shorthand for a single-column label field.

    Equivalent to :class:`Vector` without ``standardize=``; preserved so the
    pre-``fields=`` API (``dataset(columns=..., target=...)``) keeps working.
    """

    expr: str
    dtype: str = "f32"


@dataclass(frozen=True)
class Vector:
    """A one-column field. Materialised as a 1-D array of shape ``(n,)``."""

    expr: str
    dtype: str = "f32"
    standardize: bool = False


@dataclass(frozen=True)
class Matrix:
    """A multi-column field. Materialised as a 2-D array of shape ``(n, d)``.

    ``columns`` maps a short name (used as part of the SQL alias) to a
    :class:`Feature` spec. The order of columns is preserved in the stack.
    """

    columns: dict[str, Feature] = field(default_factory=dict)


Field = Vector | Matrix


@dataclass(frozen=True)
class Split:
    """Random named-fold split based on a hash bucket over row position.

    ``fractions`` maps fold name → share, and must sum to 1.0. The most common
    case (single ``"train"``/``"val"`` split) has a shorthand: pass a plain
    float to :func:`split` and we expand it to
    ``{"train": x, "val": 1 - x}``.

    Reproducible across runs given a fixed ``seed``. Stratified splits and
    user-supplied fold columns are not supported in this v0 sketch.

    Standardisation statistics (see :class:`Feature` ``standardize=True``) are
    always computed against the fold named ``"train"`` if present; otherwise
    they're computed across the whole dataset.
    """

    fractions: dict[str, float] = field(default_factory=lambda: {"train": 0.8, "val": 0.2})
    seed: int = 0


def feature(expr: str, *, dtype: str = "f32", standardize: bool = False) -> Feature:
    """Shorthand for ``Feature(expr=..., dtype=..., standardize=...)``."""
    return Feature(expr=expr, dtype=dtype, standardize=standardize)


def target(expr: str, *, dtype: str = "f32") -> Target:
    """Shorthand for ``Target(expr=..., dtype=...)``."""
    return Target(expr=expr, dtype=dtype)


def vector(expr: str, *, dtype: str = "f32", standardize: bool = False) -> Vector:
    """Shorthand for ``Vector(expr=..., dtype=..., standardize=...)``."""
    return Vector(expr=expr, dtype=dtype, standardize=standardize)


def matrix(columns: dict[str, Feature]) -> Matrix:
    """Shorthand for ``Matrix(columns=...)``."""
    if not columns:
        raise ValueError("matrix() requires at least one column")
    return Matrix(columns=dict(columns))


def split(
    fractions: float | dict[str, float] = 0.8,
    *,
    seed: int = 0,
) -> Split:
    """Shorthand for ``Split(fractions=..., seed=...)``.

    ``fractions`` is either a single float in ``(0, 1)`` (expanded to
    ``{"train": x, "val": 1 - x}``) or a dict of fold name → share that sums
    to 1.0.
    """
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


# ── Dtype mapping ──────────────────────────────────────────────────────────

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


# ── Backend dispatch ───────────────────────────────────────────────────────
# A dataset is materialised into one array library. The heavy lifting (chunk →
# column) is the streaming DLPack converter in _conversions; the helpers below
# add the few backend-specific ops the split needs: pulling the bucket column
# to the host for index maths, an integer row-gather, and an axis-1 stack.
# Imports are lazy so none of jax/torch/mlx is a hard dependency.

Backend = Literal["numpy", "jax", "torch", "mlx"]
_BACKENDS: tuple[Backend, ...] = ("numpy", "jax", "torch", "mlx")
_NO_DEVICE: tuple[Backend, ...] = ("numpy", "mlx")

# Element type of a dataset's arrays — np.ndarray / jax.Array / torch.Tensor /
# mlx.core.array, depending on the chosen backend. The dataset() overloads map
# each backend literal to the concrete type so tensors() is precisely typed.


class AbstractArray(Protocol):
    """The minimal array surface ducky relies on across all backends.

    numpy / jax / torch / mlx arrays all satisfy this structurally (it's just
    ``shape``); it bounds :data:`ArrayT` so generic code can read ``.shape``
    without knowing the concrete backend type.
    """

    @property
    def shape(self) -> tuple[int, ...]: ...


ArrayT = TypeVar("ArrayT", bound=AbstractArray)


def _materialize(result: Result, backend: Backend, device: Any) -> dict[str, Any]:
    """Stream `result` into ``{column: array}`` in `backend`.

    Native backends wrap DuckDB's chunk buffers via DLPack with no host copy
    (numpy goes through a single concatenate); see ``ducky._conversions``.
    """
    from . import _conversions as conv

    if backend == "numpy":
        return conv.to_numpy(result)
    if backend == "jax":
        return conv.to_jax(result, device=device)
    if backend == "torch":
        return conv.to_torch(result, device=device)
    return conv.to_mlx(result)


def _bucket_to_host(arr: Any, backend: Backend) -> Any:
    """Return the (small) bucket column as a numpy array for index maths."""
    import numpy as np

    if backend == "torch":
        return arr.detach().cpu().numpy()
    return np.asarray(arr)  # numpy is a no-op; jax/mlx transfer to host


def _gather(arr: Any, idx: Any, backend: Backend) -> Any:
    """Gather rows `idx` (a 1-D numpy int array) from `arr` along axis 0."""
    if backend == "torch":
        import torch

        return arr[torch.as_tensor(idx, device=arr.device)]
    if backend == "mlx":
        import mlx.core as mx

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
        import mlx.core as mx

        return mx.stack(cols, axis=1)
    import numpy as np

    return np.stack(cols, axis=1)


# ── Output: Fold + Dataset ─────────────────────────────────────────────────


@dataclass
class Fold(Generic[ArrayT]):
    """One side of a split — a dict of named field arrays.

    Each field is an array in the dataset's ``backend`` (numpy/jax/torch/mlx);
    ``ArrayT`` is that array type. A :class:`Matrix` field is materialised as a
    2-D ``(n, d)`` array, a :class:`Vector` field as a 1-D ``(n,)`` array.

    Access fields by name with ``fold["X"]`` or as attributes (``fold.X``).
    Several single-character field names can be requested at once via a
    "swizzle" attribute — ``fold.Xy`` returns ``(fold["X"], fold["y"])``.
    """

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
        # __getattr__ only fires when normal lookup misses, so dataclass
        # attributes like ``backend`` and ``_fields`` aren't routed here.
        # Skip dunder/private lookups so things like ``__deepcopy__`` raise
        # cleanly instead of trying to swizzle the leading underscore.
        if name.startswith("_"):
            raise AttributeError(name)
        fields = self.__dict__.get("_fields")
        if not fields:
            raise AttributeError(name)
        if name in fields:
            return fields[name]
        # Vector-style swizzle: every char of ``name`` must be a single-char
        # field name. Returns a tuple in the order written.
        if len(name) > 1 and all(ch in fields for ch in name):
            return tuple(fields[ch] for ch in name)
        raise AttributeError(name)

    def tensors(self) -> tuple[ArrayT, ArrayT]:
        """Sugar returning ``(self["X"], self["y"])`` for the classic case."""
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
        """Yield ``(X_batch, y_batch)`` slices of ``self.tensors()``.

        Materialises the fold's ``X``/``y`` fields once up front (cheap — same
        memory as ``tensors()``); each batch is then a gather along axis 0 in
        the dataset's backend. The shuffle permutation is always computed on
        the host (cheap, deterministic) and applied as a backend gather.

        ``shuffle=True`` permutes row order once per call (i.e. once per
        epoch if you wrap this in a ``for epoch in range(...)``). Pass a
        fresh ``seed`` per epoch for a deterministic but varying shuffle, or
        ``seed=None`` for non-deterministic.
        """
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
    """A loaded dataset, split into one or more named folds.

    ``ArrayT`` is the array type of the folds (set by the load ``backend``).

    The most common access patterns:

    - ``ds.train`` / ``ds.val`` / ``ds.test`` — convenience aliases for the
      standard fold names (return ``None`` if that fold doesn't exist).
    - ``ds.folds["custom"]`` or ``ds["custom"]`` — arbitrary fold names.
    - Iterating ``ds.folds`` gives all available folds in declaration order.

    When the spec includes no split, all rows land in a single ``"train"`` fold.
    """

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
        """Column names of the ``"X"`` matrix field (legacy shorthand path)."""
        spec = self.field_specs.get("X")
        if isinstance(spec, Matrix):
            return list(spec.columns)
        raise AttributeError("feature_names is only defined when field 'X' is a Matrix")

    @property
    def target_name(self) -> str:
        """The expression of the ``"y"`` vector field (legacy shorthand path)."""
        spec = self.field_specs.get("y")
        if isinstance(spec, Vector):
            return spec.expr
        raise AttributeError("target_name is only defined when field 'y' is a Vector")

    def __getitem__(self, name: str) -> Fold[ArrayT]:
        return self.folds[name]

    def __repr__(self) -> str:
        parts = ", ".join(f"{name}={f.n_rows}" for name, f in self.folds.items())
        return f"Dataset({parts} rows, fields={list(self.field_specs)!r})"


# ── SQL compiler ───────────────────────────────────────────────────────────


# 1000 buckets gives 0.1% resolution on fold fractions — enough for anything
# users want to express in v0.
_N_BUCKETS = 1000


def _fold_ranges(fractions: dict[str, float]) -> list[tuple[str, int, int]]:
    """Cumulative-threshold conversion: each fold gets a [lo, hi) bucket range.
    The last fold absorbs any rounding drift so the ranges always tile [0, N).
    """
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
    """Yield ``(alias, field_name, spec)`` for every materialised column.

    Vectors map to a single column whose alias is the field name; each matrix
    column ``c`` maps to alias ``{field_name}__{c}`` and its inner Feature.
    """
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
) -> tuple[str, list[tuple[str, int, int]]]:
    """Compile a dataset spec into one SQL query.

    Returns ``(sql, fold_ranges)``. The query emits one row per source row,
    with one column per (matrix-column / vector) field-qualified alias, plus
    ``_bucket`` (in ``[0, _N_BUCKETS)``). Standardisation, when requested,
    references a ``stats`` CTE computed against the ``"train"`` fold only —
    never the full dataset.
    """
    if not isinstance(source, str):
        raise NotImplementedError("source must be a string URL/path in this sketch")
    if not fields:
        raise ValueError("at least one field is required")

    columns = list(_iter_field_columns(fields))
    if not columns:
        raise ValueError("fields must contain at least one column")

    aliases = [a for a, _, _ in columns]
    if len(set(aliases)) != len(aliases):
        # Aliases are derived from field + sub-column names; collisions only
        # occur if the user picks a sub-column name with a "__" separator that
        # mirrors another field's name. Surface it explicitly.
        dupes = sorted({a for a in aliases if aliases.count(a) > 1})
        raise ValueError(f"duplicate field/column aliases: {dupes}")

    from_clause = f"'{source}'"

    if split_spec is not None:
        ranges = _fold_ranges(split_spec.fractions)
        bucket_expr = f"(hash(row_number() OVER () + {split_spec.seed}) % {_N_BUCKETS})"
    else:
        ranges = [("train", 0, _N_BUCKETS)]
        bucket_expr = "0::BIGINT"

    null_filter = (
        " WHERE " + " AND ".join(f"{c} IS NOT NULL" for c in drop_nulls) if drop_nulls else ""
    )
    raw_sql = f"SELECT *, {bucket_expr} AS _bucket FROM {from_clause}{null_filter}"

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
                stats_cols.append(f"avg({spec.expr}) AS _{alias}_mean")
                stats_cols.append(f"stddev_pop({spec.expr}) AS _{alias}_std")
        stats_sql = f"SELECT {', '.join(stats_cols)} FROM raw WHERE {is_train}"
        ctes = f"WITH raw AS ({raw_sql}), stats AS ({stats_sql})"
        from_outer = "raw, stats"
    else:
        ctes = f"WITH raw AS ({raw_sql})"
        from_outer = "raw"

    select_items: list[str] = []
    for alias, _, spec in columns:
        if spec.standardize:
            expr = f"({spec.expr} - stats._{alias}_mean) / stats._{alias}_std"
        else:
            expr = spec.expr
        select_items.append(f"CAST({expr} AS {_to_duckdb_dtype(spec.dtype)}) AS {alias}")
    select_items.append("CAST(_bucket AS BIGINT) AS _bucket")

    sql = f"{ctes} SELECT {', '.join(select_items)} FROM {from_outer}"
    return sql, ranges


def _assemble_field(name: str, spec: Field, columns_data: dict[str, Any], backend: Backend) -> Any:
    """Stack a matrix field's columns / return a vector field's lone column."""
    if isinstance(spec, Vector):
        return columns_data[name]
    cols = [columns_data[f"{name}__{c}"] for c in spec.columns]
    return _stack(cols, backend)


# ── Public API ─────────────────────────────────────────────────────────────


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
    """Load a remote / local table as a feature-engineered dataset.

    ``source`` is a string passed straight into ``FROM '…'`` — URL, local path,
    or anything DuckDB's auto-detection can read (CSV, Parquet, JSON, …).

    The dataset shape is described by ``fields`` — a dict mapping each output
    field name to either a :class:`Matrix` (several stacked feature columns) or
    a :class:`Vector` (one column). For the classic single-features /
    single-target case, ``columns=`` and ``target=`` are kept as shorthand that
    desugar to ``fields={"X": matrix(columns), "y": vector(target)}``.

    ``backend`` selects the array library each fold is materialised into —
    ``"numpy"`` (default), ``"jax"``, ``"torch"`` or ``"mlx"`` — and fixes the
    returned ``Dataset``'s element type. The source is scanned once and streamed
    into the backend via DLPack (zero host copy for the native backends); each
    fold is then split out with an on-device integer gather. ``device`` is
    forwarded to the jax/torch converters (DuckDB feeds them from the host, so
    the chunks land there); ``"numpy"`` and ``"mlx"`` do not take a device.
    """
    if backend not in _BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; pick one of {list(_BACKENDS)}")
    backend = cast(Backend, backend)  # validated above
    if device is not None and backend in _NO_DEVICE:
        raise ValueError(f"backend {backend!r} does not take a device argument")

    field_specs = _normalize_fields(fields, columns, target)

    sql, ranges = _compile_sql(source, field_specs, drop_nulls, split)
    own_connection = con is None
    if own_connection:
        con = connect()
    try:
        columns_data = _materialize(con.sql(sql), backend, device)
    finally:
        if own_connection:
            con.close()

    import numpy as np

    # The bucket column is metadata; pull it to the host to compute each fold's
    # row indices, then gather the feature/target columns on the backend device.
    bucket = _bucket_to_host(columns_data.pop("_bucket"), backend)
    folds: dict[str, Fold[Any]] = {}
    for name, lo, hi in ranges:
        idx = np.nonzero((bucket >= lo) & (bucket < hi))[0]
        gathered = {k: _gather(v, idx, backend) for k, v in columns_data.items()}
        assembled = {
            field_name: _assemble_field(field_name, spec, gathered, backend)
            for field_name, spec in field_specs.items()
        }
        folds[name] = Fold(backend=backend, _fields=assembled)
    return Dataset(folds=folds, field_specs=field_specs)
