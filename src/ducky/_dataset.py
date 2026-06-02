"""Higher-level dataset / feature API for ML pipelines on top of DuckDB.

Sketch — surface plus a minimal SQL compiler. The goal is to demonstrate the
shape of the API; advanced features (stratified split, fillna='mean', custom
fold lists, streaming batches) are deliberately left out for now.

Typical use:

    ds = ducky.dataset(
        "https://…/titanic.csv",
        columns={
            "pclass":   ducky.feature("Pclass",                  dtype="f32"),
            "sex_male": ducky.feature("Sex = 'male'",            dtype="f32"),
            "age":      ducky.feature("Age",  standardize=True,  dtype="f32"),
            "sibsp":    ducky.feature('"Siblings/Spouses Aboard"'),
            "parch":    ducky.feature('"Parents/Children Aboard"'),
            "fare":     ducky.feature("Fare", standardize=True,  dtype="f32"),
        },
        target=ducky.target("Survived", dtype="f32"),
        drop_nulls=["Age"],
        split=ducky.split(0.8, seed=0),
        backend="jax",          # materialise folds straight into JAX (numpy default)
    )
    Xtr, ytr = ds.train.tensors()
    Xval, yval = ds.val.tensors()

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
    """One column in the output table.

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
    """The label column. Always emitted to the result; not included in X."""

    expr: str
    dtype: str = "f32"


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
    """One side of a train/val split — feature columns + target column.

    The columns are arrays in the dataset's ``backend`` (numpy/jax/torch/mlx);
    ``ArrayT`` is that array type. ``arrays`` is a dict view keyed by
    feature/target name; ``tensors()`` stacks the feature columns into a single
    ``(n_rows, n_features)`` array on the backend's device.
    """

    feature_names: list[str]
    target_name: str
    backend: Backend
    _arrays: dict[str, ArrayT]

    @property
    def arrays(self) -> dict[str, ArrayT]:
        return dict(self._arrays)

    @property
    def n_rows(self) -> int:
        return int(next(iter(self._arrays.values())).shape[0])

    def tensors(self) -> tuple[ArrayT, ArrayT]:
        """Returns ``(X, y)``: features stacked into ``(n, d)``, target ``(n,)``.

        The stack runs on the backend's device — no host round-trip for the
        native backends.
        """
        X = _stack([self._arrays[name] for name in self.feature_names], self.backend)
        y = self._arrays[self.target_name]
        return X, y

    def batches(
        self,
        batch_size: int,
        *,
        shuffle: bool = False,
        seed: int | None = None,
        drop_last: bool = False,
    ) -> Iterator[tuple[ArrayT, ArrayT]]:
        """Yield ``(X_batch, y_batch)`` slices of ``self.tensors()``.

        Materialises the fold into one ``(X, y)`` pair up front (cheap — same
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
        return (
            f"Fold(n_rows={self.n_rows}, backend={self.backend!r}, "
            f"features={self.feature_names!r}, target={self.target_name!r})"
        )


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
    def feature_names(self) -> list[str]:
        return next(iter(self.folds.values())).feature_names

    @property
    def target_name(self) -> str:
        return next(iter(self.folds.values())).target_name

    def __getitem__(self, name: str) -> Fold[ArrayT]:
        return self.folds[name]

    def __repr__(self) -> str:
        parts = ", ".join(f"{name}={f.n_rows}" for name, f in self.folds.items())
        return f"Dataset({parts} rows, features={self.feature_names!r})"


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


def _compile_sql(
    source: str,
    columns: dict[str, Feature],
    target_spec: Target,
    drop_nulls: list[str] | None,
    split_spec: Split | None,
) -> tuple[str, list[str], str, list[tuple[str, int, int]]]:
    """Compile a dataset spec into one SQL query.

    Returns ``(sql, feature_names, target_name, fold_ranges)``. The query emits
    one row per source row, with feature columns + ``_y`` + ``_bucket`` (in
    ``[0, _N_BUCKETS)``). Standardisation, when requested, references a
    ``stats`` CTE computed against the ``"train"`` fold only — never the full
    dataset.
    """
    if not isinstance(source, str):
        raise NotImplementedError("source must be a string URL/path in this sketch")
    if not columns:
        raise ValueError("at least one feature column is required")

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

    needs_stats = any(f.standardize for f in columns.values())
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
        for name, feat in columns.items():
            if feat.standardize:
                stats_cols.append(f"avg({feat.expr}) AS _{name}_mean")
                stats_cols.append(f"stddev_pop({feat.expr}) AS _{name}_std")
        stats_sql = f"SELECT {', '.join(stats_cols)} FROM raw WHERE {is_train}"
        ctes = f"WITH raw AS ({raw_sql}), stats AS ({stats_sql})"
        from_outer = "raw, stats"
    else:
        ctes = f"WITH raw AS ({raw_sql})"
        from_outer = "raw"

    select_items: list[str] = []
    for name, feat in columns.items():
        if feat.standardize:
            expr = f"({feat.expr} - stats._{name}_mean) / stats._{name}_std"
        else:
            expr = feat.expr
        select_items.append(f"CAST({expr} AS {_to_duckdb_dtype(feat.dtype)}) AS {name}")
    select_items.append(f"CAST({target_spec.expr} AS {_to_duckdb_dtype(target_spec.dtype)}) AS _y")
    select_items.append("CAST(_bucket AS BIGINT) AS _bucket")

    sql = f"{ctes} SELECT {', '.join(select_items)} FROM {from_outer}"
    return sql, list(columns.keys()), "_y", ranges


# ── Public API ─────────────────────────────────────────────────────────────


@overload
def dataset(
    source: str,
    *,
    columns: dict[str, Feature],
    target: Target,
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
    columns: dict[str, Feature],
    target: Target,
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
    columns: dict[str, Feature],
    target: Target,
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
    columns: dict[str, Feature],
    target: Target,
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
    columns: dict[str, Feature],
    target: Target,
    drop_nulls: list[str] | None = ...,
    split: Split | None = ...,
    con: Connection | None = ...,
    backend: str,
    device: Any = ...,
) -> Dataset[Any]: ...
def dataset(
    source: str,
    *,
    columns: dict[str, Feature],
    target: Target,
    drop_nulls: list[str] | None = None,
    split: Split | None = None,
    con: Connection | None = None,
    backend: str = "numpy",
    device: Any = None,
) -> Dataset[Any]:
    """Load a remote / local table as a feature-engineered dataset.

    ``source`` is a string passed straight into ``FROM '…'`` — URL, local path,
    or anything DuckDB's auto-detection can read (CSV, Parquet, JSON, …).

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

    sql, feature_names, target_name, ranges = _compile_sql(
        source, columns, target, drop_nulls, split
    )
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
        folds[name] = Fold(feature_names, target_name, backend, gathered)
    return Dataset(folds=folds)
