"""Convert query results to Arrow and array-library objects."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import jax
    import mlx.core as mx
    import numpy as np
    import pandas as pd
    import polars
    import pyarrow
    import torch

    from ._core import Chunk


class ArrowSource(Protocol):
    """An object implementing the Arrow C stream interface."""

    def __arrow_c_stream__(self, requested_schema: Any = None) -> Any: ...

    def _arrow_chunk_stream(self) -> Any: ...


class _ChunkSource(Protocol):
    """Anything that produces ducky :class:`Chunk` objects on demand."""

    def to_numpy(self, columns: Iterable[str] | None = None) -> dict[str, np.ndarray]: ...

    def fetch_chunk(self) -> Chunk | None: ...


def arrow(source: ArrowSource) -> pyarrow.Table:
    """Return ``source`` as a ``pyarrow.Table``."""
    import pyarrow as pa

    return pa.table(source)


def arrow_batches(source: ArrowSource) -> pyarrow.RecordBatchReader:
    """Stream one Arrow record batch per DuckDB chunk."""
    import pyarrow as pa

    return pa.RecordBatchReader._import_from_c_capsule(source._arrow_chunk_stream())


class _CollectedArrowSource(Protocol):
    """A ducky :class:`Connection` that can execute a query straight into Arrow."""

    def _arrow_collected_stream(
        self, query: str, parameters: Any = ..., rows_per_batch: int = ...
    ) -> Any: ...


class _StreamingQuerySource(Protocol):
    """A ducky :class:`Connection` that can execute a query as a lazily-pulled Result."""

    def _query(self, query: str, parameters: Any = ..., streaming: bool = ...) -> ArrowSource: ...


def arrow_reader(
    con: _StreamingQuerySource,
    query: str,
    parameters: Any = None,
    rows_per_batch: int = 1_000_000,
) -> pyarrow.RecordBatchReader:
    """Execute ``query`` and stream batches of up to ``rows_per_batch`` rows."""
    if rows_per_batch < 1:
        raise ValueError("rows_per_batch must be positive")
    import pyarrow as pa

    # _query runs the cached statement without replacing the connection's
    # current result (what con.fetchall() reads). The reader streams: a later
    # statement on the connection closes it, and reading it then raises.
    result = con._query(query, parameters, streaming=True)
    chunk_reader = pa.RecordBatchReader._import_from_c_capsule(result._arrow_chunk_stream())

    def coalesced() -> Iterator[pyarrow.RecordBatch]:
        pending: list[pyarrow.RecordBatch] = []
        pending_rows = 0
        for batch in chunk_reader:
            pending.append(batch)
            pending_rows += batch.num_rows
            while pending_rows >= rows_per_batch:
                if len(pending) == 1:
                    head = pending[0]
                    tail = head.slice(rows_per_batch) if head.num_rows > rows_per_batch else None
                    yield head.slice(0, rows_per_batch) if tail is not None else head
                else:
                    merged = pa.Table.from_batches(pending).combine_chunks().to_batches()[0]
                    yield merged.slice(0, rows_per_batch)
                    tail = merged.slice(rows_per_batch)
                    tail = tail if tail.num_rows else None
                pending = [tail] if tail is not None else []
                pending_rows = tail.num_rows if tail is not None else 0
        if pending_rows:
            yield (
                pending[0]
                if len(pending) == 1
                else (pa.Table.from_batches(pending).combine_chunks().to_batches()[0])
            )

    return pa.RecordBatchReader.from_batches(chunk_reader.schema, coalesced())


def arrow_query(con: _CollectedArrowSource, query: str, parameters: Any = None) -> pyarrow.Table:
    """Execute ``query`` and return the complete result as an Arrow table."""
    import pyarrow as pa

    return pa.RecordBatchReader._import_from_c_capsule(
        con._arrow_collected_stream(query, parameters, 1_000_000)
    ).read_all()


def df(source: ArrowSource) -> pd.DataFrame:
    """Return ``source`` as a ``pandas.DataFrame`` (via Arrow)."""
    return arrow(source).to_pandas()


def pl(source: ArrowSource, lazy: bool = False) -> polars.DataFrame | polars.LazyFrame:
    """Return ``source`` as a polars ``DataFrame`` (or ``LazyFrame`` if ``lazy``)."""
    import polars

    # Not polars.from_arrow: since polars 2.0 it turns any non-pyarrow Arrow
    # stream into a struct Series.
    frame = polars.DataFrame(source)
    return frame.lazy() if lazy else frame


def fetchnumpy(source: _ChunkSource) -> dict[str, np.ndarray]:
    """Return numeric and temporal columns as NumPy arrays."""
    return source.to_numpy()


# Numeric / temporal columns only; for strings / nested types go through .arrow().
# NULL slots in the dense ndarrays hold raw buffer contents — filter or coalesce
# in SQL, or use iter_batches(with_validity=True), to handle them safely.


def chunks(source: _ChunkSource) -> Iterator[Chunk]:
    """Iterate over ``source`` one :class:`Chunk` at a time.

    Drains the underlying result.
    """
    while (chunk := source.fetch_chunk()) is not None:
        yield chunk


def _select(names: list[str], columns: Iterable[str] | None) -> list[str]:
    if columns is None:
        return names
    names_set = set(names)
    selected = list(columns)
    missing = [c for c in selected if c not in names_set]
    if missing:
        raise KeyError(f"unknown column(s): {missing!r}; available: {names!r}")
    return selected


def iter_batches(
    source: _ChunkSource,
    columns: Iterable[str] | None = None,
    with_validity: bool = False,
) -> Iterator[dict[str, np.ndarray] | dict[str, tuple[np.ndarray, np.ndarray | None]]]:
    """Yield NumPy columns per chunk, optionally with uint8 validity masks."""
    if columns is None and not with_validity:
        fetch_numpy_chunk = getattr(source, "fetch_numpy_chunk", None)
        if fetch_numpy_chunk is not None:
            while (batch := fetch_numpy_chunk()) is not None:
                yield batch
            return

    names: list[str] | None = None
    for chunk in chunks(source):
        if names is None:  # column names are constant across a result's chunks
            names = _select(chunk.columns, columns)
        if with_validity:
            yield {n: (chunk.column(n), chunk.validity(n)) for n in names}
        else:
            yield {n: chunk.column(n) for n in names}


def to_numpy(source: _ChunkSource, columns: Iterable[str] | None = None) -> dict[str, np.ndarray]:
    """Return selected columns as NumPy arrays."""
    return source.to_numpy(columns)


def iter_batches_torch(
    source: _ChunkSource,
    columns: Iterable[str] | None = None,
    device: torch.device | str | int | None = None,
) -> Iterator[dict[str, torch.Tensor]]:
    """Yield Torch tensors per chunk, moving them to ``device`` if set."""
    import torch

    names: list[str] | None = None
    for chunk in chunks(source):
        if names is None:
            names = _select(chunk.columns, columns)
        batch = {}
        for n in names:
            t = torch.from_dlpack(chunk.dlpack(n))
            if device is not None:
                t = t.to(device)
            batch[n] = t
        yield batch


def to_torch(
    source: _ChunkSource,
    columns: Iterable[str] | None = None,
    device: torch.device | str | int | None = None,
) -> dict[str, torch.Tensor]:
    """Return selected columns as Torch tensors."""
    import torch

    arrays = source.to_numpy(columns)
    if device is None:
        return {name: torch.from_numpy(arr) for name, arr in arrays.items()}
    return {name: torch.from_numpy(arr).to(device) for name, arr in arrays.items()}


def iter_batches_jax(
    source: _ChunkSource,
    columns: Iterable[str] | None = None,
    device: jax.Device | None = None,
) -> Iterator[dict[str, jax.Array]]:
    """Yield JAX arrays per chunk on ``device``."""
    import jax.numpy as jnp

    names: list[str] | None = None
    for chunk in chunks(source):
        if names is None:
            names = _select(chunk.columns, columns)
        yield {n: jnp.asarray(chunk.dlpack(n), device=device) for n in names}


def to_jax(
    source: _ChunkSource,
    columns: Iterable[str] | None = None,
    device: jax.Device | None = None,
) -> dict[str, jax.Array]:
    """Return selected columns as JAX arrays on ``device``."""
    import jax.numpy as jnp

    arrays = source.to_numpy(columns)
    return {name: jnp.asarray(arr, device=device) for name, arr in arrays.items()}


def iter_batches_mlx(
    source: _ChunkSource,
    columns: Iterable[str] | None = None,
) -> Iterator[dict[str, mx.array]]:
    """Yield MLX arrays per chunk. ``DOUBLE`` columns become float32."""
    names: list[str] | None = None
    for chunk in chunks(source):
        if names is None:
            names = _select(chunk.columns, columns)
        yield {n: chunk.mlx(n) for n in names}


def to_mlx(
    source: _ChunkSource,
    columns: Iterable[str] | None = None,
) -> dict[str, mx.array]:
    """Return selected columns as MLX arrays; ``DOUBLE`` becomes float32."""
    import mlx.core as mx

    arrays = source.to_numpy(columns)
    return {name: mx.array(arr) for name, arr in arrays.items()}
