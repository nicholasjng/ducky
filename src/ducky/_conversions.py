"""Conversions from a ducky result into DataFrame / array libraries.

Each function takes any object implementing the Arrow C stream interface (``__arrow_c_stream__``) — typically a ducky :class:`Result` or :class:`Connection`
— and imports its target library lazily, so none of these are hard dependencies.
The extension itself only knows how to produce the Arrow stream; everything here is plain Python on top of it.

Consuming the stream drains the underlying result, so call exactly one of these (or the ``fetch*`` methods) per executed query.
"""

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
    """Anything implementing the Arrow C stream PyCapsule interface.

    Covers ducky's own :class:`Result` / :class:`Connection` and any third-party object exposing ``__arrow_c_stream__`` — pyarrow ``Table``, polars ``DataFrame``, pandas 3.x ``DataFrame``, etc.
    """

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
    """Stream ``source`` as a ``pyarrow.RecordBatchReader`` (one batch per chunk).

    The lazy, Arrow-native counterpart of :func:`iter_batches`: it yields one
    ``RecordBatch`` per DuckDB chunk (~2048 rows) without materializing the whole
    table. Emitting a single Arrow object per chunk (rather than a dict of N
    column ndarrays) makes it cheaper than :func:`iter_batches` for wide results.
    Wraps ``source``'s per-chunk Arrow C stream, so it stays zero-copy where
    DuckDB's Arrow export is.
    """
    import pyarrow as pa

    # Use the per-chunk stream (not __arrow_c_stream__, which coalesces an
    # all-flat result into a single batch — defeating bounded-memory streaming).
    return pa.RecordBatchReader._import_from_c_capsule(source._arrow_chunk_stream())


class _CollectedArrowSource(Protocol):
    """A ducky :class:`Connection` that can execute a query straight into Arrow."""

    def _arrow_collected_stream(
        self, query: str, parameters: Any = ..., rows_per_batch: int = ...
    ) -> Any: ...


class _StreamingQuerySource(Protocol):
    """A ducky :class:`Connection` that can execute a query as a lazily-pulled Result."""

    def execute(
        self, query: str, parameters: Any = ..., streaming: bool = ...
    ) -> _StreamingQuerySource: ...

    @property
    def current_result(self) -> ArrowSource: ...


def arrow_reader(
    con: _StreamingQuerySource,
    query: str,
    parameters: Any = None,
    rows_per_batch: int = 1_000_000,
) -> pyarrow.RecordBatchReader:
    """Bounded-memory ``pyarrow.RecordBatchReader`` for ``query`` (backs Connection.arrow_reader).

    Executes ``query`` as a lazily-pulled (streaming) result and coalesces
    consecutive ~2048-row DuckDB chunks into batches of up to ``rows_per_batch``
    rows before handing them to the caller. Peak memory is bounded by one
    batch's worth of rows, not the whole result, regardless of how large the
    query is — unlike DuckDB's Arrow result collector (see :func:`arrow_query`),
    which must finish the whole query before yielding anything.
    ``parameters`` binds like ``execute()``.

    Use :func:`arrow_batches` instead if you want the smallest possible
    per-batch footprint (exactly one DuckDB chunk, ~2048 rows, no coalescing).
    """
    import pyarrow as pa

    result = con.execute(query, parameters, streaming=True).current_result
    chunk_reader = pa.RecordBatchReader._import_from_c_capsule(result._arrow_chunk_stream())

    def coalesced() -> Iterator[pyarrow.RecordBatch]:
        # Accumulate whole chunks (cheap: no copy) until they'd reach
        # rows_per_batch. The common case — rows_per_batch at or above one
        # DuckDB chunk (~2048 rows), e.g. the default — needs no copy at all:
        # a lone pending batch is sliced directly. combine_chunks only runs
        # when actually merging multiple physical chunks into one.
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
    """Born-as-Arrow ``pyarrow.Table`` for ``query`` (backs Connection.arrow(query)).

    The fastest path from a query to a whole Arrow table: executes with DuckDB's
    Arrow result collector (batches assembled *during* execution, no
    chunk->Arrow conversion pass) and concatenates the collected batches.
    Equivalent in result to ``con.execute(query, parameters).arrow()``, just
    faster. The return value is a whole Table, so memory is inherently
    unbounded here regardless of implementation — use :func:`arrow_reader` for
    a bounded, incrementally-consumed alternative. ``parameters`` binds like
    ``execute()``.
    """
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

    # polars.from_arrow is typed DataFrame | Series; an Arrow stream is always
    # the DataFrame branch.
    frame = polars.from_arrow(source)
    assert isinstance(frame, polars.DataFrame)
    return frame.lazy() if lazy else frame


def fetchnumpy(source: _ChunkSource) -> dict[str, np.ndarray]:
    """Return ``source`` as a ``{column: numpy.ndarray}`` dict.

    Delegates to the C++ :meth:`Result.to_numpy` chunk-direct path (numeric /
    temporal columns only); for strings / nested types go through :func:`arrow`.
    """
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
    """Yield one ``{name: ndarray}`` dict per chunk.

    Parameters
    ----------
    source : _ChunkSource
        A ducky :class:`Result` or :class:`Connection`.
    columns : Iterable[str], optional
        Subset of column names to yield; defaults to all columns.
    with_validity : bool, default False
        When True each value becomes a ``(values, mask)`` tuple.
        ``mask`` is a uint8 ndarray (1 = valid, 0 = NULL) or ``None`` if the chunk's column has no nulls.
    """
    # Fast path (all columns, no validity): the C++ side assembles the whole
    # {name: ndarray} dict per chunk, so we cross the binding once per chunk
    # instead of once per column. `fetch_numpy_chunk` lives on Result; the
    # _ChunkSource Protocol keeps `source` duck-typed.
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
    """Thin alias for ``source.to_numpy(columns)``.

    Kept so ``from ducky._conversions import to_numpy`` still resolves; the real
    implementation lives on :meth:`Result.to_numpy` (C++).
    """
    return source.to_numpy(columns)


def iter_batches_torch(
    source: _ChunkSource,
    columns: Iterable[str] | None = None,
    device: torch.device | str | int | None = None,
) -> Iterator[dict[str, torch.Tensor]]:
    """Yield one ``{name: torch.Tensor}`` dict per chunk.

    Each tensor is derived directly from the chunk's buffer via DLPack (zero-copy on CPU).
    If ``device`` is given each tensor is moved there before yielding.
    Chunks are released as soon as the caller advances the iterator, so peak memory is bounded to one chunk.
    """
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
    """Eagerly drain all chunks into ``{name: torch.Tensor}``.

    Goes through :func:`to_numpy` rather than accumulating :func:`iter_batches_torch`
    chunks in Python — measured ~15x faster on a multi-chunk result, since
    per-chunk dispatch overhead adds up fast. ``torch.from_numpy`` is
    zero-copy on CPU. For memory-bounded streaming use :func:`iter_batches_torch`.
    """
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
    """Yield one ``{name: jax.Array}`` dict per chunk.

    Each array is derived directly from the chunk's buffer.
    On CPU JAX shares the buffer (zero-copy); on accelerators a transfer occurs.
    Chunks are released as soon as the caller advances the iterator.
    """
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
    """Eagerly drain all chunks into ``{name: jax.Array}``.

    Goes through :func:`to_numpy` rather than per-chunk accumulation — see
    :func:`to_torch` for why. For memory-bounded streaming use :func:`iter_batches_jax`.
    """
    import jax.numpy as jnp

    arrays = source.to_numpy(columns)
    return {name: jnp.asarray(arr, device=device) for name, arr in arrays.items()}


def iter_batches_mlx(
    source: _ChunkSource,
    columns: Iterable[str] | None = None,
) -> Iterator[dict[str, mx.array]]:
    """Yield one ``{name: mlx.core.array}`` dict per chunk.

    Each array is built directly from the chunk's buffer via nanobind's native mlx ndarray dispatch (zero-copy on the CPU backend) — no separate DLPack + ``mx.array()`` step.
    MLX uses a unified-memory model, so — unlike the JAX and torch variants — there is no per-array ``device`` argument; set the active device globally with ``mlx.core.set_default_device`` if needed.
    MLX has no float64, so ``DOUBLE`` columns arrive as float32.
    Chunks are released as soon as the caller advances the iterator.
    """
    names: list[str] | None = None
    for chunk in chunks(source):
        if names is None:
            names = _select(chunk.columns, columns)
        yield {n: chunk.mlx(n) for n in names}


def to_mlx(
    source: _ChunkSource,
    columns: Iterable[str] | None = None,
) -> dict[str, mx.array]:
    """Eagerly drain all chunks into ``{name: mlx.core.array}``.

    Goes through :func:`to_numpy` rather than per-chunk accumulation — see
    :func:`to_torch` for why. MLX has no float64, so ``DOUBLE`` columns
    arrive as float32. For memory-bounded streaming use :func:`iter_batches_mlx`.
    """
    import mlx.core as mx

    arrays = source.to_numpy(columns)
    return {name: mx.array(arr) for name, arr in arrays.items()}
