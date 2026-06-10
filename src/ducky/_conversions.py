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


class _ChunkSource(Protocol):
    """Anything that produces ducky :class:`Chunk` objects on demand."""

    def to_numpy(self, columns: Iterable[str] | None = None) -> dict[str, np.ndarray]: ...

    def fetch_chunk(self) -> Chunk | None: ...


def arrow(source: ArrowSource) -> pyarrow.Table:
    """Return ``source`` as a ``pyarrow.Table``."""
    import pyarrow as pa

    return pa.table(source)


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


def fetchnumpy(source: ArrowSource) -> dict[str, np.ndarray]:
    """Return ``source`` as a ``{column: numpy.ndarray}`` dict."""
    table = arrow(source)
    return {
        name: column.to_numpy(zero_copy_only=False)
        for name, column in zip(table.column_names, table.columns, strict=True)
    }


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
    for chunk in chunks(source):
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

    for chunk in chunks(source):
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

    Loops chunks in C++ via :func:`to_numpy`; each tensor wraps the resulting numpy buffer via :func:`torch.from_numpy` (zero-copy on CPU).
    For memory-bounded streaming use :func:`iter_batches_torch`.
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

    for chunk in chunks(source):
        names = _select(chunk.columns, columns)
        yield {n: jnp.asarray(chunk.dlpack(n), device=device) for n in names}


def to_jax(
    source: _ChunkSource,
    columns: Iterable[str] | None = None,
    device: jax.Device | None = None,
) -> dict[str, jax.Array]:
    """Eagerly drain all chunks into ``{name: jax.Array}``.

    Loops chunks in C++ via :func:`to_numpy`; each array wraps the resulting numpy buffer via :func:`jax.numpy.asarray` (zero-copy on CPU).
    For memory-bounded streaming use :func:`iter_batches_jax`.
    """
    import jax.numpy as jnp

    arrays = source.to_numpy(columns)
    return {name: jnp.asarray(arr, device=device) for name, arr in arrays.items()}


def iter_batches_mlx(
    source: _ChunkSource,
    columns: Iterable[str] | None = None,
) -> Iterator[dict[str, mx.array]]:
    """Yield one ``{name: mlx.core.array}`` dict per chunk.

    Each array is built directly from the chunk's buffer via DLPack (zero-copy on the CPU backend).
    MLX uses a unified-memory model, so — unlike the JAX and torch variants — there is no per-array ``device`` argument; set the active device globally with ``mlx.core.set_default_device`` if needed.
    MLX has no float64, so ``DOUBLE`` columns arrive as float32.
    Chunks are released as soon as the caller advances the iterator.
    """
    import mlx.core as mx

    for chunk in chunks(source):
        names = _select(chunk.columns, columns)
        yield {n: mx.array(chunk.dlpack(n)) for n in names}


def to_mlx(
    source: _ChunkSource,
    columns: Iterable[str] | None = None,
) -> dict[str, mx.array]:
    """Eagerly drain all chunks into ``{name: mlx.core.array}``.

    Loops chunks in C++ via :func:`to_numpy`; each array wraps the resulting numpy buffer via :func:`mlx.core.array` (zero-copy on the CPU backend).
    MLX has no float64, so ``DOUBLE`` columns arrive as float32.
    For memory-bounded streaming use :func:`iter_batches_mlx`.
    """
    import mlx.core as mx

    arrays = source.to_numpy(columns)
    return {name: mx.array(arr) for name, arr in arrays.items()}
