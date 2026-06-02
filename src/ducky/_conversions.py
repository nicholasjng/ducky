"""Conversions from a ducky result into DataFrame/array libraries.

Each function takes any object implementing the Arrow C stream interface
(``__arrow_c_stream__``) — i.e. a ducky ``Result`` or ``Connection`` — and
imports its target library lazily, so none of these are hard dependencies. The
extension itself only knows how to produce the Arrow stream; everything here is
plain Python on top of it.

Note: consuming the stream drains the underlying result, so call exactly one of
these (or the ``fetch*`` methods) per executed query.
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

    Covers ducky's own ``Result``/``Connection`` and any third-party
    object exposing ``__arrow_c_stream__`` — pyarrow ``Table``, polars
    ``DataFrame``, pandas 3.x ``DataFrame``, etc.
    """

    def __arrow_c_stream__(self, requested_schema: Any = None) -> Any: ...


class _ChunkSource(Protocol):
    """Anything that produces ducky ``Chunk`` objects on demand."""

    def fetch_chunk(self) -> Chunk | None: ...


def arrow(source: ArrowSource) -> pyarrow.Table:
    """Return the result as a ``pyarrow.Table``."""
    import pyarrow as pa

    return pa.table(source)


def df(source: ArrowSource) -> pd.DataFrame:
    """Return the result as a ``pandas.DataFrame`` (via Arrow)."""
    return arrow(source).to_pandas()


def pl(source: ArrowSource, lazy: bool = False) -> polars.DataFrame | polars.LazyFrame:
    """Return the result as a polars ``DataFrame`` (or ``LazyFrame`` if ``lazy``)."""
    import polars

    # `polars.from_arrow` is typed `DataFrame | Series`, but an Arrow stream
    # always produces a multi-column DataFrame here.
    frame = polars.from_arrow(source)
    assert isinstance(frame, polars.DataFrame)
    return frame.lazy() if lazy else frame


def fetchnumpy(source: ArrowSource) -> dict[str, np.ndarray]:
    """Return the result as a dict of column name -> ``numpy`` array."""
    table = arrow(source)
    return {
        name: column.to_numpy(zero_copy_only=False)
        for name, column in zip(table.column_names, table.columns, strict=True)
    }


# ── ndarray / DLPack path ───────────────────────────────────────────────────
# Built on Chunk.column / Result.fetch_chunk (see src/cpp/chunk.cpp). Numeric
# and temporal columns only; for strings/nested types go through .arrow().
#
# NULL handling: the dense ndarrays are raw buffers — NULL slots hold whatever
# was in the column's storage at allocation time. If you want clean arrays,
# filter or coalesce in SQL (``WHERE x IS NOT NULL`` / ``coalesce(x, ...)``);
# DuckDB will then ship chunks with no validity mask at all. Use ``iter_batches
# (with_validity=True)`` if you need to keep the nulls in-band.


def chunks(source: _ChunkSource) -> Iterator[Chunk]:
    """Iterate over the result one ``Chunk`` at a time. Drains the result."""
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
    """Yield one dict per chunk: ``{name: ndarray}``.

    If ``with_validity`` is true, each value is a ``(values, mask)`` tuple,
    where ``mask`` is a uint8 ndarray (1=valid, 0=null) or ``None`` if the
    chunk's column has no nulls.
    """
    for chunk in chunks(source):
        names = _select(chunk.columns, columns)
        if with_validity:
            yield {n: (chunk.column(n), chunk.validity(n)) for n in names}
        else:
            yield {n: chunk.column(n) for n in names}


def to_numpy(source: _ChunkSource, columns: Iterable[str] | None = None) -> dict[str, np.ndarray]:
    """Eagerly concatenate all chunks into ``{name: numpy.ndarray}``.

    Numeric/temporal columns only — raises ``ducky.Error`` on string, list,
    struct, decimal, etc. Use ``.arrow()`` (or select only numeric columns)
    for those.
    """
    import numpy as np

    # `chunk.column(n)` is a view over the chunk's buffer, and `chunks()` frees
    # each chunk as the loop advances — so retain the chunks until after the
    # concatenate, which is then the single copy into the contiguous output.
    # (Copying per chunk first would copy every byte twice.) Peak memory is
    # unchanged: live chunks (1x) + result (1x), the same high-water mark as
    # holding per-chunk numpy copies (1x) + result (1x).
    live: list[Chunk] = []
    parts: dict[str, list[np.ndarray]] = {}
    for chunk in chunks(source):
        live.append(chunk)
        for name in _select(chunk.columns, columns):
            parts.setdefault(name, []).append(chunk.column(name))
    return {name: np.concatenate(arrs) for name, arrs in parts.items()}


def iter_batches_torch(
    source: _ChunkSource,
    columns: Iterable[str] | None = None,
    device: torch.device | str | int | None = None,
) -> Iterator[dict[str, torch.Tensor]]:
    """Yield one dict per chunk: ``{name: torch.Tensor}``.

    Each tensor is derived directly from the chunk's buffer via DLPack
    (zero-copy on CPU). If ``device`` is given each tensor is moved there
    before yielding. Chunks are released as soon as the caller advances the
    iterator, so peak memory is bounded by one chunk at a time.
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
    """Eagerly concatenate all chunks into ``{name: torch.Tensor}``.

    Uses ``iter_torch_batches`` internally, so no intermediate numpy
    materialization occurs; each chunk is converted via DLPack and released
    before the next is fetched.
    """
    import torch

    parts: dict[str, list[torch.Tensor]] = {}
    for batch in iter_batches_torch(source, columns=columns, device=device):
        for name, t in batch.items():
            parts.setdefault(name, []).append(t)
    return {name: torch.cat(ts) for name, ts in parts.items()}


def iter_batches_jax(
    source: _ChunkSource,
    columns: Iterable[str] | None = None,
    device: jax.Device | None = None,
) -> Iterator[dict[str, jax.Array]]:
    """Yield one dict per chunk: ``{name: jax.Array}``.

    Each array is derived directly from the chunk's buffer. On CPU JAX shares
    the buffer (zero-copy); on accelerators a transfer occurs. Chunks are
    released as soon as the caller advances the iterator.
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
    """Eagerly concatenate all chunks into ``{name: jax.Array}``.

    Uses ``iter_jax_batches`` internally, so no intermediate numpy
    materialization occurs.
    """
    import jax.numpy as jnp

    parts: dict[str, list[jax.Array]] = {}
    for batch in iter_batches_jax(source, columns=columns, device=device):
        for name, arr in batch.items():
            parts.setdefault(name, []).append(arr)
    return {name: jnp.concatenate(arrs) for name, arrs in parts.items()}


def iter_batches_mlx(
    source: _ChunkSource,
    columns: Iterable[str] | None = None,
) -> Iterator[dict[str, mx.array]]:
    """Yield one dict per chunk: ``{name: mlx.core.array}``.

    Each array is built directly from the chunk's buffer via DLPack (zero-copy
    on the CPU backend). MLX uses a unified-memory model, so — unlike the JAX
    and torch variants — there is no per-array ``device`` argument; set the
    active device globally with ``mlx.core.set_default_device`` if needed. MLX
    has no float64, so ``DOUBLE`` columns arrive as float32. Chunks are released
    as soon as the caller advances the iterator.
    """
    import mlx.core as mx

    for chunk in chunks(source):
        names = _select(chunk.columns, columns)
        yield {n: mx.array(chunk.dlpack(n)) for n in names}


def to_mlx(
    source: _ChunkSource,
    columns: Iterable[str] | None = None,
) -> dict[str, mx.array]:
    """Eagerly concatenate all chunks into ``{name: mlx.core.array}``.

    Uses ``iter_batches_mlx`` internally, so no intermediate numpy
    materialization occurs; each chunk is converted via DLPack and released
    before the next is fetched.
    """
    import mlx.core as mx

    parts: dict[str, list[mx.array]] = {}
    for batch in iter_batches_mlx(source, columns=columns):
        for name, arr in batch.items():
            parts.setdefault(name, []).append(arr)
    return {name: mx.concatenate(arrs) for name, arrs in parts.items()}
