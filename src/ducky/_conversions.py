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
    import numpy as np
    import pandas as pd
    import polars
    import pyarrow
    import torch

    from ._core import Chunk


class _ArrowSource(Protocol):
    """Anything implementing the Arrow C stream PyCapsule interface."""

    def __arrow_c_stream__(self, requested_schema: Any = None) -> Any: ...


class _ChunkSource(Protocol):
    """Anything that produces ducky ``Chunk`` objects on demand."""

    def fetch_chunk(self) -> Chunk | None: ...


def arrow(source: _ArrowSource) -> pyarrow.Table:
    """Return the result as a ``pyarrow.Table``."""
    import pyarrow as pa

    return pa.table(source)


def df(source: Any) -> Any:
    """Return the result as a ``pandas.DataFrame`` (via Arrow)."""
    return arrow(source).to_pandas()


def pl(source: Any, lazy: bool = False) -> Any:
    """Return the result as a polars ``DataFrame`` (or ``LazyFrame`` if ``lazy``)."""
    import polars

    frame = polars.from_arrow(source)
    return frame.lazy() if lazy else frame


def fetchnumpy(source: Any) -> dict[str, Any]:
    """Return the result as a dict of column name -> ``numpy`` array."""
    table = arrow(source)
    return {
        name: column.to_numpy(zero_copy_only=False)
        for name, column in zip(table.column_names, table.columns)
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

    parts: dict[str, list[np.ndarray]] = {}
    for batch in iter_batches(source, columns=columns):
        for name, arr in batch.items():
            parts.setdefault(name, []).append(np.array(arr, copy=True))
    return {name: np.concatenate(arrs) for name, arrs in parts.items()}


def to_torch(
    source: _ChunkSource,
    columns: Iterable[str] | None = None,
    device: torch.device | str | int | None = None,
) -> dict[str, torch.Tensor]:
    """Like ``to_numpy`` but returns ``torch.Tensor`` values.

    The CPU path goes through ``torch.from_dlpack`` (zero-copy from the numpy
    buffer); if ``device`` is given the tensor is copied onto that device.
    """
    import torch

    arrays = to_numpy(source, columns=columns)
    out = {name: torch.from_dlpack(arr) for name, arr in arrays.items()}
    if device is not None:
        out = {name: t.to(device) for name, t in out.items()}
    return out


def to_jax(
    source: _ChunkSource,
    columns: Iterable[str] | None = None,
    device: jax.Device | None = None,
) -> dict[str, jax.Array]:
    """Like ``to_numpy`` but returns ``jax.Array`` values.

    ``device`` is forwarded to ``jax.numpy.asarray`` (Array-API standard).
    On CPU JAX shares the numpy buffer; on accelerators the data is copied.
    """
    import jax.numpy as jnp

    arrays = to_numpy(source, columns=columns)
    return {name: jnp.asarray(arr, device=device) for name, arr in arrays.items()}
