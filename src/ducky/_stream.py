"""Stream shuffled, sharded training batches from DuckDB-readable data."""

from __future__ import annotations

import contextlib
import queue
import sys
import threading
from collections.abc import Callable, Generator, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, overload

from ._core import Connection, Error, connect
from ._dataset import (
    _BACKENDS,
    _N_BUCKETS,
    _NO_DEVICE,
    AbstractArray,
    Feature,
    Field,
    Matrix,
    Split,
    Target,
    _bucket_expr,
    _check_columns,
    _fold_ranges,
    _normalize_fields,
    _null_filter,
    _quote_literal,
    _resolve_categories,
    _row_id_expr,
    _select_item,
)

if TYPE_CHECKING:
    import jax
    import mlx.core as mx
    import numpy as np
    import torch


@dataclass(frozen=True)
class Shuffle:
    """Shuffle within a buffer of ``buffer`` batches, or globally if it is None."""

    buffer: int | None = 64
    seed: int = 0


def shuffle(buffer: int | Literal["global"] = 64, *, seed: int = 0) -> Shuffle:
    """Shuffle within a buffer of ``buffer`` batches, or ``"global"``-ly.

    A buffer is cheap but only mixes rows that arrive close together. A global
    shuffle sorts every epoch by a hash of each row and the epoch: exact, and
    DuckDB spills the sort to disk when it does not fit in memory, but it costs
    a full sort per epoch.
    """
    if buffer == "global":
        return Shuffle(buffer=None, seed=seed)
    if isinstance(buffer, bool) or not isinstance(buffer, int) or buffer < 1:
        raise ValueError(f"shuffle buffer must be a positive int or 'global'; got {buffer!r}")
    return Shuffle(buffer=buffer, seed=seed)


def _auto_shard() -> tuple[int, int, bool]:
    """This process's (rank, world size) across DDP ranks and DataLoader
    workers, and whether there are several DDP ranks."""
    torch = sys.modules.get("torch")
    if torch is None:  # nothing to shard across without torch
        return 0, 1, False
    import torch.utils.data

    rank, world = 0, 1
    dist = torch.distributed
    if dist.is_available() and dist.is_initialized():
        rank, world = dist.get_rank(), dist.get_world_size()
    distributed = world > 1
    info = torch.utils.data.get_worker_info()
    if info is not None:
        rank, world = rank * info.num_workers + info.id, world * info.num_workers
    return rank, world, distributed


class _Batcher:
    """Cut blocks of rows into batches, optionally through a shuffle buffer.

    Rows wait in preallocated host arrays. Without shuffling, a batch is the
    oldest ``batch_size`` rows. With a buffer of ``capacity`` rows, a batch is
    drawn at random once the buffer is full, and the newest rows fill the holes.
    """

    def __init__(
        self,
        batch_size: int,
        drop_last: bool,
        capacity: int | None,
        rng: np.random.Generator | None,
    ) -> None:
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.capacity = capacity
        self.rng = rng
        self.bufs: dict[str, np.ndarray] = {}
        self.n = 0

    def push(self, block: dict[str, np.ndarray]) -> Iterator[dict[str, np.ndarray]]:
        k = len(next(iter(block.values())))
        self._reserve(block, self.n + k)
        for name, arr in block.items():
            self.bufs[name][self.n : self.n + k] = arr
        self.n += k
        if self.capacity is None:
            while self.n >= self.batch_size:
                yield self._take_front()
        else:
            while self.n >= max(self.capacity, self.batch_size):
                yield self._take_random()

    def finish(self) -> Iterator[dict[str, np.ndarray]]:
        if self.capacity is not None and self.n and self.rng is not None:
            perm = self.rng.permutation(self.n)
            for buf in self.bufs.values():
                buf[: self.n] = buf[perm]
        while self.n >= self.batch_size:
            yield self._take_front()
        if self.n and not self.drop_last:
            yield {name: buf[: self.n].copy() for name, buf in self.bufs.items()}
            self.n = 0

    def _reserve(self, block: dict[str, np.ndarray], rows: int) -> None:
        import numpy as np

        have = len(next(iter(self.bufs.values()))) if self.bufs else 0
        if rows <= have:
            return
        cap = max(rows, (self.capacity or 0) + self.batch_size, 2 * have)
        for name, arr in block.items():
            grown = np.empty((cap, *arr.shape[1:]), dtype=arr.dtype)
            if name in self.bufs:
                grown[: self.n] = self.bufs[name][: self.n]
            self.bufs[name] = grown

    def _take_front(self) -> dict[str, np.ndarray]:
        bs, n = self.batch_size, self.n
        out = {name: buf[:bs].copy() for name, buf in self.bufs.items()}
        for buf in self.bufs.values():
            buf[: n - bs] = buf[bs:n]
        self.n = n - bs
        return out

    def _take_random(self) -> dict[str, np.ndarray]:
        import numpy as np

        assert self.rng is not None
        bs, n = self.batch_size, self.n
        picked = np.sort(self.rng.choice(n, bs, replace=False))
        out = {name: buf[picked] for name, buf in self.bufs.items()}
        # Move the unpicked rows among the last `bs` into the picked rows'
        # holes below them; both sets have the same size.
        tail = n - bs
        holes = picked[picked < tail]
        picked_in_tail = np.zeros(bs, dtype=bool)
        picked_in_tail[picked[picked >= tail] - tail] = True
        movers = tail + np.flatnonzero(~picked_in_tail)
        for buf in self.bufs.values():
            buf[holes] = buf[movers]
        self.n = tail
        return out


class _Failure:
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


_DONE = object()


def _prefetched(
    produce: Callable[[list[Connection]], Iterator[dict[str, np.ndarray]]],
    depth: int,
    interruptible: bool,
) -> Iterator[dict[str, np.ndarray]]:
    """Run ``produce`` on a thread, keeping up to ``depth`` batches ready."""
    ready: queue.Queue[Any] = queue.Queue(maxsize=depth)
    stop = threading.Event()
    connections: list[Connection] = []  # the producer's, once open

    def put(item: Any) -> bool:
        while not stop.is_set():
            try:
                ready.put(item, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def run() -> None:
        try:
            for batch in produce(connections):
                if not put(batch):
                    return
            put(_DONE)
        except BaseException as exc:  # noqa: BLE001  # re-raised by the consumer
            put(_Failure(exc))

    thread = threading.Thread(target=run, name="ducky-stream", daemon=True)
    thread.start()
    try:
        while (item := ready.get()) is not _DONE:
            if isinstance(item, _Failure):
                raise item.exc
            yield item
    finally:
        stop.set()
        if thread.is_alive() and interruptible and connections:
            # Cut short a long fetch, such as a global shuffle's sort.
            with contextlib.suppress(Error):  # the producer closed it meanwhile
                connections[0].interrupt()
        thread.join()


def _to_backend(batch: dict[str, np.ndarray], backend: str, device: Any) -> dict[str, Any]:
    if backend == "numpy":
        return batch
    if backend == "torch":
        import torch

        if device is None:
            return {name: torch.from_numpy(arr) for name, arr in batch.items()}
        return {name: torch.from_numpy(arr).to(device) for name, arr in batch.items()}
    if backend == "jax":
        import jax.numpy as jnp

        return {name: jnp.asarray(arr, device=device) for name, arr in batch.items()}
    import mlx.core as mx

    return {name: mx.array(arr) for name, arr in batch.items()}


def _register_with_torch() -> None:
    """Make streams torch ``IterableDataset``s, which DataLoader requires of
    iterable inputs, once torch is in use; ducky doesn't import torch itself."""
    if "torch" in sys.modules:
        import torch.utils.data

        torch.utils.data.IterableDataset.register(Stream)


class Stream[ArrayT: AbstractArray]:
    """Batches of one fold, streamed from the source; iterate once per epoch.

    :attr:`categories` holds each category field's categories, indexed by its
    codes. Each full pass advances :attr:`epoch`, which reseeds the shuffle. Under
    DataLoader workers, call :meth:`set_epoch` every epoch instead: workers
    iterate copies, so the stream itself never sees an epoch end.
    """

    def __init__(
        self,
        source: str,
        field_specs: dict[str, Field],
        *,
        batch_size: int,
        fold: str,
        split: Split | None,
        key: str | None,
        drop_nulls: list[str] | None,
        shuffle: Shuffle | None,
        drop_last: bool,
        shard: tuple[int, int] | None,
        even_shards: bool | None,
        backend: str,
        device: Any,
        prefetch: int,
        read_ahead: str,
        con: Connection | None,
    ) -> None:
        self.source = source
        self.field_specs = field_specs
        self.batch_size = batch_size
        self.fold = fold
        self.split = split
        self.key = key
        self.drop_nulls = drop_nulls
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.shard = shard
        self.even_shards = even_shards
        self.backend = backend
        self.device = device
        self.prefetch = prefetch
        self.read_ahead = read_ahead
        self._con = con
        self._columns = _check_columns(field_specs)
        ranges = _fold_ranges(split.fractions) if split is not None else [("train", 0, _N_BUCKETS)]
        self._ranges = {name: (lo, hi) for name, lo, hi in ranges}
        if fold not in self._ranges:
            raise ValueError(f"unknown fold {fold!r}; folds: {list(self._ranges)}")
        self.epoch = 0
        self._batch = 0  # yielded this epoch
        self._resume = 0  # to skip, set by load_state_dict
        self._plans: dict[tuple[int, bool], tuple[list[str] | None, int | None]] = {}
        # Computed here, so DataLoader workers inherit them instead of each
        # scanning the source again.
        source_sql = _quote_literal(source)
        train = self._in_fold("train") if "train" in self._ranges else None
        conn, own = self._connect()
        try:
            resolved, self.categories = _resolve_categories(
                conn, source_sql, self._from(source_sql), train, field_specs
            )
            self._columns = _check_columns(resolved)
            self._stats = self._standardization(conn)
        finally:
            if own:
                conn.close()

    def set_epoch(self, epoch: int) -> None:
        """Start the next iteration at the beginning of ``epoch``."""
        self.epoch = epoch
        self._batch = self._resume = 0

    def state_dict(self) -> dict[str, int]:
        """The epoch and the batches of it this stream has yielded, for resuming."""
        return {"epoch": self.epoch, "batch": self._batch}

    def load_state_dict(self, state: dict[str, int]) -> None:
        """Resume where :meth:`state_dict` was taken; the next iteration skips
        the batches already yielded (reading, not training on, them). Without
        it, an iteration cut short is not resumed: the next starts its epoch over."""
        self.epoch = int(state["epoch"])
        self._batch = self._resume = int(state["batch"])

    def __iter__(self) -> Generator[dict[str, ArrayT], None, None]:
        epoch, skip = self.epoch, self._resume
        self._batch, self._resume = skip, 0
        if self.prefetch:
            batches = _prefetched(
                lambda cons: self._produce(epoch, cons), self.prefetch, self._con is None
            )
        else:
            batches = self._produce(epoch, [])
        for i, batch in enumerate(batches):
            if i < skip:
                continue
            self._batch = i + 1
            yield _to_backend(batch, self.backend, self.device)
        self.epoch, self._batch = epoch + 1, 0

    def __repr__(self) -> str:
        return (
            f"Stream(fold={self.fold!r}, batch_size={self.batch_size}, "
            f"fields={list(self.field_specs)!r}, backend={self.backend!r}, epoch={self.epoch})"
        )

    def _connect(self) -> tuple[Connection, bool]:
        if self._con is not None:
            return self._con, False
        # How far DuckDB executes ahead of the reader; at the default (10 MiB)
        # wide rows such as embeddings stream mostly single-threaded.
        con = connect()
        con.execute(f"SET streaming_buffer_size = {_quote_literal(self.read_ahead)}")
        return con, True

    def _from(self, from_clause: str) -> str:
        """CTEs numbering the rows of ``from_clause`` and assigning their folds."""
        numbered = (
            f"SELECT *, {_row_id_expr(self.key)} AS _ducky_row "
            f"FROM {from_clause}{_null_filter(self.drop_nulls)}"
        )
        bucket = _bucket_expr("_ducky_row", self.split, self.key) if self.split is not None else "0"
        return (
            f"WITH numbered AS ({numbered}), "
            f"raw AS (SELECT *, {bucket} AS _ducky_bucket FROM numbered)"
        )

    def _in_fold(self, fold: str) -> str:
        lo, hi = self._ranges[fold]
        return f"_ducky_bucket >= {lo} AND _ducky_bucket < {hi}"

    def _standardization(self, con: Connection) -> dict[str, tuple[str, str]]:
        """Train-fold mean and standard deviation (as SQL literals) per standardized column."""
        wanted = [(alias, spec) for alias, _, spec in self._columns if spec.standardize]
        if not wanted:
            return {}
        if "train" not in self._ranges:
            raise ValueError(
                "standardize=True requires a fold named 'train'; either rename "
                "your training fold or compute statistics yourself in SQL"
            )
        aggregates = ", ".join(f"avg({spec.expr}), stddev_pop({spec.expr})" for _, spec in wanted)
        sql = (
            f"{self._from(_quote_literal(self.source))} "
            f"SELECT count(*), {aggregates} FROM raw WHERE {self._in_fold('train')}"
        )
        row = con.sql(sql).fetchone()
        assert row is not None
        if row[0] == 0:
            raise ValueError(
                "standardize=True requires at least one row in the 'train' fold "
                "after filtering/splitting, but it's empty; widen the split, check "
                "drop_nulls, or drop standardize=True"
            )
        return {
            alias: (f"{float(row[1 + 2 * i])!r}::DOUBLE", f"{float(row[2 + 2 * i])!r}::DOUBLE")
            for i, (alias, _) in enumerate(wanted)
        }

    def _shard_files(self, con: Connection, world: int) -> list[str] | None:
        """The source's files, if each of ``world`` shards can read only its own."""
        if world == 1 or self.key is None or not self.source.lower().endswith(".parquet"):
            return None
        # With a key, a row's fold doesn't depend on what else is scanned.
        glob = f"SELECT file FROM glob({_quote_literal(self.source)})"
        files = sorted(f for (f,) in con.sql(glob).fetchall())
        return files if len(files) >= world else None

    def _shard_source(self, files: list[str] | None, rank: int, world: int) -> tuple[str, str]:
        """The FROM clause and the row filter reading shard ``rank``."""
        if files is not None:
            mine = ", ".join(_quote_literal(f) for f in files[rank::world])
            return f"read_parquet([{mine}])", ""
        if world > 1:
            return _quote_literal(self.source), f" AND hash(_ducky_row, 'shard') % {world} = {rank}"
        return _quote_literal(self.source), ""

    def _count(self, con: Connection, from_clause: str, by_hash: int | None = None) -> list[int]:
        """The fold's rows in ``from_clause``, or per hash shard of ``by_hash``."""
        fold = self._in_fold(self.fold)
        if by_hash is None:
            row = con.sql(
                f"{self._from(from_clause)} SELECT count(*) FROM raw WHERE {fold}"
            ).fetchone()
            assert row is not None
            return [row[0]]
        shard = f"hash(_ducky_row, 'shard') % {by_hash}"
        sql = f"{self._from(from_clause)} SELECT {shard}, count(*) FROM raw WHERE {fold} GROUP BY 1"
        counts = dict(con.sql(sql).fetchall())
        return [counts.get(rank, 0) for rank in range(by_hash)]

    def _plan(self, con: Connection, world: int, even: bool) -> tuple[list[str] | None, int | None]:
        """How ``world`` shards read the fold: their files, if by file, and the
        rows each yields, if even. Decided once per world size."""
        if (world, even) in self._plans:
            return self._plans[world, even]
        files = self._shard_files(con, world)
        limit = None
        if even:
            if files is not None:
                counts = [
                    self._count(con, self._shard_source(files, rank, world)[0])[0]
                    for rank in range(world)
                ]
                # Uneven files would trim too much; shard their rows instead.
                if min(counts) * world < 0.99 * sum(counts):
                    files = None
            if files is None:
                counts = self._count(con, _quote_literal(self.source), by_hash=world)
            limit = min(counts)
        self._plans[world, even] = files, limit
        return files, limit

    def _sql(self, con: Connection, epoch: int, rank: int, world: int, even: bool) -> str:
        files, limit = self._plan(con, world, even and world > 1)
        from_clause, shard_filter = self._shard_source(files, rank, world)
        items = []
        for alias, _, spec in self._columns:
            mean, std = self._stats.get(alias, (None, None))
            items.append(_select_item(alias, spec, mean, std))
        order = ""
        if self.shuffle is not None and self.shuffle.buffer is None:
            order = f" ORDER BY hash(_ducky_row, {self.shuffle.seed}, {epoch})"
        return (
            f"{self._from(from_clause)} SELECT {', '.join(items)} FROM raw "
            f"WHERE {self._in_fold(self.fold)}{shard_filter}{order}"
            f"{f' LIMIT {limit}' if limit is not None else ''}"
        )

    def _assemble(self, chunk: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        import numpy as np

        block = {}
        for name, spec in self.field_specs.items():
            if isinstance(spec, Matrix):
                block[name] = np.stack([chunk[f"{name}__{c}"] for c in spec.columns], axis=1)
            else:
                block[name] = chunk[name]
        return block

    def _produce(
        self, epoch: int, connections: list[Connection]
    ) -> Iterator[dict[str, np.ndarray]]:
        """Run the epoch's query and cut its rows into host batches."""
        import numpy as np

        if self.shard is not None:
            (rank, world), distributed = self.shard, True
        else:
            rank, world, distributed = _auto_shard()
        even = distributed if self.even_shards is None else self.even_shards
        capacity = rng = None
        if self.shuffle is not None and self.shuffle.buffer is not None:
            capacity = self.shuffle.buffer * self.batch_size
            rng = np.random.default_rng([self.shuffle.seed, epoch, rank])
        batcher = _Batcher(self.batch_size, self.drop_last, capacity, rng)
        con, own = self._connect()
        connections.append(con)
        try:
            result = con.sql(self._sql(con, epoch, rank, world, even), streaming=True)
            while (chunk := result.fetch_numpy_chunk(fill_nulls=True)) is not None:
                yield from batcher.push(self._assemble(chunk))
            yield from batcher.finish()
        finally:
            if own:
                con.close()


@overload
def stream(
    source: str,
    *,
    batch_size: int,
    fields: Mapping[str, Field] | None = ...,
    columns: dict[str, Feature] | None = ...,
    target: Target | None = ...,
    fold: str = ...,
    split: Split | None = ...,
    key: str | None = ...,
    drop_nulls: list[str] | None = ...,
    shuffle: Shuffle | None = ...,
    drop_last: bool = ...,
    shard: tuple[int, int] | None = ...,
    even_shards: bool | None = ...,
    backend: Literal["numpy"] = ...,
    device: Any = ...,
    prefetch: int = ...,
    read_ahead: str = ...,
    con: Connection | None = ...,
) -> Stream[np.ndarray]: ...
@overload
def stream(
    source: str,
    *,
    batch_size: int,
    fields: Mapping[str, Field] | None = ...,
    columns: dict[str, Feature] | None = ...,
    target: Target | None = ...,
    fold: str = ...,
    split: Split | None = ...,
    key: str | None = ...,
    drop_nulls: list[str] | None = ...,
    shuffle: Shuffle | None = ...,
    drop_last: bool = ...,
    shard: tuple[int, int] | None = ...,
    even_shards: bool | None = ...,
    backend: Literal["jax"],
    device: Any = ...,
    prefetch: int = ...,
    read_ahead: str = ...,
    con: Connection | None = ...,
) -> Stream[jax.Array]: ...
@overload
def stream(
    source: str,
    *,
    batch_size: int,
    fields: Mapping[str, Field] | None = ...,
    columns: dict[str, Feature] | None = ...,
    target: Target | None = ...,
    fold: str = ...,
    split: Split | None = ...,
    key: str | None = ...,
    drop_nulls: list[str] | None = ...,
    shuffle: Shuffle | None = ...,
    drop_last: bool = ...,
    shard: tuple[int, int] | None = ...,
    even_shards: bool | None = ...,
    backend: Literal["torch"],
    device: Any = ...,
    prefetch: int = ...,
    read_ahead: str = ...,
    con: Connection | None = ...,
) -> Stream[torch.Tensor]: ...
@overload
def stream(
    source: str,
    *,
    batch_size: int,
    fields: Mapping[str, Field] | None = ...,
    columns: dict[str, Feature] | None = ...,
    target: Target | None = ...,
    fold: str = ...,
    split: Split | None = ...,
    key: str | None = ...,
    drop_nulls: list[str] | None = ...,
    shuffle: Shuffle | None = ...,
    drop_last: bool = ...,
    shard: tuple[int, int] | None = ...,
    even_shards: bool | None = ...,
    backend: Literal["mlx"],
    device: Any = ...,
    prefetch: int = ...,
    read_ahead: str = ...,
    con: Connection | None = ...,
) -> Stream[mx.array]: ...
@overload
def stream(
    source: str,
    *,
    batch_size: int,
    fields: Mapping[str, Field] | None = ...,
    columns: dict[str, Feature] | None = ...,
    target: Target | None = ...,
    fold: str = ...,
    split: Split | None = ...,
    key: str | None = ...,
    drop_nulls: list[str] | None = ...,
    shuffle: Shuffle | None = ...,
    drop_last: bool = ...,
    shard: tuple[int, int] | None = ...,
    even_shards: bool | None = ...,
    backend: str,
    device: Any = ...,
    prefetch: int = ...,
    read_ahead: str = ...,
    con: Connection | None = ...,
) -> Stream[Any]: ...
def stream(
    source: str,
    *,
    batch_size: int,
    fields: Mapping[str, Field] | None = None,
    columns: dict[str, Feature] | None = None,
    target: Target | None = None,
    fold: str = "train",
    split: Split | None = None,
    key: str | None = None,
    drop_nulls: list[str] | None = None,
    shuffle: Shuffle | None = None,
    drop_last: bool = False,
    shard: tuple[int, int] | None = None,
    even_shards: bool | None = None,
    backend: str = "numpy",
    device: Any = None,
    prefetch: int = 2,
    read_ahead: str = "128MB",
    con: Connection | None = None,
) -> Stream[Any]:
    """Stream one fold of a DuckDB-readable source as ``{field: array}`` batches.

    Fields, splits, and standardization work as in :func:`dataset`, and with
    the same ``split`` and ``key`` assign rows to the same folds, but rows are
    read per epoch instead of held in memory.

    ``shard=(rank, world_size)`` reads only every ``world_size``-th share of
    the fold; by default it comes from ``torch.distributed`` and DataLoader
    workers. Shards hash each row's identity, so every shard scans the whole
    source, except that with a ``key`` and at least ``world_size`` Parquet files
    each shard reads only its own files.

    ``even_shards`` trims every shard to the rows of the smallest, counted once
    per stream, so that all ranks take the same number of steps, as DDP
    requires. It defaults to True for an explicit ``shard`` or several
    ``torch.distributed`` ranks, and to False for DataLoader workers alone,
    whose batches the loader merges anyway. The trimmed rows are the last ones
    read, so without a global shuffle they are the same every epoch. Even
    shards read their own files only if the files hold nearly equal rows.

    ``prefetch`` batches are prepared on a background thread (0 disables it).
    Each iteration opens its own connection, on which DuckDB executes up to
    ``read_ahead`` of results ahead of the reader. A stream on a given ``con``
    keeps that connection's settings instead, and any other statement run on
    it meanwhile closes the stream.
    """
    if backend not in _BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; supported backends: {list(_BACKENDS)}")
    if device is not None and backend in _NO_DEVICE:
        raise ValueError(f"backend {backend!r} does not take a device argument")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0; got {batch_size}")
    if prefetch < 0:
        raise ValueError(f"prefetch must be >= 0; got {prefetch}")
    if shard is not None:
        rank, world = shard
        if world < 1 or not 0 <= rank < world:
            raise ValueError(
                f"shard must be (rank, world_size) with 0 <= rank < world_size; got {shard}"
            )
    field_specs = _normalize_fields(fields, columns, target)
    if backend == "torch":
        import torch  # noqa: F401  # registers streams as torch datasets below
    _register_with_torch()
    return Stream(
        source,
        field_specs,
        batch_size=batch_size,
        fold=fold,
        split=split,
        key=key,
        drop_nulls=drop_nulls,
        shuffle=shuffle,
        drop_last=drop_last,
        shard=shard,
        even_shards=even_shards,
        backend=backend,
        device=device,
        prefetch=prefetch,
        read_ahead=read_ahead,
        con=con,
    )
