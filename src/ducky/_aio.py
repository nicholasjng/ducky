"""Async query drivers."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from ._core import Error, PendingState

if TYPE_CHECKING:
    from ._core import Connection, PendingResult, Result


# One query runs on a connection at a time: DuckDB refuses a second one while
# the first holds the connection. Coroutines on the same connection therefore
# take turns.
# Entries are keyed by id() and dropped once unused, so nothing outlives the
# coroutines (a live coroutine keeps its connection, and so its id, alive).
_connection_locks: dict[int, tuple[asyncio.Lock, list[int]]] = {}


@contextlib.asynccontextmanager
async def _connection_turn(con: Connection) -> AsyncGenerator[None]:
    key = id(con)
    entry = _connection_locks.get(key)
    if entry is None:
        entry = _connection_locks[key] = (asyncio.Lock(), [0])
    lock, users = entry
    users[0] += 1
    try:
        async with lock:
            yield
    finally:
        users[0] -= 1
        if users[0] == 0:
            del _connection_locks[key]


# Seconds of executor work per thread hop: long enough that the hop's cost is
# negligible, short enough to resurface promptly for cancellation.
_TICK_BUDGET = 0.005


async def _drive(pending: PendingResult) -> Result:
    """Step ``pending`` to completion on the event loop, returning its result."""
    try:
        while True:
            state = await asyncio.to_thread(pending.execute_tasks, _TICK_BUDGET)
            if state == PendingState.READY:
                return pending.materialize()
            if state == PendingState.ERROR:
                raise Error(pending.error())
    except asyncio.CancelledError:
        # Drain off the event loop: it waits for the workers to stop. Shielded,
        # so a repeated cancellation can't abandon it half-way.
        await asyncio.shield(asyncio.to_thread(pending.drain))
        raise
    except KeyboardInterrupt:
        pending.drain()
        raise


async def aexecute(
    con: Connection,
    query: str,
    parameters: list | tuple | dict[str, Any] | None,
    streaming: bool,
) -> Result:
    """Async driver behind :meth:`Connection.aexecute`."""
    async with _connection_turn(con):
        return await _drive(con.make_pending(query, parameters, streaming))


async def asql(con: Connection, query: str, streaming: bool) -> Result:
    """Async driver behind :meth:`Connection.asql`."""
    async with _connection_turn(con):
        return await _drive(con.make_pending(query, None, streaming))
