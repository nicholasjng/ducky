"""Async query drivers."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from ._core import Error, PendingState

if TYPE_CHECKING:
    from ._core import Connection, PendingResult, Result


async def _drive(pending: PendingResult) -> Result:
    """Step ``pending`` to completion on the event loop, returning its result."""
    no_tasks_streak = 0
    try:
        while True:
            state = await asyncio.to_thread(pending.execute_task)
            if state == PendingState.READY:
                return pending.materialize()
            if state == PendingState.ERROR:
                raise Error(pending.error())
            if state == PendingState.NO_TASKS:
                # Back off after repeated no-op ticks.
                no_tasks_streak += 1
                await asyncio.sleep(0 if no_tasks_streak < 4 else 0.001)
            else:
                no_tasks_streak = 0
    except (asyncio.CancelledError, KeyboardInterrupt):
        pending.drain()
        raise


async def aexecute(
    con: Connection,
    query: str,
    parameters: list | tuple | dict[str, Any] | None,
    streaming: bool,
) -> Result:
    """Async driver behind :meth:`Connection.aexecute`."""
    return await _drive(con.make_pending(query, parameters, streaming))


async def asql(con: Connection, query: str, streaming: bool) -> Result:
    """Async driver behind :meth:`Connection.asql`."""
    return await _drive(con.make_pending(query, None, streaming))
