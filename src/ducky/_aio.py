"""Async drivers behind :meth:`Connection.aexecute` / :meth:`Connection.asql`.

The C++ ``aexecute`` / ``asql`` methods forward here, handing us a steppable :class:`PendingResult` (see ``connection.hpp``).
We own the executor loop from Python: each tick is offloaded to a worker thread.
The DuckDB task releases the GIL, so the event loop thread stays free — and we yield to the loop between ticks instead of blocking on it.
Cancellation (a cancelled task, or Ctrl-C) maps onto DuckDB's interrupt + drain teardown, mirroring the synchronous ``run_pending`` path.

Unlike the synchronous :meth:`Connection.execute` (which stashes ``current_result``), the async drivers resolve to the :class:`Result` directly —
no connection state is mutated from inside the coroutine, so concurrent awaits on one connection can't trample each other's "current result".
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from ._core import Error, PendingState

if TYPE_CHECKING:
    from ._core import Connection, PendingResult, Result


async def _drive(pending: PendingResult) -> Result:
    """Step ``pending`` to completion on the event loop, returning its result."""
    try:
        while True:
            state = await asyncio.to_thread(pending.execute_task)
            if state == PendingState.READY:
                return pending.materialize()
            if state == PendingState.ERROR:
                raise Error(pending.error())
            if state == PendingState.NO_TASKS:
                # Workers own the outstanding tasks; yield to the loop rather
                # than spinning (the sync path sleeps 1 ms here instead).
                await asyncio.sleep(0)
            # READY is handled above; NOT_READY means there is more for us to
            # do — tick again immediately.
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
