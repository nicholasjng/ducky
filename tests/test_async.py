"""Async execute / sql coverage (pytest-asyncio, auto mode)."""

from __future__ import annotations

import asyncio
import time

import pytest

import ducky
from ducky._core import PendingState


async def test_aexecute_basic(con):
    res = await con.aexecute("SELECT count(*) FROM range(1000)")
    assert res.fetchitem() == 1000


async def test_aexecute_with_params(con):
    res = await con.aexecute("SELECT ? + ?", [40, 2])
    assert res.fetchitem() == 42


async def test_asql_streaming_spans_chunks(con):
    res = await con.asql("SELECT * FROM range(50_000) t(i)", streaming=True)
    seen = chunks = 0
    while (chunk := res.fetch_chunk()) is not None:
        chunks += 1
        seen += len(chunk.column("i"))
    assert seen == 50_000
    assert chunks > 1, "expected the streaming result to span multiple chunks"


async def test_aexecute_error_propagates(con):
    with pytest.raises(ducky.Error, match="no_such_table"):
        await con.aexecute("SELECT * FROM no_such_table")


async def test_aexecute_loop_stays_responsive():
    # A long query must not park the event loop: a concurrent heartbeat keeps
    # ticking at roughly its own cadence while the query runs off-thread.
    con = ducky.connect(threads=2)
    ticks = 0
    interval = 0.01

    async def heartbeat():
        nonlocal ticks
        while True:
            await asyncio.sleep(interval)
            ticks += 1

    hb = asyncio.create_task(heartbeat())
    start = time.monotonic()
    await con.aexecute("SELECT count(*) FROM range(40_000_000) t(i) WHERE i % 7 = 0")
    elapsed = time.monotonic() - start
    hb.cancel()
    # Self-normalized against the query's own measured duration (not an
    # absolute machine-speed guess): a loop blocked for most of that time
    # would tick once or twice regardless of how long the query actually
    # took, so requiring ticks roughly proportional to elapsed/interval (50%
    # tolerance for scheduling jitter) catches that "ticks > 0" cannot.
    expected_min = max(1, int(elapsed / interval * 0.5))
    assert ticks >= expected_min, (
        f"event loop was blocked during the async query: {ticks} ticks over {elapsed:.3f}s "
        f"(expected >= {expected_min})"
    )


async def test_aexecute_cancellation_interrupts_and_survives():
    con = ducky.connect(threads=2)
    task = asyncio.create_task(
        con.aexecute("SELECT count(*) FROM range(10_000_000_000) t(i) WHERE i % 7 = 0")
    )
    await asyncio.sleep(0.1)  # let the query actually start
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # The connection survives an interrupted query — a fresh one still runs.
    res = await con.aexecute("SELECT 1")
    assert res.fetchitem() == 1


def test_execute_tasks_runs_to_completion(con):
    pending = con.make_pending("SELECT count(*) FROM range(5_000_000)")
    hops = 0
    while (state := pending.execute_tasks(0.01)) not in (
        PendingState.READY,
        PendingState.ERROR,
    ):
        hops += 1
    assert state == PendingState.READY
    assert pending.materialize().fetchitem() == 5_000_000


async def test_cancellation_keeps_event_loop_responsive():
    con = ducky.connect(threads=2)
    task = asyncio.create_task(
        con.aexecute("SELECT count(*) FROM range(10_000_000_000) t(i) WHERE i % 7 = 0")
    )
    await asyncio.sleep(0.1)
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0)

    heartbeat = asyncio.create_task(ticker())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    heartbeat.cancel()
    assert ticks > 0
    assert (await con.aexecute("SELECT 1")).fetchitem() == 1
