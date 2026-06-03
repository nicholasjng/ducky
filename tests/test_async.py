"""Async execute / sql coverage (pytest-asyncio, auto mode)."""

from __future__ import annotations

import asyncio

import pytest

import ducky


async def test_aexecute_basic():
    con = ducky.connect()
    res = await con.aexecute("SELECT count(*) FROM range(1000)")
    assert res.fetchitem() == 1000


async def test_aexecute_with_params():
    con = ducky.connect()
    res = await con.aexecute("SELECT ? + ?", [40, 2])
    assert res.fetchitem() == 42


async def test_asql_streaming_spans_chunks():
    con = ducky.connect()
    res = await con.asql("SELECT * FROM range(50_000) t(i)", streaming=True)
    seen = chunks = 0
    while (chunk := res.fetch_chunk()) is not None:
        chunks += 1
        seen += len(chunk.column("i"))
    assert seen == 50_000
    assert chunks > 1, "expected the streaming result to span multiple chunks"


async def test_aexecute_error_propagates():
    con = ducky.connect()
    with pytest.raises(ducky.Error, match="no_such_table"):
        await con.aexecute("SELECT * FROM no_such_table")


async def test_aexecute_loop_stays_responsive():
    # A long query must not park the event loop: a concurrent heartbeat keeps
    # ticking while the query runs off-thread.
    con = ducky.connect(threads=2)
    ticks = 0

    async def heartbeat():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    hb = asyncio.create_task(heartbeat())
    await con.aexecute("SELECT count(*) FROM range(40_000_000) t(i) WHERE i % 7 = 0")
    hb.cancel()
    assert ticks > 0, "event loop was blocked during the async query"


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
