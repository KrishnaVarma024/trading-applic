import asyncio

from websockets.asyncio.server import serve

from tradingapp.ingest import Backoff, IngestStats, stream_trades

from .helpers import msg

TS = 1_700_000_000_000


async def _collect(url, stats, max_reconnects):
    out = []
    async for t in stream_trades(url, stats=stats, backoff=Backoff(base=0.01, cap=0.01),
                                 max_future_ms=None, max_reconnects=max_reconnects, proxy=None):
        out.append(t.trade_id)
    return out


def test_reconnect_dedup_gap_and_bad_messages():
    scripts = [
        ['{"result": null, "id": 1}', msg(1, TS), msg(2, TS + 1), "not json",
         msg(3, TS + 2), msg(4, TS + 3, price="-1")],
        [msg(3, TS + 2), msg(6, TS + 5), msg(7, TS + 6)],     # replays 3, skips 4 and 5
    ]
    calls = {"n": 0}

    async def handler(ws):
        script = scripts[min(calls["n"], len(scripts) - 1)]
        calls["n"] += 1
        for m in script:
            await ws.send(m)
        # returning closes the connection, which the client must survive

    async def scenario():
        async with serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            stats = IngestStats()
            ids = await asyncio.wait_for(_collect(f"ws://127.0.0.1:{port}", stats, 1), 10)
            return ids, stats

    ids, stats = asyncio.run(scenario())
    assert ids == [1, 2, 3, 6, 7]
    assert stats.connects == 2 and stats.disconnects == 2
    assert stats.parse_errors == 2
    assert stats.duplicates == 1
    assert (stats.gaps, stats.missed_trades) == (1, 2)


def test_unreachable_server_backs_off_and_gives_up_when_told():
    async def scenario():
        stats = IngestStats()
        ids = await asyncio.wait_for(_collect("ws://127.0.0.1:1", stats, 2), 10)
        return ids, stats

    ids, stats = asyncio.run(scenario())
    assert ids == [] and stats.connects == 0 and stats.disconnects == 3


def test_far_future_timestamp_is_rejected():
    async def handler(ws):
        await ws.send(msg(1, TS))
        await ws.send(msg(2, 9_999_999_999_999))   # year 2286
        await ws.send(msg(3, TS + 1))

    async def scenario():
        async with serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            stats, out = IngestStats(), []
            async for t in stream_trades(f"ws://127.0.0.1:{port}", stats=stats,
                                         backoff=Backoff(0.01, 0.01), max_reconnects=0, proxy=None):
                out.append(t.trade_id)
            return out, stats

    ids, stats = asyncio.run(scenario())
    assert ids == [1, 3] and stats.rejected_future == 1


def test_backoff_grows_and_is_capped():
    b = Backoff(base=1.0, cap=4.0)
    ceilings = []
    for _ in range(5):
        ceilings.append(min(b.cap, b.base * 2 ** b.attempt))
        assert 0 <= b.next_delay() <= ceilings[-1]
    assert ceilings == [1.0, 2.0, 4.0, 4.0, 4.0]
    b.reset()
    assert b.attempt == 0
