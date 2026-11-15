"""WebSocket ingest: connect, parse, de-duplicate, detect gaps, reconnect.

The public function is `stream_trades`, an async generator that yields clean
Trade objects forever and hides every network failure behind a reconnect.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import WebSocketException

from .models import Trade
from .parse import ParseError, parse_trade

log = logging.getLogger(__name__)

DEFAULT_WS_BASE = "wss://stream.binance.com:9443/ws"


def trade_stream_url(symbol: str, base: str = DEFAULT_WS_BASE) -> str:
    return f"{base}/{symbol.lower()}@trade"


@dataclass
class IngestStats:
    connects: int = 0
    disconnects: int = 0
    messages: int = 0
    trades: int = 0
    parse_errors: int = 0
    duplicates: int = 0
    gaps: int = 0
    missed_trades: int = 0
    rejected_future: int = 0


class SequenceTracker:
    """Uses the exchange's trade id to spot duplicates and missed trades.

    Trade ids increase by exactly 1. After a reconnect the id tells us how
    many trades were lost while we were away, instead of leaving us guessing.
    """

    def __init__(self, stats: IngestStats) -> None:
        self.stats = stats
        self.last_id: int | None = None

    def accept(self, trade: Trade) -> bool:
        if self.last_id is not None:
            if trade.trade_id <= self.last_id:
                self.stats.duplicates += 1
                return False
            missed = trade.trade_id - self.last_id - 1
            if missed > 0:
                self.stats.gaps += 1
                self.stats.missed_trades += missed
                log.warning("gap: missed %d trades (%d -> %d)", missed, self.last_id, trade.trade_id)
        self.last_id = trade.trade_id
        return True


class Backoff:
    """Exponential backoff with full jitter, so many clients do not reconnect in step."""

    def __init__(self, base: float = 0.5, cap: float = 30.0) -> None:
        self.base, self.cap, self.attempt = base, cap, 0

    def reset(self) -> None:
        self.attempt = 0

    def next_delay(self) -> float:
        ceiling = min(self.cap, self.base * (2 ** self.attempt))
        self.attempt += 1
        return random.uniform(0, ceiling)


async def stream_trades(
    url: str,
    *,
    stats: IngestStats | None = None,
    backoff: Backoff | None = None,
    max_future_ms: int | None = 60_000,
    max_reconnects: int | None = None,
    **connect_kwargs: Any,
) -> AsyncIterator[Trade]:
    """Yield trades from `url`, reconnecting on any failure.

    max_future_ms: drop trades stamped further ahead of the local clock than
        this. One corrupt far-future timestamp would otherwise make the bar
        builder treat every later real trade as late. None disables the check.
    max_reconnects: stop after this many disconnects. None means run forever;
        tests set it so they terminate.
    """
    stats = stats if stats is not None else IngestStats()
    backoff = backoff if backoff is not None else Backoff()
    seq = SequenceTracker(stats)

    while True:
        try:
            async with ws_connect(
                url, ping_interval=20, ping_timeout=20, max_size=1 << 20, **connect_kwargs
            ) as ws:
                stats.connects += 1
                log.info("connected to %s", url)
                async for raw in ws:
                    recv_ns = time.perf_counter_ns()
                    stats.messages += 1
                    backoff.reset()  # the link is proven healthy only once data flows
                    try:
                        trade = parse_trade(raw, recv_ns)
                    except ParseError as exc:
                        stats.parse_errors += 1
                        log.debug("dropped message: %s", exc)
                        continue
                    if trade is None:
                        continue
                    if max_future_ms is not None and trade.ts_ms > time.time() * 1000 + max_future_ms:
                        stats.rejected_future += 1
                        continue
                    if not seq.accept(trade):
                        continue
                    stats.trades += 1
                    yield trade
            log.warning("server closed the connection")
        except (OSError, WebSocketException, asyncio.TimeoutError) as exc:
            log.warning("connection lost: %r", exc)

        stats.disconnects += 1
        if max_reconnects is not None and stats.disconnects > max_reconnects:
            return
        delay = backoff.next_delay()
        log.info("reconnecting in %.2fs", delay)
        await asyncio.sleep(delay)
