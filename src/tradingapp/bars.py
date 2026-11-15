"""Aggregate trades into fixed-width time bars.

Bars are aligned to the epoch on *event time* (the exchange's trade timestamp),
not on the local clock. A bar closes when the first trade of a later bar
arrives. That keeps the builder deterministic: the same trades always give
the same bars, live or replayed.
"""
from __future__ import annotations

from dataclasses import dataclass

from .models import Bar, Trade


@dataclass
class BarStats:
    late_dropped: int = 0      # trades older than the open bar
    empty_bars: int = 0        # bars emitted with no trades
    unfilled_gaps: int = 0     # gaps too long to fill with empty bars


class BarBuilder:
    def __init__(self, interval_ms: int, max_fill_bars: int = 1440) -> None:
        if interval_ms <= 0:
            raise ValueError("interval_ms must be positive")
        self.interval_ms = interval_ms
        self.max_fill_bars = max_fill_bars
        self.stats = BarStats()
        self._start: int | None = None
        self._o = self._h = self._l = self._c = 0.0
        self._vol = self._notional = 0.0
        self._n = 0

    def _bucket(self, ts_ms: int) -> int:
        return ts_ms - (ts_ms % self.interval_ms)

    def _open(self, start: int, trade: Trade) -> None:
        self._start = start
        self._o = self._h = self._l = self._c = trade.price
        self._vol = trade.qty
        self._notional = trade.price * trade.qty
        self._n = 1

    def _snapshot(self) -> Bar:
        assert self._start is not None
        return Bar(
            start_ms=self._start,
            end_ms=self._start + self.interval_ms,
            open=self._o, high=self._h, low=self._l, close=self._c,
            volume=self._vol, notional=self._notional, n_trades=self._n,
        )

    def update(self, trade: Trade) -> list[Bar]:
        """Feed one trade. Returns the bars this trade caused to close."""
        bucket = self._bucket(trade.ts_ms)

        if self._start is None:
            self._open(bucket, trade)
            return []

        if bucket < self._start:
            # The bar this trade belongs to is already closed and emitted.
            self.stats.late_dropped += 1
            return []

        if bucket == self._start:
            self._h = max(self._h, trade.price)
            self._l = min(self._l, trade.price)
            self._c = trade.price
            self._vol += trade.qty
            self._notional += trade.price * trade.qty
            self._n += 1
            return []

        closed = [self._snapshot()]
        skipped = (bucket - self._start) // self.interval_ms - 1
        if 0 < skipped <= self.max_fill_bars:
            last_close = self._c
            for i in range(1, skipped + 1):
                start = self._start + i * self.interval_ms
                closed.append(Bar(
                    start_ms=start, end_ms=start + self.interval_ms,
                    open=last_close, high=last_close, low=last_close, close=last_close,
                    volume=0.0, notional=0.0, n_trades=0,
                ))
            self.stats.empty_bars += skipped
        elif skipped > self.max_fill_bars:
            # A long outage or a corrupt timestamp. Do not invent thousands of bars.
            self.stats.unfilled_gaps += 1
        self._open(bucket, trade)
        return closed

    def flush(self) -> Bar | None:
        """Return the still-open bar, for shutdown. It may be incomplete."""
        return self._snapshot() if self._start is not None else None
