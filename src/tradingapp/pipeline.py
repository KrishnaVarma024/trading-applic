"""Wires the stages together: trade -> bar builder -> sinks."""
from __future__ import annotations

from typing import Callable

from .bars import BarBuilder
from .models import Bar, Trade
from .storage import BAR_HEADER, TICK_HEADER, DailyCsvSink


class Pipeline:
    def __init__(
        self,
        interval_ms: int,
        data_dir: str | None = None,
        symbol: str = "symbol",
        save_ticks: bool = False,
        on_bar: Callable[[Bar], None] | None = None,
    ) -> None:
        self.builder = BarBuilder(interval_ms)
        self.on_bar = on_bar
        self.bars_emitted = 0
        self._bar_sink = self._tick_sink = None
        if data_dir is not None:
            secs = interval_ms // 1000
            self._bar_sink = DailyCsvSink(data_dir, f"{symbol}-bars-{secs}s", BAR_HEADER)
            if save_ticks:
                self._tick_sink = DailyCsvSink(data_dir, f"{symbol}-ticks", TICK_HEADER, flush_every=500)

    def _emit(self, bar: Bar) -> None:
        self.bars_emitted += 1
        if self._bar_sink is not None:
            self._bar_sink.write(bar.start_ms, [
                bar.start_ms, bar.end_ms, bar.open, bar.high, bar.low, bar.close,
                bar.volume, bar.notional, bar.n_trades,
            ])
        if self.on_bar is not None:
            self.on_bar(bar)

    def on_trade(self, trade: Trade) -> None:
        if self._tick_sink is not None:
            self._tick_sink.write(trade.ts_ms, [
                trade.trade_id, trade.ts_ms, trade.price, trade.qty, int(trade.buyer_is_maker),
            ])
        for bar in self.builder.update(trade):
            self._emit(bar)

    def close(self) -> None:
        # The open bar is incomplete, so it is deliberately not written.
        for sink in (self._bar_sink, self._tick_sink):
            if sink is not None:
                sink.close()
