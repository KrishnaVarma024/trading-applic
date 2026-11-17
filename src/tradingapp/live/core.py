"""The hot path: one trade in, and on a bar close, one forecast out.

Order matters here. Everything needed for the forecast happens first and is
timed. Anything that only records what happened (CSV rows, printing) happens
after the final timestamp, so disk speed cannot delay a forecast.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

from ..bars import BarBuilder
from ..models import Bar, Trade
from .engine import FeatureEngine
from .forecaster import Forecaster
from .latency import LatencyRecorder

HOURS_PER_YEAR = 24 * 365


@dataclass(frozen=True)
class Forecast:
    bar_end_ms: int
    rv_next_hour: float       # forecast realized variance over the next 60 minutes
    degraded: bool
    latency_ns: int           # arrival of the bar-closing trade to forecast ready

    @property
    def annualised_vol_pct(self) -> float:
        return 100 * math.sqrt(self.rv_next_hour * HOURS_PER_YEAR)


class LiveForecaster:
    def __init__(
        self, forecaster: Forecaster, interval_ms: int = 60_000, skip_first_bar: bool = True,
        on_forecast: Callable[[Forecast], None] | None = None,
        on_bar: Callable[[Bar], None] | None = None,
    ) -> None:
        self.forecaster = forecaster
        self.builder = BarBuilder(interval_ms, skip_first_bar=skip_first_bar)
        self.engine = FeatureEngine(interval_ms)
        self.latency = LatencyRecorder()
        self.on_forecast, self.on_bar = on_forecast, on_bar
        self.forecasts = 0

    def warm_up(self, klines) -> None:
        """Feed historical bars (start_ms, close, volume, trades) into the engine."""
        for start_ms, close, volume, trades in klines:
            self.engine.update(start_ms, close, volume, trades)

    def join_history(self, klines, live_bars: list[Bar]) -> None:
        """Replace the engine's history with `klines` followed by the live bars seen so far.

        Called once at start-up, when the first complete live bar is known. The
        klines must end exactly one bar before `live_bars[0]`, so history and
        live data meet with no missing and no repeated minute.
        """
        klines = list(klines)
        if klines[-1][0] + self.engine.interval_ms != live_bars[0].start_ms:
            raise ValueError("history does not end one bar before the first live bar")
        self.engine = FeatureEngine(self.engine.interval_ms)
        self.warm_up(klines)
        for bar in live_bars:
            self.engine.update(bar.start_ms, bar.close, bar.volume, bar.n_trades)
        if self.engine.resets:
            raise ValueError("history is not contiguous")

    def on_trade(self, trade: Trade, t_parsed_ns: int) -> Forecast | None:
        """Handle one trade. `t_parsed_ns` is when parsing and sequencing finished."""
        bars = self.builder.update(trade)
        if not bars:
            return None
        t_bar = time.perf_counter_ns()

        feats = None
        for bar in bars:                       # more than one only after a quiet stretch
            feats = self.engine.update(bar.start_ms, bar.close, bar.volume, bar.n_trades)
        t_feat = time.perf_counter_ns()

        forecast = None
        if feats is not None:
            rv = self.forecaster.predict(feats.har_logs, feats.x)
            t_done = time.perf_counter_ns()
            self.latency.add(
                total=t_done - trade.recv_ns,
                parse=t_parsed_ns - trade.recv_ns,
                bar=t_bar - t_parsed_ns,
                features=t_feat - t_bar,
                model=t_done - t_feat,
            )
            forecast = Forecast(feats.bar_end_ms, rv, feats.degraded, t_done - trade.recv_ns)
            self.forecasts += 1

        # ---- timing has stopped; record-keeping only below this line ----
        if self.on_bar is not None:
            for bar in bars:
                self.on_bar(bar)
        if forecast is not None and self.on_forecast is not None:
            self.on_forecast(forecast)
        return forecast
