"""Live forecasting from the exchange stream.

    uv run python -m tradingapp.live.forecast --symbol btcusdt

Start-up: connect, wait for the first complete live bar, fetch the week of
klines that ends just before it, then forecast on every bar close.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import time
from datetime import datetime, timezone

from ..ingest import DEFAULT_WS_BASE, IngestStats, stream_trades, trade_stream_url
from ..models import Bar
from ..storage import BAR_HEADER, DailyCsvSink
from .backfill import DEFAULT_REST_BASE, MINUTE_MS, fetch_klines
from .core import Forecast, LiveForecaster
from .engine import WEEK
from .forecaster import Forecaster, pin_single_thread
from .scheduling import request_interactive_qos

log = logging.getLogger("tradingapp.live")
FORECAST_HEADER = ["bar_end_ms", "rv_next_hour", "annualised_vol_pct", "degraded", "latency_us"]


def _utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%H:%M:%S")


async def run(args: argparse.Namespace) -> None:
    pin_single_thread()
    if args.qos == "interactive":
        log.info("scheduling: %s", request_interactive_qos())
    forecaster = Forecaster.load(args.model).prepare(reference=args.reference_model)
    log.info("inference: %s", "scikit-learn predict" if args.reference_model else "compiled trees")
    log.info("model: %s", forecaster.meta)
    bar_sink = DailyCsvSink(args.data_dir, f"{args.symbol}-bars-60s", BAR_HEADER)
    fc_sink = DailyCsvSink(args.data_dir, f"{args.symbol}-forecasts", FORECAST_HEADER)
    pending_first: list[Bar] = []
    warmed = False

    def on_bar(bar: Bar) -> None:
        bar_sink.write(bar.start_ms, [bar.start_ms, bar.end_ms, bar.open, bar.high, bar.low, bar.close,
                                      bar.volume, bar.notional, bar.n_trades])
        if not warmed:
            pending_first.append(bar)

    def on_forecast(f: Forecast) -> None:
        fc_sink.write(f.bar_end_ms, [f.bar_end_ms, f.rv_next_hour, round(f.annualised_vol_pct, 3),
                                     int(f.degraded), round(f.latency_ns / 1000, 1)])
        flag = "  DEGRADED" if f.degraded else ""
        print(f"{_utc(f.bar_end_ms)}Z  next-hour vol {f.annualised_vol_pct:6.2f}% annualised"
              f"  latency {f.latency_ns / 1000:8.1f} us{flag}", flush=True)

    live = LiveForecaster(forecaster, on_forecast=on_forecast, on_bar=on_bar)
    stats = IngestStats()
    task = asyncio.current_task()
    try:
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
    except NotImplementedError:
        pass
    last_report = time.monotonic()
    try:
        async for trade in stream_trades(trade_stream_url(args.symbol, args.ws_base), stats=stats):
            t_parsed = time.perf_counter_ns()
            live.on_trade(trade, t_parsed)
            if not warmed and pending_first:
                # First complete live bar seen. Fetch the week that ends right before it,
                # so history and live data join with no missing minute.
                first = pending_first[0]
                log.info("first live bar at %sZ; backfilling %d klines", _utc(first.start_ms), WEEK + 1)
                klines = await asyncio.to_thread(
                    fetch_klines, args.symbol, first.start_ms - MINUTE_MS, WEEK + 1, args.rest_base)
                live.join_history(klines, pending_first)
                # Warm-up: run inference a few times on real features and discard the results,
                # so the first timed forecast does not pay for first-call setup.
                warm = live.engine._features(pending_first[-1].end_ms)
                for _ in range(5):
                    forecaster.predict(warm.har_logs, warm.x)
                warmed = True
                log.info("engine warm: %s; forecasting from the next bar close", live.engine.warm)
            if time.monotonic() - last_report >= args.report_every:
                last_report = time.monotonic()
                log.info("ingest %s | forecasts %d | resets %d", stats, live.forecasts, live.engine.resets)
                if live.latency.count():
                    log.info("latency us: %s", live.latency.summary()["total"])
    finally:
        bar_sink.close()
        fc_sink.close()
        if live.latency.count():
            live.latency.save(args.latency_out, mode="live", symbol=args.symbol, forecasts=live.forecasts,
                              inference="sklearn" if args.reference_model else "compiled", qos=args.qos,
                              disconnects=stats.disconnects, missed_trades=stats.missed_trades)
            print("\n" + live.latency.to_markdown())
            log.info("latency summary written to %s", args.latency_out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="btcusdt")
    p.add_argument("--model", default="models/har_gbm.joblib")
    p.add_argument("--data-dir", default="data/live")
    p.add_argument("--ws-base", default=DEFAULT_WS_BASE)
    p.add_argument("--rest-base", default=DEFAULT_REST_BASE)
    p.add_argument("--report-every", type=float, default=300.0)
    p.add_argument("--latency-out", default="reports/latency_live.json")
    p.add_argument("--qos", choices=["interactive", "default"], default="interactive",
                   help="macOS scheduling class for this process")
    p.add_argument("--reference-model", action="store_true", help="use scikit-learn predict, not compiled trees")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        asyncio.run(run(args))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass


if __name__ == "__main__":
    main()
