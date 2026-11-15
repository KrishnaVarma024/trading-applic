"""Command line entry point.

    uv run python -m tradingapp live --symbol btcusdt --interval 60 --data-dir data
    uv run python -m tradingapp replay tests/fixtures/sample_trades.jsonl --interval 60
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import time
from datetime import datetime, timezone

from .ingest import DEFAULT_WS_BASE, IngestStats, SequenceTracker, stream_trades, trade_stream_url
from .models import Bar
from .parse import ParseError, parse_trade
from .pipeline import Pipeline

log = logging.getLogger("tradingapp")


def print_bar(bar: Bar) -> None:
    t = datetime.fromtimestamp(bar.start_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"{t}Z  O {bar.open:.2f}  H {bar.high:.2f}  L {bar.low:.2f}  C {bar.close:.2f}"
          f"  vol {bar.volume:.5f}  trades {bar.n_trades}", flush=True)


async def run_live(args: argparse.Namespace) -> None:
    stats = IngestStats()
    pipeline = Pipeline(args.interval * 1000, args.data_dir, args.symbol, args.save_ticks, print_bar)
    url = trade_stream_url(args.symbol, args.ws_base)
    task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGTERM, task.cancel)
    except NotImplementedError:  # Windows
        pass
    last_report = time.monotonic()
    try:
        async for trade in stream_trades(url, stats=stats):
            pipeline.on_trade(trade)
            if time.monotonic() - last_report >= args.report_every:
                last_report = time.monotonic()
                log.info("ingest %s | bars %s", stats, pipeline.builder.stats)
    finally:
        pipeline.close()
        log.info("final: ingest %s | bars %s | emitted %d", stats, pipeline.builder.stats, pipeline.bars_emitted)


def run_replay(args: argparse.Namespace) -> None:
    """Feed recorded messages through the same parse/sequence/bar code as live."""
    stats = IngestStats()
    seq = SequenceTracker(stats)
    pipeline = Pipeline(args.interval * 1000, args.data_dir, "replay", False, print_bar)
    with open(args.file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stats.messages += 1
            try:
                trade = parse_trade(line, time.perf_counter_ns())
            except ParseError:
                stats.parse_errors += 1
                continue
            if trade is None or not seq.accept(trade):
                continue
            stats.trades += 1
            pipeline.on_trade(trade)
    pipeline.close()
    log.info("final: ingest %s | bars %s | emitted %d", stats, pipeline.builder.stats, pipeline.bars_emitted)


def main() -> None:
    parser = argparse.ArgumentParser(prog="tradingapp")
    sub = parser.add_subparsers(dest="cmd", required=True)

    live = sub.add_parser("live", help="ingest a live trade stream")
    live.add_argument("--symbol", default="btcusdt")
    live.add_argument("--interval", type=int, default=60, help="bar width in seconds")
    live.add_argument("--data-dir", default="data")
    live.add_argument("--save-ticks", action="store_true")
    live.add_argument("--ws-base", default=DEFAULT_WS_BASE)
    live.add_argument("--report-every", type=float, default=30.0, help="seconds between stats lines")

    replay = sub.add_parser("replay", help="replay a JSONL file of recorded messages")
    replay.add_argument("file")
    replay.add_argument("--interval", type=int, default=60)
    replay.add_argument("--data-dir", default=None)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.cmd == "live":
        try:
            asyncio.run(run_live(args))
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
    else:
        run_replay(args)


if __name__ == "__main__":
    main()
