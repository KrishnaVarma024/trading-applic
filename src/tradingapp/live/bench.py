"""Latency benchmark: replay recorded ticks through the live hot path.

    uv run python -m tradingapp.live.bench data/btcusdt-ticks-*.csv

A live run produces one forecast a minute, so a thousand samples would take
most of a day. This replays your recorded ticks with 1-second bars instead,
which exercises exactly the same code (parse, sequence, bar, features, model)
thousands of times in a few seconds.

What it measures: time from a raw message being handed to the parser to the
forecast being ready, inside this process. What it does not measure: the
network, the exchange, or the operating system's socket handling. The
forecasts it produces are meaningless as forecasts, because the model was
trained on 1-minute bars; only their timing is of interest.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import platform
import time
from pathlib import Path

import numpy as np

from ..ingest import IngestStats, SequenceTracker
from ..parse import ParseError, parse_trade
from .core import LiveForecaster
from .engine import WEEK
from .forecaster import Forecaster, pin_single_thread
from .latency import LatencyRecorder


def load_messages(path: str) -> list[str]:
    """Turn tick CSV rows back into exchange-format JSON, so parsing is part of the timing."""
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out.append(json.dumps({
                "e": "trade", "E": int(row["ts_ms"]), "s": "X", "t": int(row["trade_id"]),
                "p": row["price"], "q": row["qty"], "T": int(row["ts_ms"]),
                "m": row["buyer_is_maker"] == "1", "M": True}))
    return out


def split_sessions(messages: list[str], max_gap_ms: int = 60_000) -> list[list[str]]:
    """Cut the replay wherever recording stopped, so each run is one unbroken stretch."""
    sessions, current, last = [], [], None
    for m in messages:
        ts = json.loads(m)["T"]
        if last is not None and ts - last > max_gap_ms:
            sessions.append(current)
            current = []
        current.append(m)
        last = ts
    if current:
        sessions.append(current)
    return sessions


def run_bench(messages: list[str], forecaster: Forecaster, interval_ms: int = 1000,
              warmup_forecasts: int = 200, seed: int = 0) -> LiveForecaster:
    live = LiveForecaster(forecaster, interval_ms=interval_ms, skip_first_bar=False)
    # Pre-fill a week of bars so the engine forecasts from the first replayed bar.
    first_ts = json.loads(messages[0])["T"]
    first_bar = first_ts - first_ts % interval_ms
    rng = np.random.default_rng(seed)
    price = float(json.loads(messages[0])["p"])
    closes = price * np.exp(rng.normal(0, 2e-4, WEEK + 1).cumsum())
    for i in range(WEEK + 1):
        start = first_bar - (WEEK + 1 - i) * interval_ms
        live.engine.update(start, float(closes[i]), 1.0, 10.0)

    stats = IngestStats()
    seq = SequenceTracker(stats)
    for raw in messages:
        recv_ns = time.perf_counter_ns()
        try:
            trade = parse_trade(raw, recv_ns)
        except ParseError:
            continue
        if trade is None or not seq.accept(trade):
            continue
        live.on_trade(trade, time.perf_counter_ns())
        if live.forecasts == warmup_forecasts and warmup_forecasts:
            live.latency = type(live.latency)()      # discard warm-up samples (cold caches, first calls)
            warmup_forecasts = 0
    return live


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("ticks", nargs="+", help="tick CSV files written by `tradingapp live --save-ticks`")
    p.add_argument("--model", default="models/har_gbm.joblib")
    p.add_argument("--interval-ms", type=int, default=1000)
    p.add_argument("--no-gc", action="store_true", help="disable the garbage collector during the run")
    p.add_argument("--threads", action="store_true", help="leave the tree library's thread pool on, to see its cost")
    p.add_argument("--out", default="reports/latency_bench.json")
    args = p.parse_args()

    if not args.threads:
        pin_single_thread()
    forecaster = Forecaster.load(args.model)
    messages = [m for path in sorted(args.ticks) for m in load_messages(path)]
    sessions = [s for s in split_sessions(messages) if len(s) > 1000]
    total = LatencyRecorder()
    if args.no_gc:
        gc.disable()
    for session in sessions:
        total.extend(run_bench(session, forecaster, args.interval_ms).latency)
    gc.enable()
    context = dict(mode="replay", messages=len(messages), sessions=len(sessions), forecasts=total.count(),
                   interval_ms=args.interval_ms, gc_disabled=args.no_gc, single_thread=not args.threads,
                   machine=f"{platform.system()} {platform.machine()}", python=platform.python_version())
    total.save(args.out, **context)
    print(f"{len(messages):,} messages in {len(sessions)} sessions, {total.count():,} timed forecasts, "
          f"gc {'off' if args.no_gc else 'on'}\n")
    print(total.to_markdown())
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
