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
from .scheduling import request_interactive_qos
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


def closes_a_bar(messages: list[str], interval_ms: int) -> list[bool]:
    """True for each message whose trade falls in a later bar than the one before it."""
    out, last = [], None
    for m in messages:
        ts = json.loads(m)["T"]
        bucket = ts - ts % interval_ms
        out.append(last is not None and bucket > last)
        last = bucket if last is None else max(last, bucket)
    return out


def run_bench(messages: list[str], forecaster: Forecaster, interval_ms: int = 1000,
              warmup_forecasts: int = 200, seed: int = 0, idle_ms: float = 0.0,
              max_forecasts: int | None = None) -> LiveForecaster:
    """Replay messages through the hot path.

    idle_ms: sleep this long before each bar-closing message, before its arrival
        is timestamped. Live, the process waits on the network between bars;
        this reproduces that wait so its effect on latency can be measured.
    max_forecasts: stop after this many timed forecasts.
    """
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
    pause = closes_a_bar(messages, interval_ms) if idle_ms > 0 else None
    for i, raw in enumerate(messages):
        if pause is not None and pause[i] and not warmup_forecasts:
            time.sleep(idle_ms / 1000)
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
        if max_forecasts is not None and not warmup_forecasts and live.latency.count() >= max_forecasts:
            break
    return live


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("ticks", nargs="+", help="tick CSV files written by `tradingapp live --save-ticks`")
    p.add_argument("--model", default="models/har_gbm.joblib")
    p.add_argument("--interval-ms", type=int, default=1000)
    p.add_argument("--no-gc", action="store_true", help="disable the garbage collector during the run")
    p.add_argument("--threads", action="store_true", help="leave the tree library's thread pool on, to see its cost")
    p.add_argument("--idle-ms", type=float, default=0.0, help="sleep before each bar-closing trade")
    p.add_argument("--max-forecasts", type=int, default=None, help="stop after this many timed forecasts")
    p.add_argument("--qos", choices=["interactive", "default"], default="interactive")
    p.add_argument("--reference-model", action="store_true", help="use scikit-learn predict, not compiled trees")
    p.add_argument("--out", default="reports/latency_bench.json")
    args = p.parse_args()

    if not args.threads:
        pin_single_thread()
    qos = request_interactive_qos() if args.qos == "interactive" else "default"
    forecaster = Forecaster.load(args.model).prepare(reference=args.reference_model)
    messages = [m for path in sorted(args.ticks) for m in load_messages(path)]
    sessions = [s for s in split_sessions(messages) if len(s) > 1000]
    total = LatencyRecorder()
    if args.no_gc:
        gc.disable()
    for session in sessions:
        remaining = None if args.max_forecasts is None else args.max_forecasts - total.count()
        if remaining is not None and remaining <= 0:
            break
        total.extend(run_bench(session, forecaster, args.interval_ms, idle_ms=args.idle_ms,
                               max_forecasts=remaining).latency)
    gc.enable()
    context = dict(mode="replay", messages=len(messages), sessions=len(sessions), forecasts=total.count(),
                   interval_ms=args.interval_ms, gc_disabled=args.no_gc, single_thread=not args.threads,
                   idle_ms=args.idle_ms, qos=qos, inference="sklearn" if args.reference_model else "compiled",
                   machine=f"{platform.system()} {platform.machine()}", python=platform.python_version())
    total.save(args.out, **context)
    print(f"{len(messages):,} messages in {len(sessions)} sessions, {total.count():,} timed forecasts, "
          f"gc {'off' if args.no_gc else 'on'}, idle {args.idle_ms:g} ms, qos {qos}, "
          f"{'scikit-learn' if args.reference_model else 'compiled'} inference\n")
    print(total.to_markdown())
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
