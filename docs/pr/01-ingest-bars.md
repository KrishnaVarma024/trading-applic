# Phase 1: WebSocket ingest and bar builder

## What this adds

The first stage of the pipeline: connect to an exchange trade stream, turn raw
messages into validated trades, and aggregate them into one-minute OHLCV bars
written to CSV.

- `parse.py`: the only code that touches raw network data. Rejects non-JSON,
  missing fields, non-finite or non-positive prices, negative quantities.
- `ingest.py`: `stream_trades` async generator with reconnect (exponential
  backoff, full jitter), duplicate dropping and gap counting by trade id, and
  rejection of far-future timestamps.
- `bars.py`: pure, deterministic event-time bar builder with explicit empty
  bars for quiet intervals and a cap on gap filling.
- `storage.py`: append-only CSV sink with daily rotation.
- `pipeline.py`, `__main__.py`: wiring and a CLI with `live` and `replay` modes.

## How it was tested

21 tests, no network required:

- Parser: valid trade, non-trade messages, seven kinds of broken input.
- Bars: OHLCV and VWAP, boundary alignment, empty bars, late trades, huge gaps.
- Ingest: against a local WebSocket server that closes mid-stream, replays a
  trade and skips two, so reconnect, de-duplication and gap counting are
  exercised end to end. Also an unreachable server and a far-future timestamp.
- Pipeline: replay of a 400-message fixture checks that volume and trade count
  are conserved across bars and that the CSV matches.

## Live run

A 25-minute run against the live BTCUSDT stream on the author's machine ingested
about 50,000 trades with no disconnects, gaps or parse errors. That run exposed
one defect, fixed in this PR: the first bar after start was emitted although it
began mid-minute. Live mode now drops it and counts it in `partial_dropped`.
Details are in `BENCH.md`.

## Known limits

- No backfill of trades missed during an outage.
- No staleness alarm for a connected but silent stream.
- CSV writes are synchronous on the event loop.
- A bar only closes when a later trade arrives.

## Review checklist

- [ ] I can explain why bars use event time and not arrival time.
- [ ] I can explain what happens to `last_id` across a reconnect.
- [ ] I ran `uv run pytest -q` and the live command on my machine.
- [ ] I filled in `BENCH.md` from my own run.
