# Phase 1: WebSocket ingest and bar builder

## Architecture

```
exchange WebSocket
      |  raw JSON text
      v
 ingest.stream_trades ---- reconnect with exponential backoff + jitter
      |                    stamps recv_ns on arrival
      v
 parse.parse_trade ------- rejects non-JSON, missing fields, bad price/qty
      |  Trade
      v
 ingest.SequenceTracker -- drops duplicates, counts missed trades by trade id
      |  Trade (clean, in order, counted)
      v
 pipeline.Pipeline ------> storage.DailyCsvSink (ticks, optional)
      |
      v
 bars.BarBuilder --------- event-time bars aligned to the epoch
      |  Bar (on close)
      v
 storage.DailyCsvSink (bars) and stdout
```

Each stage has one job and trusts the stage before it. Only `parse.py` touches
raw network data. Only `ingest.py` knows there is a network. `bars.py` is pure:
no clock, no I/O, so the same trades always produce the same bars.

## Run it

```bash
uv sync
uv run pytest -q                                              # 55 tests, no network needed
uv run python -m tradingapp replay tests/fixtures/sample_trades.jsonl
uv run python -m tradingapp live --symbol btcusdt --interval 60 --save-ticks
```

`live` writes `data/btcusdt-bars-60s-YYYY-MM-DD.csv` and prints each bar as it
closes. Stop it with Ctrl-C. If the default endpoint is unreachable from your
network, pass another with `--ws-base`.

## Design decisions

- **Event time, not wall-clock time.** Bars are bucketed by the exchange's trade
  timestamp. Local arrival time varies with the network; event time does not.
- **A bar closes when a later trade arrives.** There is no timer. This keeps the
  builder deterministic and replayable. The cost: in a dead market the last bar
  stays open until the next trade.
- **Quiet minutes become explicit empty bars** carrying the last close, so the
  bar series has no holes and realized volatility in phase 2 sees a zero return
  instead of a missing row. A gap longer than 1,440 bars is counted, not filled.
- **The first live bar is dropped.** The stream is joined mid-interval, so that
  bar is missing its early trades. It is counted in `partial_dropped`, not written.
- **Trade ids detect loss.** Ids increase by one, so after a reconnect the
  pipeline knows exactly how many trades it missed and logs it.
- **Counters over silence.** Every dropped, duplicated, late or rejected message
  increments a named counter that is logged every 30 seconds.

## The four production questions

**What happens under 100x load?** Parsing and bar updates are O(1) per trade, so
CPU is not the first limit. The synchronous CSV write on the event loop is: with
`--save-ticks` every trade costs a write call. If the loop falls behind, the
WebSocket receive buffer fills and the exchange disconnects a slow consumer,
which shows up as rising `disconnects` and `missed_trades`. Phase 4 measures
this and moves writes off the hot path.

**What can a malicious or broken sender do?** Messages are capped at 1 MiB.
Non-JSON, missing fields, non-finite or non-positive prices and negative
quantities are rejected in `parse.py`. A timestamp more than 60 s ahead of the
local clock is rejected, because one far-future trade would otherwise make every
later real trade look late. A huge time gap cannot force unbounded empty bars.
Not handled: a plausible but wrong price (a bad print) passes straight through.

**What happens when the network dies mid-operation?** `stream_trades` catches the
failure, waits with exponential backoff and full jitter (0.5 s base, 30 s cap),
and reconnects. Trades during the outage are lost: this stream has no replay.
The loss is measured by the trade-id gap, and bars for the outage are emitted as
empty. Not handled: backfilling the missed trades over REST.

**How would I know it broke in production?** The periodic stats line carries
`disconnects`, `gaps`, `missed_trades`, `parse_errors`, `duplicates`,
`rejected_future`, `late_dropped` and `empty_bars`. Any of the first three
rising is an alert. Not handled yet: a staleness alarm for "connected but no
messages for N seconds", and exporting these as real metrics.

## Known limits

- Only tested against synthetic messages in the exchange's documented shape.
  The first live run is recorded in `BENCH.md`.
- Out-of-order trades older than the open bar are dropped, not re-bucketed.
- The still-open bar is discarded on shutdown.
