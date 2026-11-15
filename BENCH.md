# Benchmarks

Numbers go here only after a script on a named machine produced them.

## Phase 1: live soak run

Command: `uv run python -m tradingapp live --symbol btcusdt --interval 60 --save-ticks`

| Field | Value |
| --- | --- |
| Machine | MacBook Pro, Apple silicon (arm64), 8 cores, macOS |
| Runtime | CPython 3.12.0, websockets 17.4 |
| Date and duration | 2026-11-15, 21:23:16 to 21:50:35 UTC, 27 min 19 s |
| Trades ingested | 52,656 |
| Complete 1-minute bars | 26 |
| Disconnects | 0 |
| Gaps / missed trades | 0 / 0 |
| Parse errors | 0 |
| Late dropped | 0 |
| Empty bars | 0 |

How these were checked: the figures come from the tick and bar CSV files the run
wrote, not from the log alone. The first and last trade ids are 6865719776 and
6865772431, a span of 52,656, which equals the number of tick rows, and every
consecutive pair of ids differs by exactly 1. So no trade was missed or
duplicated. The 26 bars are contiguous at 60 s spacing.

Defect found by this run: the first bar (21:23 UTC) was written as a full bar
although the stream was joined at 21:23:16. That row was removed from the data
file and live mode now drops the first partial bar (`partial_dropped`).

Not measured here: latency. That is phase 4.
