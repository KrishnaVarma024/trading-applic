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

## Phase 2: baseline evaluation

Command: `uv run python -m tradingapp.research.evaluate --symbol BTCUSDT`
Data: BTCUSDT 1-minute klines, 2025-11 to 2026-10, from the public archive.
Full output: `reports/phase2.md`, `reports/phase2_results.json`.

| Model | QLIKE | Reduction vs tuned EWMA | 95% CI |
| --- | --- | --- | --- |
| Persistence | 0.4471 | -17.5% | -22.4% to -13.1% |
| EWMA (tuned) | 0.3804 | baseline | |
| HAR | 0.3610 | +5.1% | -2.1% to +12.4% |
| log-HAR | 0.3213 | +15.5% | +12.8% to +18.3% |

Direction check on model features: 49.5% (CI 48.4% to 50.7%), n = 7,152.

First produced in the build environment on 2026-11-16. Rerun on your own
machine and confirm the table matches before quoting it.

## Phase 3: model evaluation

Command: `uv run python -m tradingapp.research.evaluate_model --symbol BTCUSDT`
Full output: `reports/phase3.md`, `reports/phase3_results.json`.

Test period, 4,415 hours, 2026-05-01 to 2026-10-31:

| Model | QLIKE | vs tuned EWMA | vs log-HAR |
| --- | --- | --- | --- |
| EWMA (tuned) | 0.3842 | | |
| log-HAR | 0.3241 | +15.7% | |
| Seasonal log-HAR | 0.3127 | +18.6% | +3.5% |
| log-HAR + trees | 0.2860 | +25.6% (CI 22.4% to 29.2%) | +11.8% (CI 7.7% to 15.4%) |

First produced in the build environment on 2026-11-16. Rerun on your own
machine and confirm before quoting.
