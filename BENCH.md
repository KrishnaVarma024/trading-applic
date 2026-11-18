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

Reproduced on the author's MacBook Pro on 2026-11-16 with an identical table.

## Phase 3: model evaluation

Command: `uv run python -m tradingapp.research.evaluate_model --symbol BTCUSDT`
Full output: `reports/phase3.md`, `reports/phase3_results.json`.

Test period, 4,415 hours, 2026-05-01 to 2026-10-31:

| Model | QLIKE | vs tuned EWMA | vs log-HAR |
| --- | --- | --- | --- |
| EWMA (tuned) | 0.3842 | | |
| log-HAR | 0.3241 | +15.7% | |
| Seasonal log-HAR | 0.3127 | +18.6% | +3.5% |
| log-HAR + trees | 0.2862 | +25.5% (CI 22.3% to 29.2%) | +11.7% (CI 7.7% to 15.3%) |

Reproduced on the author's MacBook Pro (Python 3.12, scikit-learn 1.9.1) on
2026-11-16 with identical output.

## Phase 4: tick-to-forecast latency

In-process only: from the raw message of the bar-closing trade being in the
process to the forecast value existing. Excludes the network, the exchange,
socket handling and writing the result anywhere.

Machine: author's MacBook Pro, Apple silicon (arm64), macOS, Python 3.12.0,
scikit-learn 1.9.1, inference pinned to one thread, garbage collector on.
Raw output: `reports/latency_bench.json`, `reports/latency_live.json`.

### Replay benchmark

Command: `uv run python -m tradingapp.live.bench data/btcusdt-ticks-*.csv`
82,704 recorded trades in 2 sessions, 1-second bars, 1,518 timed forecasts
after 200 warm-up forecasts were discarded.

| Stage | n | p50 | p90 | p99 | p99.9 | max |
| --- | --- | --- | --- | --- | --- | --- |
| total | 1,518 | 567.3 | 614.9 | 933.3 | 1800.5 | 2394.0 |
| parse | 1,518 | 2.6 | 3.5 | 9.3 | 32.0 | 83.7 |
| bar | 1,518 | 1.8 | 3.4 | 6.1 | 9.8 | 11.3 |
| features | 1,518 | 18.3 | 35.2 | 69.0 | 101.0 | 219.1 |
| model | 1,518 | 538.5 | 584.6 | 881.8 | 1676.2 | 2314.0 |

Microseconds. Model inference is 95% of the median total.

### Live run

Command: `uv run python -m tradingapp.live.forecast`, stopped with Ctrl-C.
2026-11-16, 19:05 to 21:20 UTC, 2 h 15 min. 136 forecasts, 0 disconnects,
0 missed trades, 0 engine resets, 0 degraded forecasts, 137 contiguous bars
with none empty. Start-up backfill of 10,081 klines took 9.4 s.

| Stage | n | p50 | p90 | p99 | p99.9 | max |
| --- | --- | --- | --- | --- | --- | --- |
| total | 136 | 4456.8 | 9103.6 | 25812.3 | 33426.5 | 34494.4 |
| parse | 136 | 108.7 | 161.5 | 927.7 | 6379.6 | 7184.9 |
| bar | 136 | 39.8 | 106.0 | 200.6 | 283.4 | 294.2 |
| features | 136 | 356.9 | 1340.1 | 9558.2 | 10854.0 | 10975.1 |
| model | 136 | 3905.5 | 7159.1 | 14885.2 | 31437.4 | 33940.7 |

Microseconds.

- The worst value, 34.5 ms, is the first forecast after start, the first
  inference call on a freshly loaded model. Excluding it the worst is 26.6 ms.
- 11 of 136 forecasts took over 10 ms; 4 took over 20 ms, the first included.
- With 136 samples the p99 and p99.9 columns are interpolations between the
  two or three largest values, not stable statistics. Quote the median and the
  count above a threshold instead, or run much longer.

### Live is about eight times slower than replay, in every stage

| Stage | Replay p50 | Live p50 | Ratio |
| --- | --- | --- | --- |
| parse | 2.6 | 108.7 | 42x |
| bar | 1.8 | 39.8 | 22x |
| features | 18.3 | 356.9 | 20x |
| model | 538.5 | 3905.5 | 7x |
| total | 567.3 | 4456.8 | 7.9x |

The parse and bar code is identical in both runs and does no I/O, yet it
slowed down most. When code that did not change runs slower, the likeliest
cause is the machine running it slower, not the code.

Probable cause, not yet confirmed: live, the process is nearly idle (a trade
every 30 ms or so, a forecast once a minute). macOS tends to keep such a
process on efficiency cores at a low clock, and its caches go cold between
bars. The replay keeps one core busy, so it runs on a performance core at full
speed with warm caches. Cold caches hurt short code paths the most, which fits
the cheap stages slowing down 20 to 40 times and the model only 7.

Not controlled: other applications running on the Mac during the live run.

How to confirm: replay the same ticks with an idle pause of about a second
before each bar-closing trade. If replay latency rises to the live figure, the
cause is idle-state behaviour, not the live network code.

## Phase 5: compiled inference and the idle test

### Build VM measurements

Linux VM on the author's MacBook, 2026-11-17. Same hardware, but a different
operating system and scheduler, so these are indicative only.

| Replay | Inference | n | p50 | p90 | p99 |
| --- | --- | --- | --- | --- | --- |
| No idle | scikit-learn | 1,518 | 0.672 ms | 0.797 ms | 1.36 ms |
| No idle | compiled | 1,518 | 0.051 ms | 0.068 ms | 0.100 ms |
| 1 s idle before each bar close | scikit-learn | 120 | 4.80 ms | 5.81 ms | 17.8 ms |
| 1 s idle before each bar close | compiled | 120 | 0.95 ms | 1.49 ms | 11.2 ms |

Compiled inference agrees with scikit-learn on all 8,592 historical rows of
the real model to a worst relative difference of 2.0e-15.

### macOS measurements

Author's MacBook Pro, Apple silicon, macOS, Python 3.12.0, 2026-11-18.
Raw output: `reports/bench_sklearn.json`, `reports/bench_compiled.json`,
`reports/bench_idle_default.json`, `reports/bench_idle_interactive.json`.

| Replay | Inference | Scheduling class | n | p50 | p90 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| No idle | scikit-learn | interactive | 1,518 | 0.539 ms | 0.581 ms | 0.711 ms | 0.855 ms |
| No idle | compiled | interactive | 1,518 | 0.051 ms | 0.067 ms | 0.098 ms | 0.130 ms |
| 1 s idle before each bar close | compiled | default | 120 | 0.802 ms | 1.086 ms | 3.011 ms | 3.906 ms |
| 1 s idle before each bar close | compiled | interactive | 120 | 0.730 ms | 1.080 ms | 1.676 ms | 1.982 ms |

- Compiled inference: model stage 512 us to 31 us at the median, total 0.54 ms
  to 0.05 ms. About ten times faster end to end.
- The scikit-learn replay (0.54 ms) matches phase 4's (0.57 ms), so the setup
  is comparable.
- Idling still costs about fifteen times: 0.05 ms hot against 0.73 to 0.80 ms
  after a one-second pause. The cost is spread over parsing, features and the
  model, which is what cold caches and a slow-to-wake core would produce.
- Scheduling class: the user-interactive request gave a 9% lower median and a
  smaller worst case, but an identical p90. One run of 120 forecasts each, not
  interleaved, so this does not separate the effect from run-to-run noise. It
  is kept because it costs nothing, and it is not credited with any of the
  improvement.

### Live run with compiled inference

`caffeinate -i uv run python -m tradingapp.live.forecast`, stopped with Ctrl-C.
2026-11-17 19:49 to 2026-11-18 02:26 UTC, 6 h 37 min. 398 forecasts, 0
disconnects, 0 missed trades, 0 engine resets, 0 degraded forecasts.
Raw output: `reports/latency_live_compiled.json`.

| Stage | n | p50 | p90 | p99 | p99.9 | max |
| --- | --- | --- | --- | --- | --- | --- |
| total | 398 | 789.5 | 2172.0 | 9141.2 | 37012.9 | 39018.3 |
| parse | 398 | 98.6 | 154.3 | 287.8 | 1016.8 | 1400.9 |
| bar | 398 | 31.9 | 91.0 | 244.2 | 700.7 | 790.7 |
| features | 398 | 335.6 | 1349.0 | 6231.2 | 23082.4 | 26813.7 |
| model | 398 | 301.5 | 602.0 | 2397.0 | 24301.0 | 32381.0 |

Microseconds.

- Against the phase 4 live run (scikit-learn, 136 forecasts): median 4.46 ms to
  0.79 ms, worst 34.5 ms to 39.0 ms.
- 52 of 398 forecasts took over 2 ms, 10 over 5 ms, 4 over 10 ms, 2 over 20 ms.
  With 398 samples the p99 rests on the four or five slowest values.
- Three of the four slowest came in one four-minute stretch (21:41 to 21:44
  UTC: 39.0, 34.0 and 11.0 ms). The cause is not known.
- The hourly median was 1.0 to 1.6 ms until 23:00 UTC and 0.6 to 0.8 ms after.
  Whether the earlier hours were slower because the market was busier or
  because the Mac was still in use is not separated.
- The live median (0.79 ms) sits close to the idle replay (0.73 ms), which
  supports using the idle replay as a stand-in for live latency.
