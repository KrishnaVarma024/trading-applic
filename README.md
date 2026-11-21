# trading-applic

A real-time pipeline that ingests BTCUSDT trades from the exchange WebSocket,
builds one-minute bars, and forecasts next-hour realized volatility on every
bar close. The forecasting model is judged by walk-forward evaluation on a
held-out test period, and its live path is tested to compute exactly the
features it was trained on.

Built in five phases, each a pull request with its own write-up in
[`docs/design/`](docs/design). Every number below was produced by a command in
this repository; machines and raw output are in [`BENCH.md`](BENCH.md).

## Results

### Forecast accuracy

Next-hour realized variance, BTCUSDT. Models are refitted daily on an
expanding window and scored on a held-out test period, 2026-05-01 to
2026-10-31 (4,415 hours). QLIKE loss, lower is better. Intervals are 95%
moving-block bootstrap.

| Model | Test QLIKE | Reduction vs tuned EWMA | Reduction vs log-HAR |
| --- | --- | --- | --- |
| EWMA, decay tuned on training data | 0.3842 | | |
| log-HAR (Corsi, 2009) | 0.3241 | 15.7% (11.8 to 19.2) | |
| log-HAR + hour-of-day terms | 0.2959 | | 8.7% (5.6 to 12.1) |
| log-HAR + boosted-tree correction | **0.2862** | **25.5% (22.3 to 29.2)** | **11.7% (7.7 to 15.3)** |

- **Replication.** The same code on ETHUSDT, nothing retuned: the tree model
  beats log-HAR by 10.5% (6.8 to 14.2) on the test period.
- **Leakage check.** A classifier given the volatility features predicts
  next-hour price direction with 49.5% accuracy (48.4 to 50.7), as it should
  if no feature contains the future.

### Latency

In-process tick-to-forecast: from the raw message of the bar-closing trade
arriving in the process to the forecast value existing. It excludes the
network and the exchange.

Measured on the author's MacBook Pro (Apple silicon, macOS):

| Run | Inference | Forecasts | p50 | p90 | p99 |
| --- | --- | --- | --- | --- | --- |
| Replay | scikit-learn | 1,518 | 0.54 ms | 0.58 ms | 0.71 ms |
| Replay | compiled | 1,518 | 0.05 ms | 0.07 ms | 0.10 ms |
| Replay, 1 s idle before each bar | compiled | 120 | 0.73 ms | 1.08 ms | * |
| Live, 2 h 15 min | scikit-learn | 136 | 4.46 ms | 9.10 ms | * |
| **Live, 6 h 37 min overnight** | **compiled** | **398** | **0.79 ms** | **2.17 ms** | **9.1 ms** |

\* Too few samples for a 99th percentile. Neither live run had a disconnect,
missed trade or engine reset. Worst single forecast overnight: 39 ms.

## Findings

1. **Most of the model's gain is the clock.** Remove the hour-of-day and
   weekend features and the trees no longer beat log-HAR. A plain linear model
   with 24 hourly terms recovers about three quarters of their gain. What the
   trees add beyond it is about 3%: clear on the test period (0.5% to 5.9%),
   indistinguishable from zero on the development period.
   ([reports/ablation.md](reports/ablation.md))
2. **Volatility follows the US equity open.** The 14:00 and 15:00 UTC hours
   average more than twice the daily mean variance; weekends carry 46% of
   weekday variance. A market that never closes still keeps office hours.
   ([reports/ablation.md](reports/ablation.md))
3. **Fitting in logs is what beats EWMA.** HAR in levels is not significantly
   better than a tuned EWMA (5.1%, interval -2.1% to 12.4%); log-HAR is
   (15.5%). A few extreme hours dominate a least-squares fit in levels. The
   same holds on ETHUSDT.
4. **Live latency was an idle-state problem, not a code problem.** Live, every
   stage ran 7 to 42 times slower than in replay, including parsing, which does
   no I/O. Replaying with a one-second pause before each bar close reproduced
   the live median (4.8 ms against 4.46 ms, measured in a Linux VM on the same
   MacBook) with no network involved. After the fix, the overnight live median
   was 0.79 ms against 0.73 ms for the idle replay.
5. **Library overhead was the inference cost.** A one-row scikit-learn
   prediction took 0.5 to 0.6 ms, nearly all of it per-call overhead rather
   than tree walking. Walking the same 150 trees as flat arrays takes 31 us and
   matches scikit-learn on all 8,592 historical rows to 2e-15. Waking from idle
   still costs: the compiled path goes from 0.05 ms hot to 0.73 ms after a
   one-second pause.

## Architecture

```
  OFFLINE  (src/tradingapp/research)            LIVE  (src/tradingapp/live)

  kline archive, SHA-256 checked                exchange WebSocket
          |                                             |  stamped on arrival
          v                                             v
  hourly realized variance                      parse -> sequence check -> bar builder
          |                                             |  bar close
          v                                             v
  14 causal features   <---- same definitions ---> incremental feature engine
          |                  (tested to 1e-9)            ^           |
          v                                              |           v
  walk-forward fit and score                     REST: one week   compiled trees
  EWMA | HAR | log-HAR | log-HAR + trees         of history       (150 trees as arrays)
          |                                                          |
          v                                                          v
  QLIKE, Diebold-Mariano, bootstrap   train.py --> model file -->  forecast + per-stage latency
  dev/test split, leakage check
```

## How the results are protected

- **Walk-forward only.** Every forecast uses data available at its origin. The
  test corrupts all data after row k and requires every earlier forecast to be
  bit-for-bit unchanged, for every model.
- **A leakage check that can fail.** Planting the future return as a feature
  must push the direction check above 95%. It does, so its 49.5% means
  something.
- **One held-out period.** The test split was fixed before any score was seen.
  It was scored three times, each time only to add a baseline, which can only
  weaken the model's claim. Disclosed in the phase 3 write-up.
- **Baselines given a fair chance.** EWMA's decay is re-tuned at every refit; a
  linear model gets the same hour-of-day information as the trees.
- **Statistics that respect time dependence.** Diebold-Mariano with a Newey-West
  variance, and a moving-block bootstrap.
- **Train/serve parity.** Live features, computed from a rolling window of
  minute bars, equal research features at every hour boundary to 1e-9,
  including across the start-up seam where history meets the live stream.

## Reproduce

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest -q                                          # 67 tests, a few seconds, no network

uv run python -m tradingapp.research.download --symbol BTCUSDT --start 2025-11 --end 2026-10
uv run python -m tradingapp.research.evaluate             # baselines, about 2 s
uv run python -m tradingapp.research.evaluate_model       # model, development and test periods, about 30 s
uv run python -m tradingapp.research.ablation             # where the gain comes from (findings 1, 2), about 2 min
uv run python -m tradingapp.research.train                # writes models/har_gbm.joblib

uv run python -m tradingapp live --save-ticks             # record live trades to data/
uv run python -m tradingapp.live.bench data/btcusdt-ticks-*.csv
uv run python -m tradingapp.live.forecast                 # a forecast every minute; Ctrl-C prints latency
```

## Repository layout

```
src/tradingapp/
    parse.py, ingest.py, bars.py, storage.py, pipeline.py    ticks to bars
    research/     archive download, features, models, walk-forward, metrics, evaluation
    live/         feature engine, compiled trees, backfill, hot path, benchmark
tests/            67 tests
docs/design/      one write-up per phase: design, results, the four production questions, limits
docs/pr/          pull request descriptions
reports/          raw evaluation and latency output
BENCH.md          every measured number, with machine and command
BREAKS.md         what broke, the root cause, and the fix
```

## Limitations

- One year of data. ETHUSDT replicates the result, but both are crypto and
  share the same daily rhythm; an unrelated market might not.
- A lower QLIKE is not a trading profit. Nothing here tests a strategy or costs.
- Live forecast accuracy is unmeasured. That needs weeks of forecasts scored
  against the variance that followed them.
- The live 99th percentile rests on 398 forecasts from one night on one
  laptop, so on its four or five slowest values. A cluster of three slow
  forecasts at 21:41 to 21:44 UTC is unexplained.
- The model is trained once. There is no scheduled retraining or drift alarm.
- Open question: hourly return signs reverse 52.3% of the time. Four candidate
  causes were tested and ruled out; the mechanism is unknown.

## Build log

| Phase | Pull request | What it added | Write-up |
| --- | --- | --- | --- |
| 1 | [#1](https://github.com/KrishnaVarma024/trading-applic/pull/1) | WebSocket ingest, reconnects, gap detection, event-time bars | [01](docs/design/01-ingest-bars.md) |
| 2 | [#2](https://github.com/KrishnaVarma024/trading-applic/pull/2) | Archive data, baselines, walk-forward harness, QLIKE, leakage check | [02](docs/design/02-research-eval.md) |
| 3 | [#3](https://github.com/KrishnaVarma024/trading-applic/pull/3) | log-HAR plus boosted trees, held-out test period, ablation | [03](docs/design/03-model.md) |
| 4 | [#4](https://github.com/KrishnaVarma024/trading-applic/pull/4) | Live forecasting, train/serve parity, latency measurement | [04](docs/design/04-live-latency.md) |
| 5 | [#5](https://github.com/KrishnaVarma024/trading-applic/pull/5), [#6](https://github.com/KrishnaVarma024/trading-applic/pull/6) | Idle-latency diagnosis, compiled trees, warm-up, scheduling class | [05](docs/design/05-latency.md) |
| 6 | #7 | One command reproduces the ablation, hourly baseline and hour-of-day profile | [03](docs/design/03-model.md) |
