# trading-applic

A live market-data pipeline that ingests exchange trades, builds time bars, and
(in later phases) emits a realized-volatility forecast on every bar with
measured tick-to-forecast latency.

| Phase | Branch | What it adds | Status |
| --- | --- | --- | --- |
| 1 | `phase/01-ingest-bars` | WebSocket ingest, reconnects, gap detection, bar builder, CSV storage | merged |
| 2 | `phase/02-research-eval` | Baselines (persistence, EWMA, HAR-RV), walk-forward harness, QLIKE, leakage check | merged |
| 3 | `phase/03-model` | log-HAR plus boosted trees, development/test split, ablation | merged |
| 4 | `phase/04-live-latency` | Live per-bar forecasts, train/serve parity, per-stage latency | this PR |

## Phase 1 architecture

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

---

# Phase 2: baselines and walk-forward evaluation

Before any model is trained, this phase fixes how a model will be judged and
what it has to beat.

## Question

Given everything known at the end of an hour, how well can the realized
variance of the next hour be forecast, and how much of that is already
captured by simple methods?

## Research architecture

```
public kline archive (1-minute bars, SHA-256 verified)
      |
      v
 research.data ---------- hourly realized variance = sum of squared 1-min log returns
      |                   hours with missing minutes are dropped and counted
      v
 research.features ------ row t: features from hours <= t, target = RV of hour t+1
      |
      v
 research.walkforward --- fit on rows [0, start), forecast the next 24 rows, repeat
      |        \
      |         models: persistence, EWMA (decay re-tuned each refit), HAR, log-HAR
      v
 research.metrics ------- QLIKE, RMSE, Diebold-Mariano test, block-bootstrap CI
      |
      v
 research.direction ----- leakage check: can the same features predict direction?
      |
      v
 reports/phase2.md and reports/phase2_results.json
```

## Reproduce

```bash
uv run python -m tradingapp.research.download --symbol BTCUSDT --start 2025-11 --end 2026-10
uv run python -m tradingapp.research.evaluate --symbol BTCUSDT
```

The evaluation takes about two seconds and has no randomness except a seeded
bootstrap, so the numbers below should reproduce exactly.

## Results

BTCUSDT, 525,600 one-minute bars from 2025-11-01 to 2026-10-31, none missing.
The first 60 days are training only. Evaluated out of sample on 7,152 hours
from 2026-01-06 to 2026-10-31, refitting every 24 hours on an expanding window.

| Model | QLIKE | QLIKE reduction vs tuned EWMA | 95% CI | DM p-value |
| --- | --- | --- | --- | --- |
| Persistence | 0.4471 | -17.5% | -22.4% to -13.1% | 1.7e-10 |
| EWMA, decay tuned on training data | 0.3804 | baseline | | |
| HAR | 0.3610 | +5.1% | -2.1% to +12.4% | 0.18 |
| log-HAR | 0.3213 | +15.5% | +12.8% to +18.3% | 1.1e-17 |

What this says:

- log-HAR cuts QLIKE by 15.5% against a tuned EWMA, and the interval excludes zero.
- HAR in levels is not distinguishable from EWMA. Its interval spans zero.
  Fitting in logs is what makes the difference, because a few extreme hours
  dominate a least-squares fit in levels.
- The bar for phase 3 is log-HAR at 0.3213, not EWMA. A model that only beats
  EWMA has shown nothing HAR does not already give.

**Leakage check.** A classifier on the volatility model's own features predicts
next-hour direction with 49.5% accuracy (95% CI 48.4% to 50.7%). No detectable
information about direction, as expected if no feature contains the future.

**A finding I did not expect.** Adding past returns as features lifts direction
accuracy to 52.2% (CI 51.1% to 53.4%). Hourly return signs flip 52.3% of the
time in this sample. The cause is not established and nothing here shows it
survives trading costs. It is not look-ahead: the edge vanishes when features
are lagged one more hour, and the tests below prove no feature reads past its
origin. It is reported, not explained.

## How the harness is prevented from fooling me

- `test_walk_forward_cannot_see_the_future`: every row from index k onward is
  multiplied by 1000, and forecasts before k must be bit-for-bit identical, for
  all four models.
- `test_target_is_the_next_hour_and_features_stop_at_the_origin`: the same idea
  one level down, on the feature builder.
- `test_a_planted_look_ahead_feature_is_caught`: the future return is added as
  a feature on purpose and the direction check must jump above 95%. A check
  that cannot fail proves nothing.
- `test_known_ranking_on_clustered_volatility`: on simulated data with known
  volatility clustering, every model must beat persistence.
- The EWMA decay is re-tuned on training rows at every refit, so the baseline
  is not handicapped.

## The four production questions, for research code

**100x data?** One year hourly is 8,760 rows and runs in two seconds. The loader
holds every 1-minute close in memory, about 4 MB a year, so 100 symbol-years is
fine and 10,000 needs a columnar store. The EWMA recursion is a Python loop and
would be the first thing to vectorise.

**A malicious or corrupt input?** Every archive file is checked against the
published SHA-256 before it is written, and written through a temporary file so
a crash cannot leave a truncated file under the real name. Not handled: a
checksum only proves the file is what the archive published, not that the
prices in it are right.

**Network dies mid-download?** The run stops with an error. Files already
verified stay; rerunning skips them and resumes.

**How would I know the evaluation is wrong?** The three tests above fail. Beyond
that, a QLIKE reduction much larger than log-HAR's, or direction accuracy from
volatility features far from 50%, should be treated as a bug until shown
otherwise.

## Known limits of phase 2

- One symbol, one year, one horizon. No claim beyond that.
- Hourly realized variance from 1-minute returns includes microstructure noise.
- The target is close-to-close within the hour; overnight or weekend effects
  are not modelled, since this market does not close.
- Evaluation uses non-overlapping hours. Phase 4 forecasts on every bar and
  will need its own evaluation.

---

# Phase 3: the model

## Question

Phase 2 set the bar: log-HAR at a 15.5% QLIKE reduction over a tuned EWMA.
Can a model beat log-HAR on data that played no part in building it?

## Design

```
hourly series (RV, return, semivariance, volume, trade count)
      |
      v
 features_ext.build_matrix --- 14 features, each from hours <= the origin
      |
      v
 models.HarGbm
      stage 1: log-HAR, linear        carries the level, can extrapolate
      stage 2: boosted trees          predict stage 1's residual from all 14 features
      forecast = exp(stage 1 + stage 2 + half the residual variance)
      |
      v
 evaluate_model --- walk-forward, then scored on two separate periods
      development: 2026-01-06 to 2026-04-30   choices may be made here
      test:        2026-05-01 to 2026-10-31   scored at the end
```

Why two stages: trees predict by averaging training targets in a leaf, so they
cannot forecast a variance higher than any they have seen. The linear stage
can. Letting the trees correct a linear model keeps that property.

The trees are scikit-learn's `HistGradientBoostingRegressor`, the same family
as LightGBM, chosen because it needs no extra native library to install.

## Reproduce

```bash
uv run python -m tradingapp.research.evaluate_model --symbol BTCUSDT
```

About 30 seconds. `--dev-only` scores the development period without touching
the test period.

## Results

Test period, 4,415 hours from 2026-05-01 to 2026-10-31. Intervals are 95%
block-bootstrap; p-values are Diebold-Mariano.

| Model | QLIKE | vs tuned EWMA | vs log-HAR | vs seasonal log-HAR |
| --- | --- | --- | --- | --- |
| EWMA (tuned) | 0.3842 | | | |
| log-HAR | 0.3241 | +15.7% (+11.8% to +19.2%) | | |
| Seasonal log-HAR | 0.3127 | +18.6% (+15.3% to +21.6%) | +3.5% (+1.5% to +5.3%) | |
| log-HAR + trees | 0.2862 | +25.5% (+22.3% to +29.2%) | +11.7% (+7.7% to +15.3%) | +8.5% (+5.4% to +11.6%) |

The development period gave +10.4% over log-HAR (+3.8% to +16.0%), so the test
result is in line with it and not a lucky draw.

## Where the gain comes from

Ablation on the development period only, removing one feature group at a time:

| Removed | QLIKE | vs log-HAR |
| --- | --- | --- |
| nothing | 0.2840 | +10.4% |
| time of day and weekend | 0.3208 | -1.3% |
| volume and trade count | 0.2835 | +10.5% |
| returns and semivariance | 0.2876 | +9.2% |
| 4-hour and 12-hour variance | 0.2914 | +8.0% |

Almost all of the gain is the time-of-day and weekend pattern. Without it the
trees are no better than log-HAR. Volume and trade count add nothing once
variance itself is in the model. So the plain reading is: hourly volatility in
this market has a strong daily rhythm, and the trees learn its shape. A linear
model given the same rhythm as one sine wave recovers about a third of the
gain (the seasonal log-HAR row). The rest is the shape being more complicated
than one sine wave, or interactions, and this phase does not separate the two.

## Follow-up: a stronger linear opponent

After the results above, a linear log-HAR with a separate term for each hour
of the day (23 indicator variables plus the weekend flag) was scored on the
same periods. It is not in the code; it is assignment 3 of this phase.

| Test period | QLIKE | vs log-HAR |
| --- | --- | --- |
| log-HAR | 0.3241 | |
| log-HAR + hourly indicators | 0.2959 | +8.7% (+5.6% to +12.1%) |
| log-HAR + trees | 0.2862 | +11.7% (+7.7% to +15.3%) |

The trees beat this stronger linear model by 3.3% on the test period (+0.5%
to +5.9%, p = 0.016) and by 1.7% on the development period, where the interval
spans zero. So roughly three quarters of the trees' gain over log-HAR is a
daily pattern that 24 plain coefficients capture. What the trees add beyond a
well-specified linear model is small, about 3%, and only clearly present on
one of the two periods. This was a third look at the test period, again to add
a baseline.

## Reproducibility note

The tree model's numbers depend on the scikit-learn version in the fourth
decimal place. The figures here are from Python 3.12 with the versions in
`uv.lock`, which the project now requires. The linear models are identical on
any version.

## What was and was not tuned

- The tree settings are the first ones tried. Nothing was tuned on either period.
- The feature list was written before any score was seen and not changed.
- The ablation used the development period only.
- The test period was scored twice with the model unchanged: once for the
  main table, and once more to add the seasonal log-HAR row. That baseline was
  added after seeing the first test result, to check whether a simpler model
  explained it. An added baseline can only weaken the model's claim, not help it.

## Tests added

- `test_feature_matrix_shape_and_no_look_ahead` and
  `test_model_walk_forward_cannot_see_the_future`: corrupt the future, require
  earlier features and forecasts to be bit-for-bit unchanged.
- `test_model_learns_a_pattern_har_cannot_express`: on simulated data with a
  planted daily rhythm, the model must beat log-HAR by at least 5%.
- `test_model_does_not_invent_signal_where_there_is_none`: on simulated data
  with no extra structure, it must not be more than 5% worse.

## The four production questions, for the model

**100x data?** Refitting daily costs about 30 seconds a year of hourly data,
dominated by tree fitting. At 100 symbols, refit weekly or fit incrementally.

**A malicious or corrupt input?** A corrupt volume or variance value in the
latest hour reaches the trees as an out-of-range feature. Trees clamp it to the
nearest leaf, so the correction stays bounded, but the linear stage does not:
a wildly wrong last-hour variance moves the forecast in proportion. Phase 4
needs an input sanity check before inference.

**Network dies?** Not applicable offline. Live, a missing hour means stale
features. Phase 4 must decide whether to forecast on stale inputs or abstain.

**How would I know it broke?** Track live QLIKE of the model against log-HAR
on a rolling window. If the model stops beating its own first stage, the trees
have gone stale or the daily rhythm has changed.

## Known limits of phase 3

- One symbol, one year. The daily rhythm may differ in another year or market.
- The seasonal linear baseline in the table uses one sine wave. See the
  follow-up below for a stronger one.
- No hyperparameter search. The model may be better or worse than a tuned one.
- QLIKE gains are not profits. Nothing here tests a trading use.

---

# Phase 4: live forecasting and latency

## Question

Can the phase 3 model run on the live stream, produce the same features it was
trained on, and how long does a forecast take from the trade that triggers it?

## Architecture

```
            start-up, once                              every trade
 REST klines (one week) ----+            exchange WebSocket
                            |                    |  raw message, stamped recv_ns
                            v                    v
                    live.engine.FeatureEngine <- parse -> sequence check -> BarBuilder
                    (a week of per-bar history)               |
                            |                       bar closed? no -> done
                            v  14 features                    | yes
                    live.forecaster.Forecaster  <-------------+
                    log-HAR + boosted trees
                            |
                            v  forecast ready: final timestamp taken here
                    ---- timing stops ----
                            |
                            v
                    CSV rows (bars, forecasts), console line
```

Start-up joins two sources. The engine needs a week of bars, so it cannot wait
for the stream to supply them. The program connects, waits for the first
complete live bar, then fetches the week of klines that ends one minute before
that bar. `join_history` refuses to proceed unless the two meet exactly.

Each forecast is the realized variance expected over the next 60 minutes,
printed as annualised volatility. If any of the last 60 bars had no trades,
the forecast is marked degraded, because for this symbol an empty minute means
data was lost, not that nothing traded.

## Run it

```bash
uv run python -m tradingapp.research.train            # fits on data/hist, writes models/har_gbm.joblib
uv run python -m tradingapp.live.bench data/btcusdt-ticks-*.csv
uv run python -m tradingapp.live.forecast             # live; first forecast about two minutes after start
```

## What the latency number is

Tick-to-forecast: from the moment the raw message of the bar-closing trade is
in the process, to the moment the forecast value exists. It covers parsing,
the sequence check, the bar update, feature computation and model inference.

It does not include the network, the exchange, the kernel's socket handling,
or writing the result anywhere. It is an in-process figure and should always
be quoted with that word.

Two ways to measure it:

- `live.bench` replays recorded ticks with 1-second bars. Same code path,
  thousands of samples in seconds. The forecasts themselves are meaningless at
  that bar width; only the timing counts.
- `live.forecast` measures real forecasts, one a minute, and writes
  `reports/latency_live.json` on exit. An hour gives 60 samples, which is
  enough for a median and not for a 99th percentile.

Measured on the author's MacBook Pro (details in `BENCH.md`):

| Run | Forecasts | p50 | Worst | Over 10 ms |
| --- | --- | --- | --- | --- |
| Replay benchmark | 1,518 | 0.57 ms | 2.4 ms | 0 |
| Live, 2 h 15 min | 136 | 4.5 ms | 34.5 ms (first call); 26.6 ms after | 11 |

The live run had no disconnects, missed trades, resets or degraded forecasts.

## What the measurements showed while building

- Model inference is about 95% of the total. Features take tens of
  microseconds; parsing and the bar update take single digits.
- Most of the inference time is the library validating its input on every
  call, not walking the trees.
- The tree library hands even a one-row prediction to a thread pool. Pinning
  it to one thread cut the 99th percentile roughly threefold in the build
  environment, with no change to the median. `bench --threads` turns the pool
  back on so the effect can be seen.

## Train/serve parity

Research computes features from hourly arrays. Live computes them from a
rolling window of minute bars. They are two implementations of one definition,
and a silent difference between them would make every live forecast wrong
while every offline number stayed good.

`test_live_features_equal_research_features_on_the_hour` feeds nine days of
minute bars through both and requires all 14 features to agree to 1e-9 at
every hour boundary. `test_start_up_seam_joins_history_and_live_bars_without_a_gap`
does the same across the start-up join.

## The four production questions

**100x load?** Per-trade work is parse plus a bar update, a few microseconds,
so 100 times the trade rate is still far from saturating one core. The model
runs once per bar regardless of trade rate. The limit is the single event
loop: a forecast blocks ingest for its duration, about a millisecond.

**A malicious or corrupt input?** Trades pass the phase 1 parser. Backfilled
klines are checked for holes and for ending at the expected minute. The model
file is a joblib pickle, which can execute code when loaded, so only a file
this project wrote should ever be loaded; the loader also refuses a file from
a different library version or feature list. Not handled: a plausible but
wrong price still reaches the model.

**Network dies mid-operation?** The stream reconnects as in phase 1. Missed
minutes become empty bars, forecasts in the following hour are marked
degraded, and a hole longer than the builder will fill resets the engine,
which then needs a new week of history. Not handled: re-running the backfill
automatically after such a reset. If the start-up backfill fails, the program
stops with an error rather than forecasting on partial history.

**How would I know it broke?** `degraded` forecasts, `resets`, and the phase 1
counters are logged every five minutes with the latency summary. Not built
yet: scoring live forecasts against what then happened, which is the only
check that the model still works.

## Known limits of phase 4

- Live latency is about eight times the replay figure, in every stage. The
  probable cause is the process idling between bars; it is not confirmed.
- The live run was 2 h 15 min, 136 forecasts: enough for a median, not for a
  99th percentile.
- The model was trained on forecasts made on the hour. Live forecasts are made
  every minute, with windows that end mid-hour. The features are defined the
  same way, but accuracy at off-hour origins has not been evaluated.
- Live forecast accuracy is not measured at all yet. That needs weeks of
  forecasts and the realized variance that followed each one.
- The model is trained once. There is no scheduled retraining.
