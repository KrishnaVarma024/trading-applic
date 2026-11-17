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
