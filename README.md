# trading-applic

A live market-data pipeline that ingests exchange trades, builds time bars, and
(in later phases) emits a realized-volatility forecast on every bar with
measured tick-to-forecast latency.

| Phase | Branch | What it adds | Status |
| --- | --- | --- | --- |
| 1 | `phase/01-ingest-bars` | WebSocket ingest, reconnects, gap detection, bar builder, CSV storage | merged |
| 2 | `phase/02-research-eval` | Baselines (persistence, EWMA, HAR-RV), walk-forward harness, QLIKE, leakage check | this PR |
| 3 | `phase/03-model` | Gradient-boosted forecast, comparison against each baseline | planned |
| 4 | `phase/04-live-latency` | Live per-bar inference, per-stage latency histogram | planned |

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
uv run pytest -q                                              # 37 tests, no network needed
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
that, a QLIKE reduction far outside the 10 to 20% range reported for this kind
of forecast, or direction accuracy from volatility features far from 50%,
should be treated as a bug until shown otherwise.

## Known limits of phase 2

- One symbol, one year, one horizon. No claim beyond that.
- Hourly realized variance from 1-minute returns includes microstructure noise.
- The target is close-to-close within the hour; overnight or weekend effects
  are not modelled, since this market does not close.
- Evaluation uses non-overlapping hours. Phase 4 forecasts on every bar and
  will need its own evaluation.
