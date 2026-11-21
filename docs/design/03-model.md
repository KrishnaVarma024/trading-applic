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
same periods. It was first run as a one-off script (assignment 3 of this phase);
it is now `uv run python -m tradingapp.research.ablation`, which also produces the
ablation table above and the hour-of-day profile, and writes `reports/ablation.md`.

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
