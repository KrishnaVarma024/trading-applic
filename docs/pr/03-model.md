# Phase 3: log-HAR plus boosted trees

## What this adds

- `research/features_ext.py`: 14 features aligned row-for-row with the phase 2
  dataset, each built from hours up to the forecast origin only.
- `research/data.py`: the loader now also returns volume and trade count, and
  the hourly series carries downside semivariance.
- `research/models.py`: `HarGbm` (log-HAR, then boosted trees on its residual)
  and `HarLogSeasonal` (a linear opponent with time-of-day terms).
- `research/evaluate_model.py`: walk-forward evaluation scored separately on a
  development period and a held-out test period. Output floats are rounded so
  the JSON is identical across machines.

## Result on the test period (4,415 hours, 2026-05-01 to 2026-10-31)

- Against tuned EWMA: QLIKE down 25.5% (95% CI 22.3% to 29.2%).
- Against log-HAR: down 11.7% (7.7% to 15.3%), DM p = 6.6e-11.
- Against seasonal log-HAR: down 8.5% (5.4% to 11.6%).
- Development period agreed: +10.4% over log-HAR.

## The finding that matters

An ablation shows nearly all of the gain is the time-of-day and weekend
pattern. Remove those three features and the model is no better than log-HAR.
Volume and trade count add nothing.

## Discipline

Tree settings and features are the first ones tried. Nothing was tuned. The
test period was scored twice with the model unchanged; the second time only
added a baseline. Details in the README.

## Tests

7 new, 44 in total, including future-corruption tests for the feature matrix
and the model, and a pair of tests on simulated data: one where the model must
find a planted pattern and one where it must not invent one.

## Review checklist

- [ ] I reran the command and got the same two tables.
- [ ] I can explain why the model is two stages and what trees cannot do.
- [ ] I can explain why there is a development period and a test period.
- [ ] I can say in one sentence where the gain comes from.
