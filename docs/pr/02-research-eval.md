# Phase 2: baselines and walk-forward evaluation

## What this adds

The evaluation that every later model will be judged by, and the baselines it
must beat. No machine-learning model yet, on purpose.

- `research/download.py`: monthly 1-minute klines from the public archive,
  SHA-256 verified, atomic writes, resumable.
- `research/data.py`: hourly realized variance and return; hours with missing
  minutes are dropped and counted.
- `research/features.py`: HAR features (hour, day, week) with the target
  defined as the next hour, so row t never contains anything after hour t.
- `research/models.py`: persistence, EWMA with the decay re-tuned at each
  refit, HAR and log-HAR.
- `research/walkforward.py`: expanding-window walk-forward harness.
- `research/metrics.py`: QLIKE, RMSE, Diebold-Mariano test with a Newey-West
  variance, moving-block bootstrap interval, Wilson interval.
- `research/direction.py`: the leakage check.
- `research/evaluate.py`: runs everything, writes `reports/phase2.md` and JSON.

## Result

On BTCUSDT, 7,152 out-of-sample hours (2026-01-06 to 2026-10-31):

- log-HAR reduces QLIKE by 15.5% against a tuned EWMA (95% CI 12.8% to 18.3%,
  DM p = 1.1e-17).
- HAR in levels is not significantly better than EWMA (p = 0.18).
- Direction classifier on the model's features: 49.5% (CI 48.4% to 50.7%).
- With past returns added: 52.2%. Investigated, shown not to be look-ahead,
  cause not established. Reported in the README.

## How it was tested

16 new tests, 37 in total. The important ones corrupt the future and require
past forecasts to be unchanged, and plant a look-ahead feature and require the
leakage check to catch it.

## Not done here

- No model beyond the baselines. That is phase 3, and its bar is log-HAR.
- No live forecasting or latency. That is phase 4.

## Review checklist

- [ ] I reran both commands and got the same table.
- [ ] I can explain why row t's target is hour t+1 and what would break if it were hour t.
- [ ] I can explain why QLIKE and not RMSE.
- [ ] I can explain why the 52.2% is not a leak, and what would convince me it was.
