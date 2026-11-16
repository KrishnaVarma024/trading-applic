# Phase 2 results: BTCUSDT

Data: 525,600 one-minute bars, 2025-11-01 to 2026-10-31. 0 minutes missing, 0 hours dropped, 8,760 hours used.

Setup: target is realized variance of the next hour, from 1-minute log returns. Expanding window, first 60 days for training only, refit every 24 hours. Evaluated on 7,152 hours, 2026-01-06 to 2026-10-31.

| Model | QLIKE | RMSE | QLIKE reduction vs EWMA | 95% CI | DM p-value |
| --- | --- | --- | --- | --- | --- |
| persistence | 0.4471 | 4.170e-05 | -17.5% | -22.4% to -13.1% | 1.7e-10 |
| ewma (tuned, final lambda 0.54) | 0.3804 | 3.638e-05 | baseline | | |
| har | 0.3610 | 3.536e-05 | +5.1% | -2.1% to +12.4% | 0.18 |
| har_log | 0.3213 | 3.507e-05 | +15.5% | +12.8% to +18.3% | 1.1e-17 |

## Direction leakage check

A logistic classifier on the volatility model's own features predicted next-hour direction with accuracy 49.5% (95% CI 48.4% to 50.7%, n = 7,152). Always guessing the majority class scores 50.2%. The interval contains 50%, so these features carry no detectable information about direction.

## Diagnostic: adding past returns

With the last-hour and last-day returns added, accuracy is 52.2% (95% CI 51.1% to 53.4%). The sign of the hourly return flips from one hour to the next 52.3% of the time, a small reversal tendency in this sample. The cause is not established here and nothing shows it survives trading costs. Look-ahead is ruled out separately: the harness tests show no feature reads past its origin, and the edge vanishes when the features are lagged one more hour.
