# Phase 3 results: BTCUSDT

Expanding window, first 60 days training only, refit every 24 hours. Development: 2,737 hours before 2026-05-01. Test: 4,415 hours from 2026-05-01.

## Development period

| Model | QLIKE | vs tuned EWMA | vs log-HAR | vs seasonal log-HAR |
| --- | --- | --- | --- | --- |
| ewma | 0.3742 |  |  |  |
| har_log | 0.3168 | +15.3% (+11.4% to +18.9%), p = 1.6e-09 |  |  |
| har_log_seasonal | 0.3102 | +17.1% (+13.5% to +20.5%), p = 1.2e-13 | +2.1% (-0.3% to +4.5%), p = 0.087 |  |
| har_gbm | 0.2848 | +23.9% (+17.4% to +29.7%), p = 3e-13 | +10.1% (+3.4% to +15.9%), p = 0.00071 | +8.2% (+2.0% to +13.4%), p = 0.0041 |

## Test period

| Model | QLIKE | vs tuned EWMA | vs log-HAR | vs seasonal log-HAR |
| --- | --- | --- | --- | --- |
| ewma | 0.3842 |  |  |  |
| har_log | 0.3241 | +15.7% (+11.8% to +19.2%), p = 6.4e-10 |  |  |
| har_log_seasonal | 0.3127 | +18.6% (+15.3% to +21.6%), p = 6.2e-15 | +3.5% (+1.5% to +5.3%), p = 0.00064 |  |
| har_gbm | 0.2860 | +25.6% (+22.4% to +29.2%), p = 7e-29 | +11.8% (+7.7% to +15.4%), p = 1e-10 | +8.6% (+5.4% to +11.8%), p = 1.3e-09 |
