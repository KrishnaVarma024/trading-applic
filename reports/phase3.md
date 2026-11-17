# Phase 3 results: BTCUSDT

Expanding window, first 60 days training only, refit every 24 hours. Development: 2,737 hours before 2026-05-01. Test: 4,415 hours from 2026-05-01.

## Development period

| Model | QLIKE | vs tuned EWMA | vs log-HAR | vs seasonal log-HAR |
| --- | --- | --- | --- | --- |
| ewma | 0.3742 |  |  |  |
| har_log | 0.3168 | +15.3% (+11.4% to +18.9%), p = 1.6e-09 |  |  |
| har_log_seasonal | 0.3102 | +17.1% (+13.5% to +20.5%), p = 1.2e-13 | +2.1% (-0.3% to +4.5%), p = 0.087 |  |
| har_gbm | 0.2840 | +24.1% (+17.6% to +29.9%), p = 2.2e-13 | +10.4% (+3.8% to +16.0%), p = 0.00043 | +8.5% (+2.4% to +13.8%), p = 0.0028 |

## Test period

| Model | QLIKE | vs tuned EWMA | vs log-HAR | vs seasonal log-HAR |
| --- | --- | --- | --- | --- |
| ewma | 0.3842 |  |  |  |
| har_log | 0.3241 | +15.7% (+11.8% to +19.2%), p = 6.4e-10 |  |  |
| har_log_seasonal | 0.3127 | +18.6% (+15.3% to +21.6%), p = 6.2e-15 | +3.5% (+1.5% to +5.3%), p = 0.00064 |  |
| har_gbm | 0.2862 | +25.5% (+22.3% to +29.2%), p = 3.1e-27 | +11.7% (+7.7% to +15.3%), p = 6.6e-11 | +8.5% (+5.4% to +11.6%), p = 5.4e-10 |
