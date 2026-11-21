# Where the gain comes from: BTCUSDT

Walk-forward, first 60 days training only, refit every 24 hours. Development: before 2026-05-01. Test: from 2026-05-01.

## Feature-group ablation (development period only)

| Removed | Development QLIKE | vs log-HAR |
| --- | --- | --- |
| none removed | 0.2840 | +10.4% |
| time of day and weekend | 0.3208 | -1.3% |
| volume and trade count | 0.2835 | +10.5% |
| returns and semivariance | 0.2876 | +9.2% |
| 4-hour and 12-hour variance | 0.2914 | +8.0% |

## A linear model given the same clock information

| Period | log-HAR | log-HAR + hourly terms | log-HAR + trees | hourly vs log-HAR | trees vs hourly |
| --- | --- | --- | --- | --- | --- |
| development | 0.3168 | 0.2889 | 0.2840 | +8.8% (+2.9% to +14.0%), p = 0.0011 | +1.7% (-2.1% to +5.6%), p = 0.39 |
| test | 0.3241 | 0.2959 | 0.2862 | +8.7% (+5.6% to +12.1%), p = 1.8e-08 | +3.3% (+0.5% to +5.9%), p = 0.016 |

## Hour-of-day profile

Mean hourly realized variance by UTC hour, as a multiple of the all-day mean. Peak: 14:00 UTC. Weekend variance is 46% of weekday.

| UTC hour | 00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multiple | 1.06 | 1.11 | 0.88 | 0.68 | 0.59 | 0.64 | 0.59 | 0.66 | 0.71 | 0.64 | 0.53 | 0.65 | 0.94 | 1.42 | 2.24 | 2.16 | 1.46 | 1.24 | 1.26 | 0.94 | 0.93 | 0.82 | 1.06 | 0.78 |
