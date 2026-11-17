# Phase 4: live forecasting and latency

## What this adds

- `live/engine.py`: incremental feature engine holding a week of minute bars.
- `live/forecaster.py`: the trained model reduced to what inference needs,
  with save, load, and guards against a stale or mismatched model file.
- `live/backfill.py`: REST kline fetch for start-up, refusing any hole.
- `live/core.py`: the hot path, with a timestamp at each stage boundary and
  all record-keeping moved after the final one.
- `live/forecast.py`: the live program.
- `live/bench.py`: replay benchmark for latency.
- `live/latency.py`: percentile summary.
- `research/train.py`: fits on all history and writes the model file.
- Python is pinned to 3.12 and the phase 3 figures are corrected to match it.

## What was verified

- 11 new tests, 55 in total.
- Live features equal research features to 1e-9 at every hour boundary over
  nine simulated days.
- The start-up join of REST history and live bars produces the same engine
  state as an unbroken series.
- Single-row inference equals batch inference.
- The REST backfill was run against the real endpoint: 10,081 gap-free klines.

## Live run on the author's machine

2 h 15 min against the BTCUSDT stream: 136 forecasts, 0 disconnects, 0 missed
trades, 0 resets, 0 degraded forecasts. Start-up backfill took 9.4 s.

| Run | Forecasts | p50 | Worst |
| --- | --- | --- | --- |
| Replay benchmark | 1,518 | 0.57 ms | 2.4 ms |
| Live | 136 | 4.5 ms | 34.5 ms (first call) |

Live is about eight times slower than replay in every stage, including parsing
and bar building, which do no I/O. Probable cause: the idle process runs on
efficiency cores with cold caches. Not yet confirmed. Details in `BENCH.md`.

## Not verified

- Whether live forecasts are accurate. Two hours of overlapping forecasts is
  about two independent observations.
- A stable 99th percentile for live latency. That needs a much longer run.

## Findings

- Inference is about 95% of tick-to-forecast time.
- Pinning the tree library to one thread cut the 99th percentile about
  threefold in the build environment.

## Review checklist

- [ ] I trained the model and ran the benchmark and the live program myself.
- [ ] `BENCH.md` holds latency numbers from my machine, not the build environment.
- [ ] I can say exactly what tick-to-forecast includes and excludes.
- [ ] I can explain what train/serve skew is and which test guards against it.
