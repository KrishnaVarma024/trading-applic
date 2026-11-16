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

## What was not verified

The live program end to end. In the build environment the stream connected
but delivered nothing. The first complete live run is recorded in `BENCH.md`.

## Findings

- Inference is about 95% of tick-to-forecast time.
- Pinning the tree library to one thread cut the 99th percentile about
  threefold in the build environment.

## Review checklist

- [ ] I trained the model and ran the benchmark and the live program myself.
- [ ] `BENCH.md` holds latency numbers from my machine, not the build environment.
- [ ] I can say exactly what tick-to-forecast includes and excludes.
- [ ] I can explain what train/serve skew is and which test guards against it.
