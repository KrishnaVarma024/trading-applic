# Phase 5: diagnosing and cutting live latency

## Problem

Phase 4 measured tick-to-forecast latency on the author's MacBook two ways.
The replay benchmark gave a 0.57 ms median. The live run gave 4.46 ms, about
eight times slower, and every stage slowed down, including parsing and bar
building, which do no I/O and did not change.

## Diagnosis

Hypothesis: live, the process is nearly idle between bars, and an idle
process runs slower when it wakes (slower core, lower clock, cold caches). The
replay never idles.

Test: `live.bench --idle-ms 1000` replays the same ticks but sleeps one second
before each bar-closing trade, before its arrival is timestamped. That
reproduces the live waiting pattern with no network involved.

Measured in a Linux VM on the author's MacBook, scikit-learn inference:

| Replay | n | p50 | p90 |
| --- | --- | --- | --- |
| No idle | 1,518 | 0.67 ms | 0.80 ms |
| 1 s idle before each bar close | 120 | 4.80 ms | 5.81 ms |
| For comparison: phase 4 live run, macOS | 136 | 4.46 ms | 9.10 ms |

The idle replay lands on the live figure. The slowdown comes from idling, not
from the WebSocket path. Confirmed in the VM; the macOS-native rerun is listed
in `BENCH.md`.

## Changes

**1. Compiled trees** (`live/compiled.py`). scikit-learn's `predict` spent most
of its time validating and converting a one-row input. The 150 fitted trees
are copied into flat NumPy arrays and walked one depth level at a time across
all trees at once. Same arithmetic, no per-call overhead.

**2. Warm-up** (`live/forecast.py`). After start-up history is loaded, inference
runs five times on real features and the results are discarded, so the first
timed forecast does not pay one-off setup costs. Phase 4's worst value, 34.5
ms, was that first call.

**3. Scheduling class** (`live/scheduling.py`). On macOS the process asks to be
treated as user-interactive, the class macOS prefers performance cores for.
A request, not a guarantee, and untested until the macOS rerun. `--qos default`
turns it off so its effect can be measured.

## Verification

- The compiled trees equal scikit-learn to 1e-12 on synthetic models at depths
  2, 3 and 5, on inputs far outside the training range, and with missing values.
- On the real model file, the two paths agree on all 8,592 historical rows to
  a worst relative difference of 2.0e-15.
- The compiled form is rebuilt at load and never saved, so the model file
  format and its version checks are unchanged.
- 64 tests in total (67 after the ablation command was added).

## Results in the build VM

Linux VM on the author's MacBook. Indicative only; the macOS numbers are the
ones to quote.

| Replay | Inference | n | p50 | p90 | p99 |
| --- | --- | --- | --- | --- | --- |
| No idle | scikit-learn | 1,518 | 0.672 ms | 0.797 ms | 1.36 ms |
| No idle | compiled | 1,518 | 0.051 ms | 0.068 ms | 0.100 ms |
| 1 s idle | scikit-learn | 120 | 4.80 ms | 5.81 ms | 17.8 ms |
| 1 s idle | compiled | 120 | 0.95 ms | 1.49 ms | 11.2 ms |

Model inference fell from 643 us to 33 us at the median, about 20 times.

## Results on the author's MacBook (macOS)

| Replay | Inference | Scheduling class | n | p50 | p90 | p99 |
| --- | --- | --- | --- | --- | --- | --- |
| No idle | scikit-learn | interactive | 1,518 | 0.539 ms | 0.581 ms | 0.711 ms |
| No idle | compiled | interactive | 1,518 | 0.051 ms | 0.067 ms | 0.098 ms |
| 1 s idle | compiled | default | 120 | 0.802 ms | 1.086 ms | 3.011 ms |
| 1 s idle | compiled | interactive | 120 | 0.730 ms | 1.080 ms | 1.676 ms |

The scheduling-class request did not show a clear effect: 9% lower median,
same p90, one run each. It stays as a harmless default and is not credited.

## Live, overnight, on the author's MacBook

398 forecasts over 6 h 37 min, compiled inference, interactive scheduling
class. No disconnects, missed trades, resets or degraded forecasts.

| Live run | Inference | n | p50 | p90 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- |
| Phase 4 | scikit-learn | 136 | 4.46 ms | 9.10 ms | * | 34.5 ms |
| Phase 5, overnight | compiled | 398 | 0.79 ms | 2.17 ms | 9.1 ms | 39.0 ms |

\* Too few samples. The median fell 5.6 times. The live median (0.79 ms)
matches the idle replay (0.73 ms), so the idle replay is a usable stand-in
for live latency. The tail did not improve: the worst value is about the same,
and three of the four slowest came in one unexplained four-minute stretch.

## What is left

With compiled inference, the idle-replay median on macOS is still about 15
times the hot one (0.73 ms against 0.05 ms). The remainder is parsing (170 us) and features (370 us), code that is
fast when warm and slow after a second of idling. Two options, neither built:

- Keep the hot path warm: a timer that runs a dummy feature and model pass a
  few milliseconds before each minute boundary, when the next bar close is due.
  The forecast would genuinely arrive sooner, but this must be disclosed with
  any latency figure, since it changes what the number means.
- Make features incremental: running sums instead of re-summing a week of bars
  on every close. Less memory touched, so fewer cold-cache misses. Needs care
  to keep train/serve parity within 1e-9.

The tail is now the open problem. Features (p99 6.2 ms) and the model (p99
2.4 ms) both have long tails live, which points at the operating system pausing
the process rather than at any one stage. Finding out needs a timestamp of
every pause, not just stage durations.

## The four production questions

**100x load?** Inference is no longer the bottleneck. At 33 us per forecast one
core could serve thousands of symbols per minute; per-trade work and the single
event loop are now the limits.

**Malicious input?** Unchanged from phase 4. The compiled trees refuse model
types they cannot reproduce exactly (non-squared-error loss, categorical
splits) rather than guessing.

**Network dies?** Unchanged from phase 4.

**How would I know it broke?** If the compiled path ever disagreed with the
reference, forecasts would drift silently. The equality tests run on every
change; a live guard that re-checks one forecast an hour against scikit-learn
would catch a drift in production and is not built.
