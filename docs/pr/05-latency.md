# Phase 5: why live latency was eight times the replay, and the fix

## Diagnosis

Phase 4's live median (4.46 ms) was eight times the replay median (0.57 ms),
in every stage including parsing. New flag `live.bench --idle-ms 1000` sleeps
one second before each bar-closing trade. With it, the replay median rises to
4.80 ms, matching live with no network involved. The cause is the process
idling between bars, not the live code path.

## Changes

- `live/compiled.py`: the 150 fitted trees as flat arrays, walked one level at
  a time across all trees. Replaces scikit-learn's per-call input validation.
- `live/forecaster.py`: builds the compiled trees at load; `--reference-model`
  keeps scikit-learn's path for comparison. The model file format is unchanged.
- `live/forecast.py`: five discarded warm-up inferences before the first timed
  forecast; `--qos` scheduling-class option.
- `live/scheduling.py`: asks macOS for the user-interactive QoS class.
- `live/bench.py`: `--idle-ms`, `--max-forecasts`, `--qos`, `--reference-model`.
- README rewritten as a landing page. Per-phase detail moved to
  `docs/design/01` to `05`. `BREAKS.md` records the five real incidents so far.

## Verification

- 7 new tests, 64 in total. Compiled trees equal scikit-learn to 1e-12 at
  depths 2, 3 and 5, outside the training range, and with missing values.
- Real model file: identical on all 8,592 rows to 2.0e-15.

## Results in the build VM (Linux VM on the author's MacBook)

| Replay | Inference | p50 |
| --- | --- | --- |
| No idle | scikit-learn | 0.672 ms |
| No idle | compiled | 0.051 ms |
| 1 s idle | scikit-learn | 4.80 ms |
| 1 s idle | compiled | 0.95 ms |

## Results on the author's MacBook (macOS)

| Replay | Inference | p50 | p90 |
| --- | --- | --- | --- |
| No idle | scikit-learn | 0.539 ms | 0.581 ms |
| No idle | compiled | 0.051 ms | 0.067 ms |
| 1 s idle, default scheduling | compiled | 0.802 ms | 1.086 ms |
| 1 s idle, interactive scheduling | compiled | 0.730 ms | 1.080 ms |

The scheduling-class request showed no clear effect and is not credited.

## Live overnight on the author's MacBook

398 forecasts over 6 h 37 min, no disconnects, gaps or resets.

| Live run | Inference | n | p50 | p90 | p99 |
| --- | --- | --- | --- | --- | --- |
| Phase 4 | scikit-learn | 136 | 4.46 ms | 9.10 ms | too few samples |
| Phase 5 | compiled | 398 | 0.79 ms | 2.17 ms | 9.1 ms |

## Not verified

- The cause of the slowest forecasts: three of the four slowest came in one
  four-minute stretch.
- Live forecast accuracy, which needs weeks of forecasts.

## Review checklist

- [x] I ran the four bench commands in `BENCH.md` and the numbers went in.
- [ ] I can explain why an idle process runs slower when it wakes.
- [ ] I can explain why the compiled trees give the same answer, and which test proves it.
- [ ] I read the new README as a stranger would and every number has a source.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01Jyjdgap7o32HDKVcXjMdP5
