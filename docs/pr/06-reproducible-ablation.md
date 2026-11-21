# Make every attribution claim reproducible from one command

## Why

The README's first two findings (most of the gain is the clock; volatility peaks
at the US equity open) came from one-off scripts. Every other number in the repo
has a command that regenerates it. These two did not, so a reader could not check
them.

## What changed

- `src/tradingapp/research/ablation.py`: one command,
  `uv run python -m tradingapp.research.ablation --symbol BTCUSDT`, which runs
  1. the feature-group ablation, scored on the development period only;
  2. log-HAR with 23 hour-of-day indicators and a weekend flag, scored on both
     periods against log-HAR and against the trees, with block-bootstrap
     intervals and Diebold-Mariano p-values;
  3. the hour-of-day variance profile and the weekend-to-weekday ratio.
  It writes `reports/ablation.md` and `reports/ablation.json`.
- `tests/test_ablation.py`: the profile finds a planted daily peak; the hour
  indicators avoid the dummy-variable trap; removing the clock hurts on data
  whose variance has a daily cycle. 67 tests pass.
- README: the ablation command in Reproduce, links from findings 1 and 2, build
  log fixed (phase 5 was PRs #5 and #6).

## Result

Every figure matches what was reported before, to the fourth decimal:

| | Before (one-off) | This command |
| --- | --- | --- |
| Dev QLIKE without time features | 0.3208 (-1.3%) | 0.3208 (-1.3%) |
| Hourly indicators vs log-HAR, test | +8.7% (+5.6% to +12.1%) | +8.7% (+5.6% to +12.1%) |
| Trees vs hourly indicators, test | +3.3% (+0.5% to +5.9%), p = 0.016 | same |
| Trees vs hourly indicators, dev | +1.7%, interval spans zero | +1.7% (-2.1% to +5.6%), p = 0.39 |
| Peak hours | 14:00, 15:00 UTC above 2x mean | 2.24x, 2.16x |
| Weekend / weekday variance | 46% | 46% |

No model, feature or threshold changed. The test period was not used to choose anything.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01Jyjdgap7o32HDKVcXjMdP5
