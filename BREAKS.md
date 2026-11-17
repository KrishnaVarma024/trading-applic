# What broke, and what fixed it

Every entry here was found by running the system, not by reading the code.

| Phase | What happened | Root cause | Fix |
| --- | --- | --- | --- |
| 1 | First live bar was written with a 21:23 open but held only 44 seconds of trades | The stream is joined mid-minute; the builder emitted the partial bar as complete | Live mode drops the first bar and counts it in `partial_dropped` |
| 3 | Committed model scores differed from a clean checkout in the fourth decimal | Build environment used Python 3.10, which resolved an older scikit-learn | Pinned Python 3.12; all reported figures regenerated |
| 2-3 | Per-minute evaluation made log-HAR look worse than a plain daily average | A 90-minute stretch with no price change gave near-zero variance forecasts, and QLIKE explodes as the forecast approaches zero | Excluded the stretch; recorded that ten origins out of 429,059 reversed the conclusion |
| 4 | Live latency was eight times the replay figure in every stage | The process idles between bars and runs slower on waking | Reproduced with an idle replay; compiled inference and a scheduling hint (phase 5) |
| 4 | First live forecast took 34.5 ms | One-off setup on the first inference call | Warm-up inference at start-up (phase 5) |

## Assignment experiments

Add one row per experiment: what you did, what you expected, what happened.
