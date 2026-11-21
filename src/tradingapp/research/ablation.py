"""Where does the model's gain come from? Reproduces every attribution claim.

    uv run python -m tradingapp.research.ablation --symbol BTCUSDT

Three parts, all written to reports/ablation.md and reports/ablation.json:

1. Feature-group ablation. The tree model is refitted with one group of
   features removed at a time and scored on the DEVELOPMENT period only, so
   the test period plays no part in deciding what matters.
2. An hourly-indicator baseline: log-HAR plus 23 hour-of-day indicators and a
   weekend flag, a plain linear model given the same clock information the
   trees get. Scored on both periods against log-HAR and against the trees.
3. The hour-of-day profile: mean hourly realized variance per UTC hour as a
   multiple of the all-day mean, and the weekend-to-weekday ratio.

Takes a few minutes: each ablation is a full walk-forward refit.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data import HOUR_MS
from .evaluate_model import _ms, _round, load
from .features import Dataset
from .features_ext import FEATURE_NAMES
from .metrics import block_bootstrap_ci, dm_test, qlike_losses
from .models import HarGbm, HarLog, HarLogSeasonal
from .walkforward import walk_forward

GROUPS = {
    "time of day and weekend": ["hour_sin", "hour_cos", "weekend"],
    "volume and trade count": ["log_volume_h", "log_volume_rel", "log_trades_h"],
    "returns and semivariance": ["neg_share", "ret_h", "abs_ret_h"],
    "4-hour and 12-hour variance": ["log_rv_4h", "log_rv_12h"],
}


def hour_indicators(start_ms: np.ndarray, weekend: np.ndarray) -> np.ndarray:
    """23 hour-of-day indicator columns (hour 0 is the reference) plus the weekend flag."""
    hour = ((start_ms // HOUR_MS) % 24).astype(int)
    return np.column_stack([(hour == k).astype(float) for k in range(1, 24)] + [weekend])


def hour_profile(start_ms: np.ndarray, rv: np.ndarray) -> dict:
    """Mean variance per UTC hour divided by the overall mean, and weekend over weekday."""
    hour = (start_ms // HOUR_MS) % 24
    weekday = (start_ms // (24 * HOUR_MS)) % 7          # 1970-01-01 was a Thursday: 2, 3 = Sat, Sun
    overall = float(rv.mean())
    by_hour = [float(rv[hour == h].mean() / overall) for h in range(24)]
    weekend = np.isin(weekday, (2, 3))
    return {"multiple_of_mean_by_utc_hour": by_hour,
            "peak_hour_utc": int(np.argmax(by_hour)),
            "weekend_over_weekday": float(rv[weekend].mean() / rv[~weekend].mean())}


def _versus(a: np.ndarray, b: np.ndarray) -> dict:
    lo, hi = block_bootstrap_ci(a, b)
    _, p = dm_test(a, b)
    return {"qlike_reduction": float(1 - a.mean() / b.mean()), "ci95": [lo, hi], "dm_p": p}


def run_ablation(ds: Dataset, x: np.ndarray, initial: int, refit: int, is_test: np.ndarray,
                 **gbm_params) -> dict:
    """All model comparisons. `is_test` marks evaluation rows (from `initial` on) in the test period."""
    y = ds.y[initial:]
    dev = ~is_test
    loss = lambda model: qlike_losses(y, walk_forward(ds, model, initial, refit)[initial:])

    l_har = loss(HarLog())
    l_full = loss(HarGbm(x, **gbm_params))
    ablation = {"none removed": {"dev_qlike": float(l_full[dev].mean()),
                                 "vs_har_log": float(1 - l_full[dev].mean() / l_har[dev].mean())}}
    for name, cols in GROUPS.items():
        keep = [i for i, f in enumerate(FEATURE_NAMES) if f not in cols]
        l = loss(HarGbm(x[:, keep], **gbm_params))
        ablation[name] = {"dev_qlike": float(l[dev].mean()),
                          "vs_har_log": float(1 - l[dev].mean() / l_har[dev].mean())}

    ind = hour_indicators(ds.start_ms, x[:, FEATURE_NAMES.index("weekend")])
    l_ind = loss(HarLogSeasonal(ind))
    periods = {}
    for part, mask in (("development", dev), ("test", is_test)):
        periods[part] = {
            "qlike": {"har_log": float(l_har[mask].mean()), "har_log_hourly": float(l_ind[mask].mean()),
                      "har_gbm": float(l_full[mask].mean())},
            "hourly_vs_har_log": _versus(l_ind[mask], l_har[mask]),
            "gbm_vs_hourly": _versus(l_full[mask], l_ind[mask]),
            "gbm_vs_har_log": _versus(l_full[mask], l_har[mask]),
        }
    return {"ablation_dev": ablation, "hourly_baseline": periods}


def to_markdown(r: dict) -> str:
    s = r["setup"]
    lines = [f"# Where the gain comes from: {r['symbol']}", "",
             f"Walk-forward, first {s['initial_train_days']} days training only, refit every {s['refit_every_hours']} "
             f"hours. Development: before {s['test_start']}. Test: from {s['test_start']}.", "",
             "## Feature-group ablation (development period only)", "",
             "| Removed | Development QLIKE | vs log-HAR |", "| --- | --- | --- |"]
    for name, v in r["ablation_dev"].items():
        lines.append(f"| {name} | {v['dev_qlike']:.4f} | {v['vs_har_log']:+.1%} |")
    lines += ["", "## A linear model given the same clock information", "",
              "| Period | log-HAR | log-HAR + hourly terms | log-HAR + trees | hourly vs log-HAR | trees vs hourly |",
              "| --- | --- | --- | --- | --- | --- |"]
    for part, v in r["hourly_baseline"].items():
        q, a, b = v["qlike"], v["hourly_vs_har_log"], v["gbm_vs_hourly"]
        cell = lambda c: f"{c['qlike_reduction']:+.1%} ({c['ci95'][0]:+.1%} to {c['ci95'][1]:+.1%}), p = {c['dm_p']:.2g}"
        lines.append(f"| {part} | {q['har_log']:.4f} | {q['har_log_hourly']:.4f} | {q['har_gbm']:.4f} | {cell(a)} | {cell(b)} |")
    p = r["hour_profile"]
    hours = " | ".join(f"{h:02d}" for h in range(24))
    vals = " | ".join(f"{m:.2f}" for m in p["multiple_of_mean_by_utc_hour"])
    lines += ["", "## Hour-of-day profile", "",
              f"Mean hourly realized variance by UTC hour, as a multiple of the all-day mean. Peak: "
              f"{p['peak_hour_utc']:02d}:00 UTC. Weekend variance is {p['weekend_over_weekday']:.0%} of weekday.", "",
              f"| UTC hour | {hours} |", "| --- |" + " --- |" * 24, f"| multiple | {vals} |", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--data-dir", default="data/hist")
    ap.add_argument("--initial-days", type=int, default=60)
    ap.add_argument("--refit-hours", type=int, default=24)
    ap.add_argument("--test-start", default="2026-05-01")
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()

    ds, x = load(a.data_dir, a.symbol)
    initial = a.initial_days * 24
    is_test = ds.start_ms[initial:] >= _ms(a.test_start)
    result = run_ablation(ds, x, initial, a.refit_hours, is_test)
    # The profile uses every hour in the archive; it describes the data, not a forecast.
    from .data import load_klines, to_hourly
    k = load_klines(a.data_dir, a.symbol)
    h = to_hourly(k["t_ms"], k["close"])
    result = {"symbol": a.symbol,
              "setup": {"initial_train_days": a.initial_days, "refit_every_hours": a.refit_hours,
                        "test_start": a.test_start,
                        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d")},
              **result, "hour_profile": hour_profile(h.start_ms, h.rv)}
    result = _round(result)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablation.json").write_text(json.dumps(result, indent=2) + "\n")
    md = to_markdown(result)
    (out / "ablation.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
