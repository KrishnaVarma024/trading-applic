"""Phase 3 evaluation: does the model beat log-HAR on data it has never touched?

    uv run python -m tradingapp.research.evaluate_model --symbol BTCUSDT

The out-of-sample period is split in two. The development part is where
choices may be made and compared. The test part is scored once, at the end.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data import load_klines, to_hourly
from .features import build_dataset
from .features_ext import FEATURE_NAMES, build_matrix
from .metrics import block_bootstrap_ci, dm_test, qlike_losses
from .models import Ewma, HarGbm, HarLog, HarLogSeasonal
from .walkforward import walk_forward


def _ms(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def _round(obj, digits: int = 10):
    """Round floats so the JSON is identical across machines and numpy versions."""
    if isinstance(obj, float):
        return float(f"{obj:.{digits}g}")
    if isinstance(obj, dict):
        return {k: _round(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v, digits) for v in obj]
    return obj


def compare(losses: dict[str, np.ndarray], rows: np.ndarray) -> dict:
    out = {}
    for name, loss in losses.items():
        entry = {"qlike": float(loss[rows].mean())}
        order = ["ewma", "har_log", "har_log_seasonal", "har_gbm"]
        for ref in ("ewma", "har_log", "har_log_seasonal"):
            if order.index(name) <= order.index(ref):
                continue
            lo, hi = block_bootstrap_ci(loss[rows], losses[ref][rows])
            stat, p = dm_test(loss[rows], losses[ref][rows])
            entry[f"vs_{ref}"] = {
                "qlike_reduction": float(1 - loss[rows].mean() / losses[ref][rows].mean()),
                "ci95": [lo, hi], "dm_stat": stat, "dm_p": p,
            }
        out[name] = entry
    return out


def load(data_dir: str, symbol: str):
    k = load_klines(data_dir, symbol)
    hourly = to_hourly(k["t_ms"], k["close"], volume=k["volume"], trades=k["trades"])
    ds = build_dataset(hourly.start_ms, hourly.rv, hourly.ret)
    return ds, build_matrix(ds, hourly)


def run(data_dir: str, symbol: str, initial_days: int, refit_hours: int, test_start: str,
        include_test: bool = True) -> dict:
    ds, x = load(data_dir, symbol)
    initial = initial_days * 24

    losses = {}
    seasonal = x[:, [FEATURE_NAMES.index(n) for n in ("hour_sin", "hour_cos", "weekend")]]
    for model in (Ewma(), HarLog(), HarLogSeasonal(seasonal), HarGbm(x)):
        f = walk_forward(ds, model, initial, refit_hours)
        losses[model.name] = qlike_losses(ds.y[initial:], f[initial:])

    origin = ds.start_ms[initial:]
    is_test = origin >= _ms(test_start)
    result = {
        "symbol": symbol,
        "setup": {
            "initial_train_days": initial_days, "refit_every_hours": refit_hours,
            "test_start": test_start, "dev_hours": int((~is_test).sum()), "test_hours": int(is_test.sum()),
            "features": FEATURE_NAMES,
        },
        "development": compare(losses, ~is_test),
    }
    if include_test:
        result["test"] = compare(losses, is_test)
    return _round(result)


def to_markdown(r: dict) -> str:
    s = r["setup"]
    lines = [f"# Phase 3 results: {r['symbol']}", "",
             f"Expanding window, first {s['initial_train_days']} days training only, refit every "
             f"{s['refit_every_hours']} hours. Development: {s['dev_hours']:,} hours before {s['test_start']}. "
             f"Test: {s['test_hours']:,} hours from {s['test_start']}.", ""]
    for part in ("development", "test"):
        if part not in r:
            continue
        lines += [f"## {part.capitalize()} period", "",
                  "| Model | QLIKE | vs tuned EWMA | vs log-HAR | vs seasonal log-HAR |",
                  "| --- | --- | --- | --- | --- |"]

        def cell(e: dict, ref: str) -> str:
            c = e.get(f"vs_{ref}")
            if c is None:
                return ""
            return f"{c['qlike_reduction']:+.1%} ({c['ci95'][0]:+.1%} to {c['ci95'][1]:+.1%}), p = {c['dm_p']:.2g}"

        for name in ("ewma", "har_log", "har_log_seasonal", "har_gbm"):
            e = r[part][name]
            lines.append(f"| {name} | {e['qlike']:.4f} | {cell(e, 'ewma')} | {cell(e, 'har_log')} | "
                         f"{cell(e, 'har_log_seasonal')} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--data-dir", default="data/hist")
    p.add_argument("--initial-days", type=int, default=60)
    p.add_argument("--refit-hours", type=int, default=24)
    p.add_argument("--test-start", default="2026-05-01")
    p.add_argument("--dev-only", action="store_true", help="do not score the test period")
    p.add_argument("--out", default="reports")
    args = p.parse_args()
    result = run(args.data_dir, args.symbol, args.initial_days, args.refit_hours, args.test_start,
                 include_test=not args.dev_only)
    md = to_markdown(result)
    if not args.dev_only:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "phase3_results.json").write_text(json.dumps(result, indent=2) + "\n")
        (out / "phase3.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
