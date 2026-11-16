"""Fit the phase 3 model on all downloaded history and save it for live use.

    uv run python -m tradingapp.research.train --symbol BTCUSDT
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

from ..live.forecaster import Forecaster
from .evaluate_model import load
from .models import HarGbm


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--data-dir", default="data/hist")
    p.add_argument("--out", default="models/har_gbm.joblib")
    args = p.parse_args()
    ds, x = load(args.data_dir, args.symbol)
    model = HarGbm(x)
    model.fit(ds, len(ds))
    last = datetime.fromtimestamp(int(ds.start_ms[-1]) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    f = Forecaster.from_model(model, symbol=args.symbol, rows=len(ds), trained_through_utc=last)
    f.save(args.out)
    print(f"saved {args.out}: {len(ds):,} hourly rows through {last} UTC, scikit-learn {f.meta['sklearn']}")


if __name__ == "__main__":
    main()
