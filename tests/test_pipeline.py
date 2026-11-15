import csv
import json
from pathlib import Path

import pytest

from tradingapp.parse import parse_trade
from tradingapp.pipeline import Pipeline

FIXTURE = Path(__file__).parent / "fixtures" / "sample_trades.jsonl"


def test_replay_fixture_conserves_volume_and_writes_csv(tmp_path):
    trades = [parse_trade(line, 0) for line in FIXTURE.read_text().splitlines()]
    seen = []
    p = Pipeline(60_000, str(tmp_path), "test", save_ticks=True, on_bar=seen.append)
    for t in trades:
        p.on_trade(t)
    open_bar = p.builder.flush()
    p.close()

    # every trade is in exactly one bar: nothing lost, nothing double counted
    assert sum(b.volume for b in seen) + open_bar.volume == pytest.approx(sum(t.qty for t in trades))
    assert sum(b.n_trades for b in seen) + open_bar.n_trades == len(trades)
    # bars are contiguous, with no holes in time
    assert all(b.start_ms - a.start_ms == 60_000 for a, b in zip(seen, seen[1:]))
    assert p.builder.stats.empty_bars >= 2          # the fixture has a quiet stretch
    assert all(b.low <= b.open <= b.high and b.low <= b.close <= b.high for b in seen)

    bar_rows = list(csv.DictReader(next(tmp_path.glob("test-bars-60s-*.csv")).open()))
    assert len(bar_rows) == len(seen)
    assert float(bar_rows[0]["open"]) == seen[0].open
    tick_rows = list(csv.DictReader(next(tmp_path.glob("test-ticks-*.csv")).open()))
    assert len(tick_rows) == len(trades)
    assert int(tick_rows[0]["trade_id"]) == json.loads(FIXTURE.read_text().splitlines()[0])["t"]
