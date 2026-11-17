import json

import numpy as np
import pytest

from tradingapp.live.backfill import fetch_klines
from tradingapp.live.bench import run_bench, split_sessions
from tradingapp.live.core import LiveForecaster
from tradingapp.live.engine import HOUR_MS, WEEK, FeatureEngine
from tradingapp.live.forecaster import Forecaster
from tradingapp.live.latency import LatencyRecorder
from tradingapp.research.data import to_hourly
from tradingapp.research.features import build_dataset
from tradingapp.research.features_ext import build_matrix
from tradingapp.research.models import HarGbm

from .helpers import msg, trade
from .test_model import dataset, make_hourly

MIN = 60_000
T0 = 1_790_812_800_000          # an hour boundary


def minute_klines(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = T0 + np.arange(n, dtype=np.int64) * MIN
    close = 60_000 * np.exp(rng.normal(0, 6e-4, n).cumsum())
    volume = np.exp(rng.normal(1, 0.7, n))
    trades = rng.integers(1, 3000, n).astype(float)
    return t, close, volume, trades


@pytest.fixture(scope="module")
def forecaster():
    ds, x = dataset(make_hourly(1500, seed=11))
    model = HarGbm(x, max_iter=30)
    model.fit(ds, len(ds))
    return Forecaster.from_model(model, note="test"), model, ds, x


# ---- the test that matters most: live features equal research features ------

def test_live_features_equal_research_features_on_the_hour():
    t, close, volume, trades = minute_klines(9 * 1440 + 60)
    hourly = to_hourly(t, close, volume=volume, trades=trades)
    ds = build_dataset(hourly.start_ms, hourly.rv, hourly.ret)
    x = build_matrix(ds, hourly)
    row_of_hour = {int(s): i - (168 - 1) for i, s in enumerate(hourly.start_ms)}

    engine, compared = FeatureEngine(MIN), 0
    for i in range(len(t)):
        f = engine.update(int(t[i]), float(close[i]), float(volume[i]), float(trades[i]))
        if f is None or f.bar_end_ms % HOUR_MS:
            continue
        row = row_of_hour[f.bar_end_ms - HOUR_MS]
        if not 0 <= row < len(ds):
            continue
        np.testing.assert_allclose(f.x, x[row], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(f.har_logs, np.log([ds.rv_h[row], ds.rv_d[row], ds.rv_w[row]]), rtol=1e-9)
        compared += 1
    assert compared >= 40


def test_engine_warms_after_a_week_resets_on_a_hole_and_flags_lost_data():
    t, close, volume, trades = minute_klines(WEEK + 200, seed=2)
    engine = FeatureEngine(MIN)
    out = [engine.update(int(t[i]), float(close[i]), float(volume[i]), float(trades[i])) for i in range(WEEK + 1)]
    assert all(o is None for o in out[:WEEK]) and out[WEEK] is not None      # needs WEEK returns
    assert not out[WEEK].degraded

    f = engine.update(int(t[WEEK + 1]), float(close[WEEK + 1]), 0.0, 0.0)    # a bar with no trades
    assert f.degraded
    for i in range(WEEK + 2, WEEK + 62):                                     # stays flagged for an hour
        f = engine.update(int(t[i]), float(close[i]), 1.0, 5.0)
    assert not f.degraded

    assert engine.update(int(t[WEEK + 62]) + 5 * MIN, 1.0, 1.0, 1.0) is None  # a hole: history discarded
    assert engine.resets == 1 and not engine.warm


# ---- forecaster --------------------------------------------------------------

def test_single_row_inference_matches_batch_inference(forecaster):
    f, model, ds, x = forecaster
    rows = slice(700, 720)
    batch = model.predict(ds, rows)
    for i, r in enumerate(range(700, 720)):
        har = np.log([ds.rv_h[r], ds.rv_d[r], ds.rv_w[r]])
        assert f.predict(har, x[r]) == pytest.approx(batch[i], rel=1e-12)


def test_save_load_round_trip_and_version_guard(forecaster, tmp_path):
    f, _, ds, x = forecaster
    path = tmp_path / "m.joblib"
    f.save(path)
    g = Forecaster.load(path)
    har = np.log([ds.rv_h[5], ds.rv_d[5], ds.rv_w[5]])
    assert g.predict(har, x[5]) == f.predict(har, x[5])

    stale = Forecaster(f.beta, f.gbm, f.resid_var, {**f.meta, "sklearn": "0.0.1"})
    stale.save(path)
    with pytest.raises(RuntimeError, match="retrain"):
        Forecaster.load(path)


# ---- backfill ----------------------------------------------------------------

def fake_exchange(first_ms: int, n: int, drop: int | None = None):
    rows = {first_ms + i * MIN: [first_ms + i * MIN, "1", "1", "1", f"{100 + i}", "2.5", 0, "0", 7, "0", "0", "0"]
            for i in range(n)}
    if drop is not None:
        rows.pop(first_ms + drop * MIN)

    def get(url: str) -> str:
        q = dict(kv.split("=") for kv in url.split("?")[1].split("&"))
        end, limit = int(q["endTime"]), int(q["limit"])
        return json.dumps([rows[k] for k in sorted(rows) if k <= end][-limit:])
    return get


def test_backfill_pages_backwards_and_returns_a_gap_free_run():
    get = fake_exchange(T0, 5000)
    last = T0 + 4999 * MIN
    k = fetch_klines("btcusdt", last, 2500, get=get)
    assert len(k) == 2500 and k[-1][0] == last and k[0][0] == last - 2499 * MIN
    assert k[-1][1:] == (100 + 4999, 2.5, 7.0)


def test_backfill_refuses_a_hole_or_a_short_history():
    with pytest.raises(RuntimeError, match="hole"):
        fetch_klines("btcusdt", T0 + 2999 * MIN, 2500, get=fake_exchange(T0, 3000, drop=2000))
    with pytest.raises(RuntimeError, match="incomplete"):
        fetch_klines("btcusdt", T0 + 99 * MIN, 500, get=fake_exchange(T0, 100))


# ---- hot path ----------------------------------------------------------------

def test_forecast_is_emitted_on_bar_close_with_every_stage_timed(forecaster):
    f, *_ = forecaster
    seen = []
    live = LiveForecaster(f, skip_first_bar=False, on_forecast=seen.append)
    t, close, volume, trades = minute_klines(WEEK + 1, seed=3)
    live.warm_up(zip(t.tolist(), close.tolist(), volume.tolist(), trades.tolist()))
    assert live.engine.warm                           # WEEK + 1 closes hold WEEK returns

    nxt = int(t[-1]) + MIN
    assert live.on_trade(trade(1, nxt + 100, 60_000.0), 1) is None       # opens the bar
    assert live.on_trade(trade(2, nxt + 30_000, 60_010.0), 1) is None    # same bar
    out = live.on_trade(trade(3, nxt + MIN + 5, 60_020.0), 1)            # closes it
    assert out is not None and seen == [out]
    assert out.bar_end_ms == nxt + MIN and out.rv_next_hour > 0 and out.annualised_vol_pct > 0
    s = live.latency.summary()
    assert set(s) == {"total", "parse", "bar", "features", "model"} and s["total"]["n"] == 1


def test_bench_replays_messages_and_times_forecasts(forecaster):
    f, *_ = forecaster
    messages = [msg(i, T0 + i * 250, price=f"{60_000 + (i % 7)}") for i in range(1, 4001)]   # 1000 seconds
    live = run_bench(messages, f, interval_ms=1000, warmup_forecasts=50)
    assert live.latency.count() > 800
    assert live.latency.summary()["total"]["p50"] > 0


def test_sessions_split_where_recording_stopped():
    a = [msg(i, T0 + i * 100) for i in range(1, 11)]
    b = [msg(i, T0 + 10_000_000 + i * 100) for i in range(11, 16)]
    assert [len(s) for s in split_sessions(a + b)] == [10, 5]


def test_latency_percentiles_and_merge():
    r, other = LatencyRecorder(), LatencyRecorder()
    for ns in range(1000, 101_000, 1000):
        r.add(total=ns)
    other.add(total=1_000_000)
    s = r.summary()["total"]
    assert s["n"] == 100 and s["p50"] == pytest.approx(50.5) and s["max"] == 100.0
    r.extend(other)
    assert r.summary()["total"]["max"] == 1000.0 and r.count() == 101


def test_start_up_seam_joins_history_and_live_bars_without_a_gap(forecaster):
    """Simulate start-up mid-minute and check the joined engine equals one fed a continuous series."""
    f, *_ = forecaster
    t, close, volume, trades = minute_klines(WEEK + 5, seed=4)
    as_bars = list(zip(t.tolist(), close.tolist(), volume.tolist(), trades.tolist()))
    m = WEEK + 2                                       # index of the first complete live minute
    first_live = int(t[m])

    seen, bars = [], []
    live = LiveForecaster(f, skip_first_bar=True, on_forecast=seen.append, on_bar=bars.append)
    tid = iter(range(1, 100))
    live.on_trade(trade(next(tid), first_live - 20_000, 1.0), 1)             # joined mid-minute: partial bar
    for k in (m, m + 1):                                                     # two complete live minutes
        live.on_trade(trade(next(tid), int(t[k]) + 10, float(close[k]), float(volume[k])), 1)
        if k == m + 1 and len(bars) == 1:                                    # first complete bar just closed
            assert bars[0].start_ms == first_live
            live.join_history(as_bars[m - WEEK - 1:m], bars)
    assert live.builder.stats.partial_dropped == 1 and seen == []
    out = live.on_trade(trade(next(tid), int(t[m + 2]) + 10, float(close[m + 2])), 1)   # closes minute m+1
    assert out is not None and seen == [out] and out.bar_end_ms == int(t[m + 2])

    reference = FeatureEngine(MIN)                     # the same minutes, fed without any seam
    for k in range(m - WEEK - 1, m + 2):
        trd = float(trades[k]) if k < m else 1.0       # each live bar here was built from one trade
        ref = reference.update(int(t[k]), float(close[k]), float(volume[k]), trd)
    np.testing.assert_allclose(live.engine._features(int(t[m + 2])).x, ref.x, rtol=1e-12)

    with pytest.raises(ValueError, match="one bar before"):
        live.join_history(as_bars[m - WEEK - 2:m - 1], bars)
