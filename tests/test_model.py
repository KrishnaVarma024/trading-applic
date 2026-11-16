import copy
import zipfile

import numpy as np
import pytest

from tradingapp.research.data import Hourly, load_klines, to_hourly
from tradingapp.research.evaluate_model import _round
from tradingapp.research.features import build_dataset
from tradingapp.research.features_ext import FEATURE_NAMES, build_matrix
from tradingapp.research.metrics import qlike
from tradingapp.research.models import HarGbm, HarLog
from tradingapp.research.synthetic import simulate_hourly
from tradingapp.research.walkforward import walk_forward

FAST = dict(max_iter=40)


def make_hourly(n: int, seed: int, seasonal: float = 0.0) -> Hourly:
    """Synthetic hours. `seasonal` adds a time-of-day pattern HAR cannot express."""
    start, rv, ret = simulate_hourly(n, seed=seed)
    rng = np.random.default_rng(seed + 100)
    hour = (start // 3_600_000) % 24
    rv = rv * np.exp(seasonal * np.sin(2 * np.pi * hour / 24))
    return Hourly(start_ms=start, rv=rv, ret=ret, dropped_hours=0, missing_minutes=0,
                  rs_neg=rv * rng.uniform(0.3, 0.7, n),
                  volume=np.exp(rng.normal(3, 0.5, n)), trades=np.exp(rng.normal(8, 0.5, n)))


def dataset(h: Hourly):
    ds = build_dataset(h.start_ms, h.rv, h.ret)
    return ds, build_matrix(ds, h)


def test_loader_sums_volume_and_trades_per_hour(tmp_path):
    h0 = 1_790_812_800_000
    rows = [f"{(h0 + m * 60_000) * 1000},0,0,0,{100 + m * 0.01:.8f},2.5,0,0,7,0,0,0" for m in range(120)]
    with zipfile.ZipFile(tmp_path / "TEST-1m-2026-10.zip", "w") as zf:
        zf.writestr("x.csv", "\n".join(rows))
    k = load_klines(tmp_path, "TEST")
    h = to_hourly(k["t_ms"], k["close"], volume=k["volume"], trades=k["trades"])
    assert h.volume.tolist() == [150.0, 150.0] and h.trades.tolist() == [420.0, 420.0]
    assert (h.rs_neg == 0).all()               # the price only rises in this file


def test_feature_matrix_shape_and_no_look_ahead():
    h = make_hourly(600, seed=5)
    ds, x = dataset(h)
    assert x.shape == (len(ds), len(FEATURE_NAMES)) and np.isfinite(x).all()
    k = 400                                    # an hour index; corrupt everything after it
    h2 = copy.deepcopy(h)
    for name in ("rv", "ret", "rs_neg", "volume", "trades"):
        getattr(h2, name)[k + 1:] *= 37.0
    _, x2 = dataset(h2)
    rows = slice(0, k - 167 + 1)               # dataset row i is hour i + 167
    assert np.array_equal(x[rows], x2[rows])
    assert not np.array_equal(x[rows.stop:], x2[rows.stop:])


def test_misaligned_inputs_are_refused():
    h = make_hourly(400, seed=6)
    ds, _ = dataset(h)
    shorter = make_hourly(390, seed=6)
    with pytest.raises(ValueError):
        build_matrix(ds, shorter)


def test_model_walk_forward_cannot_see_the_future():
    ds, x = dataset(make_hourly(1400, seed=7))
    initial, refit, k = 500, 150, 900
    base = walk_forward(ds, HarGbm(x, **FAST), initial, refit)
    bad, bad_x = copy.deepcopy(ds), x.copy()
    for name in ("rv_h", "rv_d", "rv_w", "rv_all", "y", "ret_h", "ret_d", "y_ret"):
        getattr(bad, name)[k:] *= 1000.0
    bad_x[k:] += 5.0
    again = walk_forward(bad, HarGbm(bad_x, **FAST), initial, refit)
    assert np.array_equal(base[initial:k], again[initial:k])
    assert (base[initial:] > 0).all() and np.isfinite(base[initial:]).all()


def test_model_learns_a_pattern_har_cannot_express():
    ds, x = dataset(make_hourly(3000, seed=8, seasonal=0.8))
    initial = 1000
    y = ds.y[initial:]
    har = qlike(y, walk_forward(ds, HarLog(), initial, 250)[initial:])
    gbm = qlike(y, walk_forward(ds, HarGbm(x, **FAST), initial, 250)[initial:])
    assert gbm < 0.95 * har


def test_model_does_not_invent_signal_where_there_is_none():
    """With no extra structure, the trees should add little and must not blow up."""
    ds, x = dataset(make_hourly(3000, seed=9, seasonal=0.0))
    initial = 1000
    y = ds.y[initial:]
    har = qlike(y, walk_forward(ds, HarLog(), initial, 250)[initial:])
    gbm = qlike(y, walk_forward(ds, HarGbm(x, **FAST), initial, 250)[initial:])
    assert gbm < 1.05 * har


def test_rounding_makes_output_machine_independent():
    assert _round({"a": [0.38038559784041137, 0.38038559784041115], "n": 3}) == {"a": [0.3803855978, 0.3803855978], "n": 3}
