import copy
import zipfile

import numpy as np
import pytest

from tradingapp.research.data import load_minutes, to_hourly
from tradingapp.research.direction import walk_forward_direction
from tradingapp.research.download import month_range
from tradingapp.research.features import build_dataset, ewma, trailing_mean
from tradingapp.research.metrics import block_bootstrap_ci, dm_test, qlike, qlike_losses, wilson_ci
from tradingapp.research.models import Ewma, Har, HarLog, Persistence
from tradingapp.research.synthetic import simulate_hourly
from tradingapp.research.walkforward import walk_forward

INITIAL, REFIT = 600, 24


@pytest.fixture(scope="module")
def ds():
    return build_dataset(*simulate_hourly(3000, seed=1))


# ---- data loading -----------------------------------------------------------

def test_loader_handles_microseconds_gaps_and_short_hours(tmp_path):
    h0 = 1_790_812_800_000                     # an hour boundary, in ms
    rows, price = [], 100.0
    for m in range(180):                       # three hours of minutes
        if 70 <= m < 90:                       # a 20-minute hole in hour 2
            continue
        price *= 1.001 if m % 2 else 0.999
        ts_us = (h0 + m * 60_000) * 1000       # archive uses microseconds
        rows.append(f"{ts_us},0,0,0,{price:.8f},1,0,0,1,0,0,0")
    with zipfile.ZipFile(tmp_path / "TEST-1m-2026-10.zip", "w") as zf:
        zf.writestr("TEST-1m-2026-10.csv", "\n".join(rows))

    t, c = load_minutes(tmp_path, "TEST")
    assert t[0] == h0 and len(t) == 160
    hourly = to_hourly(t, c)
    assert hourly.missing_minutes == 20
    assert hourly.dropped_hours == 1           # the hour with the hole
    assert list(hourly.start_ms) == [h0, h0 + 2 * 3_600_000]
    # hour 3 by hand: 60 one-minute returns ending inside the hour
    r = np.diff(np.log(c))[-60:]
    assert hourly.rv[-1] == pytest.approx(float(r @ r))
    assert hourly.ret[-1] == pytest.approx(float(r.sum()))


def test_month_range_crosses_year_end():
    assert month_range("2025-11", "2026-02") == ["2025-11", "2025-12", "2026-01", "2026-02"]


# ---- features ---------------------------------------------------------------

def test_trailing_mean_and_ewma_by_hand():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isnan(trailing_mean(x, 3)[:2]).all()
    assert trailing_mean(x, 3)[2:].tolist() == [2.0, 3.0]
    assert ewma(x, 0.5).tolist() == [1.0, 1.5, 2.25, 3.125]


def test_target_is_the_next_hour_and_features_stop_at_the_origin():
    start, rv, ret = simulate_hourly(400, seed=3)
    d = build_dataset(start, rv, ret)
    lo = 167
    assert d.y[0] == rv[lo + 1] and d.rv_h[0] == rv[lo]
    assert d.rv_d[0] == pytest.approx(rv[lo - 23: lo + 1].mean())
    assert d.rv_w[0] == pytest.approx(rv[: lo + 1].mean())
    # changing the future must not change any feature at or before the origin
    k = 300
    rv2, ret2 = rv.copy(), ret.copy()
    rv2[k + 1:] *= 50
    ret2[k + 1:] += 1
    d2 = build_dataset(start, rv2, ret2)
    rows = slice(0, k - lo + 1)
    for name in ("rv_h", "rv_d", "rv_w", "ret_h", "ret_d"):
        assert np.array_equal(getattr(d, name)[rows], getattr(d2, name)[rows]), name


# ---- the harness itself -----------------------------------------------------

@pytest.mark.parametrize("make", [Persistence, Ewma, Har, HarLog])
def test_walk_forward_cannot_see_the_future(ds, make):
    """Corrupt every row from k onward. Forecasts before k must not move at all."""
    k = 1500
    base = walk_forward(ds, make(), INITIAL, REFIT)
    bad = copy.deepcopy(ds)
    for name in ("rv_h", "rv_d", "rv_w", "rv_all", "y", "ret_h", "ret_d", "y_ret"):
        getattr(bad, name)[k:] *= 1000.0
    again = walk_forward(bad, make(), INITIAL, REFIT)
    assert np.array_equal(base[INITIAL:k], again[INITIAL:k])
    assert not np.array_equal(base[k:], again[k:])     # the corruption was real


def test_rows_before_the_first_origin_get_no_forecast(ds):
    f = walk_forward(ds, Har(), INITIAL, REFIT)
    assert np.isnan(f[:INITIAL]).all() and np.isfinite(f[INITIAL:]).all() and (f[INITIAL:] > 0).all()


def test_known_ranking_on_clustered_volatility(ds):
    y = ds.y[INITIAL:]
    score = {m.name: qlike(y, walk_forward(ds, m, INITIAL, REFIT)[INITIAL:])
             for m in (Persistence(), Ewma(), Har(), HarLog())}
    assert score["har_log"] < score["persistence"]
    assert score["ewma"] < score["persistence"]
    assert score["har"] < score["persistence"]


# ---- metrics ----------------------------------------------------------------

def test_qlike_is_zero_at_truth_and_punishes_underforecasting_more():
    y = np.array([4.0])
    assert qlike(y, y) == 0.0
    assert qlike(y, y / 2) > qlike(y, y * 2) > 0


def test_dm_and_bootstrap_agree_on_an_obviously_better_forecast():
    rng = np.random.default_rng(0)
    y = np.exp(rng.normal(size=4000))
    good, bad = y * np.exp(rng.normal(0, 0.1, 4000)), y * np.exp(rng.normal(0, 0.6, 4000))
    la, lb = qlike_losses(y, good), qlike_losses(y, bad)
    stat, p = dm_test(la, lb)
    assert stat < 0 and p < 1e-6
    lo, hi = block_bootstrap_ci(la, lb, n_boot=500)
    assert 0 < lo < hi < 1
    stat0, p0 = dm_test(la, la + rng.normal(0, 1e-3, 4000))
    assert p0 > 0.01                                   # no real difference, no rejection


def test_wilson_interval_contains_the_estimate():
    lo, hi = wilson_ci(520, 1000)
    assert lo < 0.52 < hi and hi - lo < 0.07


# ---- the leakage check must be able to fail ---------------------------------

@pytest.mark.parametrize("with_returns", [False, True])
def test_direction_is_a_coin_flip_on_clean_features(ds, with_returns):
    pred, up = walk_forward_direction(ds, INITIAL, 168, with_returns=with_returns)
    lo, hi = wilson_ci(int((pred == up).sum()), len(up))
    assert lo < 0.5 < hi


def test_a_planted_look_ahead_feature_is_caught(ds):
    pred, up = walk_forward_direction(ds, INITIAL, 168, extra=ds.y_ret)   # the future, as a feature
    assert (pred == up).mean() > 0.95
