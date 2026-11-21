import numpy as np

from tradingapp.research.ablation import hour_indicators, hour_profile, run_ablation

from .test_model import dataset, make_hourly


def test_hour_profile_finds_a_planted_daily_peak():
    h = make_hourly(3000, seed=31, seasonal=0.8)          # variance scaled by exp(0.8 sin(2 pi h / 24))
    p = hour_profile(h.start_ms, h.rv)
    assert p["peak_hour_utc"] in (5, 6, 7)                # sin peaks at hour 6
    assert abs(np.mean(p["multiple_of_mean_by_utc_hour"]) - 1) < 0.1


def test_hour_indicators_avoid_the_dummy_trap():
    start = np.arange(48, dtype=np.int64) * 3_600_000
    d = hour_indicators(start, np.zeros(48))
    assert d.shape == (48, 24)                            # 23 hours + weekend; hour 0 is the reference
    assert d[:, :23].sum(axis=1).tolist() == [0.0 if i % 24 == 0 else 1.0 for i in range(48)]


def test_removing_the_clock_hurts_when_the_clock_matters():
    h = make_hourly(2500, seed=32, seasonal=0.8)
    ds, x = dataset(h)
    initial = 800
    is_test = np.arange(len(ds) - initial) >= (len(ds) - initial) // 2
    r = run_ablation(ds, x, initial, 250, is_test, max_iter=40)
    a = r["ablation_dev"]
    assert a["time of day and weekend"]["dev_qlike"] > a["none removed"]["dev_qlike"]
    assert r["hourly_baseline"]["test"]["hourly_vs_har_log"]["qlike_reduction"] > 0
