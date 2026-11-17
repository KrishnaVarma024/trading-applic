import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from tradingapp.live.bench import closes_a_bar, run_bench
from tradingapp.live.compiled import CompiledTrees
from tradingapp.live.forecaster import Forecaster
from tradingapp.live.scheduling import request_interactive_qos
from tradingapp.research.models import HarGbm

from .helpers import msg
from .test_live import T0
from .test_model import dataset, make_hourly


def fitted(x, y, **kw):
    params = dict(max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=40,
                  l2_regularization=1.0, early_stopping=False, random_state=0)
    params.update(kw)
    return HistGradientBoostingRegressor(**params).fit(x, y)


def test_compiled_trees_equal_sklearn_inside_and_outside_the_training_range():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3000, 14))
    y = np.sin(x[:, 0]) + x[:, 1] * x[:, 2] + rng.normal(0, 0.1, 3000)
    g = fitted(x, y)
    trees = CompiledTrees(g)
    probe = np.vstack([rng.normal(size=(1000, 14)),            # inside the range
                       rng.normal(0, 25, size=(500, 14))])     # far outside it
    expected = g.predict(probe)
    got = np.array([trees.predict_one(row) for row in probe])
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("depth", [2, 3, 5])
def test_compiled_trees_equal_sklearn_at_other_depths(depth):
    rng = np.random.default_rng(depth)
    x = rng.normal(size=(2000, 6))
    g = fitted(x, x[:, 0] - 2 * x[:, 3] + rng.normal(0, 0.2, 2000), max_depth=depth, max_iter=60)
    trees = CompiledTrees(g)
    probe = rng.normal(0, 3, size=(500, 6))
    np.testing.assert_allclose([trees.predict_one(r) for r in probe], g.predict(probe), rtol=1e-12, atol=1e-12)


def test_missing_values_follow_the_learned_direction():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3000, 4))
    x[rng.random(3000) < 0.1, 2] = np.nan                     # train with gaps so splits learn a side
    g = fitted(x, np.nan_to_num(x[:, 2], nan=3.0) + x[:, 0])
    trees = CompiledTrees(g)
    probe = rng.normal(size=(300, 4))
    probe[::3, 2] = np.nan
    np.testing.assert_allclose([trees.predict_one(r) for r in probe], g.predict(probe), rtol=1e-12, atol=1e-12)


def test_forecaster_fast_path_equals_reference_path():
    ds, x = dataset(make_hourly(1500, seed=21))
    model = HarGbm(x, max_iter=60)
    model.fit(ds, len(ds))
    fast = Forecaster.from_model(model)
    ref = Forecaster.from_model(model).prepare(reference=True)
    for r in range(0, len(ds), 37):
        har = np.log([ds.rv_h[r], ds.rv_d[r], ds.rv_w[r]])
        assert fast.predict(har, x[r]) == pytest.approx(ref.predict(har, x[r]), rel=1e-12)


def test_compiled_form_is_not_saved(tmp_path):
    ds, x = dataset(make_hourly(800, seed=22))
    model = HarGbm(x, max_iter=20)
    model.fit(ds, len(ds))
    f = Forecaster.from_model(model)
    f.save(tmp_path / "m.joblib")
    assert "_trees" in f.__dict__                              # still usable after saving
    import joblib
    assert "_trees" not in joblib.load(tmp_path / "m.joblib").__dict__
    assert "_trees" in Forecaster.load(tmp_path / "m.joblib").__dict__   # load prepares it


def test_bench_idle_pause_and_forecast_cap():
    ds, x = dataset(make_hourly(800, seed=23))
    model = HarGbm(x, max_iter=20)
    model.fit(ds, len(ds))
    f = Forecaster.from_model(model)
    messages = [msg(i, T0 + i * 250, price=f"{60_000 + (i % 7)}") for i in range(1, 2001)]
    flags = closes_a_bar(messages, 1000)
    assert sum(flags) == 500 and not flags[0]                  # trades span 501 one-second bars
    live = run_bench(messages, f, interval_ms=1000, warmup_forecasts=10, idle_ms=2, max_forecasts=15)
    assert live.latency.count() == 15


def test_qos_request_never_raises():
    assert isinstance(request_interactive_qos(), str)
