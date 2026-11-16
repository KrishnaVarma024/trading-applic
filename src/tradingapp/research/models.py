"""Volatility forecasters. Each predicts next-hour realized variance.

Common interface:
    fit(ds, end)      learn from rows [0, end) only
    predict(ds, rows) forecast for the given rows using their features
"""
from __future__ import annotations

import numpy as np

from .features import Dataset, ewma
from .metrics import qlike


class Persistence:
    """Next hour's variance equals this hour's. The no-skill reference."""
    name = "persistence"

    def fit(self, ds: Dataset, end: int) -> None:
        pass

    def predict(self, ds: Dataset, rows: slice) -> np.ndarray:
        return ds.rv_h[rows]


class Ewma:
    """Exponentially weighted average of past variance.

    The decay is chosen on the training rows by QLIKE at every refit. A baseline
    with a fixed, untuned decay would be easy to beat and prove little.
    """
    name = "ewma"

    def __init__(self, grid: tuple[float, ...] = tuple(np.round(np.arange(0.50, 0.99, 0.02), 2))) -> None:
        self.grid = grid
        self.lam: float | None = None
        self._paths: dict[float, np.ndarray] = {}
        self._source: int | None = None

    def _ensure_paths(self, ds: Dataset) -> None:
        if self._source != id(ds):
            self._paths = {lam: ewma(ds.rv_all, lam) for lam in self.grid}
            self._source = id(ds)

    def fit(self, ds: Dataset, end: int) -> None:
        self._ensure_paths(ds)
        self.lam = min(self.grid, key=lambda lam: qlike(ds.y[:end], self._paths[lam][:end]))

    def predict(self, ds: Dataset, rows: slice) -> np.ndarray:
        return self._paths[self.lam][rows]


class Har:
    """HAR-RV (Corsi, 2009): linear in last-hour, last-day and last-week variance.

    Three horizons stand in for traders acting on different time scales, and
    together they reproduce the slow decay of volatility autocorrelation.
    """
    name = "har"

    def __init__(self) -> None:
        self.beta: np.ndarray | None = None
        self.floor = 0.0

    @staticmethod
    def _x(ds: Dataset, rows: slice) -> np.ndarray:
        return np.column_stack([np.ones_like(ds.rv_h[rows]), ds.rv_h[rows], ds.rv_d[rows], ds.rv_w[rows]])

    def fit(self, ds: Dataset, end: int) -> None:
        self.beta, *_ = np.linalg.lstsq(self._x(ds, slice(0, end)), ds.y[:end], rcond=None)
        self.floor = 0.5 * float(ds.y[:end].min())   # a variance forecast must stay positive

    def predict(self, ds: Dataset, rows: slice) -> np.ndarray:
        return np.maximum(self._x(ds, rows) @ self.beta, self.floor)


class HarLog(Har):
    """HAR fitted on log variance. Positive by construction and less driven by spikes."""
    name = "har_log"

    def __init__(self) -> None:
        super().__init__()
        self.resid_var = 0.0

    @staticmethod
    def _x(ds: Dataset, rows: slice) -> np.ndarray:
        return np.column_stack([np.ones_like(ds.rv_h[rows]), np.log(ds.rv_h[rows]),
                                np.log(ds.rv_d[rows]), np.log(ds.rv_w[rows])])

    def fit(self, ds: Dataset, end: int) -> None:
        x, ly = self._x(ds, slice(0, end)), np.log(ds.y[:end])
        self.beta, *_ = np.linalg.lstsq(x, ly, rcond=None)
        self.resid_var = float(np.var(ly - x @ self.beta))

    def predict(self, ds: Dataset, rows: slice) -> np.ndarray:
        # exp(mean of log) underestimates the mean; half the residual variance corrects it.
        return np.exp(self._x(ds, rows) @ self.beta + 0.5 * self.resid_var)


class HarGbm:
    """log-HAR plus gradient-boosted trees on what log-HAR gets wrong.

    Stage 1 is the linear log-HAR fit. Stage 2 is a boosted-tree model trained
    to predict stage 1's residual from a wider feature set. Trees cannot
    extrapolate beyond the range they were trained on, so the linear stage
    carries the level and the trees only add corrections.
    """
    name = "har_gbm"

    def __init__(self, x: np.ndarray, **gbm_params) -> None:
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.x = x
        self.har = HarLog()
        params = dict(max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=40,
                      l2_regularization=1.0, early_stopping=False, random_state=0)
        params.update(gbm_params)
        self._make = lambda: HistGradientBoostingRegressor(**params)
        self.gbm = None
        self.resid_var = 0.0

    def fit(self, ds: Dataset, end: int) -> None:
        self.har.fit(ds, end)
        rows = slice(0, end)
        linear = HarLog._x(ds, rows) @ self.har.beta
        resid = np.log(ds.y[:end]) - linear
        self.gbm = self._make().fit(self.x[:end], resid)
        self.resid_var = float(np.var(resid - self.gbm.predict(self.x[:end])))

    def predict(self, ds: Dataset, rows: slice) -> np.ndarray:
        linear = HarLog._x(ds, rows) @ self.har.beta
        return np.exp(linear + self.gbm.predict(self.x[rows]) + 0.5 * self.resid_var)


class HarLogSeasonal(HarLog):
    """log-HAR plus time-of-day and weekend terms, still a plain linear regression.

    This is the fair opponent for the tree model. If most of the trees' gain is
    seasonality, this model will recover it with three extra coefficients.
    """
    name = "har_log_seasonal"

    def __init__(self, seasonal: np.ndarray) -> None:
        super().__init__()
        self.seasonal = seasonal        # columns: hour_sin, hour_cos, weekend

    def _xs(self, ds: Dataset, rows: slice) -> np.ndarray:
        return np.column_stack([HarLog._x(ds, rows), self.seasonal[rows]])

    def fit(self, ds: Dataset, end: int) -> None:
        x, ly = self._xs(ds, slice(0, end)), np.log(ds.y[:end])
        self.beta, *_ = np.linalg.lstsq(x, ly, rcond=None)
        self.resid_var = float(np.var(ly - x @ self.beta))

    def predict(self, ds: Dataset, rows: slice) -> np.ndarray:
        return np.exp(self._xs(ds, rows) @ self.beta + 0.5 * self.resid_var)
