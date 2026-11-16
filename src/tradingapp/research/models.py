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
