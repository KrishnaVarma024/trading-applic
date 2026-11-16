"""Synthetic hourly data with clustered volatility and unpredictable direction.

Used by tests: we know the truth about this data, so we know what a correct
harness must report on it.
"""
from __future__ import annotations

import numpy as np


def simulate_hourly(n_hours: int, seed: int = 0, phi: float = 0.97, vol_of_vol: float = 0.25):
    rng = np.random.default_rng(seed)
    mu = np.log(4e-5)                       # typical hourly variance
    h = np.empty(n_hours)
    h[0] = mu
    shocks = rng.normal(size=n_hours) * vol_of_vol
    for t in range(1, n_hours):
        h[t] = mu + phi * (h[t - 1] - mu) + shocks[t]
    minute = rng.normal(size=(n_hours, 60)) * np.sqrt(np.exp(h) / 60)[:, None]
    start_ms = np.arange(n_hours, dtype=np.int64) * 3_600_000
    return start_ms, (minute ** 2).sum(axis=1), minute.sum(axis=1)
