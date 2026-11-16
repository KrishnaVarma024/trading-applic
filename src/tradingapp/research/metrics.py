"""Loss functions and the statistics used to compare two forecasts."""
from __future__ import annotations

import math

import numpy as np


def qlike_losses(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    """QLIKE per observation: y/f - log(y/f) - 1. Zero when f == y, else positive.

    It penalises under-forecasting variance more than over-forecasting, and it
    ranks forecasts correctly even though realized variance is a noisy proxy
    for the true variance (Patton, 2011). Squared error does too, but it is
    dominated by a handful of extreme hours.
    """
    ratio = y / f
    return ratio - np.log(ratio) - 1.0


def qlike(y: np.ndarray, f: np.ndarray) -> float:
    return float(qlike_losses(y, f).mean())


def rmse(y: np.ndarray, f: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - f) ** 2)))


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray) -> tuple[float, float]:
    """Diebold-Mariano test that two forecasts have equal expected loss.

    Returns (statistic, two-sided p-value). Negative statistic: A has lower loss.
    The variance uses a Newey-West estimate because loss differences are
    autocorrelated in time; treating them as independent overstates significance.
    """
    d = loss_a - loss_b
    n = len(d)
    lags = int(np.floor(n ** (1 / 3)))
    dc = d - d.mean()
    lrv = float(dc @ dc) / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        lrv += 2 * w * float(dc[k:] @ dc[:-k]) / n
    stat = d.mean() / math.sqrt(lrv / n)
    p = math.erfc(abs(stat) / math.sqrt(2))
    return float(stat), float(p)


def block_bootstrap_ci(
    loss_model: np.ndarray, loss_base: np.ndarray,
    block: int = 24, n_boot: int = 2000, seed: int = 0, level: float = 0.95,
) -> tuple[float, float]:
    """CI for the fractional loss reduction 1 - mean(model)/mean(base).

    Resamples whole blocks of consecutive hours, so the dependence between
    neighbouring hours is kept inside each block.
    """
    rng = np.random.default_rng(seed)
    n = len(loss_model)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    red = 1 - loss_model[idx].mean(axis=1) / loss_base[idx].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(red, a)), float(np.quantile(red, 1 - a))


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half
