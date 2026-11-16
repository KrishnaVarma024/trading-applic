"""Features and targets, built so that row t uses nothing after hour t.

Row t is the forecast origin "end of hour t". Its features summarise hours
<= t. Its target is the realized variance of hour t+1. The last hour has no
target, so the arrays are one shorter than the hourly series.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DAY, WEEK = 24, 168
EPS = 1e-14


def trailing_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Mean of x[t-window+1 .. t]. NaN until a full window exists."""
    out = np.full(len(x), np.nan)
    c = np.cumsum(np.insert(x, 0, 0.0))
    out[window - 1:] = (c[window:] - c[:-window]) / window
    return out


def ewma(x: np.ndarray, lam: float) -> np.ndarray:
    """e[t] = lam * e[t-1] + (1 - lam) * x[t]. e[t] depends only on x[<= t]."""
    out = np.empty(len(x))
    acc = x[0]
    for i, v in enumerate(x):
        acc = lam * acc + (1 - lam) * v
        out[i] = acc
    return out


@dataclass
class Dataset:
    start_ms: np.ndarray    # hour start of the origin hour t
    rv_h: np.ndarray        # RV of hour t
    rv_d: np.ndarray        # mean RV over the last 24 hours, ending at t
    rv_w: np.ndarray        # mean RV over the last 168 hours, ending at t
    ret_h: np.ndarray       # return of hour t
    ret_d: np.ndarray       # return over the last 24 hours, ending at t
    rv_all: np.ndarray      # full RV history up to t, for recursive models
    y: np.ndarray           # target: RV of hour t+1
    y_ret: np.ndarray       # return of hour t+1 (for the direction check)

    def __len__(self) -> int:
        return len(self.y)


def build_dataset(start_ms: np.ndarray, rv: np.ndarray, ret: np.ndarray) -> Dataset:
    rv = np.maximum(rv, EPS)
    rv_d = trailing_mean(rv, DAY)
    rv_w = trailing_mean(rv, WEEK)
    ret_d = trailing_mean(ret, DAY) * DAY
    # Drop the warm-up rows without a full week, and the last row without a target.
    lo, hi = WEEK - 1, len(rv) - 1
    s = slice(lo, hi)
    return Dataset(
        start_ms=start_ms[s], rv_h=rv[s], rv_d=rv_d[s], rv_w=rv_w[s],
        ret_h=ret[s], ret_d=ret_d[s], rv_all=rv[s],
        y=rv[lo + 1: hi + 1], y_ret=ret[lo + 1: hi + 1],
    )
