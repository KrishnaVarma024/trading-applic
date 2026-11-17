"""Incremental feature engine: the live twin of research.features_ext.

Research builds features for a whole year at once from hourly arrays. Live has
one new 1-minute bar at a time. This class keeps a week of per-bar history and
produces the same 14 features, for a 60-bar window ending at the newest bar.

When the newest bar ends exactly on an hour, the output must equal the
research feature row for that hour. tests/test_live.py checks that to 1e-9.
Without that test the model would be trained on one definition of a feature
and served another, which is the most common way a live model quietly fails.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..research.features import EPS

HOUR, H4, H12, DAY, WEEK = 60, 240, 720, 1440, 10080     # window lengths in bars
HOUR_MS = 3_600_000


@dataclass(frozen=True)
class Features:
    bar_end_ms: int
    har_logs: np.ndarray      # log rv_h, log rv_d, log rv_w
    x: np.ndarray             # the 14 model features, in FEATURE_NAMES order
    degraded: bool            # an empty bar sits inside the last hour: data was lost


class _Ring:
    """Fixed-size history where the last k values are always one contiguous slice.

    Each value is written twice, at i and i + size, so a window never wraps.
    """

    def __init__(self, size: int) -> None:
        self.size = size
        self.buf = np.zeros(2 * size)
        self.i = 0

    def push(self, v: float) -> None:
        self.buf[self.i] = v
        self.buf[self.i + self.size] = v
        self.i = (self.i + 1) % self.size

    def last(self, k: int) -> np.ndarray:
        end = self.i + self.size
        return self.buf[end - k:end]


class FeatureEngine:
    def __init__(self, interval_ms: int = 60_000) -> None:
        self.interval_ms = interval_ms
        self.resets = 0
        self._reset()

    def _reset(self) -> None:
        self._r2, self._r, self._neg = _Ring(WEEK), _Ring(WEEK), _Ring(WEEK)
        self._vol, self._trd = _Ring(WEEK), _Ring(WEEK)
        self._n = 0                      # returns held, capped at WEEK
        self._last_close: float | None = None
        self._last_start: int | None = None

    @property
    def warm(self) -> bool:
        return self._n >= WEEK

    def update(self, start_ms: int, close: float, volume: float, trades: float) -> Features | None:
        """Add one closed bar. Returns features once a full week of history is held."""
        if self._last_start is not None and start_ms != self._last_start + self.interval_ms:
            # History is no longer contiguous. Old windows would mix two periods.
            self.resets += 1
            self._reset()
        if self._last_close is not None:
            r = np.log(close / self._last_close)
            self._r2.push(r * r)
            self._r.push(r)
            self._neg.push(r * r if r < 0 else 0.0)
            self._vol.push(volume)
            self._trd.push(trades)
            self._n = min(self._n + 1, WEEK)
        self._last_close, self._last_start = close, start_ms
        if not self.warm:
            return None
        return self._features(start_ms + self.interval_ms)

    def _features(self, bar_end_ms: int) -> Features:
        r2 = self._r2
        rv_h = max(float(r2.last(HOUR).sum()), EPS)
        rv_4 = max(float(r2.last(H4).sum()) / 4, EPS)
        rv_12 = max(float(r2.last(H12).sum()) / 12, EPS)
        rv_d = max(float(r2.last(DAY).sum()) / 24, EPS)
        rv_w = max(float(r2.last(WEEK).sum()) / 168, EPS)
        ret_h = float(self._r.last(HOUR).sum())
        vol_h = max(float(self._vol.last(HOUR).sum()), EPS)
        vol_w = max(float(self._vol.last(WEEK).sum()) / 168, EPS)
        trd_window = self._trd.last(HOUR)
        trd_h = max(float(trd_window.sum()), 1.0)

        window_start = bar_end_ms - HOUR * self.interval_ms
        hour_of_day = (window_start / HOUR_MS) % 24
        weekday = (window_start // (24 * HOUR_MS)) % 7        # 1970-01-01 was a Thursday
        angle = 2 * np.pi * hour_of_day / 24

        har = np.log(np.array([rv_h, rv_d, rv_w]))
        x = np.array([
            har[0], har[1], har[2],
            np.log(rv_4), np.log(rv_12),
            float(self._neg.last(HOUR).sum()) / rv_h,
            ret_h, abs(ret_h),
            np.log(vol_h), np.log(vol_h) - np.log(vol_w),
            np.log(trd_h),
            np.sin(angle), np.cos(angle),
            1.0 if weekday in (2, 3) else 0.0,
        ])
        return Features(bar_end_ms, har, x, degraded=bool((trd_window == 0).any()))
