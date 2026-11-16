"""Extra features for the phase 3 model, aligned row-for-row with a Dataset.

Row i of the matrix is the same forecast origin as row i of the Dataset, and
like it uses only hours up to and including the origin hour.
"""
from __future__ import annotations

import numpy as np

from .data import HOUR_MS, Hourly
from .features import EPS, WEEK, Dataset, trailing_mean

FEATURE_NAMES = [
    "log_rv_h", "log_rv_d", "log_rv_w",        # the HAR terms
    "log_rv_4h", "log_rv_12h",                 # horizons between hour and day
    "neg_share",                               # share of the hour's variance from falling prices
    "ret_h", "abs_ret_h",                      # signed and unsigned last-hour move
    "log_volume_h", "log_volume_rel",          # activity, and activity against its weekly norm
    "log_trades_h",
    "hour_sin", "hour_cos",                    # time of day, as a circle so 23:00 sits next to 00:00
    "weekend",
]


def build_matrix(ds: Dataset, hourly: Hourly) -> np.ndarray:
    if hourly.volume is None or hourly.trades is None or hourly.rs_neg is None:
        raise ValueError("hourly series lacks volume, trades or semivariance")
    rv = np.maximum(hourly.rv, EPS)
    lo, hi = WEEK - 1, len(rv) - 1               # the same slice build_dataset uses
    s = slice(lo, hi)
    if hi - lo != len(ds):
        raise ValueError("hourly series and dataset are not aligned")

    vol = np.maximum(hourly.volume, EPS)
    hour_of_day = (hourly.start_ms // HOUR_MS) % 24
    # 1970-01-01 was a Thursday: day index 0 -> Thursday, so 2 and 3 are Saturday and Sunday.
    weekday = (hourly.start_ms // (24 * HOUR_MS)) % 7
    angle = 2 * np.pi * hour_of_day / 24

    cols = [
        np.log(ds.rv_h), np.log(ds.rv_d), np.log(ds.rv_w),
        np.log(trailing_mean(rv, 4))[s], np.log(trailing_mean(rv, 12))[s],
        (hourly.rs_neg / rv)[s],
        ds.ret_h, np.abs(ds.ret_h),
        np.log(vol)[s], (np.log(vol) - np.log(trailing_mean(vol, WEEK)))[s],
        np.log(np.maximum(hourly.trades, 1.0))[s],
        np.sin(angle)[s], np.cos(angle)[s],
        np.isin(weekday, (2, 3)).astype(float)[s],
    ]
    x = np.column_stack(cols)
    assert x.shape[1] == len(FEATURE_NAMES)
    return x
