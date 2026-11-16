"""Leakage check: can the same features predict next-hour direction?

They should not, by much. Volatility clusters, but the sign of the next return
is close to a coin flip. If a classifier on these features scores far above
the base rate, the likeliest cause is a feature that contains the future.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import Dataset


def direction_features(ds: Dataset, with_returns: bool = False) -> np.ndarray:
    """The volatility model's own inputs. `with_returns` adds past returns.

    The leakage check uses the model's inputs only. Past returns are a separate
    diagnostic, because hourly return signs can carry a little serial
    dependence of their own, which would blur what the check is for.
    """
    cols = [np.log(ds.rv_h), np.log(ds.rv_d), np.log(ds.rv_w)]
    if with_returns:
        cols += [ds.ret_h, ds.ret_d]
    return np.column_stack(cols)


def walk_forward_direction(
    ds: Dataset, initial: int, refit_every: int, extra: np.ndarray | None = None,
    with_returns: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (predicted_up, actual_up) for rows [initial, len(ds)).

    `extra` adds a column to the features. Tests use it to plant a leak on
    purpose and confirm this check catches it.
    """
    x = direction_features(ds, with_returns)
    if extra is not None:
        x = np.column_stack([x, extra])
    up = ds.y_ret > 0
    n = len(ds)
    pred = np.zeros(n, dtype=bool)
    for start in range(initial, n, refit_every):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
        clf.fit(x[:start], up[:start])           # scaler statistics come from the past only
        rows = slice(start, min(start + refit_every, n))
        pred[rows] = clf.predict(x[rows])
    return pred[initial:], up[initial:]
