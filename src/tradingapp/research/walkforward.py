"""Walk-forward evaluation: fit on the past, forecast the next block, move on.

This mimics live use. At origin `start` the model has seen rows [0, start).
Row s holds the target RV of hour s+1, so the newest target in training is the
RV of hour `start`, which is known at that origin. Nothing later is touched.
"""
from __future__ import annotations

import numpy as np

from .features import Dataset


def walk_forward(ds: Dataset, model, initial: int, refit_every: int) -> np.ndarray:
    """Return forecasts for rows [initial, len(ds)). Earlier rows are NaN."""
    n = len(ds)
    if not 0 < initial < n:
        raise ValueError("initial must be inside the dataset")
    preds = np.full(n, np.nan)
    for start in range(initial, n, refit_every):
        model.fit(ds, start)
        rows = slice(start, min(start + refit_every, n))
        preds[rows] = model.predict(ds, rows)
    return preds
