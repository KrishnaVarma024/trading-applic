"""A trained model reduced to what inference needs, plus save and load."""
from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from ..research.features_ext import FEATURE_NAMES
from ..research.models import HarGbm
from .compiled import CompiledTrees


def pin_single_thread() -> None:
    """Keep inference on the calling thread for the life of the process.

    The tree library can hand even a one-row prediction to a thread pool.
    Waking pool threads costs more than the prediction and, worse, the cost
    varies, which shows up as a long latency tail. One row needs one thread.
    """
    threadpool_limits(limits=1)


@dataclass
class Forecaster:
    beta: np.ndarray          # log-HAR coefficients: intercept, hour, day, week
    gbm: object               # fitted boosted trees for the residual
    resid_var: float
    meta: dict

    @classmethod
    def from_model(cls, model: HarGbm, **meta) -> "Forecaster":
        meta = {"features": FEATURE_NAMES, "sklearn": sklearn.__version__,
                "python": platform.python_version(), **meta}
        return cls(beta=model.har.beta.copy(), gbm=model.gbm, resid_var=model.resid_var, meta=meta).prepare()

    def prepare(self, reference: bool = False) -> "Forecaster":
        """Build the fast tree evaluator, or choose scikit-learn's own predict.

        Called once at start-up so that no forecast pays for it. The compiled
        trees are rebuilt from the fitted model each time, never saved, so the
        model file format does not change.
        """
        self._trees = None if reference else CompiledTrees(self.gbm)
        return self

    def predict(self, har_logs: np.ndarray, x: np.ndarray) -> float:
        """Forecast realized variance over the next hour."""
        linear = self.beta[0] + float(self.beta[1:] @ har_logs)
        trees = self.__dict__.get("_trees")
        if trees is None:
            correction = float(self.gbm.predict(x.reshape(1, -1))[0])
        else:
            correction = trees.predict_one(x)
        return float(np.exp(linear + correction + 0.5 * self.resid_var))

    def save(self, path: str | Path) -> None:
        state = dict(self.__dict__)
        self.__dict__.pop("_trees", None)       # never persist the compiled form
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self, path)
        finally:
            self.__dict__.update(state)

    @staticmethod
    def load(path: str | Path) -> "Forecaster":
        # joblib files can run code when loaded. Only load a file this project wrote.
        obj = joblib.load(path)
        if not isinstance(obj, Forecaster):
            raise TypeError(f"{path} is not a Forecaster")
        if obj.meta.get("sklearn") != sklearn.__version__:
            raise RuntimeError(
                f"model was saved with scikit-learn {obj.meta.get('sklearn')}, this is {sklearn.__version__}; "
                "retrain with: uv run python -m tradingapp.research.train")
        if obj.meta.get("features") != FEATURE_NAMES:
            raise RuntimeError("model was trained on a different feature list; retrain")
        return obj.prepare()
