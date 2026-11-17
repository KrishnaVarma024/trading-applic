"""The boosted trees as flat arrays, evaluated for one row without scikit-learn.

Why: on the author's machine, scikit-learn's `predict` took about 540 us for one
row, most of it spent validating and converting the input, not walking trees.
The model is 150 trees of depth at most 3. Walking them is a few hundred
comparisons, which NumPy does in tens of microseconds.

Correctness: `tests/test_compiled.py` checks the result equals scikit-learn's
to 1e-12 on thousands of rows, including values outside the training range and
missing values. The reference path stays available for that test and for the
benchmark's `--reference-model` flag.
"""
from __future__ import annotations

import numpy as np


class CompiledTrees:
    def __init__(self, gbm) -> None:
        loss = type(getattr(gbm, "_loss", None)).__name__
        if loss != "HalfSquaredError":
            raise ValueError(f"only squared-error regression is supported, got {loss}")
        trees = [stage[0].nodes for stage in gbm._predictors]
        if any(t["is_categorical"].any() for t in trees):
            raise ValueError("categorical splits are not supported")

        n_trees, width = len(trees), max(len(t) for t in trees)
        shape = (n_trees, width)
        self.feature = np.zeros(shape, dtype=np.intp)
        self.threshold = np.zeros(shape)
        self.left = np.zeros(shape, dtype=np.intp)
        self.right = np.zeros(shape, dtype=np.intp)
        self.leaf = np.ones(shape, dtype=bool)            # padding behaves as a leaf
        self.value = np.zeros(shape)
        self.missing_left = np.zeros(shape, dtype=bool)
        for i, t in enumerate(trees):
            k = len(t)
            self.feature[i, :k] = t["feature_idx"]
            self.threshold[i, :k] = t["num_threshold"]
            self.left[i, :k] = t["left"]
            self.right[i, :k] = t["right"]
            self.leaf[i, :k] = t["is_leaf"].astype(bool)
            self.value[i, :k] = t["value"]
            self.missing_left[i, :k] = t["missing_go_to_left"].astype(bool)

        self.depth = int(max(t["depth"].max() for t in trees))
        self.baseline = float(np.asarray(gbm._baseline_prediction).ravel()[0])
        self.n_trees = n_trees
        self._rows = np.arange(n_trees)

    def predict_one(self, x: np.ndarray) -> float:
        """Sum of all trees' leaf values for one feature vector, plus the baseline."""
        rows = self._rows
        node = np.zeros(self.n_trees, dtype=np.intp)       # every tree starts at its root
        for _ in range(self.depth):                        # one level of every tree per step
            v = x[self.feature[rows, node]]
            go_left = np.where(np.isnan(v), self.missing_left[rows, node], v <= self.threshold[rows, node])
            step = np.where(go_left, self.left[rows, node], self.right[rows, node])
            node = np.where(self.leaf[rows, node], node, step)   # a tree that reached a leaf stays there
        return self.baseline + float(self.value[rows, node].sum())
