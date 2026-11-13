"""Experimental direction diagnostic -- NOT part of the public API in v0.1.

This module is intentionally not wired into :func:`sharedimp.shared_importance`
and is not exported. A direction diagnostic was removed from the common R/Python
API for v0.1 because its definition is not settled: this module computes the
Spearman correlation between the anchor and the Full model's out-of-sample
prediction, the R package computes the Pearson correlation between the anchor
and ``pred_full - pred_reduced``, and neither equals the direction coefficient
proposed originally. It is kept only as a starting point for a later version
and may change or disappear without notice.

Whatever definition is adopted, such a diagnostic is a descriptive statistic of
a fitted predictive model, **not** a causal effect and not an inferential
quantity.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import rankdata
from sklearn.base import is_classifier

__all__ = ["direction_diagnostic", "prediction_for_direction"]


def prediction_for_direction(fitted: Any, X_test: Any) -> np.ndarray | None:
    """One-dimensional out-of-sample prediction used by the direction diagnostic.

    * Classifier with exactly two classes: predicted probability of
      ``classes_[1]`` (or ``decision_function`` if ``predict_proba`` is absent).
    * Classifier with more than two classes: ``None`` (direction is undefined).
    * Anything else (regressors): ``predict``; ``None`` if the output is not
      one-dimensional.
    """
    if is_classifier(fitted):
        classes = getattr(fitted, "classes_", None)
        if classes is None or len(classes) != 2:
            return None
        if hasattr(fitted, "predict_proba"):
            return np.asarray(fitted.predict_proba(X_test), dtype=float)[:, 1]
        if hasattr(fitted, "decision_function"):
            pred = np.asarray(fitted.decision_function(X_test), dtype=float)
            return pred if pred.ndim == 1 else None
        return None
    pred = np.asarray(fitted.predict(X_test), dtype=float)
    if pred.ndim == 2 and pred.shape[1] == 1:
        pred = pred[:, 0]
    return pred if pred.ndim == 1 else None


def direction_diagnostic(anchor_values: np.ndarray, prediction: np.ndarray | None) -> float:
    """Spearman rank correlation between anchor values and model predictions.

    Returns ``nan`` when the prediction is unavailable, when fewer than three
    samples are available, or when either input is constant.
    """
    if prediction is None:
        return float("nan")
    a = np.asarray(anchor_values, dtype=float)
    p = np.asarray(prediction, dtype=float)
    if a.shape != p.shape or a.size < 3:
        return float("nan")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(p))):
        return float("nan")
    ra = rankdata(a)
    rp = rankdata(p)
    ra -= ra.mean()
    rp -= rp.mean()
    denom = float(np.sqrt((ra @ ra) * (rp @ rp)))
    if denom == 0.0:
        return float("nan")
    return float(np.clip((ra @ rp) / denom, -1.0, 1.0))
