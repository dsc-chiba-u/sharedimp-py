"""Estimator cloning and fitting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.base import clone, is_classifier
from sklearn.dummy import DummyClassifier, DummyRegressor

__all__ = [
    "clone_with_seed",
    "fit_clone",
    "fit_null_model",
    "index_rows",
    "slice_fit_params",
]


def clone_with_seed(estimator: Any, seed: int | None) -> Any:
    """Clone ``estimator``; optionally seed its unset ``random_state`` parameters.

    Hyperparameters are never changed. The only parameters touched are those
    named ``random_state`` (at any nesting depth, e.g. inside a ``Pipeline``)
    whose value is ``None``, and only when ``seed`` is not ``None``. Parameters
    that the user set explicitly (an ``int`` or a ``RandomState``) are kept.
    """
    est = clone(estimator)
    if seed is None:
        return est
    updates = {
        name: seed
        for name, value in est.get_params(deep=True).items()
        if (name == "random_state" or name.endswith("__random_state")) and value is None
    }
    if updates:
        est.set_params(**updates)
    return est


def index_rows(obj: Any, idx: np.ndarray) -> Any:
    """Select rows ``idx`` of an array-like, preserving pandas containers."""
    if hasattr(obj, "iloc"):
        return obj.iloc[idx]
    return np.asarray(obj)[idx]


def slice_fit_params(
    fit_params: Mapping[str, Any] | None, idx: np.ndarray, n_samples: int
) -> dict[str, Any]:
    """Subset per-sample fit parameters (e.g. ``sample_weight``) to rows ``idx``.

    A value is treated as per-sample when it is array-like with first dimension
    equal to ``n_samples``; everything else is passed through unchanged.
    """
    if not fit_params:
        return {}
    out: dict[str, Any] = {}
    for key, value in fit_params.items():
        per_sample = (
            not isinstance(value, (str, bytes, Mapping))
            and hasattr(value, "__len__")
            and (hasattr(value, "shape") or isinstance(value, (list, tuple)))
            and len(value) == n_samples
        )
        out[key] = index_rows(value, idx) if per_sample else value
    return out


def fit_clone(
    estimator: Any,
    X_train: Any,
    y_train: Any,
    *,
    seed: int | None,
    fit_params: Mapping[str, Any] | None = None,
) -> Any:
    """Clone ``estimator`` (see :func:`clone_with_seed`) and fit the clone."""
    est = clone_with_seed(estimator, seed)
    est.fit(X_train, y_train, **dict(fit_params or {}))
    return est


def fit_null_model(
    estimator: Any,
    n_train: int,
    y_train: Any,
    *,
    fit_params: Mapping[str, Any] | None = None,
) -> Any:
    """Fit the null model used when a representation has zero columns.

    This happens only for the Reduced model of a single-feature ``X``. The
    null model ignores the features:

    * classifier: predicts the training class frequencies as probabilities
      (and the most frequent class; ties go to the first class in sorted
      order) -- ``DummyClassifier(strategy="prior")``;
    * anything else: predicts the training mean of ``y`` --
      ``DummyRegressor(strategy="mean")``.

    Of ``fit_params`` only ``sample_weight`` is forwarded (weighted mean /
    weighted class frequencies), mirroring a weighted fit of the Full model.
    """
    null = (
        DummyClassifier(strategy="prior")
        if is_classifier(estimator)
        else DummyRegressor(strategy="mean")
    )
    weights = (fit_params or {}).get("sample_weight")
    null.fit(np.empty((n_train, 0)), y_train, sample_weight=weights)
    return null
