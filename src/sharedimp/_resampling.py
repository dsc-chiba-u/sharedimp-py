"""CV / resampling engine: turn a ``cv`` specification into concrete splits."""

from __future__ import annotations

import numbers
from typing import Any

import numpy as np
from sklearn.base import is_classifier
from sklearn.model_selection import KFold, StratifiedKFold, check_cv
from sklearn.utils.multiclass import type_of_target

__all__ = ["make_splits", "resolve_cv"]

Split = tuple[np.ndarray, np.ndarray]


def resolve_cv(cv: Any, y: Any, estimator: Any, cv_seed: int | None) -> Any:
    """Resolve ``cv`` into a scikit-learn splitter.

    * ``None`` is treated as ``5``.
    * An integer ``k`` gives a *shuffled* ``StratifiedKFold`` when the estimator
      is a classifier and ``y`` is binary or multiclass, and a *shuffled*
      ``KFold`` otherwise. The shuffle is seeded with ``cv_seed``.
    * A splitter object or an iterable of ``(train, test)`` index pairs is used
      as given (via :func:`sklearn.model_selection.check_cv`); its own
      ``random_state`` is never modified.
    """
    if cv is None:
        cv = 5
    if isinstance(cv, (bool, np.bool_)):
        raise TypeError("`cv` must be an int, a CV splitter, or an iterable of splits.")
    if isinstance(cv, numbers.Integral):
        k = int(cv)
        if k < 2:
            raise ValueError(f"An integer `cv` must be at least 2, got {k}.")
        stratify = is_classifier(estimator) and type_of_target(y) in ("binary", "multiclass")
        if stratify:
            return StratifiedKFold(n_splits=k, shuffle=True, random_state=cv_seed)
        return KFold(n_splits=k, shuffle=True, random_state=cv_seed)
    return check_cv(cv, y, classifier=is_classifier(estimator))


def make_splits(cv: Any, X: np.ndarray, y: Any, groups: Any) -> list[Split]:
    """Materialize the ``(train, test)`` index arrays of a resolved splitter."""
    n_samples = X.shape[0]
    splits: list[Split] = []
    for train, test in cv.split(X, y, groups):
        train = np.asarray(train)
        test = np.asarray(test)
        if train.dtype == bool:
            train = np.flatnonzero(train)
        if test.dtype == bool:
            test = np.flatnonzero(test)
        train = train.astype(np.intp, copy=False)
        test = test.astype(np.intp, copy=False)
        if train.size == 0 or test.size == 0:
            raise ValueError("Every CV split must have non-empty training and assessment sets.")
        for name, idx in (("training", train), ("assessment", test)):
            if idx.min() < 0 or idx.max() >= n_samples:
                raise ValueError(
                    f"A CV split has {name} indices out of bounds for {n_samples} samples."
                )
        splits.append((train, test))
    if not splits:
        raise ValueError("`cv` produced no splits.")
    return splits
