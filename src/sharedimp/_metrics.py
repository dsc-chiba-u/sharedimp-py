"""Common metrics with definitions pinned across the R and Python packages.

scikit-learn scorer names and R ``yardstick`` metric names do not always denote
the same quantity (the best-known case: ``yardstick::rsq`` is a squared
correlation, scikit-learn's ``"r2"`` is the coefficient of determination).
:func:`common_scorer` returns scikit-learn compatible scorers whose definitions
are fixed by the common specification (section 9.1; see
``docs/specification.md``) and verified against
``yardstick`` in ``tests/test_cross_language.py``.

Every scorer follows the scikit-learn convention *greater is better*: losses
are returned negated.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

__all__ = ["COMMON_METRICS", "common_scorer"]

_EPS = float(np.finfo(np.float64).eps)

#: Common metric names -> one-line definition (see ``docs/specification.md``).
COMMON_METRICS: dict[str, str] = {
    "mse": "regression; -(mean squared error)",
    "rmse": "regression; -(root mean squared error)",
    "mae": "regression; -(mean absolute error)",
    "r2": "regression; 1 - SS_res / SS_tot (coefficient of determination)",
    "rsq": "regression; squared Pearson correlation between truth and prediction",
    "log_loss": "classification; -(mean multinomial log loss), probabilities clipped to "
    "[eps, 1 - eps] with eps = 2^-52",
    "accuracy": "classification; fraction of correct labels, label = argmax of the "
    "predicted probabilities (ties: first class)",
    "roc_auc": "classification; binary: area under the ROC curve; multiclass: "
    "Hand & Till (2001) average of pairwise AUCs",
}


# ------------------------------------------------------------------ regression
def _prediction(estimator: Any, X: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y, dtype=float).ravel()
    pred = np.asarray(estimator.predict(X), dtype=float).ravel()
    if truth.shape != pred.shape:
        raise ValueError("Predictions and y have inconsistent shapes.")
    return truth, pred


def _mse(estimator: Any, X: Any, y: Any) -> float:
    truth, pred = _prediction(estimator, X, y)
    return -float(np.mean((truth - pred) ** 2))


def _rmse(estimator: Any, X: Any, y: Any) -> float:
    truth, pred = _prediction(estimator, X, y)
    return -float(np.sqrt(np.mean((truth - pred) ** 2)))


def _mae(estimator: Any, X: Any, y: Any) -> float:
    truth, pred = _prediction(estimator, X, y)
    return -float(np.mean(np.abs(truth - pred)))


def _r2(estimator: Any, X: Any, y: Any) -> float:
    truth, pred = _prediction(estimator, X, y)
    if truth.max() == truth.min():
        return float("nan")  # undefined for a constant truth
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    return 1.0 - float(np.sum((truth - pred) ** 2)) / ss_tot


def _rsq(estimator: Any, X: Any, y: Any) -> float:
    truth, pred = _prediction(estimator, X, y)
    # Exact constancy test (max == min): centring a constant vector with its
    # rounded mean can leave non-zero noise.
    if truth.max() == truth.min() or pred.max() == pred.min():
        return float("nan")  # undefined for a constant truth or prediction
    tc = truth - truth.mean()
    pc = pred - pred.mean()
    return float(tc @ pc) ** 2 / (float(tc @ tc) * float(pc @ pc))


# -------------------------------------------------------------- classification
def _aligned_proba(estimator: Any, X: Any, y: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(labels, proba, truth_index)`` on a common, sorted label set.

    The label set is the sorted union of ``estimator.classes_`` and the labels
    in ``y``. A class that the estimator has never seen gets probability 0.
    """
    if not hasattr(estimator, "predict_proba"):
        raise TypeError(
            "This metric needs class probabilities, but the estimator has no `predict_proba`."
        )
    truth = np.asarray(y).ravel()
    classes = np.asarray(estimator.classes_)
    labels = np.unique(np.concatenate([classes.astype(object), truth.astype(object)]))
    raw = np.asarray(estimator.predict_proba(X), dtype=float)
    if raw.ndim != 2 or raw.shape != (truth.shape[0], classes.shape[0]):
        raise ValueError("`predict_proba` returned an array of unexpected shape.")
    proba = np.zeros((raw.shape[0], labels.shape[0]))
    position = {label: k for k, label in enumerate(labels)}
    for col, label in enumerate(classes):
        proba[:, position[label]] = raw[:, col]
    truth_index = np.array([position[label] for label in truth], dtype=np.intp)
    return labels, proba, truth_index


def _log_loss(estimator: Any, X: Any, y: Any) -> float:
    _, proba, truth_index = _aligned_proba(estimator, X, y)
    p_true = proba[np.arange(proba.shape[0]), truth_index]
    return float(np.mean(np.log(np.clip(p_true, _EPS, 1.0 - _EPS))))


def _accuracy(estimator: Any, X: Any, y: Any) -> float:
    _, proba, truth_index = _aligned_proba(estimator, X, y)
    return float(np.mean(np.argmax(proba, axis=1) == truth_index))  # argmax: first maximum


def _roc_auc(estimator: Any, X: Any, y: Any) -> float:
    labels, proba, truth_index = _aligned_proba(estimator, X, y)
    if labels.shape[0] == 2:
        # Symmetric in the choice of the event class when p_0 = 1 - p_1.
        return float(roc_auc_score(truth_index, proba[:, 1]))
    return float(
        roc_auc_score(
            truth_index,
            proba,
            multi_class="ovo",
            average="macro",
            labels=np.arange(labels.shape[0]),
        )
    )


_SCORERS: dict[str, Callable[[Any, Any, Any], float]] = {
    "mse": _mse,
    "rmse": _rmse,
    "mae": _mae,
    "r2": _r2,
    "rsq": _rsq,
    "log_loss": _log_loss,
    "accuracy": _accuracy,
    "roc_auc": _roc_auc,
}


class _CommonScorer:
    """Picklable scorer ``scorer(estimator, X, y) -> float`` with a stable name."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.__name__ = name

    def __call__(self, estimator: Any, X: Any, y: Any) -> float:
        return _SCORERS[self.name](estimator, X, y)

    def __repr__(self) -> str:
        return f"common_scorer({self.name!r})"


def common_scorer(name: str) -> Callable[[Any, Any, Any], float]:
    """Return a scorer whose definition is shared with the R package.

    Parameters
    ----------
    name : {"mse", "rmse", "mae", "r2", "rsq", "log_loss", "accuracy", "roc_auc"}
        Common metric name; see :data:`COMMON_METRICS`.

    Returns
    -------
    scorer : callable
        ``scorer(estimator, X, y) -> float``, usable as the ``scoring``
        argument of :func:`sharedimp.shared_importance` and of scikit-learn
        tools. Greater is better: ``mse``, ``rmse``, ``mae`` and ``log_loss``
        are returned negated.

    Notes
    -----
    Relation to scikit-learn scorer names (on valid inputs):

    =============  ==================================  ===========================
    common name    scikit-learn                        R ``yardstick``
    =============  ==================================  ===========================
    ``mse``        ``neg_mean_squared_error``          ``-mean((truth - est)^2)``
    ``rmse``       ``neg_root_mean_squared_error``     ``-rmse``
    ``mae``        ``neg_mean_absolute_error``         ``-mae``
    ``r2``         ``r2``                              ``rsq_trad`` (not ``rsq``)
    ``rsq``        (none)                              ``rsq``
    ``log_loss``   ``neg_log_loss``                    ``-mn_log_loss``
    ``accuracy``   ``accuracy`` if ``predict`` is the  ``accuracy`` on argmax labels
                   argmax of ``predict_proba``
    ``roc_auc``    ``roc_auc`` (binary) /              ``roc_auc`` (binary /
                   ``roc_auc_ovo`` (multiclass)        ``hand_till``)
    =============  ==================================  ===========================

    The classification scorers use ``predict_proba`` and align its columns with
    the sorted union of the classes seen in training and in ``y``; a class
    missing from the training rows gets probability 0.

    Examples
    --------
    >>> from sharedimp import common_scorer
    >>> common_scorer("rmse")
    common_scorer('rmse')
    """
    if not isinstance(name, str) or name not in _SCORERS:
        raise ValueError(
            f"Unknown common metric {name!r}. Available: {', '.join(sorted(_SCORERS))}."
        )
    return _CommonScorer(name)
