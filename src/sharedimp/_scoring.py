"""Scoring: resolve a scikit-learn scorer and evaluate fitted models with it."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sklearn.base import is_classifier
from sklearn.metrics import check_scoring

from ._metrics import common_scorer

__all__ = ["resolve_scorer", "score_model"]

Scorer = Callable[[Any, Any, Any], float]


def resolve_scorer(estimator: Any, scoring: Any) -> tuple[Scorer, str]:
    """Resolve ``scoring`` into a single scikit-learn scorer and a display name.

    ``scoring`` may be a scorer name such as ``"neg_root_mean_squared_error"``
    or ``"roc_auc"``, a scorer callable with signature
    ``scorer(estimator, X, y)``, or ``None`` for the default of the common
    R/Python specification:

    * classifiers: the multinomial log loss, reported as ``"neg_log_loss"``;
    * anything else: the mean squared error, ``"neg_mean_squared_error"``.

    The default log loss is :func:`sharedimp.common_scorer` ``("log_loss")``.
    It equals scikit-learn's ``"neg_log_loss"`` and additionally stays defined
    when a class is missing from a training or assessment set.

    Scikit-learn scorers follow the convention *greater is better* (losses are
    exposed negated, e.g. ``neg_log_loss``). The returned scorer is used as is;
    no sign is ever flipped by this package.
    """
    if scoring is not None and not isinstance(scoring, str) and not callable(scoring):
        raise ValueError(
            "`scoring` must be None, a scorer name, or a single scorer callable; "
            "multi-metric scoring (list/tuple/dict) is not supported."
        )
    if scoring is None:
        if is_classifier(estimator):
            if not hasattr(estimator, "predict_proba"):
                raise TypeError(
                    "The default metric for classifiers is the log loss, which needs "
                    "`predict_proba`, but the estimator does not provide it. Pass an "
                    "explicit `scoring` (e.g. 'accuracy' or 'roc_auc')."
                )
            return common_scorer("log_loss"), "neg_log_loss"
        scoring = "neg_mean_squared_error"
    scorer = check_scoring(estimator, scoring=scoring)
    if isinstance(scoring, str):
        name = scoring
    else:
        name = getattr(scoring, "__name__", None) or repr(scoring)
    return scorer, name


def score_model(scorer: Scorer, fitted: Any, X_test: Any, y_test: Any) -> float:
    """Evaluate one fitted model on assessment data; returns a Python float."""
    value = scorer(fitted, X_test, y_test)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"The scorer must return a single number, got {type(value).__name__}."
        ) from exc
