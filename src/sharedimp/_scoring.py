"""Scoring: resolve a scikit-learn scorer and evaluate fitted models with it."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sklearn.metrics import check_scoring

__all__ = ["resolve_scorer", "score_model"]

Scorer = Callable[[Any, Any, Any], float]


def resolve_scorer(estimator: Any, scoring: Any) -> tuple[Scorer, str]:
    """Resolve ``scoring`` into a single scikit-learn scorer and a display name.

    ``scoring`` may be ``None`` (the estimator's own ``score`` method), a scorer
    name such as ``"neg_root_mean_squared_error"`` or ``"neg_log_loss"``, or a
    scorer callable with signature ``scorer(estimator, X, y)``.

    Scikit-learn scorers follow the convention *greater is better* (losses are
    exposed negated, e.g. ``neg_log_loss``). The returned scorer is used as is;
    no sign is ever flipped by this package.
    """
    if scoring is not None and not isinstance(scoring, str) and not callable(scoring):
        raise ValueError(
            "`scoring` must be None, a scorer name, or a single scorer callable; "
            "multi-metric scoring (list/tuple/dict) is not supported."
        )
    if scoring is None and not hasattr(estimator, "score"):
        raise TypeError(
            "`scoring=None` requires the estimator to implement `score`; "
            "pass an explicit `scoring` instead."
        )
    scorer = check_scoring(estimator, scoring=scoring)
    if isinstance(scoring, str):
        name = scoring
    elif scoring is None:
        name = "score"
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
