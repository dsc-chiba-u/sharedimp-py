"""scikit-learn API conformance of the public transformer."""

from __future__ import annotations

from sklearn.utils.estimator_checks import parametrize_with_checks

from sharedimp import AnchorResidualizer


@parametrize_with_checks([AnchorResidualizer()])
def test_sklearn_estimator_checks(estimator, check):
    check(estimator)
