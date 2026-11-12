"""scikit-learn API conformance of the public transformer."""

from __future__ import annotations

import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from sharedimp import AnchorResidualizer

# scikit-learn 1.3's own test helper ``create_memmap_backed_data`` cannot handle
# an estimator when it forces ``aligned=True``, which it does whenever OpenBLAS
# reports a "Prescott" or unknown CPU architecture (hardware dependent, seen on
# some CI runners). The failure happens before any sharedimp code runs.
_UPSTREAM_HELPER_BUG = "When creating aligned memmap-backed arrays"


@parametrize_with_checks([AnchorResidualizer()])
def test_sklearn_estimator_checks(estimator, check):
    try:
        check(estimator)
    except ValueError as exc:
        if _UPSTREAM_HELPER_BUG in str(exc):
            pytest.skip(f"scikit-learn test-helper limitation on this CPU: {exc}")
        raise
