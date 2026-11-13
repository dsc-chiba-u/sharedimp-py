"""Smoke test of the sharedimp container image.

Run at image build time (see ``Dockerfile``) and again by the publishing
workflow on the built image::

    docker run --rm ghcr.io/dsc-chiba-u/sharedimp-py python /opt/sharedimp/smoke_test.py

It imports the package and runs a small regression example with scikit-learn
whose answer is known, so a broken installation fails loudly.
"""

from __future__ import annotations

import sys

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import sharedimp
from sharedimp import common_scorer, shared_importance


def main() -> int:
    """Run the checks; return the process exit status."""
    print(
        f"sharedimp {sharedimp.__version__} | Python {sys.version.split()[0]} | "
        f"scikit-learn {sklearn.__version__} | NumPy {np.__version__}"
    )

    # y depends on x0 and x2; x1 is correlated with x0; x3 is pure noise.
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 4))
    X[:, 1] = 0.8 * X[:, 0] + 0.6 * rng.normal(size=n)
    y = 1.5 * X[:, 0] - 1.0 * X[:, 2] + 0.5 * rng.normal(size=n)

    linear = shared_importance(
        LinearRegression(), X, y, scoring=common_scorer("mse"), cv=5, random_state=0
    )
    print(linear)
    imp = linear.importances_mean
    assert list(linear.importances.columns) == [
        "feature",
        "importance",
        "importance_std",
        "score_full",
        "score_reduced",
        "n_splits",
    ]
    assert linear.fold_results.shape == (5 * 4, 7)
    assert np.all(np.isfinite(imp))
    assert imp[0] > 0.5 and imp[2] > 0.5, "informative features must be important"
    assert abs(imp[3]) < 0.1, "the noise feature must be unimportant"

    forest = shared_importance(
        RandomForestRegressor(n_estimators=50),
        X,
        y,
        scoring="neg_root_mean_squared_error",
        cv=3,
        n_jobs=2,
        random_state=0,
    )
    print(forest)
    assert np.all(np.isfinite(forest.importances_mean))
    assert forest.importances_mean[0] > forest.importances_mean[3]

    print("sharedimp smoke test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
