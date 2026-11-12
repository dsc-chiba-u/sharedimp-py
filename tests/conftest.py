from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin


@pytest.fixture
def tiny():
    """Tiny deterministic design with hand-computable regressions.

    anchor a = [0,1,2,3,4]           (mean 2, SS 10)
    b = [1,3,2,5,4]  -> slope 0.8, intercept 1.4 on a
    c = [2,0,1,0,2]  -> slope 0.0, intercept 1.0 on a
    """
    X = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 3.0, 0.0],
            [2.0, 2.0, 1.0],
            [3.0, 5.0, 0.0],
            [4.0, 4.0, 2.0],
        ]
    )
    return X


@pytest.fixture
def reg_data():
    """Correlated regression data: x1 ~ x0, x3 is pure noise."""
    rng = np.random.default_rng(12345)
    n = 240
    X = rng.normal(size=(n, 4))
    X[:, 1] = 0.8 * X[:, 0] + 0.6 * rng.normal(size=n)
    y = 1.5 * X[:, 0] - 1.0 * X[:, 2] + 0.5 * rng.normal(size=n)
    return X, y


@pytest.fixture
def clf_data():
    rng = np.random.default_rng(2024)
    n = 300
    X = rng.normal(size=(n, 4))
    X[:, 1] = 0.7 * X[:, 0] + 0.7 * rng.normal(size=n)
    logit = 2.0 * X[:, 0] - 1.5 * X[:, 2]
    y_bin = (logit + rng.logistic(size=n) > 0).astype(int)
    score = X[:, 0] + 0.8 * X[:, 2] + 0.5 * rng.normal(size=n)
    y_multi = np.digitize(score, np.quantile(score, [1 / 3, 2 / 3]))
    return X, y_bin, y_multi


class RecordingRegressor(RegressorMixin, BaseEstimator):
    """Mean predictor that records every design matrix it is given.

    Records are appended to a class-level list so that they survive ``clone``.
    Only usable with ``n_jobs=1``.
    """

    log: list = []  # noqa: RUF012

    def __init__(self, tag: str = "rec"):
        self.tag = tag

    def fit(self, X, y, **fit_params):
        self.mean_ = float(np.mean(y))
        type(self).log.append(("fit", X, np.asarray(y), dict(fit_params)))
        return self

    def predict(self, X):
        type(self).log.append(("predict", X))
        return np.full(len(X), self.mean_)


@pytest.fixture
def recorder():
    RecordingRegressor.log = []
    yield RecordingRegressor
    RecordingRegressor.log = []
