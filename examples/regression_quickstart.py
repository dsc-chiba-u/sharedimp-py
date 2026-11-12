"""Regression with correlated and duplicated predictors."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from sharedimp import shared_importance

rng = np.random.default_rng(0)
n = 500
signal = rng.normal(size=n)
X = pd.DataFrame(
    {
        "signal": signal,
        "signal_copy": signal,  # exact duplicate
        "signal_noisy": signal + 0.5 * rng.normal(size=n),  # correlated proxy
        "independent": rng.normal(size=n),
        "noise": rng.normal(size=n),
    }
)
y = 2.0 * X["signal"] + 1.0 * X["independent"] + 0.5 * rng.normal(size=n)

result = shared_importance(
    estimator=RandomForestRegressor(n_estimators=200, random_state=42),
    X=X,
    y=y,
    scoring="neg_root_mean_squared_error",
    cv=5,
    n_jobs=-1,
    random_state=42,
)

print(result)
print("\nPer-split importances (features x splits):")
print(result.fold_matrix().round(3))
