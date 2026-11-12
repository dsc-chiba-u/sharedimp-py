"""Grouped cross-validation, a Pipeline estimator, and the stand-alone residualizer."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sharedimp import AnchorResidualizer, shared_importance

rng = np.random.default_rng(1)
n_subjects, n_per = 40, 10
subject = np.repeat(np.arange(n_subjects), n_per)
subject_effect = rng.normal(size=n_subjects)[subject]
X = pd.DataFrame(
    {
        "dose": rng.normal(size=subject.size) + 0.5 * subject_effect,
        "age": rng.normal(size=subject.size),
        "weight": rng.normal(size=subject.size) + 0.8 * subject_effect,
    }
)
y = 1.0 * X["dose"] + 0.5 * X["weight"] + subject_effect + 0.5 * rng.normal(size=subject.size)

result = shared_importance(
    make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    X,
    y,
    scoring="r2",
    cv=GroupKFold(n_splits=5),  # all rows of a subject stay in the same fold
    groups=subject,
    features=["dose", "weight"],
)
print(result)

# The residualizer is an ordinary scikit-learn transformer.
train, test = next(GroupKFold(n_splits=5).split(X, y, subject))
res = AnchorResidualizer(anchor="dose").fit(X.iloc[train])
print("\nslopes on 'dose':", pd.Series(res.coef_, index=X.columns).round(3).to_dict())
print(res.transform(X.iloc[test]).head().round(3))
