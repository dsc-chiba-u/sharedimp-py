"""Binary and multiclass classification with probability-based scoring."""

from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sharedimp import shared_importance

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

# Binary: log loss on predicted probabilities, plus the exploratory direction diagnostic.
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X = X.iloc[:, :8]
binary = shared_importance(
    model,
    X,
    y,
    scoring="neg_log_loss",
    cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0),
    direction=True,
    n_jobs=-1,
)
print(binary.to_frame(sort=True).round(4).to_string(index=False))

# Multiclass: one-vs-rest AUC.
X, y = load_wine(return_X_y=True, as_frame=True)
multi = shared_importance(model, X, y, scoring="roc_auc_ovr", cv=5, random_state=0, n_jobs=-1)
print(multi.to_frame(sort=True).round(4).to_string(index=False))
