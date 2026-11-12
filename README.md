# sharedimp

[![CI](https://github.com/dsc-chiba-u/sharedimp-py/actions/workflows/ci.yml/badge.svg)](https://github.com/dsc-chiba-u/sharedimp-py/actions/workflows/ci.yml)

Cross-validated **UMFI-LR** feature importance for scikit-learn workflows.

`sharedimp` implements ultra-marginal feature importance with linear-regression
dependence removal (UMFI-LR), introduced by
[Janssen, Guan & Robeva (AISTATS 2023)](https://proceedings.mlr.press/v206/janssen23a.html),
as a leakage-free, out-of-sample procedure that works with any scikit-learn
compatible estimator, scorer and CV splitter.

> This package does **not** propose a new feature-importance method. It is a
> software implementation of an existing one, designed to be easy to use
> correctly. See [docs/relation_to_umfi.md](docs/relation_to_umfi.md).

## What it computes

For each *anchor* feature `x_i` and each train/assessment split:

1. On the **training** rows, every other feature `x_j` is regressed on `x_i`
   (simple OLS) and replaced by its residual. The **same coefficients** are
   applied to the assessment rows.
2. Two clones of your estimator are fitted on the training rows:
   **Full** (`x_i` + residualized others) and **Reduced** (residualized others).
3. Both are scored on the assessment rows; `importance = score_full − score_reduced`.

The reported importance is the mean over splits. Because the other features no
longer carry the linear information they share with `x_i`, the anchor is
credited with everything it shares with them — correlated or duplicated
features are not "split" or masked the way they are with permutation or
drop-column importance.

Importance is positive when keeping the anchor improves out-of-sample
performance. It can be negative, and negative values are reported as they are.

## Installation

```bash
pip install sharedimp          # once released on PyPI
pip install git+https://github.com/dsc-chiba-u/sharedimp-py.git
```

Requires Python ≥ 3.10, scikit-learn ≥ 1.3, NumPy, SciPy, pandas and joblib.

## Quick start

```python
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sharedimp import shared_importance

X, y = load_diabetes(return_X_y=True, as_frame=True)

result = shared_importance(
    estimator=RandomForestRegressor(n_estimators=500, random_state=42),
    X=X,
    y=y,
    scoring="neg_root_mean_squared_error",
    cv=5,
    n_jobs=-1,
    random_state=42,
)

result.importances  # one row per feature
result.fold_results  # one row per (split, feature)
result.to_frame(sort=True)  # sorted by decreasing importance
result.fold_matrix()  # features x splits table of per-split importances
```

`result.importances` has the columns `feature`, `importance`, `importance_std`,
`score_full`, `score_reduced` and `n_splits`.

### Classification and probability-based scoring

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold

result = shared_importance(
    LogisticRegression(max_iter=1000),
    X,
    y_class,
    scoring="neg_log_loss",  # or "roc_auc", "roc_auc_ovr", "neg_brier_score", ...
    cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0),
)
```

Binary and multiclass targets are supported. Scorers that need probabilities or
decision values obtain them through scikit-learn's scorer machinery.

### Grouped data, pipelines, feature subsets

```python
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

result = shared_importance(
    make_pipeline(StandardScaler(), SVR()),
    X,
    y,
    scoring="r2",
    cv=GroupKFold(n_splits=5),
    groups=subject_id,
    features=["bmi", "bp"],  # evaluate only these anchors
)
```

### The residualizer on its own

```python
from sharedimp import AnchorResidualizer

r = AnchorResidualizer(anchor="bmi").fit(X_train)  # coefficients from training rows
X_test_full = r.transform(X_test)  # applied to held-out rows
r.coef_, r.intercept_
```

## Things to know

* **Scoring sign.** scikit-learn scorers are "greater is better" (losses are
  exposed as `neg_*`). `sharedimp` uses them as they are, so
  `importance = score_full − score_reduced` always reads "positive = the anchor
  helps".
* **Fixed hyperparameters.** The estimator is cloned and fitted as given; no
  tuning is performed.
* **`cv=<int>`** means *shuffled* K-fold (stratified for classifiers), seeded
  by `random_state`. Pass a splitter object for full control.
* **`random_state`** also seeds estimators whose own `random_state` is `None`
  (Full and Reduced get the same seed). Seeds you set yourself are kept.
  Results do not depend on `n_jobs`.
* **Numeric features only.** Encode categorical variables first. One-hot
  columns are treated as separate anchors.
* **Linear dependence only.** Residualization removes the linear dependence of
  the other features on the anchor; this is the definition of UMFI-LR.
* **Cost.** `2 × n_features × n_splits` model fits.
* **`importance_std`** is the standard deviation of the per-split importances.
  It is descriptive; it is not a standard error and the package provides no
  confidence intervals or p-values.
* **`direction=True`** adds an exploratory diagnostic (Spearman correlation
  between the anchor and the Full model's out-of-sample prediction). It is
  **not** a causal effect.

## Documentation

* [docs/specification.md](docs/specification.md) — algorithm, conventions and
  output format (provisional common specification for the Python and R packages).
* [docs/relation_to_umfi.md](docs/relation_to_umfi.md) — relation to the UMFI
  paper and its official implementations, differences, and parity results.
* [examples/](examples) — runnable scripts.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy
```

## Citation

If you use this package, please cite the method:

```bibtex
@inproceedings{janssen2023ultra,
  title     = {Ultra-marginal Feature Importance: Learning from Data with Causal Guarantees},
  author    = {Janssen, Joseph and Guan, Vincent and Robeva, Elina},
  booktitle = {Proceedings of The 26th International Conference on Artificial Intelligence and Statistics},
  series    = {Proceedings of Machine Learning Research},
  volume    = {206},
  pages     = {10782--10814},
  year      = {2023},
  publisher = {PMLR}
}
```

## License

MIT — see [LICENSE](LICENSE).
