# sharedimp

[![DOI](https://zenodo.org/badge/1455260199.svg)](https://zenodo.org/doi/10.5281/zenodo.24220839)
[![PyPI version](https://badge.fury.io/py/sharedimp.svg)](https://badge.fury.io/py/sharedimp)
[![CI](https://github.com/dsc-chiba-u/sharedimp-py/actions/workflows/ci.yml/badge.svg)](https://github.com/dsc-chiba-u/sharedimp-py/actions/workflows/ci.yml)
[![DockerHub/GHCR](https://github.com/dsc-chiba-u/sharedimp-py/actions/workflows/build_test_push.yml/badge.svg)](https://github.com/dsc-chiba-u/sharedimp-py/actions/workflows/build_test_push.yml)

Cross-validated **UMFI-LR** feature importance for scikit-learn workflows.

`sharedimp` implements ultra-marginal feature importance with linear-regression
dependence removal (UMFI-LR), introduced by
[Janssen, Guan & Robeva (AISTATS 2023)](https://proceedings.mlr.press/v206/janssen23a.html),
as a leakage-free, out-of-sample procedure that works with any scikit-learn
compatible estimator, scorer and CV splitter.

> This package does **not** propose a new feature-importance method. It is a
> software implementation of an existing one, designed to be easy to use
> correctly. See [docs/relation_to_umfi.md](https://github.com/dsc-chiba-u/sharedimp-py/blob/main/docs/relation_to_umfi.md).

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
pip install sharedimp
```

The development version can be installed from GitHub:

```bash
pip install git+https://github.com/dsc-chiba-u/sharedimp-py.git
```

Requires Python ≥ 3.10, scikit-learn ≥ 1.3, NumPy, SciPy, pandas and joblib.

### Docker

A container image with `sharedimp` and its dependencies preinstalled is
published to two registries (the same image under two names):

| Registry | Image |
| --- | --- |
| GitHub Container Registry | `ghcr.io/dsc-chiba-u/sharedimp-py` |
| Docker Hub | `koki/sharedimp-py` |

Tags: `latest` (the most recent successful build of the `main` branch) and the
short commit SHA of every published build (e.g. `:1bb2ce8`), for
reproducibility. Release tags `vX.Y.Z` will additionally publish `X.Y.Z` and
`X.Y`.

```bash
docker pull ghcr.io/dsc-chiba-u/sharedimp-py:latest     # or: koki/sharedimp-py:latest

# built-in smoke test (a small scikit-learn regression example)
docker run --rm ghcr.io/dsc-chiba-u/sharedimp-py python /opt/sharedimp/smoke_test.py

# interactive Python
docker run --rm -it ghcr.io/dsc-chiba-u/sharedimp-py

# run your own script on your own data (the working directory is /work)
docker run --rm -v "$PWD":/work ghcr.io/dsc-chiba-u/sharedimp-py python analysis.py
```

The image is based on the official `python:3.13-slim` image, runs as an
unprivileged user and contains only the installed package and its runtime
dependencies. To build it yourself: `docker build -t sharedimp-py .` (the build
runs the smoke test and fails if it fails).

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
result.fold_results  # one row per (feature, split)
result.to_frame(sort=True)  # sorted by decreasing importance
result.fold_matrix()  # features x splits table of per-split importances
```

`result.importances` has the columns `feature`, `importance`, `importance_std`,
`score_full`, `score_reduced` and `n_splits`; `result.fold_results` has
`feature`, `split`, `n_train`, `n_test`, `score_full`, `score_reduced`,
`importance` and `error`.

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

* **Default metric.** Without `scoring`, regressors are scored by the mean
  squared error (`neg_mean_squared_error`) and classifiers by the multinomial
  log loss (`neg_log_loss`) — the same defaults as the R package. This is *not*
  the estimator's own `score` (R² / accuracy); ask for those with
  `scoring="r2"` or `scoring="accuracy"`.
* **Scoring sign.** scikit-learn scorers are "greater is better" (losses are
  exposed as `neg_*`). `sharedimp` uses them as they are, so
  `importance = score_full − score_reduced` always reads "positive = the anchor
  helps", for every metric.
* **Failed fits.** If the estimator fails on some (split, feature) pair —
  an exception, non-finite predictions, or a classifier's training set with a
  single class — that pair is recorded as
  `NaN` with its message in `result.fold_results["error"]`, a warning reports
  how many pairs failed, and each feature is averaged over its successful
  splits (`n_splits` tells you how many; `result.n_failed` counts failures).
  Pass `on_error="error"` to stop at the first failure instead.
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
* **Degenerate features.** A feature that is an exact affine function of the
  anchor (e.g. a duplicate) gets a residual of exactly zero; the column is kept.
* **Single feature.** With one column the Reduced model is the null model
  (training mean, or training class frequencies).
* **No direction diagnostic in v0.1.** Importances say how much a feature
  helps prediction, not in which direction it acts, and they are not causal
  effects.

## Same numbers as the R package

The R package `sharedimp` follows the same specification. To compare results
across languages, fix what differs between the ecosystems: pass explicit
splits and use a metric whose definition is pinned in both languages.

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import PredefinedSplit
from sharedimp import common_scorer, shared_importance

result = shared_importance(
    LinearRegression(),
    X,
    y,
    scoring=common_scorer("r2"),  # = yardstick::rsq_trad, not yardstick::rsq
    cv=PredefinedSplit(fold_labels - 1),  # the same fold labels as in R
    on_error="error",  # a failure must not be mistaken for a value
)
```

`common_scorer()` knows `mse`, `rmse`, `mae`, `r2`, `rsq`, `log_loss`,
`accuracy` and `roc_auc`. Names alone are not enough: scikit-learn's `"r2"` is
the coefficient of determination, `yardstick::rsq` is a squared correlation.

The agreement is a regression test (`tests/test_cross_language.py`): residualized
data, predictions, both result tables, the default metric, exact collinearity,
single-feature input and failed fits are compared with output produced by
running the R package, to `1e-8` for closed-form learners. Two limits are worth
knowing: iteratively fitted models agree only as far as their optimisers
converge (multinomial logistic regression: about `3e-5`), and on rank-deficient
designs (e.g. two features that are exact copies of each other) ordinary least
squares is not unique, so R's `lm` and `LinearRegression` can differ — see
[docs/rank_deficiency.md](https://github.com/dsc-chiba-u/sharedimp-py/blob/main/docs/rank_deficiency.md). Random forests and other
stochastic learners are not numerically comparable across languages.

## Documentation

* The specification is shared with the R package and lives in its repository:
  [sharedimp-r/docs/specification.md](https://github.com/dsc-chiba-u/sharedimp-r/blob/main/docs/specification.md).
* [docs/specification.md](https://github.com/dsc-chiba-u/sharedimp-py/blob/main/docs/specification.md) — how this package realises
  each section of it, Python-specific behaviour and recorded deviations.
* [docs/rank_deficiency.md](https://github.com/dsc-chiba-u/sharedimp-py/blob/main/docs/rank_deficiency.md) — what least-squares
  solvers do on collinear designs, and why that is a learner matter.
* [docs/r_side_changes.md](https://github.com/dsc-chiba-u/sharedimp-py/blob/main/docs/r_side_changes.md) — record of the items raised
  with the R package during the reconciliation (all resolved).
* [docs/relation_to_umfi.md](https://github.com/dsc-chiba-u/sharedimp-py/blob/main/docs/relation_to_umfi.md) — relation to the UMFI
  paper and its official implementations, differences, and parity results.
* [examples/](https://github.com/dsc-chiba-u/sharedimp-py/tree/main/examples) — runnable scripts.

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

The software itself is archived on Zenodo. To refer to the implementation, use
[10.5281/zenodo.24220839](https://doi.org/10.5281/zenodo.24220839) (all
versions) or the DOI of the version you used
(0.1.0: [10.5281/zenodo.24220840](https://doi.org/10.5281/zenodo.24220840)).

## License

MIT — see [LICENSE](https://github.com/dsc-chiba-u/sharedimp-py/blob/main/LICENSE).
