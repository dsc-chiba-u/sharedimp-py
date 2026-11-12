# Relation to UMFI and its official implementations

## The method

Ultra-marginal feature importance (UMFI) was introduced by

> Joseph Janssen, Vincent Guan, Elina Robeva.
> *Ultra-marginal Feature Importance: Learning from Data with Causal Guarantees.*
> Proceedings of the 26th International Conference on Artificial Intelligence
> and Statistics (AISTATS), PMLR 206:10782–10814, 2023.
> <https://proceedings.mlr.press/v206/janssen23a.html> · arXiv:2204.09938

For a feature `x`, a feature set `F`, a response `y` and an evaluation function
`ν` (predictive power of a model trained on a feature set), UMFI is

```
U(x) = ν(S_x ∪ {x}) − ν(S_x)
```

where `S_x` is a preprocessing of `F \ {x}` from which the dependence on `x`
has been removed. The paper proposes two dependence-removal procedures taken
from the AI-fairness literature: pairwise optimal transport (UMFI-OT) and
linear regression (UMFI-LR), in which every other feature is replaced by the
residual of its regression on `x`.

`sharedimp` implements **UMFI-LR only**, and claims no new feature-importance
method. Its Full model is `S_x ∪ {x}` and its Reduced model is `S_x`.

## Official implementations that were examined

| Repository | Language | Commit examined | License |
| --- | --- | --- | --- |
| [`HydroML/UMFI`](https://github.com/HydroML/UMFI) (package + `UMFI_aistats_an.R`) | R | `54b9647` (2024-06-24) | MIT |
| [`guanton/umfi_python`](https://github.com/guanton/umfi_python) (linked from the R repository's README) | Python | `4db1c23` (2024-06-13) | no license file |
| [`joej1997/UMFI`](https://github.com/joej1997/UMFI) (original experiment scripts) | R | `16ef563` (2022-07-16) | no license file |

What they do (UMFI-LR path):

* `preprocess_lr(dat, protect)`: for every other column, simple linear
  regression on the protected column **using all rows**; the column is replaced
  by its residual **only if** the regression passes a gate — slope p-value
  `< 0.01` in R, `R² > 0.01` in Python. Near-constant columns get Gaussian noise
  added before the regression, and a zero-variance result is replaced by
  Gaussian noise.
* `umfi(X, y)`: for every feature, fit two random forests (R: `ranger`, 100
  trees; Python: `ExtraTrees`/`RandomForest`, 100 trees) on the preprocessed
  data with and without the feature and take the difference of their
  **out-of-bag** accuracies (`R²`, or classification accuracy). Each accuracy
  is truncated from below (at 0, or at 0.5 for classification in R), and
  negative importances are set to 0.

## Differences introduced by `sharedimp`

All differences are software-engineering choices about *how the UMFI-LR
quantity is estimated and exposed*; none changes the definition
`U(x) = ν(S_x ∪ {x}) − ν(S_x)`.

| Aspect | Official implementations | `sharedimp` |
| --- | --- | --- |
| Where residualization is fitted | all rows, once | training rows of each split; coefficients re-used on assessment rows |
| Evaluation of `ν` | random-forest out-of-bag score, in-sample preprocessing | any scikit-learn scorer on held-out assessment rows |
| Learner | random forest / extra trees, hard-coded | any scikit-learn estimator or pipeline, cloned, fixed hyperparameters |
| Resampling | none (OOB), optional repetitions | any scikit-learn CV splitter (grouped, stratified, repeated, …) |
| Significance / `R²` gate | yes (and different in R and Python) | none: always residualize |
| Random-noise injection for degenerate columns | yes | none; deterministic rules (`docs/specification.md` §3.1) |
| Truncation of scores and of negative importances | yes | none |
| Full/Reduced pairing | independent unseeded forests | same rows, same hyperparameters, same seed |
| Output | vector / data frame of importances | summary table + per-split table |
| Reproducibility | global RNG | `random_state`, independent of `n_jobs` |

Consequences worth knowing:

* Removing the gate means that weakly correlated features are residualized too.
  With the gate, a feature whose correlation with the anchor is below the
  threshold is left untouched; without it, a (small) linear component is
  removed. The gate depends on `n` in R (p-value) but not in Python (`R²`).
  Whether to offer the gate as an option is an open item for the common
  specification (§9, item 1).
* Because nothing is truncated, importances of irrelevant features scatter
  around zero and can be negative; this is intended.
* Residualization removes *linear* dependence on the anchor only (this is the
  definition of UMFI-LR, not a property of this package). Non-linear dependence
  between features remains in the Reduced model.

## Parity with the official implementation

Parity tests live in `tests/test_parity_official.py` (pytest marker `parity`).
The official Python sources are not vendored because the upstream repository
has no license; the two files needed (`umfi/preprocess_LR.py`, `umfi/UMFI.py`)
are downloaded at test time from the pinned commit
`4db1c23f2355d69fc1471ae8df89c196d7a9a347`, verified by SHA-256, and the tests
are skipped (not faked) if they cannot be obtained. A local checkout can be
used with `SHAREDIMP_UMFI_PYTHON_DIR=/path/to/umfi_python`.

`UMFI.py` imports `minepy`, `tqdm` and the optimal-transport module at import
time although the UMFI-LR path with `select_optimal_preprocessing=False` uses
none of them. The test supplies inert stand-ins for these three imports; the
UMFI-LR code path itself runs unmodified.

Results obtained on 2026-11-12 (Python 3.11, scikit-learn 1.9.1, NumPy 2.4.6;
synthetic data, `n = 400`, 4 correlated features, seed fixed in the test):

1. **Residualization** — `AnchorResidualizer(anchor=c).fit_transform(X)` equals
   the official `preprocess_lr(X, c)` for every anchor within `1e-12` (absolute),
   on data where every feature pair passes the official `R² > 0.01` gate.
2. **Documented difference** — for a feature with `R² ≤ 0.01`, the official
   function leaves the column unchanged while `sharedimp` residualizes it; the
   test asserts both behaviours.
3. **Official UMFI-LR score reproduced** — applying the official scoring rule
   (random forest, 100 trees, OOB `R²`, truncation) to `sharedimp`'s Full and
   Reduced representations, with the same global seed and the same sequence of
   fits, gives the official `UMFI(...)` output with maximum absolute difference
   `0.0` (official: `[0.94755, 0.69779, 0.67584, 0.10420]`).
4. **Cross-validated estimate** — `shared_importance(RandomForestRegressor(100),
   scoring="r2", cv=5)` gives `[0.931, 0.670, 0.639, 0.106]` against the
   official 5-iteration average `[0.948, 0.695, 0.685, 0.103]` (maximum absolute
   difference `0.046`). These are two different estimators of the same quantity
   (out-of-sample CV vs. out-of-bag with in-sample residualization), so exact
   agreement is not expected; the test only asserts agreement within `0.1` and
   the same top-2 ranking.

### What was *not* verified

* **The R package `HydroML/UMFI` was not executed.** R is not installed in the
  development environment used for this work. Its source was read and its
  UMFI-LR logic is described above, but no numerical comparison against the R
  implementation (including its p-value gate and `ranger` forests) has been
  made. This should be done from `sharedimp-r`.
* The classification path of the official implementations (OOB accuracy) was
  not compared numerically.
* The paper's experiments were not reproduced here; benchmarks belong in
  `sharedimp-experiments`.
