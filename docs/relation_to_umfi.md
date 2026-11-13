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
| Column layout | all columns, anchor in place | the same (specification §3) |
| Truncation of scores and of negative importances | yes | none |
| Full/Reduced pairing | independent unseeded forests | same rows, same hyperparameters, same seed |
| Output | vector / data frame of importances | summary table + per-split table |
| Reproducibility | global RNG | `random_state`, independent of `n_jobs` |

Consequences worth knowing:

* Removing the gate means that weakly correlated features are residualized too.
  With the gate, a feature whose correlation with the anchor is below the
  threshold is left untouched; without it, a (small) linear component is
  removed. The gate depends on `n` in R (p-value) but not in Python (`R²`).
  Both sharedimp packages always residualize.
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

Because the two implementations differ by design, the tests are split by what
can be expected.

### Conditions under which exact agreement is expected

All three hold only when sharedimp's building blocks are used *the way the
official code uses them*: all rows at once, no cross-validation.

| Quantity | Condition | Test (`TestExactAgreementExpected`) | Result |
| --- | --- | --- | --- |
| Residualized features | same rows; every feature pair passes the official gate (`R² > 0.01`); no constant column | `test_residualization_when_the_gate_is_passed` | equal within `1e-12` |
| Layout of the representation | — | `test_representation_layout` | same columns, same order, raw anchor in its original position |
| Official UMFI-LR score | official scoring rule (random forest, 100 trees, OOB `R²` on all rows, floors at 0) applied to sharedimp's Full/Reduced representations, same global seed and order of fits; gate passed for all pairs | `test_upstream_score_from_sharedimp_representation` | maximum absolute difference `0.0` (official: `[0.94755, 0.69779, 0.67584, 0.10420]`) |

### Conditions under which agreement is not expected

Each is a deliberate difference and is pinned by a test that asserts that the
difference exists.

| Difference | What happens | Test (`TestAgreementNotExpected`) |
| --- | --- | --- |
| Gate | a feature with `R² ≤ 0.01` (Python) / slope p-value `≥ 0.01` (R) is left raw by the official code and residualized by sharedimp | `test_gate_upstream_skips_weakly_correlated_features` |
| Truncation | the official code floors scores and importances at 0; with a response independent of all features it returns exactly 0 for every feature, while sharedimp reports negative scores and non-zero, partly negative importances | `test_truncation_upstream_never_returns_negative_importance` |
| CV vs. OOB | `shared_importance()` residualizes within training folds and scores held-out rows; the official code residualizes on all rows and uses the forest's out-of-bag score | `test_cross_validation_vs_oob_agree_only_approximately` |
| Noise injection | the official code adds Gaussian noise to (near-)constant columns; sharedimp uses the deterministic rules of `specification.md` §3.1 | not compared (the official result is random) |
| Learner and metric | official: fixed forest, OOB `R²` or accuracy; sharedimp: any estimator and scorer | — |

For the third row the numbers obtained on the test data (`n = 400`, four
correlated features) were `[0.931, 0.670, 0.639, 0.106]` for
`shared_importance(RandomForestRegressor(100), scoring="r2", cv=5)` against the
official 5-iteration average `[0.948, 0.695, 0.685, 0.103]` (maximum absolute
difference `0.046`; Python 3.11, scikit-learn 1.9.1, 2026-11-12). The test
asserts only that the two are not identical, that they agree within `0.1`, and
that the top-2 ranking is the same. It is a sanity check, not a claim of
numerical equivalence.

### What was *not* verified

* **The official R package `HydroML/UMFI` was not executed from this
  repository.** Its source was read and its UMFI-LR logic is described above.
  A numerical comparison against it is recorded on the R side
  (`sharedimp-r/docs/umfi_comparison.md`).
* The classification path of the official implementations (OOB accuracy) was
  not compared numerically.
* The paper's experiments were not reproduced here; benchmarks belong in
  `sharedimp-experiments`.

## Parity with the R package `sharedimp-r`

Separate from the above: `tests/test_cross_language.py` compares this package
with `yardstick` and with the R package on shared deterministic inputs, using
reference numbers produced by running R. See `specification.md` §9 and
`r_side_changes.md`.
