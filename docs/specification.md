# sharedimp — specification (Python side)

**The specification of `sharedimp` is a single document shared by the R and the
Python package:**

> `sharedimp-r/docs/specification.md` —
> <https://github.com/dsc-chiba-u/sharedimp-r/blob/main/docs/specification.md>
> ("common algorithm specification (R / Python)", v0.1, reconciled 2026-11-13)

That document is the single source of truth. This file does not restate it. It
records, for the Python package:

1. which revision of the common specification the package was checked against
   (§1) and a short orientation (§2);
2. how each section is realised in Python and where it is tested (§3);
3. Python-specific behaviour and every intentional deviation (§4) — the common
   specification requires deviations to be recorded in the deviating
   repository;
4. the status of the changes the common specification asked of this package
   (§5) and what is still open between the two repositories (§6).

Section numbers "spec §n" refer to the common specification.

## 1. Revision checked

`sharedimp-r` commit `eaafaee` (2026-11-13), in which the default metric (spec
§9.1) and the default handling of learner failures (spec §10) are normative
for both languages and spec §17 describes the Docker images. The file was only
read; nothing in that repository was modified. Earlier Python-side texts
(`0.1-provisional` of 2026-11-12, `0.2-draft` of 2026-11-13) are superseded and
remain only in the git history of this file.

## 2. Orientation (non-normative summary)

For every anchor feature and every train/assessment split: every other feature
is regressed on the anchor by OLS on the training rows and replaced by its
residual, with the same coefficients applied to the assessment rows; a Full
model (raw anchor in its original column position + residualized others) and a
Reduced model (the same without the anchor column) are fitted with identical
settings and scored on the assessment rows; `importance = score_full −
score_reduced`, where scores are oriented greater-is-better. The reported
importance is the unweighted mean over the splits in which it is defined.
Negative values are kept. There is no inference and no direction diagnostic.

## 3. Conformance

| Spec § | Topic | Python realisation | Tests |
| --- | --- | --- | --- |
| 3 | Inputs; `p ≥ 1` | `_core._validate_X`; numeric, finite, unique names; bool/int → float | `test_api.py::test_error_handling`, `test_common_spec.py` |
| 4 | Resampling | `_resampling.py`; `cv=<int>` → shuffled `KFold`/`StratifiedKFold` seeded by `random_state`; any scikit-learn splitter or iterable of `(train, test)` | `test_api.py::test_cv_splitters` |
| 5 | Fold-internal OLS residualization | `AnchorResidualizer` (`fit` on training rows, `transform` on any rows) | `test_residualizer.py`, `test_core_exact.py` |
| 5.1 | D1 constant anchor, D2 constant feature (`max = min`), D3 exact collinearity (`SS_res ≤ 1e-10·S_jj` → residual exactly 0, column kept) | `_residualizer._fit_coefficients`; `collinear_`, `constant_anchor_`; one warning naming constant anchors | `test_common_spec.py` (rules, unit invariance, tolerance boundary), `test_residualizer.py` |
| 6 | Full / Reduced, anchor in its original position | `_core._evaluate_anchor` | `test_common_spec.py::test_full_keeps_anchor_in_original_position` |
| 8.3 | Null model for `p = 1` | `_fitting.fit_null_model` (`DummyRegressor("mean")` / `DummyClassifier("prior")`) | `test_common_spec.py` (single-feature tests) |
| 8.4 | User learners | any scikit-learn estimator, cloned per fit, fixed hyperparameters | `test_api.py` |
| 9 | Scores and sign | scikit-learn scorers used as they are; `importance = score_full − score_reduced`; nothing truncated | `test_core_exact.py::test_scorer_sign_is_not_double_flipped` |
| 9.1 | Common metrics | `common_scorer(name)`: `mse, rmse, mae, r2, rsq, log_loss, accuracy, roc_auc` | `test_cross_language.py` (vs. `yardstick`) |
| 9.1 | **Default metric** | `scoring=None` → `neg_mean_squared_error` (regressors) / log loss, reported as `neg_log_loss` (classifiers) | `test_api.py::test_default_scoring_*` |
| 9.2 | Aggregation over successful splits; `n_splits` | `_result.aggregate_fold_results` | `test_failures.py`, `test_common_spec.py::test_output_schema` |
| 10 | **Learner failures** | `on_error="warn"` (default) / `"error"`; `error` column | `test_failures.py` |
| 11 | Output schema | `SharedImportanceResult.importances`, `.fold_results` | `test_common_spec.py::test_output_schema` |
| 12 | Differences from reference UMFI-LR | see `relation_to_umfi.md` | `test_parity_official.py` |
| 13 | No direction diagnostic | no `direction` argument or column; `_direction.py` is internal and unused | `test_common_spec.py::test_direction_is_not_part_of_the_public_api` |
| 14 | Reproducibility | seeds fixed per (split, column) before dispatch; independent of `n_jobs` and of the `features` subset | `test_api.py`, `test_common_spec.py::test_parallel_and_serial_are_identical` |
| 15.1 | Parity fixture, tolerance `1e-8` | reproduced, including the collinear variant and the default metric | `test_cross_language.py::test_r_specification_parity_fixture*` |
| 17 | Docker images: `latest` + short SHA on `main`, `X.Y.Z` + `X.Y` + short SHA on `vX.Y.Z`, nothing on pull requests, one build for GHCR and Docker Hub | `Dockerfile`, `.github/workflows/build_test_push.yml` | the workflow itself (build-time smoke test) |

## 4. Python-specific behaviour and recorded deviations

### 4.1 API mapping

| R | Python |
| --- | --- |
| `shared_importance(x, y, learner, metric, task, v, repeats, resamples, features, seed, on_error)` | `shared_importance(estimator, X, y, *, scoring=None, cv=5, groups=None, features=None, fit_params=None, on_error="warn", n_jobs=None, random_state=None, verbose=0)` |
| `learner` (default `"rand_forest"`) | `estimator` — required; there is no default learner |
| `metric` | `scoring` — scikit-learn scorer name, scorer callable, or `common_scorer(name)` |
| `task` | taken from the estimator (`sklearn.base.is_classifier`) |
| `v`, `repeats`, `resamples` | `cv` (+ `groups`); fold labels as `PredefinedSplit(labels − 1)` |
| `seed` | `random_state` |
| `on_error = "warn"` / `"error"` | `on_error="warn"` / `"error"` (same names, same default) |
| `residualize_fit()` / `residualize_apply()` | `AnchorResidualizer(anchor, *, fit_intercept=True, drop_anchor=False, tol=1e-10)` |
| `$importance`, `fold_results()` | `.importances`, `.fold_results` (+ `.fold_matrix()`, `.n_failed`) |

### 4.2 Default metric (spec §9.1)

`scoring=None` selects the common default: mean squared error for regressors,
multinomial log loss for classifiers, both as greater-is-better scores, so a
call without `scoring` computes the same quantity as an R call without
`metric`. Details:

* The classifier default is `common_scorer("log_loss")`, reported under the
  name `neg_log_loss`. It gives the same value as scikit-learn's
  `"neg_log_loss"` scorer and stays defined when a class is missing from a
  training or assessment set (such a class has probability 0, clipped at
  `2⁻⁵²`), as the R implementation does.
* A classifier without `predict_proba` has no default metric: a `TypeError`
  asks for an explicit `scoring`.
* The estimator's own `score` method (R² / accuracy) is no longer used; request
  it with `scoring="r2"` / `scoring="accuracy"`.
* The sign never depends on the metric: `importance = score_full −
  score_reduced` with greater-is-better scores, so for both defaults a positive
  importance means a lower loss with the anchor.

### 4.3 Learner failures (spec §10)

A *failure* is an exception raised while fitting, predicting or scoring the
Full or the Reduced model of one (split, feature) pair, or a classifier's
training set that contains a single class (raised by the package itself, also
for learners that would accept it).

* `on_error="warn"` (default): both scores and the importance of that pair are
  `NaN`, the message is stored in the `error` column as
  `"<ExceptionType>: <message>"`, the remaining pairs are evaluated, and one
  `UserWarning` reports the number of failures and the first message. Each
  feature is aggregated over its successful splits; `n_splits` is their number;
  with none, the summary values are `NaN` and `n_splits = 0`.
* `on_error="error"`: the original exception is re-raised immediately.
  Cross-language comparisons should use this mode, so that a failure cannot be
  mistaken for a value.
* Not failures: errors in the inputs or in the residualization (raised before
  any learner runs), and a metric that is undefined on a split without raising
  (e.g. `rsq` for a constant prediction): its `NaN` is excluded from the
  aggregation and from `n_splits`, with `error` empty — as in R.

**Deviation.** The specification also counts *non-finite predictions* as a
failure. The Python package does not inspect predictions (scorers receive the
fitted model, not its predictions), so non-finite predictions surface either
as an exception from the scorer (the usual case; recorded as a failure) or as
a non-finite score (excluded from the aggregation, but with an empty `error`).

### 4.4 Other language-specific points

| Topic | Python | R |
| --- | --- | --- |
| `split` label (spec §11.2) | 0-based | 1-based |
| Extra per-split columns | `error` | `id`, `id2`, `error` |
| Names for unnamed input | `x0, x1, …` | `V1, V2, …` |
| Class level order | sorted labels (`estimator.classes_`) | factor levels |
| Overlapping training/assessment rows | accepted (used for in-sample closed-form tests) | rejected |
| Fit-time weights | `fit_params={"sample_weight": …}` (also applied to the null model); metrics are unweighted | not available |
| `accuracy` | scikit-learn's `"accuracy"` scorer uses `predict`; the spec's "argmax of the probabilities" is `common_scorer("accuracy")`. They coincide when `predict` is that argmax (not for `SVC(probability=True)`) | argmax labels |
| Collinear designs with OLS | `LinearRegression` does not reliably detect rank deficiency on older stacks (scikit-learn 1.3.2 / NumPy 1.23 gave 41.79 instead of 45.55 for `x3` in the collinear variant of spec §15.1); the parity tests use an explicit minimum-norm least-squares fit for collinear designs | `lm` aliases the column |
| Manual Docker run | publishes only from `main` or a release tag (input `publish`, default true) | publishes from the selected ref (input `publish`, default true) |

For a cross-language comparison, pass explicit splits and a deterministic
learner, and use `on_error="error"`. The default metric now agrees; the learner
(none by default in Python) and randomly generated folds still differ.

## 5. Changes requested of the Python package (spec §16)

| # | Item | Status |
| --- | --- | --- |
| 1 | Remove the direction diagnostic from the API | done |
| 2 | Accept `p = 1` with the null model | done |
| 3 | Default scoring: MSE / log loss | **done** (§4.2) |
| 4 | Learner failures: `NaN` + warning by default, aggregate over the rest, option to raise | **done** (§4.3) |
| 5 | Replace the Python specification by a pointer plus Python notes | **done** (this file) |
| 6 | Document the scorer ↔ common-name mapping; scorer for `rsq` | done (`common_scorer`, README) |
| 7 | Parity fixture of spec §15.1 at `1e-8` | done |
| 7b | Regenerate `tests/fixtures/cross_language/r_reference.json` from the current R commit and remove the compensating conversions | **deferred** — see §6 |

## 6. Open between the two repositories

* **R reference values.** `r_reference.json` was generated by running R at
  commit `6c5631d`, which predates the reconciled specification (anchor first,
  dropped columns, raw metric values, feature-major rows).
  `tests/test_cross_language.py` compensates (comparison by column name, sign
  of raw metrics, row reordering) and restricts itself to quantities those
  layout differences do not change. Regeneration is deliberately postponed
  until the R-side revision of the specification is declared complete (its
  text was still being revised on 2026-11-13, and section B of
  `r_side_changes.md` proposes further corrections); the fixture of spec
  §15.1, whose numbers come from the reconciled R package, is already
  reproduced without any conversion, also with the default metric.
* **Text of the common specification.** It still describes the Python package
  as it was before these changes, and contains a few statements that need
  correcting. They are listed in [`r_side_changes.md`](r_side_changes.md).
