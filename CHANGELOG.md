# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-11-14

First release on PyPI. The entries below record the alignment with the R
package (`sharedimp-r`) on a common specification, relative to the initial
implementation listed at the end of this section.

### Added

- `common_scorer()` / `COMMON_METRICS`: scorers for `mse`, `rmse`, `mae`, `r2`,
  `rsq`, `log_loss`, `accuracy`, `roc_auc` with definitions pinned across R and
  Python (`r2` is `yardstick::rsq_trad`; `rsq` is `yardstick::rsq`).
- Single-feature input (`p = 1`): the Reduced model is the null model (training
  mean / training class frequencies) instead of an error.
- Cross-language parity tests against reference values generated with R
  (`yardstick` and `sharedimp-r`), and `docs/r_side_changes.md`.
- A warning when an anchor is constant on a training set.
- Docker image (`Dockerfile`, `docker/smoke_test.py`) and a workflow that
  publishes it to GHCR (`ghcr.io/dsc-chiba-u/sharedimp-py`) and Docker Hub
  (`koki/sharedimp-py`).
- `docs/specification.md` rewritten as the common R/Python specification
  (v0.2-draft), aligned with the R draft.
- `AnchorResidualizer.constant_anchor_`.
- A workflow that publishes the package to PyPI when a tag `vX.Y.Z` is pushed
  (TestPyPI on a manual run), using trusted publishing.

### Changed

- **Non-finite predictions are learner failures.** The assessment predictions
  of every fitted model are inspected (one extra prediction pass); `NaN`/`inf`
  predictions are recorded as a failure with a message in `error`, also when
  the scorer would have returned `NaN` silently.
- Cross-language tests now compare directly with output of the reconciled R
  package (`sharedimp-r` `92f9ecb`); the conversions for the old R layout are
  gone, and predictions, the default metric and failing splits are covered.
  `docs/rank_deficiency.md` records how least-squares solvers behave on
  rank-deficient designs.

- **Default metric.** `scoring=None` now means the mean squared error for
  regressors (`neg_mean_squared_error`) and the multinomial log loss for
  classifiers (`neg_log_loss`), as in the R package, instead of
  `estimator.score` (R² / accuracy). The importance remains
  `score_full - score_reduced`.
- **Learner failures.** A failed fit for a (split, feature) pair no longer
  aborts the run: it is recorded as `NaN` with its message in the new `error`
  column of `fold_results`, one warning is emitted, and each feature is
  aggregated over its successful splits (`n_splits` is their number;
  `SharedImportanceResult.n_failed`). `on_error="error"` restores the previous
  behaviour. A classifier's single-class training set counts as a failure.
- Splits on which the metric is undefined (`NaN`) are excluded from the
  summary instead of making it `NaN`.
- `docs/specification.md` is now the Python-side record (conformance,
  deviations) of the common specification kept in `sharedimp-r`.

- Constant columns are detected exactly (`max == min`) instead of through a
  rounded mean; the exact-collinearity rule (`SS_res <= 1e-10 * SS_tot`) is
  unchanged.
- Official-UMFI parity tests are grouped into conditions where exact agreement
  is expected and conditions where it is not.

### Removed

- The `direction` argument of `shared_importance()` and the `direction` result
  columns. A direction diagnostic is not part of the common API in v0.1; the
  previous helper remains as an internal, unexported, experimental module.

### Initial implementation (2026-11-12, not released separately)

- `shared_importance()`: cross-validated UMFI-LR with fold-internal
  residualization and paired Full/Reduced evaluation for any scikit-learn
  estimator, scorer and CV splitter; regression, binary and multiclass
  classification; pandas and NumPy input; `n_jobs`, `random_state`,
  `features`, `groups`, `fit_params`.
- `AnchorResidualizer`: scikit-learn transformer for the linear
  dependence-removal step.
- `SharedImportanceResult` with a summary table and per-split results.
- Provisional common specification (`docs/specification.md`) and notes on the
  relation to the official UMFI implementations (`docs/relation_to_umfi.md`).
- Test suite, including parity tests against the official UMFI Python code.
