# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

Alignment with the R package (`sharedimp-r`) on a common specification
(`docs/specification.md`, v0.2-draft).

### Added

- `common_scorer()` / `COMMON_METRICS`: scorers for `mse`, `rmse`, `mae`, `r2`,
  `rsq`, `log_loss`, `accuracy`, `roc_auc` with definitions pinned across R and
  Python (`r2` is `yardstick::rsq_trad`; `rsq` is `yardstick::rsq`).
- Single-feature input (`p = 1`): the Reduced model is the null model (training
  mean / training class frequencies) instead of an error.
- Cross-language parity tests against reference values generated with R
  (`yardstick` and `sharedimp-r`), and `docs/r_side_changes.md`.
- A warning when an anchor is constant on a training set.
- `docs/specification.md` rewritten as the common R/Python specification
  (v0.2-draft), aligned with the R draft.
- `AnchorResidualizer.constant_anchor_`.

### Changed

- Constant columns are detected exactly (`max == min`) instead of through a
  rounded mean; the exact-collinearity rule (`SS_res <= 1e-10 * SS_tot`) is
  unchanged.
- Official-UMFI parity tests are grouped into conditions where exact agreement
  is expected and conditions where it is not.

### Removed

- The `direction` argument of `shared_importance()` and the `direction` result
  columns. A direction diagnostic is not part of the common API in v0.1; the
  previous helper remains as an internal, unexported, experimental module.

## [0.1.0] - 2026-11-12

Initial implementation (not yet released on PyPI).

### Added

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
