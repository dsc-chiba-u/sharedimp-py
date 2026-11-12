# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
- Optional exploratory direction diagnostic (`direction=True`).
- Provisional common specification (`docs/specification.md`) and notes on the
  relation to the official UMFI implementations (`docs/relation_to_umfi.md`).
- Test suite, including parity tests against the official UMFI Python code.
