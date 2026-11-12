# CLAUDE.md

Guidance for working in this repository (`sharedimp-py`, PyPI/import name `sharedimp`).

## What this project is

A Python package, aimed at a SoftwareX submission, that makes **UMFI-LR**
(ultra-marginal feature importance with linear-regression dependence removal;
Janssen, Guan & Robeva, AISTATS 2023) easy and safe to use inside scikit-learn
workflows.

Positioning — keep this wording consistent everywhere (README, docs, docstrings, paper):

* We do **not** claim a new feature-importance method. The quantity is UMFI-LR.
* The contribution is software engineering: fold-internal residualization (no
  leakage), any estimator / scorer / CV splitter, paired evaluation,
  reproducibility, per-fold results, tests.
* The direction diagnostic is exploratory and is **not** a causal effect.
* No confidence intervals, p-values or other formal inference. Do not add them
  unless explicitly asked. `importance_std` is a descriptive between-split SD.

Sibling repositories (in the parent directory): `sharedimp-r` (R package) and
`sharedimp-experiments` (paper benchmarks). **Never modify them from here.**
Large benchmarks belong in `sharedimp-experiments`, not in this repository.

## Specification

* `docs/specification.md` is the **provisional** common spec, written on the
  Python side because `sharedimp-r/docs/specification.md` did not exist on
  2026-11-12. If that file appears, read it first and treat it as authoritative;
  reconcile differences explicitly (§9 of the provisional spec lists the open
  items) rather than silently changing behaviour.
* `docs/relation_to_umfi.md` records what the paper and the official
  implementations do, how this package differs, and what was and was not
  verified numerically.
* Never change the statistical quantity for implementation convenience. If an
  implementation constraint seems to require it, stop and raise it.

## Algorithm (summary; the spec is normative)

For every CV split and every anchor feature `x_i`:

1. Fit, on **training rows only**, the simple OLS regression (with intercept) of
   each other feature `x_j` on `x_i`; residualize training **and assessment**
   rows with those same coefficients.
2. Full = raw `x_i` (original column position) + residualized others;
   Reduced = residualized others only.
3. Clone the estimator twice (same hyperparameters, same seed), fit on the two
   training representations, score both on the assessment rows with one
   scikit-learn scorer.
4. `importance = score_full − score_reduced` per split; report the unweighted
   mean over splits.

Invariants that must hold:

* Scorers are "greater is better" already. Never negate a score; never
  special-case `neg_*` names.
* Never truncate scores or importances (negative importance is valid output).
* Nothing is estimated from assessment rows.
* Results are independent of `n_jobs` and of which subset of `features` is
  evaluated (seeds are pre-assigned per (split, column)).
* The user's estimator is never mutated; explicitly set `random_state` values
  and user-supplied CV splitters are never modified.
* Residualization is deterministic (no noise injection, no significance gate).

## Architecture (`src/sharedimp/`)

| Module | Responsibility |
| --- | --- |
| `_residualizer.py` | `AnchorResidualizer`, a stand-alone scikit-learn transformer (`coef_`, `intercept_`, `collinear_`) |
| `_resampling.py` | `resolve_cv`, `make_splits`: CV spec → concrete index pairs |
| `_fitting.py` | `clone_with_seed`, `fit_clone`, per-fold slicing of `fit_params` |
| `_scoring.py` | `resolve_scorer`, `score_model` |
| `_direction.py` | optional direction diagnostic |
| `_result.py` | `SharedImportanceResult`, `aggregate_fold_results` |
| `_core.py` | `shared_importance()`: validation, seeding, joblib dispatch, assembly |

Keep these concerns separate. `_core.py` orchestrates; it should not grow
statistics of its own. Public API = what `__init__.py` exports
(`shared_importance`, `AnchorResidualizer`, `SharedImportanceResult`).

## Public API

```python
shared_importance(estimator, X, y, *, scoring=None, cv=5, groups=None,
                  features=None, direction=False, fit_params=None,
                  n_jobs=None, random_state=None, verbose=0)
```

Returns `SharedImportanceResult` with `.importances` (one row per feature:
`feature, importance, importance_std, score_full, score_reduced, n_splits[, direction]`)
and `.fold_results` (one row per split × feature:
`feature, split, n_train, n_test, score_full, score_reduced, importance[, direction]`).
Column names are part of the cross-language contract — do not rename them
without updating the spec and coordinating with the R package.

An integer `cv` means *shuffled* (Stratified)KFold seeded by `random_state`;
this differs from scikit-learn's unshuffled default on purpose (see spec §4).

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest              # tests + doctests (parity tests need network)
.venv/bin/python -m pytest -m "not parity"
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/mypy
.venv/bin/python -m build && .venv/bin/twine check dist/*
```

Supported: Python ≥ 3.10, scikit-learn ≥ 1.3, NumPy ≥ 1.23, pandas ≥ 1.5,
SciPy ≥ 1.9, joblib ≥ 1.2. CI has a job pinned to exactly these minimums; when
you use a newer API, either raise the minimum deliberately (and update
`pyproject.toml`, CI and the changelog) or avoid it.

## Testing policy

* Start from small deterministic synthetic data with a known answer; prefer
  exact checks (hand calculation, closed form, independent re-computation with
  `numpy.linalg.lstsq`) over "looks plausible" thresholds.
* Test layout:
  * `tests/test_residualizer.py` — coefficients vs. hand calculation, train /
    assessment separation, Full/Reduced shapes, degenerate cases, pandas.
  * `tests/test_core_exact.py` — what the estimator actually receives (spy
    estimator), OLS + MSE closed forms, exact duplicates, orthogonal
    predictors, scorer sign.
  * `tests/test_api.py` — regression / binary / multiclass smoke tests,
    pandas vs. NumPy, CV splitters, scorers, parallel vs. serial,
    reproducibility, negative importance, error handling, direction.
  * `tests/test_sklearn_compat.py` — scikit-learn estimator checks for the
    transformer.
  * `tests/test_parity_official.py` — parity with the official UMFI Python
    code, downloaded from a pinned commit (not vendored: upstream has no
    license). Skips without network; never replace a skipped comparison with
    made-up numbers.
* Every behavioural change needs a test; every bug fix needs a regression test.
* Report results honestly: if something could not be verified (e.g. the R
  implementation, because R is unavailable), write that down in
  `docs/relation_to_umfi.md` instead of implying it was checked.

## Conventions

* `src/` layout, hatchling build, version in `src/sharedimp/_version.py`.
* Full type hints (`mypy` strict-ish on `src/`), NumPy-style docstrings, ruff
  for lint and format (line length 100).
* Update `CHANGELOG.md` for user-visible changes.
* Do not commit `dist/`, virtual environments or downloaded upstream sources.
