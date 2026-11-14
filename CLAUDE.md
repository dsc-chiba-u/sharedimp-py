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
* There is **no direction diagnostic** in the v0.1 API (removed 2026-11-13; the
  R and Python definitions differed). Do not re-expose `sharedimp._direction`
  without an agreed common definition. Importances are not causal effects.
* No confidence intervals, p-values or other formal inference. Do not add them
  unless explicitly asked. `importance_std` is a descriptive between-split SD.

Sibling repositories (in the parent directory): `sharedimp-r` (R package) and
`sharedimp-experiments` (paper benchmarks). **Never modify them from here.**
Large benchmarks belong in `sharedimp-experiments`, not in this repository.

## Specification

* **The single source of truth is `../sharedimp-r/docs/specification.md`**
  (common R/Python specification). Read it before touching behaviour, and
  check `git -C ../sharedimp-r status` first: that repository is edited in
  parallel and its specification has been ahead of its last commit before.
* `docs/specification.md` in this repository is *not* a second specification.
  It is the Python-side record: revision checked, conformance table (section →
  code → test), Python-specific behaviour, recorded deviations, and what is
  open. Every intentional deviation from the common specification must be
  written there.
* The R/Python reconciliation is complete as of R commit `cf2e1cc`
  (2026-11-13): no open item in the common specification, none in this
  implementation. `docs/specification.md` §7 separates language-specific
  differences (by design) and limits of numerical agreement (learner
  properties) from open problems (none). `docs/r_side_changes.md` is the
  closed record of what was raised with the R side; add a new section there
  only if a new discrepancy is found.
* `docs/rank_deficiency.md` records why OLS results on collinear designs depend
  on the solver. It is a learner matter: never "fix" it by changing what
  estimator the package runs for the user.
* Decisions taken by the project owner (2026-11-13) — do not reopen them on
  your own: anchor stays in its original column position; exactly collinear
  residuals are fixed at zero and kept; `p = 1` uses the null model; common
  column names; no direction diagnostic in v0.1; default metric MSE / log
  loss; learner failures are recorded as `NaN` with a warning by default
  (`on_error="error"` stops); Docker tags `latest` + SHA on `main`,
  `X.Y.Z` + SHA on `vX.Y.Z`.
* `tests/fixtures/cross_language/r_reference.json` was generated from R commit
  `92f9ecb` (recorded in its `meta` block); R commits up to `cf2e1cc` changed
  documentation only. Regenerate it whenever the R
  package changes numbers or tables, and only by running R — never edit it by
  hand. If R cannot be run, leave the file alone and report why.
* `docs/relation_to_umfi.md` records what the paper and the official
  implementations do, how this package differs, and what was and was not
  verified numerically.
* Never change the statistical quantity for implementation convenience. If an
  implementation constraint seems to require it, stop and raise it.
* A change to anything in the spec (algorithm, degenerate rules, metric
  definitions, column names, row order) is a cross-language change: update the
  spec, `docs/r_side_changes.md` and the cross-language test together.

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
* Full keeps every column in the original order with the raw anchor in place;
  Reduced is Full without the anchor column. Nothing is dropped.
* Degenerate cases (spec §5.1): constant anchor or feature is detected by
  `max == min` (never via a computed mean); a feature with
  `SS_res <= 1e-10 * SS_tot` gets a residual of exactly 0 on all rows. No
  absolute thresholds — the rules must stay invariant to units.
* `p = 1`: the Reduced model is the null model (`_fitting.fit_null_model`).
* `scoring=None` means MSE for regressors and log loss for classifiers
  (`_scoring.resolve_scorer`), never `estimator.score`. The importance is
  `score_full - score_reduced` for every metric.
* Learner failures (`_core._evaluate_anchor`): with `on_error="warn"` a failed
  (split, feature) pair gets `NaN` scores and its message in the `error`
  column; the run continues, warns once, and aggregates over the splits with a
  defined importance (`n_splits` counts them). `on_error="error"` re-raises.
  A failure of either model discards the whole pair. Input and
  residualization errors are never swallowed. Non-finite assessment
  predictions are failures: `_scoring.check_finite_predictions` inspects them
  once per fitted model (the scorer alone cannot be trusted to raise). A `NaN`
  score from finite predictions (undefined metric) is not a failure.
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
| `_metrics.py` | `common_scorer`: metric definitions pinned across R and Python |
| `_direction.py` | internal, unexported, experimental; not called by anything |
| `_result.py` | `SharedImportanceResult`, `aggregate_fold_results` |
| `_core.py` | `shared_importance()`: validation, seeding, joblib dispatch, assembly |

Keep these concerns separate. `_core.py` orchestrates; it should not grow
statistics of its own. Public API = what `__init__.py` exports
(`shared_importance`, `AnchorResidualizer`, `SharedImportanceResult`,
`common_scorer`, `COMMON_METRICS`).

## Public API

```python
shared_importance(estimator, X, y, *, scoring=None, cv=5, groups=None,
                  features=None, fit_params=None, on_error="warn",
                  n_jobs=None, random_state=None, verbose=0)
```

Returns `SharedImportanceResult` with `.importances` (one row per feature:
`feature, importance, importance_std, score_full, score_reduced, n_splits`)
and `.fold_results` (one row per split × feature, split-major, `split` 0-based — R uses 1:
`feature, split, n_train, n_test, score_full, score_reduced, importance, error`).
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
docker build -t sharedimp-py .   # runs docker/smoke_test.py; needs Docker access
```

Supported: Python ≥ 3.10, scikit-learn ≥ 1.3, NumPy ≥ 1.23, pandas ≥ 1.5,
SciPy ≥ 1.9, joblib ≥ 1.2. CI has a job pinned to exactly these minimums; when
you use a newer API, either raise the minimum deliberately (and update
`pyproject.toml`, CI and the changelog) or avoid it.

## Docker image and publishing

* `Dockerfile` (two stages, `python:3.13-slim`): builds the wheel, installs it
  with its runtime dependencies, then runs `docker/smoke_test.py` **during the
  build**, as an unprivileged user. `.dockerignore` is an allow-list
  (`pyproject.toml`, `README.md`, `LICENSE`, `src/`, `docker/smoke_test.py`);
  a new file needed in the image must be added there. Keep tests, docs, dev
  tools and caches out of the image.
* `.github/workflows/build_test_push.yml` ("DockerHub/GHCR"):
  * pull requests: build + test only, no login, no push;
  * push to `main`: push `latest` and the short commit SHA;
  * push of a tag `vX.Y.Z`: push `X.Y.Z`, `X.Y` and the short SHA (not `latest`);
  * targets `ghcr.io/dsc-chiba-u/sharedimp-py` (auth: `GITHUB_TOKEN`) and
    `koki/sharedimp-py` on Docker Hub (auth: repository secrets
    `DOCKER_USERNAME`, `DOCKER_PASSWORD`); one build, one push, both names;
  * if the Docker Hub secrets are missing the job fails *before* building,
    with an explicit error, and nothing is pushed anywhere.
* The same publishing policy (registries, tags, PR behaviour, secret names) is
  intended for `sharedimp-r` (`ghcr.io/dsc-chiba-u/sharedimp-r`,
  `koki/sharedimp-r`); keep the two workflows aligned. The reference this was
  modelled on is `dsc-chiba-u/fbrglm`, with these deliberate differences: no
  push from pull requests, a single build for both registries, tags from
  `docker/metadata-action`.
* **PyPI publishing is separate**: `.github/workflows/publish.yml` ("PyPI").
  It must not be coupled to the Docker workflow in either direction.
  * push of a tag `vX.Y.Z`: build, check, test the built wheel, upload to PyPI
    (the tag must equal `v` + the version in `_version.py`, or the job fails);
  * manual run: the same build, uploaded to TestPyPI only;
  * authentication is PyPI trusted publishing (OIDC, environments `pypi` /
    `testpypi`); there is no API token and no secret. The publisher must be
    registered on PyPI / TestPyPI by the project owner.
  * A PyPI upload cannot be undone and a version number cannot be reused.
    Never push a `v*` tag without the owner's explicit go-ahead.
* Docker is not usable by the development account on this machine (no access
  to the daemon socket); verify image builds through the pull-request run of
  the workflow, and say so when a local build was not possible.

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
    reproducibility, negative importance, error handling.
  * `tests/test_sklearn_compat.py` — scikit-learn estimator checks for the
    transformer.
  * `tests/test_common_spec.py` — conventions fixed by the common spec: column
    order, degenerate rules and their unit invariance, `p = 1`, sign, schema,
    common scorers, parallel = serial.
  * `tests/test_failures.py` — learner failures: recording, warning,
    aggregation over successful splits, `on_error="error"`, single-class
    training sets, parallel = serial.
  * `tests/test_cross_language.py` — the cross-language regression test. The
    R output in `r_reference.json` (representations, predictions, both result
    tables, default metric, collinearity, `p = 1`, failures) is compared
    **directly**; the only adaptation is the split label (R 1-based). Do not
    add conversions to make a comparison pass: a mismatch is either a real
    difference or a learner effect, and must be understood. Tolerance `1e-8`
    for closed-form learners; multinomial logistic regression `1e-4` (R's
    optimiser). Rank-deficient designs use the test-only `AliasingOLS` (R's
    `lm.fit` convention), full-rank designs `LinearRegression`.
  * `tests/fixtures/cross_language/` — `inputs.json` (shared inputs, from
    `make_inputs.py`; append new draws at the end so existing values do not
    change), `generate_r_reference.R`, `r_reference.json`,
    `rank_deficiency_study.py`.
  * `tests/test_parity_official.py` — parity with the official UMFI Python
    code, downloaded from a pinned commit (not vendored: upstream has no
    license). Split into "exact agreement expected" and "agreement not
    expected". Skips without network; never replace a skipped comparison with
    made-up numbers.
* Every behavioural change needs a test; every bug fix needs a regression test.
* Report results honestly: if something could not be verified, write that down
  in `docs/relation_to_umfi.md` / `docs/specification.md` §6 instead of implying
  it was checked.

### Regenerating the R reference

Do this after any change of the R package that affects numbers or tables, and
after changing `inputs.json`. Check `git -C ../sharedimp-r status` first and
use a committed state.

R is not on the PATH; it lives in a conda environment. Load the R package from
a *copy* so that nothing is written into `../sharedimp-r`:

```bash
R=/home/koki/anaconda3/envs/sharedimp-dev/bin/Rscript
COPY=$(mktemp -d) && git -C ../sharedimp-r archive HEAD | tar -x -C "$COPY"
$R tests/fixtures/cross_language/generate_r_reference.R \
   tests/fixtures/cross_language/inputs.json \
   tests/fixtures/cross_language/r_reference.json \
   "$COPY" "$(git -C ../sharedimp-r rev-parse HEAD)"
```

## Conventions

* `src/` layout, hatchling build, version in `src/sharedimp/_version.py`.
* Full type hints (`mypy` strict-ish on `src/`), NumPy-style docstrings, ruff
  for lint and format (line length 100).
* Update `CHANGELOG.md` for user-visible changes.
* Do not commit `dist/`, virtual environments or downloaded upstream sources.
