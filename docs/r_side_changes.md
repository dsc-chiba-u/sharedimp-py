# Open items for `sharedimp-r`

Written from `sharedimp-py` on 2026-11-13 (third revision). Nothing in
`sharedimp-r` was modified; its files were only read. "Spec §" refers to the
common specification `sharedimp-r/docs/specification.md`; "Python notes §" to
[`specification.md`](specification.md) in this repository.

State of the R repository that was read: commit `eaafaee` (2026-11-13), clean
working tree.

## A. Implementation

The R package at `eaafaee` implements what the earlier revisions of this list
asked for: anchor in its original position, exactly collinear columns kept as
zeros with the `SS_res ≤ 1e-10 · S_jj` rule, constants tested by `max == min`,
the common column names with greater-is-better scores, split-major rows, `r2`
/ `rsq` as separate metrics, no direction diagnostic. **No implementation
change is requested.**

One difference in behaviour is recorded rather than requested:

| Topic | R | Python | Note |
| --- | --- | --- | --- |
| Non-finite predictions | a learner failure (`check_prediction()`) | not inspected: scorers receive the fitted model, not its predictions. They become a failure when the scorer raises (the usual case), otherwise a non-finite score that is excluded from the aggregation with an empty `error` | Python notes §4.3. Same summary numbers in both cases; only the `error` text can differ. |

## B. Text of the common specification

Statements that are outdated now that the Python package has changed (Python
commit of this revision; see the git log), or that need correcting:

| # | Spec | Issue |
| --- | --- | --- |
| 1 | §3 | "(Python currently rejects `p = 1`; see Section 16.)" — `p = 1` is accepted since Python `febd7ea`. |
| 2 | §9.1, §10, §16 items 3–4 | "Python implements this since commit `692c3fe`": the default metric and `on_error` are **not** in `692c3fe`. They were observed in the Python working tree while it was being edited (the package is installed in editable mode) and are committed in the first Python commit after `692c3fe` ("Adopt the common default metric and learner-failure handling"). The description itself is accurate: `scoring=None` → `neg_mean_squared_error` / `common_scorer("log_loss")` (reported as `neg_log_loss`); `on_error = "warn"` (default) / `"error"`; `error` column; `n_failed`. |
| 3 | §10 | Python stores the message too, as `"<ExceptionType>: <message>"` in an `error` column appended after the common columns ("R additionally stores the error message" suggests otherwise). Non-finite predictions: see section A. |
| 4 | §11.2 | Python appends `error` after the common columns (it has no `id`/`id2`). |
| 5 | §16 | item 5 (specification pointer) is done in that same commit: the Python `docs/specification.md` is now a pointer plus conformance table, Python notes and recorded deviations, and no longer describes the old default or failure behaviour. Items 1, 2, 6, 7 are committed, not "in the working tree". Remaining: the regeneration under item 7 (section C). |
| 6 | §9.1 `log_loss` row; §16 item 6 | "scikit-learn ≥ 1.5 clips at machine epsilon; earlier versions used `1e-15`": scikit-learn **1.3.2** already clips at machine epsilon (checked: the log loss of a zero probability is `−log(2⁻⁵²)/2 = 18.0218…`, not `17.2694…`). The Python package requires scikit-learn ≥ 1.3 and its `common_scorer("log_loss")` clips at `2⁻⁵²` itself. |
| 7 | §9.1 `accuracy` row | scikit-learn's `"accuracy"` scorer uses the estimator's `predict`, which is the argmax of the probabilities for most but not all classifiers (not for `SVC(probability=True)`). The exact counterpart of "argmax labels" is `common_scorer("accuracy")`. |
| 8 | §9.1 `rsq` row | Python provides `common_scorer("rsq")`; like `yardstick::rsq` it is undefined (`NaN`) for a constant prediction or truth. |
| 9 | §15.1 | the collinear variant "also agreed to 2.9e-14" with `LinearRegression()`: true for scikit-learn 1.9.1 / NumPy 2.4, but with scikit-learn 1.3.2 / NumPy 1.23 `LinearRegression` gives 41.79 instead of 45.55 for `x3`. For the anchors `x2` and `x3` the residuals of `x1` and `x1_dup` are collinear with each other, and the rank deficiency is not reliably detected. With an explicit minimum-norm least-squares fit the agreement holds in both environments. The specification should state that requirement on the learner. |
| 10 | §17 | consistent with the Python workflow. One nuance: a manual run of the Python workflow publishes only from `main` or a release tag. |

## C. After the specification revision is committed

Deliberately not done yet from the Python side:

1. Regenerate `tests/fixtures/cross_language/r_reference.json` from the then
   current R commit (script: `tests/fixtures/cross_language/generate_r_reference.R`,
   which must first be adapted to the new R output: `score_*` columns,
   `collinear` flags instead of `dropped`, no `drop_constant` argument).
2. Remove the conversions in `tests/test_cross_language.py` that compensate for
   the layout of the old R commit (sign of raw metrics, comparison by column
   name, row reordering).
3. Extend the comparison to the default metric of both languages and to a
   failing split (`on_error = "warn"` in both), which the old reference cannot
   cover.

## D. Verified to agree

* Against the R package at `6c5631d`, by running R (`r_reference.json`):
  residualization (`1e-10`), the null model, `shared_importance(learner =
  "linear")` on explicit folds for regression (`1e-9`), binary (`1e-6`) and
  multiclass (`1e-4`) targets, and all common metric values against
  `yardstick` 1.4.0 (`1e-12`).
* Against the numbers recorded in spec §15.1 (from the reconciled R package):
  the parity fixture and its collinear variant, every column of both tables, to
  `1e-8` — with `scoring=None` (the common default), with
  `"neg_mean_squared_error"` and with `common_scorer("mse")`.

Not verified from this repository: the R package's behaviour for learner
failures and its default-metric path were read in the source, not executed.
