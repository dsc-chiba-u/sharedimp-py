# Changes required in `sharedimp-r`

Written from `sharedimp-py` on 2026-11-13. Nothing in `sharedimp-r` was
modified; its files were only read, and its code was only executed from a
copy. "Spec §" refers to [`specification.md`](specification.md) in this
repository, "R spec §" to `sharedimp-r/docs/specification.md`.

Two states of the R repository are distinguished:

* **Committed**: commit `6c5631d` (2026-11-12). This is the state that can be
  cited and reproduced, and the state from which the Python cross-language
  reference `tests/fixtures/cross_language/r_reference.json` was generated.
* **Working tree**: on 2026-11-13 the R repository contained a large set of
  **uncommitted** changes (`R/*.R`, `docs/specification.md`, tests) that
  implement the same common specification. They were read to align the two
  specifications, but were not executed and may still change. The column
  "working tree" below records what was seen (around 13:00); it is not a
  verification.

## A. Changes that affect numbers or the result tables

| # | Topic | Committed `6c5631d` | Required (spec) | Working tree |
| --- | --- | --- | --- | --- |
| 1 | Column order of the Full representation | anchor first, then the residualized columns | all columns in the original order, raw anchor **in its original position**; Reduced = Full without the anchor column (§3) | implemented |
| 2 | Exactly collinear columns | residual columns that are "constant" on the training rows are **dropped** from both representations | never drop; residual **exactly 0 on training and assessment rows**, column kept (§3.1 D3) | implemented |
| 3 | Collinearity test | `is_constant()` on the residual: `max − min ≤ 1e-10 · max(1, max\|v\|)` | `S_jj > 0` and `SS_res_j ≤ 1e-10 · S_jj`; no absolute threshold (§3.1) | implemented |
| 4 | Constant anchor / constant feature | `is_constant()` (same tolerant test) | exactly constant (`S = 0`), **evaluated as `max == min`** (§3.1 D1, D2) | `ss_i == 0` after centering with `mean()` — see note |
| 5 | Summary table | `feature, importance, sd, n_folds` | `feature, importance, importance_std, score_full, score_reduced, n_splits` (§6.1) | implemented |
| 6 | Per-split table | `feature, id, id2, metric_full, metric_reduced, importance, error`; rows feature-major | `feature, split, n_train, n_test, score_full, score_reduced, importance` (+ `id, id2, error`); rows **split-major**; `split` 1-based in R (§6.2) | columns implemented; row order to be confirmed |
| 7 | Orientation of scores | `metric_full`/`metric_reduced` are raw metric values | `score_*` are greater-is-better: `−ν` for metrics to minimize (§2, §6) | implemented |
| 8 | R² | built-in `"rsq"` = `yardstick::rsq`, described as the R² both languages must use | two common metrics: `r2` = `yardstick::rsq_trad` (= scikit-learn `"r2"`) and `rsq` = `yardstick::rsq` (§5) | implemented (`"r2"`, alias `"rsq_trad"`; `"log_loss"`, alias `"mn_log_loss"`) |
| 9 | Direction diagnostic | `direction = TRUE` argument, `direction` columns, `$direction` | not part of the v0.1 API (§8) | removed from the API |

Notes:

**1–3** change results for learners whose behaviour depends on the number or
order of columns, in particular the built-in random forest (`mtry` depends on
the number of representation columns, which varied with the data when columns
were dropped). For ordinary least squares they change no prediction, which is
why the Python reference generated from the committed state is still valid.
The committed rule also has an absolute floor (`max(1, ·)`), which made it
depend on the units of the data: residuals of features recorded in small units
(values around `1e-12`) were all declared constant and dropped.

**4** is the one place where the working tree and the Python package differ in
*how* a rule is evaluated, not in the rule. `sum((x − mean(x))^2) == 0` equals
`max(x) == min(x)` whenever the computed mean of a constant vector is exactly
that constant. R's `mean()` usually achieves this through long-double
accumulation, but that is not guaranteed on platforms without extended
precision (e.g. arm64), and with plain double summation it fails for simple
values (`mean(c(0.1, 0.1, 0.1)) != 0.1`). When the test misses a constant
anchor, the slope is computed from round-off and is arbitrary; applied to
assessment rows where the anchor does vary, it produces arbitrary residuals.
Recommended: `constant <- max(x) == min(x)` for the anchor and for each
feature, and for a constant feature slope 0 with the intercept set to that
constant (training residual exactly 0).

**6** With `on_error = "warn"`, `n_splits` is the number of successful splits.

## B. Corrections to the text of the R specification

Against the working-tree revision:

| # | R spec | Issue |
| --- | --- | --- |
| 10 | §5.1 | D1/D2 are stated as `S_ii = 0` / `S_jj = 0` "(exactly)"; add that this must be evaluated as `max = min` (note on item 4). |
| 11 | §9.1, `log_loss` row | "scikit-learn ≥ 1.5 clips at machine epsilon; earlier versions used `1e-15`": scikit-learn **1.3.2** already clips at machine epsilon (checked: `log_loss` of a zero probability gives `−log(2⁻⁵²)/2 = 18.0218…`, not `17.2694…`). The Python package requires scikit-learn ≥ 1.3, and `common_scorer("log_loss")` clips at `2⁻⁵²` itself, independently of the scikit-learn version. |
| 12 | §9.1, `accuracy` row | scikit-learn's `"accuracy"` scorer uses the estimator's `predict`, which is the argmax of the probabilities for most but not all classifiers (not for `SVC(probability=True)`). The exact counterpart of "argmax labels" is `common_scorer("accuracy")`. |
| 13 | §9.1, `rsq` row | Python provides `common_scorer("rsq")`; it returns `NaN` when the prediction (or the truth) is constant on the assessment rows — always the case for the null model — matching `yardstick::rsq` returning `NA`. |
| 14 | §15, §16 | written against `sharedimp-py` at `bfcb2a5`. Since then: `direction` removed, `p = 1` supported, `common_scorer()` added, the §15.1 fixture added as a test. Status of every §16 item: spec §11. |
| 15 | §15.1 | the collinear variant (`x1_dup = 3·x1 − 2`) "also agreed to 2.9e-14" with `LinearRegression()`: true for scikit-learn 1.9.1, but with scikit-learn 1.3.2 / NumPy 1.23 `LinearRegression` gives a different result for `x3` (41.79 instead of 45.55), because for the anchors `x2`, `x3` the two residual columns of `x1` and `x1_dup` are collinear with each other and the rank deficiency is not reliably detected. The agreement holds in both environments with an explicitly rank-revealing least-squares fit. State the learner requirement (spec §9). |
| 16 | header | the statement that `sharedimp-py` "contains only a README" (committed text) is outdated. |

Against the committed text only (already addressed in the working tree):
§6/§7 (column order, tolerant constant test), §9 (`rsq` as the common R²;
`mn_log_loss` "clipped to `[1e-15, 1 − 1e-15]`" — `yardstick` 1.4.0 clips at
`.Machine$double.eps`), §11 (column names), §13 (direction), and §4 (the Python
package does not accept an integer vector of fold labels; use
`PredefinedSplit(labels − 1)` or a list of `(train, test)` pairs).

## C. Verified to agree (no change needed)

Against the committed R package, by running R (reference
`r_reference.json`; test `tests/test_cross_language.py`):

* residualization slopes, intercepts and transformed training/assessment data
  (`1e-10`), and the set of exactly collinear features (R drops exactly the
  columns Python sets to zero);
* the null model for a single feature (`mse`, `r2`, `log_loss`, `accuracy`);
* `shared_importance(learner = "linear")` on explicit folds: per-split scores,
  importances, their mean, sample standard deviation and count, for regression
  (`mse`, `rmse`, `mae`, `rsq`, `rsq_trad`; `1e-9`), binary (`mn_log_loss`,
  `accuracy`, `roc_auc`; `1e-6`) and multiclass (`1e-4`) targets, including a
  design with an exact duplicate and an affine copy;
* metric values on fixed predictions, `yardstick` 1.4.0 vs. the Python common
  scorers (`1e-12`): `mse`, `rmse`, `mae`, `rsq`, `rsq_trad`, `mn_log_loss`
  (with a clipped zero probability), `accuracy` (with an argmax tie), binary
  `roc_auc` (either event level), multiclass `roc_auc` (Hand–Till).

Against the numbers recorded in the working-tree R specification (§15.1):
Python reproduces the summary and per-split values of the parity fixture, and
of its collinear variant, to `1e-8`.

Already as specified in the committed package: sign of importance, no
truncation, per-split evaluation with an unweighted mean, sample standard
deviation.

## D. Next steps on the R side

1. Commit the working-tree revision (after its own checks).
2. Apply item 4 / 10.
3. Correct the specification text (items 11–16).
4. Then, from `sharedimp-py`, regenerate `r_reference.json` against the new
   commit and remove the conversions in `tests/test_cross_language.py` (sign of
   raw metrics, comparison by column name, row reordering). Forest learners
   can only be compared qualitatively.
5. Optionally add an R test that reads
   `sharedimp-py/tests/fixtures/cross_language/inputs.json`, so that both
   packages are tested on identical inputs in both directions.

## E. Open on the Python side (requested by R spec §16)

Not changed here because they are API decisions outside the items decided on
2026-11-13; see spec §11.

* Default metric: Python `estimator.score` (R² / accuracy) vs. R `mse` /
  `log_loss`.
* Learner failure in a split: Python raises; R records `NA` and continues by
  default.
* Replacing the Python specification by a pointer to the R one.
