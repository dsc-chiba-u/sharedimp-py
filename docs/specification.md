# sharedimp — common specification (R / Python)

**Version `0.2-draft`, 2026-11-13. Status: reconciled with the R draft; a few
items open (§11).**

This is the Python-side statement of the specification that the Python package
(`sharedimp-py`) and the R package (`sharedimp-r`) must both satisfy, so that
the two compute the *same* UMFI-LR-type importance. It supersedes the
`0.1-provisional` text of 2026-11-12 and incorporates the decisions taken on
2026-11-13 (marked **[decided]**).

**Relation to the R specification.** `sharedimp-r/docs/specification.md`
declares itself the single source of truth for both packages. Two states of it
were read (nothing in that repository was modified):

* the committed v0.1 draft (R commit `6c5631d`, 2026-11-12), which differs from
  this document in column order, the collinearity rule, the output schema, the
  definition of R² and the direction diagnostic;
* an **uncommitted** revision in the R working tree (read 2026-11-13), which
  adopts the same decisions as this document. This document was aligned with
  that revision section by section; the cross-references "R spec §n" below
  point to it.

Once the R revision is committed, the two texts should be merged into one file
kept identical in both repositories (the R draft proposes that this file
become a pointer to it). Until then: what the R package still has to do is
listed in [`r_side_changes.md`](r_side_changes.md), and what the R draft asks
of the Python package is tracked in §11.

The Python package implements everything in this document. Cross-language
agreement is tested against numbers produced by R (§9).

## 1. Scope

`sharedimp` computes ultra-marginal feature importance with linear-regression
dependence removal (UMFI-LR; Janssen, Guan & Robeva, AISTATS 2023) as an
out-of-sample, resampling-based quantity for arbitrary supervised learners.

It does **not** define a new feature-importance method. The contribution is
software engineering: leakage-free resampling, integration with the modelling
ecosystem of each language, reproducibility, and tests.

Out of scope for v0.1: confidence intervals, p-values and any other formal
inference; a direction diagnostic (§8); optimal-transport dependence removal;
hyperparameter tuning; non-numeric features; missing values.

## 2. Notation

* `X`: `n × p` numeric feature matrix, columns `x_1 … x_p`, `p ≥ 1`.
* `y`: response (continuous, binary or multiclass).
* A *split* `s` is a pair `(T_s, A_s)` of training and assessment rows.
* A *score* is a performance metric oriented so that **greater is better**:
  `score = ν` for metrics to maximize and `score = −ν` for losses.

## 3. Algorithm

For every anchor feature `x_i` and every split `s`:

1. **Residualization on training rows only.** For every `j ≠ i`, ordinary
   least squares with intercept on the rows `T_s`:

   ```
   b_ij = Σ_T (x_i − m_i)(x_j − m_j) / Σ_T (x_i − m_i)²
   a_ij = m_j − b_ij · m_i                      (m = mean over T_s)
   ```

2. **Same coefficients for both row sets.** `r_j = x_j − a_ij − b_ij · x_i` for
   the rows in `T_s` and for the rows in `A_s`. Nothing is estimated from `A_s`.

3. **Representations [decided].** Columns keep the order of `X`:

   * *Full*: all `p` columns; column `i` is the raw anchor, **in its original
     position**; every other column `j` is `r_j`.
   * *Reduced*: the Full representation without column `i` (`p − 1` columns).

   Column names are the original names. No column is ever dropped or added, so
   the layout does not depend on the data.

4. **Paired fits.** Two fresh copies of the same learner, with identical fixed
   hyperparameters and (where applicable) the identical seed, are fitted on the
   Full and on the Reduced training representation.

5. **Paired evaluation** with one metric on the assessment rows:
   `score_full[i,s]`, `score_reduced[i,s]`.

6. **Per-split importance:** `importance[i,s] = score_full[i,s] − score_reduced[i,s]`.

**Aggregation.** `importance[i] = mean_s importance[i,s]`: unweighted mean over
all splits (all repeats). Metrics are computed per split, never on pooled
predictions.

**Sign.** `importance > 0` ⇔ keeping the anchor improves out-of-sample
performance. Negative values are kept: neither scores nor importances are ever
truncated, per split or after aggregation.

### 3.1 Degenerate cases [decided]

All rules are deterministic (no noise injection) and evaluated on the training
rows `T_s` only.

| # | Case on `T_s` | Test | Rule |
| --- | --- | --- | --- |
| D1 | constant anchor: `S_ii = 0` | `max(x_i) = min(x_i)` | `b_ij = 0` for all `j`; features are only centered. A warning names the anchor. |
| D2 | constant feature: `S_jj = 0` | `max(x_j) = min(x_j)` | `b_ij = 0`, `a_ij =` that constant; hence `r_j = 0` exactly on `T_s` and `r_j = x_j − a_ij` on `A_s`. |
| D3 | feature exactly collinear with the anchor | `SS_tot_j > 0` and `SS_res_j ≤ τ · SS_tot_j`, `τ = 1e-10` | `r_j` is set to **exactly 0 for every row** of `T_s` and `A_s`. The column is kept. |

with `SS_tot_j = S_jj = Σ_T (x_j − m_j)²` and `SS_res_j = Σ_T r_j²`. `b_ij`,
`a_ij` are still reported as estimated. These are the rules of R spec §5.1;
`τ` is the only numerical tolerance in the algorithm.

*Why D3.* If `x_j` is an exact affine function of `x_i` (a duplicate, a unit
conversion, `1 − x_i`, …) its residual is pure floating-point round-off. That
round-off is a deterministic function of the anchor, so a flexible learner
could recover the anchor from it in the Reduced model. Fixing it at zero
removes the leak; keeping the (zero) column keeps the layout and learner
settings that depend on the number of columns (e.g. `mtry`) data-independent.

*Criterion.* D3 is the statement `1 − R²_j ≤ τ`, where `R²_j` is the coefficient
of determination of the regression of `x_j` on `x_i`.

*Tolerance.* `τ = 1e-10` on the ratio of sums of squares, i.e. the residual
standard deviation is at most `1e-5` of the feature's. Round-off produces
ratios of about `(ε·|m_j|/sd_j)²` with `ε = 2.2e-16` (`≈ 5e-32` for centered
data), ten or more orders of magnitude below `τ`; a feature that genuinely
differs from an affine function of the anchor by a relative `1e-5` or more is
not flagged.

*Scale dependence.* D1–D3 are invariant to any change of units and origin
(`x → c·x + d`, `c ≠ 0`) of the feature and of the anchor, because D1/D2
compare values for equality and D3 compares two sums of squares of the same
centered variable. There is deliberately **no absolute threshold** anywhere: a
rule of the form "range ≤ 1e-10 · max(1, |x|)" would flag every feature
recorded in small units (values around `1e-12`) and miss collinearity in
features recorded in large units. The only limit is numerical: with an extreme
offset, `|m_j|/sd_j ≳ 1/(ε·√τ) ≈ 4.5e10`, the data no longer carry enough
significant digits and an exactly collinear feature may not be recognised.

*How D1/D2 are evaluated.* "`S = 0`" is meant exactly. In floating point it
must be evaluated as `max = min`, not by centering with a computed mean: the floating-point mean
of a constant vector is not always that constant (`mean(0.1, 0.1, 0.1) ≠ 0.1`
in IEEE double arithmetic with plain summation), and a slope computed from the
resulting noise is arbitrary. Near-constant but not constant features are
ordinary data.

### 3.2 Single feature (`p = 1`) [decided]

The Reduced representation has zero columns. The Reduced model is then the
**null model**, fitted on the training rows:

* regression: predict the training mean of `y`;
* classification: predict the training class frequencies as class
  probabilities; the predicted label is the most frequent class, ties going to
  the first class in level order.

The Full model is the learner fitted on the anchor alone. Everything else
(scoring, schema) is unchanged. (Python: `DummyRegressor(strategy="mean")` /
`DummyClassifier(strategy="prior")`; a `sample_weight` passed to the learner
is also applied to the null model.)

### 3.3 Input requirements

* All features numeric (integer/logical are converted to double). Categorical
  features must be encoded by the user. No missing or infinite values.
* `p ≥ 1`, `n ≥ 2`, unique feature names.

## 4. Resampling

* Any scheme yielding `(training, assessment)` index pairs: k-fold, repeated,
  stratified, grouped, or explicit user-supplied splits.
* Default: 5-fold cross-validation with shuffled folds, stratified on `y` for
  classification.
* Splits are generated once and shared by all anchors and by the Full and
  Reduced model of each anchor.
* Fold generators differ between languages. **Explicit splits are the
  mechanism for cross-language equality.**

## 5. Metrics [decided]

One metric per call. A metric name does not define a quantity across
languages, so the common metrics are defined here; "identical metric" in a
cross-language comparison means one row of this table.

| Common name | Definition (per split, on `A_s`) | Score | Python | R (`yardstick`) |
| --- | --- | --- | --- | --- |
| `mse` | `mean((y − ŷ)²)` | `−ν` | `common_scorer("mse")` = `"neg_mean_squared_error"` | package metric `mse` |
| `rmse` | `sqrt(mse)` | `−ν` | `common_scorer("rmse")` = `"neg_root_mean_squared_error"` | `rmse` |
| `mae` | `mean(\|y − ŷ\|)` | `−ν` | `common_scorer("mae")` = `"neg_mean_absolute_error"` | `mae` |
| `r2` | `1 − Σ(y − ŷ)² / Σ(y − mean_A(y))²` | `ν` | `common_scorer("r2")` = `"r2"` | **`rsq_trad`** |
| `rsq` | `cor(y, ŷ)²` (Pearson) | `ν` | `common_scorer("rsq")` (no scikit-learn name) | **`rsq`** |
| `log_loss` | `−mean(log p_true)`, `p` clipped to `[ε, 1 − ε]`, `ε = 2⁻⁵²` | `−ν` | `common_scorer("log_loss")` = `"neg_log_loss"` | `mn_log_loss` |
| `accuracy` | fraction of correct labels; label = argmax of the class probabilities, ties → first level | `ν` | `common_scorer("accuracy")` | `accuracy` on argmax labels |
| `roc_auc` | binary: area under the ROC curve; multiclass: Hand & Till (2001) mean of pairwise AUCs | `ν` | `common_scorer("roc_auc")` = `"roc_auc"` / `"roc_auc_ovo"` | `roc_auc` (default `hand_till`) |

Notes:

* **`r2` ≠ `rsq`.** scikit-learn's `"r2"` and `yardstick::rsq` are different
  quantities (coefficient of determination vs. squared correlation; the latter
  cannot be negative and ignores bias and scale of the predictions).
  `yardstick::rsq_trad` is the one that equals scikit-learn's `"r2"`.
  `rsq` is undefined (`NaN`/`NA`) when the predictions are constant on `A_s`,
  which always happens for the null model; `r2` is undefined only when `y` is
  constant on `A_s`.
* `log_loss`: natural logarithm; `ε` is the double-precision machine epsilon in
  both `yardstick` (`.Machine$double.eps`) and scikit-learn. A class absent
  from the training rows has probability 0 (then clipped).
* `accuracy`: defined on argmax labels so that it does not depend on a
  learner's own label rule. scikit-learn's `"accuracy"` scorer calls `predict`
  and coincides whenever `predict` is the argmax of `predict_proba`.
* `roc_auc`, binary: the value does not depend on which class is called the
  event as long as the matching probability is used
  (`AUC(event = first, p_first) = AUC(event = second, p_second)`), so the
  `yardstick` default (first level) and scikit-learn (second class) agree.
  Multiclass: `yardstick`'s `hand_till` equals scikit-learn's one-vs-one macro
  average (`"roc_auc_ovo"`), **not** `"roc_auc_ovr"`.
* Scores are used as they are: an implementation never negates a score a
  second time.
* Other metrics (any scikit-learn scorer, any `yardstick` metric with a
  declared direction) are allowed in each language but are outside the
  cross-language guarantee.

## 6. Output schema [decided]

Names are `snake_case` and identical in both languages. Scores are
greater-is-better (§2), so `score_full` for `mse` is `−MSE`.

### 6.1 Summary table — one row per evaluated anchor

| Column | Type | Meaning |
| --- | --- | --- |
| `feature` | label | Feature name. |
| `importance` | double | Mean over splits of `score_full − score_reduced`. |
| `importance_std` | double | Sample standard deviation (denominator `n_splits − 1`) of the per-split importances; `NaN`/`NA` with one split. Descriptive only — not a standard error. |
| `score_full` | double | Mean Full score over splits. |
| `score_reduced` | double | Mean Reduced score over splits. |
| `n_splits` | integer | Number of splits that entered the means. |

Rows follow the order of the features in `X` (or of the `features` argument);
they are not sorted by importance.

### 6.2 Per-split table — one row per (split, anchor)

| Column | Type | Meaning |
| --- | --- | --- |
| `feature` | label | Anchor feature. |
| `split` | integer | Index of the split in generation order, across repeats. **Python: 0-based; R: 1-based.** |
| `n_train` | integer | Size of `T_s`. |
| `n_test` | integer | Size of `A_s`. |
| `score_full` | double | Full score on `A_s`. |
| `score_reduced` | double | Reduced score on `A_s`. |
| `importance` | double | `score_full − score_reduced`. |

Rows are ordered by `split`, then by feature (in the order of the summary).
The `split` label is the one place where the tables differ between languages
(each uses its own index base); cross-language comparisons match rows by
position, not by this label. A language may append columns after these seven
(R: `id`, `id2`, `error`); the common columns must be present, in this order,
with these meanings.

### 6.3 Metadata

Metric name, number of splits, `n`, `p`, all feature names, a description of
the resampling scheme.

## 7. Randomness and reproducibility

* One top-level seed (`random_state` / `seed`) determines the fold assignment
  when folds are generated internally, and the seeds of stochastic learners
  whose own seed is unset. The Full and Reduced model of a pair get the same
  seed.
* Results must not depend on the number of workers. (Python: seeds are fixed
  per (split, feature column) before dispatch, so they also do not depend on
  which subset of anchors is evaluated.)
* A learner seed set by the user and a user-supplied resampling object are
  never modified.
* Equality across languages is required only for deterministic learners on
  explicit splits (§9); random number generators differ.

## 8. Direction diagnostic — not part of v0.1 [decided]

No direction diagnostic belongs to the common API in v0.1. The two packages had
implemented different statistics (Python: Spearman correlation between the
anchor and the Full model's prediction; R: Pearson correlation between the
anchor and `pred_full − pred_reduced`), and neither is the direction
coefficient proposed originally. Until a single definition is agreed,
`shared_importance()` takes no `direction` argument and no result table has a
`direction` column. The Python package keeps an internal, unexported
experimental helper (`sharedimp._direction`) that nothing calls.

Whatever is adopted later must be labelled exploratory: a description of a
fitted predictive model, not a causal effect.

## 9. Cross-language acceptance test

Equality across languages is defined for deterministic learners on explicit
splits with a common metric (§5). Two sets of reference numbers, both produced
by R, are checked by `tests/test_cross_language.py`.

**(a) Parity fixture of the R specification** (R spec §15.1): 12 rows, 3
features, 3 explicit splits, ordinary least squares, `mse`; plus a variant with
an exactly collinear column `x1_dup = 3·x1 − 2`. Every column of both tables
must agree to `1e-8`. The expected numbers are those recorded in the R
specification; Python reproduces them.

**(b) Shared inputs** `tests/fixtures/cross_language/inputs.json`, with the
reference `r_reference.json` generated by `generate_r_reference.R`
(R 4.4.3, `yardstick` 1.4.0, `sharedimp-r` at commit `6c5631d`):

1. the metric values of §5 on fixed predictions (`common_scorer` vs.
   `yardstick`), to `1e-12`;
2. slopes, intercepts and transformed training/assessment data of §3, and the
   set of features flagged by D3, on a design containing an exact duplicate
   and an exact affine copy of a feature, to `1e-10`;
3. per-split and summary results for ordinary least squares (`mse`, `rmse`,
   `mae`, `r2`, `rsq`; `1e-9`), unpenalized binary logistic regression
   (`log_loss`, `accuracy`, `roc_auc`; `1e-6`) and multinomial logistic
   regression (`1e-4`; iterative fits stopped by different optimisers),
   including a single-feature design (§3.2) and the degenerate design of
   item 2.

**Collinear designs need a rank-revealing learner.** When `X` contains two
features that are exactly collinear with each other, their residuals with
respect to any *third* anchor are collinear too (rule D3 only concerns
collinearity with the anchor). The least-squares fit is then not unique, and
predictions agree across implementations only if the learner detects the rank
deficiency. R's `lm` does (pivoted QR, tolerance `1e-7`). scikit-learn's
`LinearRegression` leaves it to LAPACK's default threshold, which is
borderline: on the collinear variant of fixture (a) it reproduced the R
numbers with scikit-learn 1.9.1 / NumPy 2.4 but not with scikit-learn 1.3.2 /
NumPy 1.23 (importance of `x3`: 41.79 instead of 45.55). The Python tests
therefore use an explicit minimum-norm least-squares fit (`rcond = 1e-10`) for
the two collinear designs. This is a property of the learner, not of the
importance definition, but it limits what "same learner" means across
languages.

The R commit behind (b) predates the decisions of §3 and §6: it places the
anchor first, drops D3 columns, reports raw metric values under other column
names and lists per-split rows feature-major. The test therefore compares by
column name, converts R's raw metrics with the metric's direction, reorders
rows, and uses only learners for which the layout differences do not change
any number (they do change results for forests). Once the R revision is
committed, `r_reference.json` must be regenerated and these conversions
removed.

## 10. Language-specific behaviour outside the common core

None of these changes a number in the acceptance test, but the first two
change what a call *without explicit arguments* computes.

| Topic | Python | R |
| --- | --- | --- |
| **Default metric** | `scoring=None` → `estimator.score`: **R²** for regressors, **accuracy** for classifiers (scikit-learn convention) | **`mse`** (regression), **`log_loss`** (classification) |
| **Learner** | any scikit-learn estimator; required argument | `parsnip` specs / `sharedimp_learner()`; default `"rand_forest"` |
| Default splits | 5-fold, shuffled, `KFold`/`StratifiedKFold` | 5-fold, `rsample::vfold_cv()` |
| `split` label | 0-based | 1-based |
| Names for unnamed input | `x0, x1, …` | `V1, V2, …` |
| Class level order | sorted labels (`estimator.classes_`) | factor levels (sorted for character `y`) |
| Learner failure in a split | the exception propagates | `on_error = "warn"` (default) records `NA`, warns and aggregates over the remaining splits; `"error"` aborts |
| Overlapping training/assessment rows | accepted (used for in-sample closed-form tests) | rejected |
| Fit-time weights | `fit_params={"sample_weight": …}` | not available |

**A cross-language comparison must therefore always pass the metric, the
learner and the splits explicitly**, and use `on_error = "error"` in R if a
learner failure is possible.

Python API: `shared_importance(estimator, X, y, *, scoring=None, cv=5,
groups=None, features=None, fit_params=None, n_jobs=None, random_state=None,
verbose=0)`; `AnchorResidualizer(anchor, *, fit_intercept=True,
drop_anchor=False, tol=1e-10)`; `common_scorer(name)`.

## 11. Requests of the R draft to the Python package

R spec §16 (uncommitted revision) lists changes it expects on the Python side.
Status in this repository:

| R spec §16 item | Status |
| --- | --- |
| 1. Remove the direction diagnostic from the API | done (§8) |
| 2. Accept `p = 1` with the null model | done (§3.2) |
| 3. Default scoring `neg_mean_squared_error` / `neg_log_loss` instead of `estimator.score`, *or* document the difference prominently | **open** — documented (§10, README); the default was not changed, because that is an API decision outside the items decided on 2026-11-13 |
| 4. Record a failed (split, feature) fit as `NaN` with a warning, with an option to raise | **open** — Python propagates the exception (§10) |
| 5. Replace this file by a pointer to the R specification | **open** — to be done when the R revision is committed (see the header) |
| 6. Document the scorer ↔ common-name mapping; provide a scorer for `rsq` | done (§5, `common_scorer`) |
| 7. Add the parity fixture of R spec §15.1, tolerance `1e-8` | done (§9a) |
