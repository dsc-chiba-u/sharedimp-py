# sharedimp — provisional common specification

**Status: PROVISIONAL (Python side).**
At the time of writing (2026-11-12) `sharedimp-r/docs/specification.md` does not
exist; the sibling repositories `sharedimp-r` and `sharedimp-experiments` contain
only a README and a LICENSE. This document was therefore written from the Python
side and is intended to become the common specification for both languages once
it has been reconciled with the R package. Items that need an explicit decision
on the R side are collected in [§9](#9-items-to-reconcile-with-the-r-package).
If a common specification later appears in `sharedimp-r`, that document takes
precedence and this file should be reduced to a pointer plus Python-specific
notes.

The specification version is `0.1-provisional`.

## 1. Scope

`sharedimp` computes ultra-marginal feature importance with linear-regression
dependence removal (UMFI-LR) of Janssen, Guan & Robeva (AISTATS 2023) as an
out-of-sample, cross-validated quantity for arbitrary supervised learners.

The package does **not** define a new feature-importance method. Its
contribution is software engineering: leakage-free resampling, integration with
the modelling ecosystem of each language, reproducibility, and tests.

Out of scope for version 0.1: confidence intervals, p-values and any other
formal inference; optimal-transport dependence removal (UMFI-OT);
hyperparameter tuning; non-numeric features; missing values.

## 2. Notation

* `X`: `n × p` numeric feature matrix with columns `x_1, …, x_p`; `p ≥ 2`.
* `y`: target (continuous, binary or multiclass).
* A *split* `s` is a pair of index sets `(T_s, A_s)`: training and assessment
  rows. Splits come from a resampling scheme (k-fold, repeated k-fold, grouped
  k-fold, …).
* `score(model, X, y)`: a performance metric oriented so that **greater is
  better** (losses are negated).

## 3. Algorithm

For every split `s` and every anchor feature `x_i`:

1. **Residualization (training rows only).** For every `j ≠ i`, fit by ordinary
   least squares the simple regression with intercept

   `x_j = a_ij + b_ij · x_i + e`  on rows `T_s`,

   i.e. `b_ij = S_ij / S_ii` and `a_ij = mean(x_j) − b_ij · mean(x_i)` with
   means and (co)sums of squares computed on `T_s`.
2. **Apply to both row sets with the same coefficients.**
   `r_j = x_j − a_ij − b_ij · x_i` for rows in `T_s` and for rows in `A_s`.
   Nothing is ever estimated from `A_s`.
3. **Representations.**
   * *Full*: `x_i` (raw, in its original column position) together with
     `{r_j : j ≠ i}`.
   * *Reduced*: `{r_j : j ≠ i}` only (the anchor column is removed).
4. **Paired fits.** Fit two fresh copies of the same learner, with identical
   fixed hyperparameters and (where applicable) the identical random seed, on
   the Full and Reduced training representations.
5. **Paired evaluation.** Evaluate both on the corresponding assessment
   representations with the same metric:
   `score_full[s,i]`, `score_reduced[s,i]`.
6. **Per-split importance.**
   `importance[s,i] = score_full[s,i] − score_reduced[s,i]`.

**Aggregation.** `importance[i] = mean_s importance[s,i]` (unweighted
arithmetic mean over all splits, including all repeats).

Sign convention: `importance > 0` ⇔ keeping the anchor improves out-of-sample
performance. Negative values are legitimate (noise features, overfitting) and
are **never truncated**, neither per split nor after aggregation. Scores are not
truncated either.

### 3.1 Degenerate cases (deterministic rules)

| Case (within `T_s`) | Rule |
| --- | --- |
| Anchor `x_i` is constant | `b_ij = 0` for all `j` (features are only centered). |
| `x_j` is constant | `b_ij = 0`, `a_ij = mean(x_j)`. |
| `x_j` is an exact affine function of `x_i`: `SS_res ≤ tol · SS_tot`, `tol = 1e-10` | `r_j` is set to exactly `0` for training and assessment rows, so that floating-point round-off of the residual cannot carry the anchor into the Reduced model. |

No random noise is injected in any case; residualization is deterministic.

### 3.2 Input requirements

* All features numeric (integer / logical are converted to double). Categorical
  features must be encoded by the user beforehand.
* No missing or infinite values in `X`.
* `p ≥ 2`, feature names unique.

## 4. Resampling

* Any resampling scheme that yields `(training, assessment)` index pairs is
  admissible, including grouped, stratified and repeated schemes, and
  user-supplied explicit splits.
* Default: 5-fold cross-validation with **shuffled** folds; stratified on `y`
  for classification. The shuffle is controlled by the top-level seed.
* Splits are generated once and shared by all anchors (every anchor is evaluated
  on exactly the same splits).

## 5. Scoring

* One metric per call.
* The metric is used in its "greater is better" orientation exactly once: the
  package never negates a score itself. In Python this is the scikit-learn
  scorer convention (`neg_root_mean_squared_error`, `neg_log_loss`, …).
* Probability-based metrics (log loss, Brier score, AUC) are computed from
  predicted probabilities on the assessment rows.

## 6. Randomness and reproducibility

* A single top-level seed (`random_state` in Python) determines (a) the fold
  assignment when the resampling scheme is given as a number of folds and
  (b) the seeds of stochastic learners whose own seed is unset.
* The two models of a pair (Full, Reduced) receive the same seed.
* Seeds are assigned per (split, feature column) before any work is dispatched;
  results therefore do not depend on the number of workers or on which subset
  of anchors is evaluated.
* A learner seed set explicitly by the user and a user-supplied resampling
  object are never modified.
* Bitwise equality between R and Python is **not** expected for stochastic
  learners or for randomly generated folds (different RNGs). Cross-language
  parity is defined for deterministic learners on explicitly supplied splits
  (see §8).

## 7. Output

Field names are `snake_case` and shared by both languages.

### 7.1 Summary table — one row per evaluated anchor

| Column | Type | Meaning |
| --- | --- | --- |
| `feature` | label | Feature name (Python arrays without names: `x0`, `x1`, …). |
| `importance` | double | Mean over splits of `score_full − score_reduced`. |
| `importance_std` | double | Sample standard deviation (denominator `n_splits − 1`) of the per-split importances; `NaN`/`NA` with one split. Descriptive only — not a standard error. |
| `score_full` | double | Mean Full score over splits. |
| `score_reduced` | double | Mean Reduced score over splits. |
| `n_splits` | integer | Number of splits. |
| `direction` | double | Only when the direction diagnostic is requested (§7.3). |

Row order: order of the features in `X` (or of the user's `features` argument);
not sorted by importance.

### 7.2 Per-split table — one row per (split, anchor)

| Column | Type | Meaning |
| --- | --- | --- |
| `feature` | label | Anchor feature. |
| `split` | integer | Index of the split in generation order (Python: 0-based). |
| `n_train`, `n_test` | integer | Sizes of `T_s` and `A_s`. |
| `score_full`, `score_reduced` | double | Scores on `A_s`. |
| `importance` | double | `score_full − score_reduced`. |
| `direction` | double | Only when requested. |

Rows are ordered by `split`, then by feature.

### 7.3 Optional direction diagnostic

Off by default. When requested, for every (split, anchor):

`direction[s,i] = Spearman rank correlation, over rows in A_s, between x_i and
the Full model's prediction`,

where the prediction is the predicted value (regression) or the predicted
probability of the second class level (binary classification). It is undefined
(`NaN`/`NA`) for multiclass targets, for fewer than three assessment rows, and
when either vector is constant. The summary value is the mean over the splits in
which it is defined.

The diagnostic is **exploratory**. It describes how the fitted predictive model
co-varies with the anchor; it is not a causal effect, not an effect size and
comes with no inferential guarantee. It never influences `importance`.

### 7.4 Metadata

Scorer name, number of splits, `n`, `p`, all feature names, and a textual
description of the resampling scheme.

## 8. Python API (normative for Python, reference for R)

```python
shared_importance(
    estimator, X, y, *,
    scoring=None,        # str | callable | None (estimator.score)
    cv=5,                # int | sklearn splitter | iterable of (train, test)
    groups=None,
    features=None,       # subset of anchors (names or positions)
    direction=False,
    fit_params=None,
    n_jobs=None,
    random_state=None,
    verbose=0,
) -> SharedImportanceResult   # .importances, .fold_results, metadata

AnchorResidualizer(anchor=0, *, fit_intercept=True, drop_anchor=False, tol=1e-10)
# scikit-learn transformer: fit() learns coef_/intercept_, transform() applies them
```

**Cross-language parity fixture (proposed).** Deterministic data, explicit
splits, ordinary least squares, negated mean squared error. Under these
conditions both implementations must agree to about `1e-8` on every column of
both tables. A closed form is available for the in-sample single split
`T = A = all rows`: `importance_i = cov(x_i, y)² / var(x_i)` (population
moments), which follows from the Frisch–Waugh–Lovell theorem and is tested in
`tests/test_core_exact.py`.

## 9. Items to reconcile with the R package

These are decisions taken provisionally on the Python side. Each must be
confirmed or changed jointly; none of them may be changed silently in one
language only.

1. **Unconditional residualization.** The official implementations residualize
   `x_j` only if the regression is "significant" (R: slope p-value `< 0.01`;
   Python: `R² > 0.01`), and the two official gates differ from each other.
   This spec always residualizes (§3). See `relation_to_umfi.md`.
2. **Degenerate-case rules** of §3.1, in particular the exact-collinearity rule
   (`tol = 1e-10`, residual set to exactly 0) instead of the official
   random-noise injection.
3. **Aggregation**: unweighted mean over splits (not pooled predictions, not
   weighted by `n_test`).
4. **`importance_std`**: sample SD with `n_splits − 1`; column name.
5. **Default resampling**: 5-fold, shuffled, stratified for classification.
6. **Split index base**: 0-based in Python; R may prefer 1-based or
   `rsample`-style ids (`Fold1`, `Repeat1`). The parity fixture should compare
   by order, not by label.
7. **Default feature names** for unnamed input (`x0…` in Python; R matrices
   without column names would more naturally give `V1…`).
8. **Direction diagnostic** definition (§7.3): Spearman vs. another statistic,
   the choice of the "positive" class for binary targets (Python: second level
   in sorted order, `estimator.classes_[1]`; R `yardstick` defaults to the
   *first* factor level), and whether a multiclass version is wanted.
9. **Default metric** when none is given (Python: `estimator.score`, i.e. `R²`
   for regressors and accuracy for classifiers).
10. **Seeding policy** of §6 and the fact that cross-language bitwise equality is
    limited to deterministic learners on explicit splits.
11. **Function and argument names** on the R side (`shared_importance()`;
    R analogues of `estimator`/`scoring`/`cv`, e.g. a `parsnip`/`workflows`
    model, a `yardstick` metric and an `rsample` object, or `mlr3` objects).
12. **Fit-time weights** (`fit_params`/`sample_weight` in Python): whether the R
    package supports case weights in v0.1. Metrics are unweighted in Python.
