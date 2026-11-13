# Open items for `sharedimp-r`

Written from `sharedimp-py` on 2026-11-13 (fourth revision), against
`sharedimp-r` commit `92f9ecb` (clean working tree). Nothing in that
repository was modified; its files were read and its code was executed from a
copy (`git archive`). "Spec §" refers to the common specification
`sharedimp-r/docs/specification.md`; "Python notes §" to
[`specification.md`](specification.md) in this repository.

## A. Implementation

**No change is requested.** Running the R package at `92f9ecb` and comparing
its output directly with the Python package gave agreement on every item
listed in Python notes §6: residualization, Full/Reduced matrices and their
column order, predictions, both result tables, the default metric, exact
collinearity, `p = 1`, and learner failures (failing fit, non-finite
predictions, single-class training set).

The ten text corrections of the previous revision of this file were all
applied in `92f9ecb`. The deviation recorded there (non-finite predictions not
inspected in Python) no longer exists: the Python package now inspects the
assessment predictions and records non-finite ones as a failure, as R does.

## B. Text of the common specification

Two statements about rank-deficient designs, found while regenerating the
reference (record: [`rank_deficiency.md`](rank_deficiency.md)):

| # | Spec | Issue |
| --- | --- | --- |
| 1 | §8.1 ("every least-squares solver that recognises the deficiency gives the same predictions whatever its coefficient convention"), §15.4, `docs/parity/rank_deficiency.md` | True for the fitted values, and for new rows that satisfy the same linear dependency (two features collinear everywhere). **Not true when the assessment rows leave the dependency**: an anchor that is constant on the training rows (rule D1) but varies on the assessment rows. `lm.fit` aliases the anchor column (coefficient 0); a minimum-norm solution gives it weight, so its assessment predictions depend on the anchor. Measured on the shared `X_constant` design: `numpy.linalg.lstsq` (`rcond=None` and `1e-7`) and `scipy.linalg.lstsq` differ from R by `1.5e-2` in the Full score of that cell, on both dependency stacks, although they recognise the rank deficiency. |
| 2 | §8.1 ("Python a least-squares solve with an explicit relative tolerance (e.g. `numpy.linalg.lstsq(rcond = 1e-7)` or an SVD minimum-norm fit …)") | Follows from item 1: a minimum-norm fit is not a counterpart of `lm.fit` on every rank-deficient design. The Python parity tests use a learner that reproduces R's convention instead (`AliasingOLS` in `tests/test_cross_language.py`: columns of `[1, X]` in order, a column whose component orthogonal to the accepted columns is at most `1e-7` of its norm gets coefficient 0). With it all rank-deficient designs agree with R to `1e-8` on both stacks. Also measured: `scipy.linalg.lstsq` with its default cut-off is wrong in one cell of the collinear variant of §15.1 on *both* stacks (scikit-learn 1.9.1's `LinearRegression` is not), so "SciPy `lstsq` in scikit-learn 1.9.1" should not be read as "SciPy `lstsq`". |

Suggested wording for §8.1: predictions on new rows are determined by the
least-squares problem only where those rows satisfy the dependencies of the
training design; otherwise they depend on the solver's convention, and a
parity test must reproduce R's (aliasing in column order) rather than merely
recognise the rank deficiency.

Minor, for information: the R reference script needed two details of the R
package that are not in the specification — `residualize_fit()` returns
`collinear` as a logical vector over the non-anchor features, and the
constant-anchor condition is signalled as a warning of class
`sharedimp_constant_anchor`. Both are relied upon by
`tests/fixtures/cross_language/generate_r_reference.R`; changing them requires
adapting that script.

## C. Keeping the two repositories in step

`tests/fixtures/cross_language/r_reference.json` records the R commit it was
generated from (`meta.sharedimp_r_commit`). Whenever the R package changes
something that affects numbers or the result tables, regenerate it (procedure:
`CLAUDE.md`, "Regenerating the R reference") and run the Python tests. An R
test that reads the same `inputs.json` would close the loop in the other
direction; none exists yet.
