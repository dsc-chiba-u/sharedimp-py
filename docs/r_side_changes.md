# Items raised with `sharedimp-r` — all resolved

Written from `sharedimp-py`; fifth and closing revision, 2026-11-13, against
`sharedimp-r` commit `cf2e1cc` (clean working tree). Nothing in that
repository was modified from here; its files were read and its code was
executed from a copy (`git archive`). "Spec §" refers to the common
specification `sharedimp-r/docs/specification.md`; "Python notes §" to
[`specification.md`](specification.md) in this repository.

**There is no open item.** This file is kept as the record of what was raised
and where it was resolved.

## Resolved

| Raised | Item | Resolved in R commit | Checked |
| --- | --- | --- | --- |
| rev. 1–2 | Anchor in its original column position; exactly collinear columns kept as zeros with the `SS_res ≤ 1e-10 · S_jj` rule; constants tested by `max == min`; common column names with greater-is-better scores; split-major rows; `r2` and `rsq` as separate metrics; no direction diagnostic | `977e07f` … `86d8c1d` | by running R at `92f9ecb` and comparing its output directly (Python notes §6) |
| rev. 3 | Ten corrections to the specification text (status of the Python package, log-loss clipping, `accuracy` on argmax labels, `rsq`, the learner needed for the collinear variant of §15.1, Docker manual runs) | `92f9ecb` | read |
| rev. 4, item 1 | Spec §8.1 / §15.4 stated that every least-squares solver which recognises a rank deficiency gives the same predictions. Not true when the assessment rows leave the training dependency (anchor constant on the training rows) | `cf2e1cc`: §8.1 now distinguishes assessment rows that satisfy the dependency from rows that leave it, states that the coefficients are not unique and that recognising the deficiency is necessary but not sufficient to reproduce `lm.fit`; §15.4 cites the Python-side measurements | read |
| rev. 4, item 2 | Spec §8.1 named `numpy.linalg.lstsq(rcond = 1e-7)` / an SVD minimum-norm fit as the Python counterpart of `lm.fit` | `cf2e1cc`: §8.1 now requires a learner that reproduces R's aliasing convention, names the test-only `AliasingOLS`, and states that `LinearRegression`, `numpy.linalg.lstsq` and SVD minimum-norm fits are not counterparts of `lm.fit` on every rank-deficient design | read |
| rev. 4, note | The reference generator relies on `residualize_fit()$collinear` and on the warning class `sharedimp_constant_anchor` | `cf2e1cc`: recorded in spec §16 item 7 | read |

`cf2e1cc` changes documentation only: `git diff 92f9ecb cf2e1cc -- R tests
DESCRIPTION NAMESPACE` is empty. The R reference values of the Python tests,
generated from `92f9ecb`, therefore still correspond to the current R code.

## What the R side says about the Python package

Spec §16 and `docs/python_changes.md` at `cf2e1cc` list every item requested of
the Python package as done, with the implementing commits, and record two
deviations as accepted (fit-time weights in Python only; overlapping
training/assessment rows accepted in Python only). Both are in Python notes
§4.4.

## Keeping the two repositories in step

`tests/fixtures/cross_language/r_reference.json` records the R commit it was
generated from (`meta.sharedimp_r_commit`). Whenever the R package changes
something that affects numbers or the result tables, regenerate it (procedure:
`CLAUDE.md`, "Regenerating the R reference") and run the Python tests. An R
test that reads the same `inputs.json` would close the loop in the other
direction; none exists yet.
