"""Cross-language parity with the R package and with yardstick.

Reference values in ``tests/fixtures/cross_language/r_reference.json`` were
produced by *running R* on a copy of ``sharedimp-r`` at the commit recorded in
the ``meta`` block of that file (see ``generate_r_reference.R`` in the same
directory). They are committed so that these tests need no R installation.
Both languages read the same ``inputs.json``.

The R output is compared **directly**: same column names, same column order of
the representations, same row order of the per-split table, same orientation
of the scores. The single adaptation is the split label, which the common
specification defines as 1-based in R and 0-based in Python.

Tolerance: ``1e-8`` wherever the learner is a closed-form computation (OLS,
null model, class frequencies). Iteratively fitted models are compared at the
accuracy their optimisers reach (``LOGISTIC_TOL``, ``MULTINOMIAL_TOL``); that
is a property of the learners, not of the importance definition.

Rank-deficient designs (two features collinear with each other, a zero
column, a constant anchor) are fitted with ``AliasingOLS``, which resolves the
deficiency by R's ``lm.fit`` convention. ``LinearRegression`` leaves that
decision to library defaults and is not used for those designs; the reasons
and the measurements are in ``docs/rank_deficiency.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import PredefinedSplit

from sharedimp import AnchorResidualizer, common_scorer, shared_importance

TOL = 1e-8  # closed-form learners
LOGISTIC_TOL = 1e-8  # glm.fit (IRLS) vs. Newton-Cholesky: observed 2.5e-9
MULTINOMIAL_TOL = 1e-4  # nnet::multinom (BFGS) vs. L-BFGS: observed 2.9e-5 to 4.0e-5

FIXTURES = Path(__file__).parent / "fixtures" / "cross_language"
INPUTS = json.loads((FIXTURES / "inputs.json").read_text())
REF = json.loads((FIXTURES / "r_reference.json").read_text())


class _FixedRegressor(RegressorMixin, BaseEstimator):
    """Regressor returning fixed predictions, to evaluate a scorer in isolation."""

    def __init__(self, pred=None):
        self.pred = pred

    def predict(self, X):
        return np.asarray(self.pred)


class _FixedClassifier(ClassifierMixin, BaseEstimator):
    """Classifier returning fixed probabilities, to evaluate a scorer in isolation."""

    def __init__(self, proba=None, classes=None):
        self.proba = proba
        self.classes = classes

    @property
    def classes_(self):
        return np.asarray(self.classes)

    def predict_proba(self, X):
        return np.asarray(self.proba)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def _Fixed(pred=None, proba=None, classes=None):
    if proba is None:
        return _FixedRegressor(pred=pred)
    return _FixedClassifier(proba=proba, classes=classes)


def test_reference_metadata_is_recorded():
    meta = REF["meta"]
    assert meta["r_version"].startswith("R version")
    assert len(meta["sharedimp_r_commit"]) == 40
    assert meta["yardstick"]


# ----------------------------------------------------------- metric definitions
@pytest.mark.parametrize(
    ("common", "r_name", "sign"),
    [
        ("mse", "mse", -1),
        ("mse", "sharedimp_mse", -1),
        ("rmse", "rmse", -1),
        ("mae", "mae", -1),
        ("r2", "rsq_trad", 1),
        ("rsq", "rsq", 1),
    ],
)
def test_regression_metrics_match_yardstick(common, r_name, sign):
    d = INPUTS["metrics"]["regression"]
    est = _Fixed(pred=d["estimate"])
    got = common_scorer(common)(est, None, np.asarray(d["truth"]))
    assert got == pytest.approx(sign * REF["metrics"]["regression"][r_name], rel=1e-12)


def test_r2_and_rsq_are_different_metrics():
    ref = REF["metrics"]["regression"]
    assert abs(ref["rsq"] - ref["rsq_trad"]) > 1e-3  # yardstick::rsq is not R^2
    # scikit-learn's "r2" is the traditional R^2, i.e. yardstick::rsq_trad
    from sklearn.metrics import r2_score

    d = INPUTS["metrics"]["regression"]
    assert r2_score(d["truth"], d["estimate"]) == pytest.approx(ref["rsq_trad"], rel=1e-12)


@pytest.mark.parametrize(
    ("common", "r_name", "sign"),
    [
        ("log_loss", "mn_log_loss", -1),
        ("accuracy", "accuracy", 1),
        ("roc_auc", "roc_auc_event_first", 1),
        ("roc_auc", "roc_auc_event_second", 1),
    ],
)
def test_binary_metrics_match_yardstick(common, r_name, sign):
    d = INPUTS["metrics"]["binary"]
    est = _Fixed(proba=np.column_stack([d["prob_a"], d["prob_b"]]), classes=d["levels"])
    got = common_scorer(common)(est, None, np.asarray(d["truth"]))
    assert got == pytest.approx(sign * REF["metrics"]["binary"][r_name], rel=1e-12)


@pytest.mark.parametrize(
    ("common", "r_name", "sign"),
    [
        ("log_loss", "mn_log_loss", -1),
        ("accuracy", "accuracy", 1),
        ("roc_auc", "roc_auc_hand_till", 1),
    ],
)
def test_multiclass_metrics_match_yardstick(common, r_name, sign):
    d = INPUTS["metrics"]["multiclass"]
    est = _Fixed(proba=np.asarray(d["prob"]), classes=d["levels"])
    got = common_scorer(common)(est, None, np.asarray(d["truth"]))
    assert got == pytest.approx(sign * REF["metrics"]["multiclass"][r_name], rel=1e-12)


def test_sklearn_scorer_names_agree_with_common_scorers_on_valid_input():
    """Documented equivalences between scikit-learn names and common names."""
    from sklearn.metrics import get_scorer

    d = INPUTS["metrics"]["multiclass"]
    est = _Fixed(proba=np.asarray(d["prob"]), classes=d["levels"])
    y = np.asarray(d["truth"])
    assert get_scorer("neg_log_loss")(est, None, y) == pytest.approx(
        common_scorer("log_loss")(est, None, y), rel=1e-12
    )
    assert get_scorer("roc_auc_ovo")(est, None, y) == pytest.approx(
        common_scorer("roc_auc")(est, None, y), rel=1e-12
    )
    # one-vs-rest AUC is a different quantity
    assert get_scorer("roc_auc_ovr")(est, None, y) != pytest.approx(
        common_scorer("roc_auc")(est, None, y), rel=1e-6
    )
    r = INPUTS["metrics"]["regression"]
    reg = _Fixed(pred=r["estimate"])
    for sk, common in [
        ("neg_mean_squared_error", "mse"),
        ("neg_root_mean_squared_error", "rmse"),
        ("neg_mean_absolute_error", "mae"),
        ("r2", "r2"),
    ]:
        assert get_scorer(sk)(reg, None, np.asarray(r["truth"])) == pytest.approx(
            common_scorer(common)(reg, None, np.asarray(r["truth"])), rel=1e-12
        )


# ------------------------------------------------------------------- learners
class AliasingOLS(RegressorMixin, BaseEstimator):
    """OLS with intercept that resolves rank deficiency the way R's ``lm.fit`` does.

    Columns of ``[1, X]`` are taken in order; a column whose part orthogonal
    to the columns accepted so far is at most ``tol`` times its own norm is
    *aliased*: its coefficient is fixed at zero (the limited pivoting of
    LINPACK ``dqrdc2`` with ``tol = 1e-7``, plus sharedimp-r setting ``NA``
    coefficients to 0). The remaining columns are fitted by least squares.

    Why not a minimum-norm solution (``numpy.linalg.lstsq``)? On the training
    rows every least-squares solution has the same fitted values. On new rows
    the predictions agree only if those rows satisfy the same linear
    dependency. They do for two features that are collinear everywhere; they
    do not for a column that is constant on the training rows but varies on
    the assessment rows (rule D1: constant anchor), where aliasing ignores the
    column and the minimum-norm solution gives it weight. See
    ``docs/rank_deficiency.md``.

    Test-only: the package itself never replaces the user's estimator.
    """

    def __init__(self, tol=1e-7):
        self.tol = tol

    def fit(self, X, y):
        A = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=float)])
        y = np.asarray(y, dtype=float)
        basis: list[np.ndarray] = []  # orthonormal basis of the accepted columns
        kept: list[int] = []
        for j in range(A.shape[1]):
            v = A[:, j].copy()
            for q in basis:
                v -= (q @ v) * q
            for q in basis:  # second pass: re-orthogonalisation
                v -= (q @ v) * q
            norm = np.linalg.norm(A[:, j])
            if norm > 0 and np.linalg.norm(v) > self.tol * norm:
                basis.append(v / np.linalg.norm(v))
                kept.append(j)
        self.coef_ = np.zeros(A.shape[1])
        self.coef_[kept] = np.linalg.lstsq(A[:, kept], y, rcond=None)[0]
        self.aliased_ = [j - 1 for j in range(1, A.shape[1]) if j not in kept]
        return self

    def predict(self, X):
        return np.column_stack([np.ones(len(X)), np.asarray(X, dtype=float)]) @ self.coef_


class FailsWithoutMarker(AliasingOLS):
    """OLS that refuses to fit unless the marker value is among the training targets."""

    def __init__(self, marker=None, tol=1e-7):
        super().__init__(tol=tol)
        self.marker = marker

    def fit(self, X, y):
        if self.marker not in np.asarray(y):
            raise RuntimeError("marker row is not in the training set")
        return super().fit(X, y)


class NaNWithoutMarker(AliasingOLS):
    """OLS whose predictions are NaN unless the marker was among the training targets."""

    def __init__(self, marker=None, tol=1e-7):
        super().__init__(tol=tol)
        self.marker = marker

    def fit(self, X, y):
        self.ok_ = self.marker in np.asarray(y)
        return super().fit(X, y)

    def predict(self, X):
        pred = super().predict(X)
        return pred if self.ok_ else np.full_like(pred, np.nan)


def _logistic():
    # Unpenalized maximum likelihood with a Newton solver (R: glm.fit, IRLS).
    return LogisticRegression(C=np.inf, solver="newton-cholesky", tol=1e-12, max_iter=1000)


def _multinomial():
    # Multinomial model (R: nnet::multinom). "newton-cholesky" is one-vs-rest
    # for multiclass targets in older scikit-learn, so L-BFGS is used here.
    return LogisticRegression(C=np.inf, tol=1e-12, max_iter=20000)


# --------------------------------------------------------------------- designs
IM = INPUTS["importance"]
FOLDS = np.asarray(IM["folds"])
TRAIN, ASSESS = FOLDS != 1, FOLDS == 1  # the split used for the representation block


def _frame(key):
    names = {
        "X": IM["feature_names"],
        "X_degenerate": IM["degenerate_feature_names"],
        "X_constant": IM["constant_feature_names"],
    }
    if key == "X1":
        return pd.DataFrame(np.asarray(IM["X"])[:, :1], columns=IM["feature_names"][:1])
    if key in ("spec", "spec_collinear"):
        sf = INPUTS["spec_fixture"]
        if key == "spec":
            return pd.DataFrame(sf["X"], columns=sf["feature_names"])
        return pd.DataFrame(sf["X_collinear"], columns=sf["collinear_feature_names"])
    return pd.DataFrame(np.asarray(IM[key]), columns=names[key])


# ---------------- 1-3, 7: coefficients, representations, predictions, collinearity
@pytest.mark.parametrize(
    ("block", "x_key"),
    [("regular", "X"), ("degenerate", "X_degenerate"), ("constant", "X_constant")],
)
def test_representations_and_predictions_match_r(block, x_key):
    X = _frame(x_key)
    y = np.asarray(IM["y_regression"])
    names = list(X.columns)
    refs = REF["representations"][block]
    assert [r["anchor"] for r in refs] == names
    for i, ref in enumerate(refs):
        others = [n for n in names if n != names[i]]
        assert ref["features"] == others

        # 1. residualization coefficients and degeneracy flags
        r = AnchorResidualizer(anchor=names[i]).fit(X[TRAIN])
        keep = np.arange(len(names)) != i
        np.testing.assert_allclose(r.coef_[keep], ref["slope"], rtol=0, atol=TOL)
        np.testing.assert_allclose(r.intercept_[keep], ref["intercept"], rtol=0, atol=TOL)
        assert list(r.collinear_[keep]) == ref["collinear"]  # 7. exact collinearity (D3)
        assert r.constant_anchor_ is ref["constant_anchor"]  # D1

        # 2. Full / Reduced matrices: same columns, same order, same numbers
        full_train, full_assess = r.transform(X[TRAIN]), r.transform(X[ASSESS])
        reduced = AnchorResidualizer(anchor=names[i], drop_anchor=True).fit(X[TRAIN])
        reduced_train, reduced_assess = reduced.transform(X[TRAIN]), reduced.transform(X[ASSESS])
        assert list(full_train.columns) == ref["full_columns"] == names
        assert list(reduced_train.columns) == ref["reduced_columns"] == others
        for got, key in [
            (full_train, "full_train"),
            (full_assess, "full_assess"),
            (reduced_train, "reduced_train"),
            (reduced_assess, "reduced_assess"),
        ]:
            np.testing.assert_allclose(got.to_numpy(), np.asarray(ref[key]), rtol=0, atol=TOL)
        # collinear residuals are exactly zero on both sides
        flagged = [n for n, c in zip(others, ref["collinear"], strict=True) if c]
        assert (full_train[flagged].to_numpy() == 0).all()
        assert (np.asarray(ref["full_assess"])[:, [names.index(n) for n in flagged]] == 0).all()

        # 3. predictions of the Full and the Reduced OLS model
        pred_full = AliasingOLS().fit(full_train, y[TRAIN]).predict(full_assess)
        pred_reduced = AliasingOLS().fit(reduced_train, y[TRAIN]).predict(reduced_assess)
        np.testing.assert_allclose(pred_full, ref["pred_full"], rtol=0, atol=TOL)
        np.testing.assert_allclose(pred_reduced, ref["pred_reduced"], rtol=0, atol=TOL)


def test_constant_design_exercises_d1_and_d2():
    """Sanity check of the fixture: the R reference really contains D1 and D2 cases."""
    refs = REF["representations"]["constant"]
    assert [r["constant_anchor"] for r in refs] == [False, False, False, False, True]
    assert refs[4]["slope"] == [0, 0, 0, 0]  # D1: slopes are zero
    assert refs[0]["slope"][-1] == 0 and refs[0]["intercept"][-1] == 2.5  # D2
    assert sum(sum(r["collinear"]) for r in REF["representations"]["degenerate"]) > 0  # D3
    assert sum(sum(r["collinear"]) for r in REF["representations"]["regular"]) == 0


def test_null_model_matches_r():
    """8. p = 1: the Reduced model is the null model."""
    from sharedimp._fitting import fit_null_model

    ref = REF["null_model"]
    empty = np.empty((int(ASSESS.sum()), 0))
    y = np.asarray(IM["y_regression"])
    reg = fit_null_model(LinearRegression(), int(TRAIN.sum()), y[TRAIN])
    np.testing.assert_allclose(reg.predict(empty), ref["regression_prediction"], atol=TOL)
    yb = np.asarray(IM["y_binary"])
    clf = fit_null_model(LogisticRegression(), int(TRAIN.sum()), yb[TRAIN])
    assert list(clf.classes_) == ref["binary_levels"]
    np.testing.assert_allclose(clf.predict_proba(empty), ref["binary_probabilities"], atol=TOL)


# ----------------------------------- 4-6, 8: per-split scores, summary, defaults
def _assert_tables_equal(res, ref, tol):
    """Compare both result tables with the R tables, column by column."""
    # per-split table: same row order (split-major), same columns
    fr, rf = res.fold_results, ref["folds"]
    assert list(fr["feature"]) == rf["feature"]
    # the split label is 1-based in R and 0-based in Python (specification 11.2)
    assert list(fr["split"] + 1) == rf["split"]
    assert list(fr["n_train"]) == rf["n_train"]
    assert list(fr["n_test"]) == rf["n_test"]
    for col in ("score_full", "score_reduced", "importance"):
        expected = np.array([np.nan if v is None else v for v in rf[col]], dtype=float)
        np.testing.assert_allclose(fr[col].to_numpy(float), expected, rtol=0, atol=tol, err_msg=col)
    assert list(fr["error"].notna()) == rf["failed"]

    # summary table
    imp, rs = res.importances, ref["summary"]
    assert list(imp["feature"]) == rs["feature"]
    assert list(imp["n_splits"]) == rs["n_splits"]
    for col in ("importance", "importance_std", "score_full", "score_reduced"):
        expected = np.array([np.nan if v is None else v for v in rs[col]], dtype=float)
        np.testing.assert_allclose(
            imp[col].to_numpy(float), expected, rtol=0, atol=tol, err_msg=col
        )


CASES = [
    # (reference block, case, X, y, estimator factory, tolerance)
    ("regression", "default", "X", "y_regression", LinearRegression, TOL),
    ("regression", "mse", "X", "y_regression", LinearRegression, TOL),
    ("regression", "rmse", "X", "y_regression", LinearRegression, TOL),
    ("regression", "mae", "X", "y_regression", LinearRegression, TOL),
    ("regression", "r2", "X", "y_regression", LinearRegression, TOL),
    ("regression", "rsq", "X", "y_regression", LinearRegression, TOL),
    ("regression_degenerate", "mse", "X_degenerate", "y_regression", AliasingOLS, TOL),
    ("regression_single_feature", "default", "X1", "y_regression", LinearRegression, TOL),
    ("regression_single_feature", "r2", "X1", "y_regression", LinearRegression, TOL),
    ("binary", "default", "X", "y_binary", _logistic, LOGISTIC_TOL),
    ("binary", "log_loss", "X", "y_binary", _logistic, LOGISTIC_TOL),
    ("binary", "accuracy", "X", "y_binary", _logistic, TOL),
    ("binary", "roc_auc", "X", "y_binary", _logistic, TOL),
    ("binary_single_feature", "default", "X1", "y_binary", _logistic, LOGISTIC_TOL),
    ("binary_single_feature", "accuracy", "X1", "y_binary", _logistic, TOL),
    ("multiclass", "default", "X", "y_multiclass", _multinomial, MULTINOMIAL_TOL),
    ("multiclass", "accuracy", "X", "y_multiclass", _multinomial, TOL),
    ("multiclass", "roc_auc", "X", "y_multiclass", _multinomial, TOL),
    ("spec_fixture", "default", "spec", "spec_y", LinearRegression, TOL),
    ("spec_fixture_collinear", "default", "spec_collinear", "spec_y", AliasingOLS, TOL),
]


def _target_and_cv(y_key):
    if y_key == "spec_y":
        sf = INPUTS["spec_fixture"]
        return np.asarray(sf["y"]), PredefinedSplit(np.asarray(sf["folds"]) - 1)
    return np.asarray(IM[y_key]), PredefinedSplit(FOLDS - 1)


@pytest.mark.parametrize(
    ("block", "case", "x_key", "y_key", "make", "tol"),
    CASES,
    ids=[f"{c[0]}-{c[1]}" for c in CASES],
)
def test_shared_importance_matches_r(block, case, x_key, y_key, make, tol):
    ref = REF["importance"][block][case]
    y, cv = _target_and_cv(y_key)
    # "default": neither language is told which metric to use
    scoring = None if case == "default" else common_scorer(ref["metric"])
    res = shared_importance(make(), _frame(x_key), y, scoring=scoring, cv=cv, on_error="error")
    if case == "default":
        assert (res.scoring, ref["metric"]) in {
            ("neg_mean_squared_error", "mse"),
            ("neg_log_loss", "log_loss"),
        }
    assert res.n_failed == 0 and ref["n_warnings"] == 0
    _assert_tables_equal(res, ref, tol)


def test_default_metric_equals_the_named_common_metric_in_r_too():
    for block, named in [("regression", "mse"), ("binary", "log_loss")]:
        a, b = REF["importance"][block]["default"], REF["importance"][block][named]
        assert a["summary"] == b["summary"] and a["folds"] == b["folds"]


def test_constant_anchor_matches_r():
    """D1/D2 end to end: one warning naming the constant anchor on both sides."""
    ref = REF["importance"]["regression_constant"]["mse"]
    y, cv = _target_and_cv("y_regression")
    with pytest.warns(UserWarning, match="constant on at least one training set") as rec:
        res = shared_importance(
            AliasingOLS(), _frame("X_constant"), y, scoring=common_scorer("mse"), cv=cv
        )
    assert len(rec) == ref["n_warnings"] == 1 and "ctrain" in str(rec[0].message)
    _assert_tables_equal(res, ref, TOL)


def test_collinear_copy_gets_the_importance_of_its_original():
    """7. Exact collinearity end to end, in the R numbers themselves."""
    for block, twin, original in [
        ("regression_degenerate", "dup", "x0"),
        ("regression_degenerate", "affine", "x0"),
        ("spec_fixture_collinear", "x1_dup", "x1"),
    ]:
        case = next(iter(REF["importance"][block].values()))
        imp = dict(zip(case["summary"]["feature"], case["summary"]["importance"], strict=True))
        assert imp[twin] == pytest.approx(imp[original], abs=TOL)


# ------------------------------------------------------- 9. learner failures
def _failure_estimators():
    marker = IM["marker"]
    return {
        "fit_error": (FailsWithoutMarker(marker=marker), "y_marker", "mse", "marker row"),
        "non_finite_predictions": (
            NaNWithoutMarker(marker=marker),
            "y_marker",
            "mse",
            "non-finite",
        ),
        "single_class": (DummyClassifier(strategy="prior"), "y_single_class", None, "single class"),
    }


@pytest.mark.parametrize("kind", ["fit_error", "non_finite_predictions", "single_class"])
def test_learner_failures_match_r(kind):
    estimator, y_key, metric, message = _failure_estimators()[kind]
    ref = REF["failures"][kind]
    y, cv = _target_and_cv(y_key)
    scoring = None if metric is None else common_scorer(metric)

    with pytest.warns(UserWarning, match="fits failed") as rec:
        res = shared_importance(estimator, _frame("X"), y, scoring=scoring, cv=cv)
    assert len(rec) == ref["n_warnings"] == 1  # one summary warning in both languages
    _assert_tables_equal(res, ref, TOL)  # NaN rows, scores of the other splits, n_splits

    failed = res.fold_results["error"].notna()
    assert failed.sum() == sum(ref["folds"]["failed"]) == 4  # split 1 (R) / 0 (Python)
    assert res.fold_results.loc[failed, "error"].str.contains(message).all()
    assert all(
        message.split()[0] in e
        for e, f in zip(ref["folds"]["error"], ref["folds"]["failed"], strict=True)
        if f
    )
    assert list(res.importances["n_splits"]) == [3, 3, 3, 3]

    # on_error = "error" aborts in both languages
    assert REF["failures"]["on_error_error_aborts"][kind] is True
    with pytest.raises((RuntimeError, ValueError), match=message):
        shared_importance(estimator, _frame("X"), y, scoring=scoring, cv=cv, on_error="error")
