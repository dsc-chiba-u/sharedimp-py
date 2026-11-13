"""Cross-language parity with the R package and with yardstick.

Reference values in ``tests/fixtures/cross_language/r_reference.json`` were
produced by *running R* (see ``generate_r_reference.R`` in that directory and
the ``meta`` block of the JSON file for versions and the commit of
``sharedimp-r``). They are committed so that these tests need no R
installation. Both languages read the same ``inputs.json``.

What is compared, and why exact agreement is expected:

* metric definitions: ``common_scorer`` vs. yardstick on fixed predictions;
* residualization coefficients and transformed data on explicit rows;
* ``shared_importance`` with deterministic learners (OLS, unpenalized logistic
  regression) on explicit folds.

The R commit used for the reference still placed the anchor first and dropped
exactly collinear columns, whereas the common specification keeps the anchor
in place and keeps collinear columns as zeros. For OLS and for the columns
compared here this does not change any number, so the comparison is valid on
both sides of that change.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import PredefinedSplit

from sharedimp import AnchorResidualizer, common_scorer, shared_importance

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


# ------------------------------------------------------------- residualization
def test_residualization_matches_r():
    im = INPUTS["importance"]
    names = im["degenerate_feature_names"]
    X = np.asarray(im["X_degenerate"])
    folds = np.asarray(im["folds"])
    train, assess = X[folds != 1], X[folds == 1]
    assert len(REF["residualization"]) == len(names)
    for i, ref in enumerate(REF["residualization"]):
        assert ref["anchor"] == names[i]
        r = AnchorResidualizer(anchor=i).fit(train)
        others = [names.index(f) for f in ref["features"]]
        np.testing.assert_allclose(r.coef_[others], ref["slope"], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(r.intercept_[others], ref["intercept"], rtol=1e-10, atol=1e-12)
        # R (reference commit) dropped exactly the columns Python sets to zero.
        assert sorted(np.asarray(names)[r.collinear_]) == sorted(ref["dropped"])
        full_train, full_assess = r.transform(train), r.transform(assess)
        # Compare by column name: the reference commit put the anchor first.
        cols = [names.index(c) for c in ref["full_columns"]]
        np.testing.assert_allclose(full_train[:, cols], np.asarray(ref["full_train"]), atol=1e-10)
        np.testing.assert_allclose(full_assess[:, cols], np.asarray(ref["full_assess"]), atol=1e-10)
        assert np.all(full_train[:, r.collinear_] == 0.0)
        assert np.all(full_assess[:, r.collinear_] == 0.0)


# ------------------------------------------------------------ shared_importance
class _RankRevealingOLS(RegressorMixin, BaseEstimator):
    """OLS with intercept that treats near-zero singular values as zero.

    Needed for designs with exactly collinear columns. There the least-squares
    fit is not unique and predictions agree across implementations only if the
    rank deficiency is detected. R's ``lm`` (pivoted QR, tolerance 1e-7) always
    detects it; ``LinearRegression`` leaves it to LAPACK's default threshold
    (machine epsilon), which is borderline: with scikit-learn 1.3.2 / NumPy
    1.23 it gave different predictions on the collinear variant of the R
    parity fixture (importance 41.79 instead of 45.55), with scikit-learn
    1.9.1 it did not.
    """

    def __init__(self, rcond=1e-10):
        self.rcond = rcond

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.x_mean_ = X.mean(axis=0)
        self.y_mean_ = y.mean()
        self.coef_ = np.linalg.lstsq(X - self.x_mean_, y - self.y_mean_, rcond=self.rcond)[0]
        return self

    def predict(self, X):
        return self.y_mean_ + (np.asarray(X, dtype=float) - self.x_mean_) @ self.coef_


def _logistic():
    # Unpenalized maximum likelihood, converged tightly with a Newton solver
    # (R: glm, i.e. IRLS). L-BFGS in older scikit-learn stops ~1e-5 early.
    return LogisticRegression(C=np.inf, solver="newton-cholesky", tol=1e-12, max_iter=1000)


def _multinomial():
    # Multinomial model (R: nnet::multinom). "newton-cholesky" is one-vs-rest
    # for multiclass targets in older scikit-learn, so L-BFGS is used here.
    return LogisticRegression(C=np.inf, tol=1e-12, max_iter=20000)


CASES = [
    # (reference block, metric key, X key, y key, estimator factory, common metric, rtol/atol)
    ("regression", "mse", "X", "y_regression", LinearRegression, "mse", 1e-9),
    ("regression", "rmse", "X", "y_regression", LinearRegression, "rmse", 1e-9),
    ("regression", "mae", "X", "y_regression", LinearRegression, "mae", 1e-9),
    ("regression", "rsq", "X", "y_regression", LinearRegression, "rsq", 1e-9),
    ("regression", "r2", "X", "y_regression", LinearRegression, "r2", 1e-9),
    (
        "regression_degenerate",
        "mse",
        "X_degenerate",
        "y_regression",
        _RankRevealingOLS,
        "mse",
        1e-8,
    ),
    ("regression_single_feature", "mse", "X1", "y_regression", LinearRegression, "mse", 1e-9),
    ("regression_single_feature", "r2", "X1", "y_regression", LinearRegression, "r2", 1e-9),
    ("binary", "log_loss", "X", "y_binary", _logistic, "log_loss", 1e-6),
    ("binary", "accuracy", "X", "y_binary", _logistic, "accuracy", 1e-12),
    ("binary", "roc_auc", "X", "y_binary", _logistic, "roc_auc", 1e-12),
    ("binary_single_feature", "log_loss", "X1", "y_binary", _logistic, "log_loss", 1e-6),
    ("binary_single_feature", "accuracy", "X1", "y_binary", _logistic, "accuracy", 1e-12),
    # nnet::multinom stops at a looser optimum (BFGS, reltol 1e-8) than glm
    ("multiclass", "log_loss", "X", "y_multiclass", _multinomial, "log_loss", 1e-4),
    ("multiclass", "accuracy", "X", "y_multiclass", _multinomial, "accuracy", 1e-12),
    ("multiclass", "roc_auc", "X", "y_multiclass", _multinomial, "roc_auc", 1e-12),
]


def _design(key):
    im = INPUTS["importance"]
    if key == "X1":
        return pd.DataFrame(np.asarray(im["X"])[:, :1], columns=im["feature_names"][:1])
    names = im["degenerate_feature_names"] if key == "X_degenerate" else im["feature_names"]
    return pd.DataFrame(np.asarray(im[key]), columns=names)


@pytest.mark.parametrize(
    ("block", "key", "x_key", "y_key", "make", "metric", "tol"),
    CASES,
    ids=[f"{c[0]}-{c[1]}" for c in CASES],
)
def test_shared_importance_matches_r(block, key, x_key, y_key, make, metric, tol):
    im = INPUTS["importance"]
    ref = REF["importance"][block][key]
    X = _design(x_key)
    y = np.asarray(im[y_key])
    cv = PredefinedSplit(np.asarray(im["folds"]) - 1)  # same explicit folds as R
    res = shared_importance(make(), X, y, scoring=common_scorer(metric), cv=cv)

    # R reports the raw metric and applies the sign afterwards; the common
    # schema reports greater-is-better scores.
    s = 1.0 if ref["metric_direction"] == "maximize" else -1.0
    # The R reference commit lists per-split rows feature-major with labels
    # "Fold<k>"; Python lists them split-major with a 0-based index. Compare
    # by (feature, split), i.e. reorder the Python rows.
    order = {f: k for k, f in enumerate(res.features)}
    fr = res.fold_results.assign(_pos=lambda d: d["feature"].map(order))
    fr = fr.sort_values(["_pos", "split"], kind="stable")
    assert list(fr["feature"]) == ref["fold_feature"]
    assert [f"Fold{k + 1}" for k in fr["split"]] == ref["fold_id"]
    np.testing.assert_allclose(fr["score_full"], s * np.asarray(ref["fold_metric_full"]), atol=tol)
    np.testing.assert_allclose(
        fr["score_reduced"], s * np.asarray(ref["fold_metric_reduced"]), atol=tol
    )
    np.testing.assert_allclose(fr["importance"], ref["fold_importance"], atol=2 * tol)

    assert res.features == ref["feature"]
    np.testing.assert_allclose(res.importances["importance"], ref["importance"], atol=2 * tol)
    np.testing.assert_allclose(res.importances["importance_std"], ref["sd"], atol=2 * tol)
    np.testing.assert_array_equal(res.importances["n_splits"], ref["n_folds"])


# ------------------------------------- parity fixture of the R specification
# Deterministic fixture defined in sharedimp-r/docs/specification.md, section
# 15.1 (working draft of 2026-11-13). The expected numbers below are the R
# results recorded there (learner = "linear", metric = "mse"); they were
# copied, not recomputed in Python.
R_SPEC_SUMMARY = {
    #        importance         importance_std     score_full           score_reduced
    "x1": (48.3228798721808, 6.17147177856012, -0.276470061784757, -48.5993499339656),
    "x2": (34.4600258422062, 29.5939201894648, -0.276470061784757, -34.7364959039909),
    "x3": (45.5507348310811, 22.4063405773906, -0.276470061784757, -45.8272048928659),
}
R_SPEC_PER_SPLIT = {
    "x1": [53.5652538573545, 49.8820278139317, 41.5213579452562],
    "x2": [61.4017818839248, 39.1937200003847, 2.78457564230898],
    "x3": [36.9725234594595, 70.9787812500000, 28.7008997837838],
}


def _r_spec_fixture():
    i = np.arange(1, 13)
    X = pd.DataFrame(
        {
            "x1": i.astype(float),
            "x2": (i % 4) + 0.5 * i,
            "x3": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0, 10.0, 9.0, 12.0, 11.0],
        }
    )
    noise = np.array([0.3, -0.2, 0.1, 0.4, -0.5, 0.2, -0.1, 0.3, -0.4, 0.1, 0.2, -0.3])
    y = (2 * X["x1"] - X["x2"] + 0.5 * X["x3"] + noise).to_numpy()
    cv = [(np.flatnonzero((i - 1) % 3 != s), np.flatnonzero((i - 1) % 3 == s)) for s in range(3)]
    return X, y, cv


def test_r_specification_parity_fixture():
    X, y, cv = _r_spec_fixture()
    np.testing.assert_allclose(
        y, [1.8, 1.3, 3.6, 7.9, 9.0, 9.7, 11.4, 15.8, 17.1, 17.6, 19.7, 23.2], atol=1e-12
    )
    # None: the default metric (MSE for regressors) is the R default as well
    for scoring in (None, "neg_mean_squared_error", common_scorer("mse")):
        res = shared_importance(LinearRegression(), X, y, scoring=scoring, cv=cv)
        imp = res.importances.set_index("feature")
        for feature, expected in R_SPEC_SUMMARY.items():
            got = imp.loc[feature, ["importance", "importance_std", "score_full", "score_reduced"]]
            np.testing.assert_allclose(got.to_numpy(float), expected, rtol=0, atol=1e-8)
            assert imp.loc[feature, "n_splits"] == 3
            np.testing.assert_allclose(
                res.fold_matrix().loc[feature].to_numpy(), R_SPEC_PER_SPLIT[feature], atol=1e-8
            )
        assert (res.fold_results["n_train"] == 8).all()
        assert (res.fold_results["n_test"] == 4).all()


def test_r_specification_parity_fixture_with_collinear_column():
    """Variant of the fixture with ``x1_dup = 3 * x1 - 2`` (zero-residual rule)."""
    X, y, cv = _r_spec_fixture()
    X["x1_dup"] = 3 * X["x1"] - 2
    # x1 and x1_dup are collinear with each other for every other anchor, so the
    # learner must handle a rank-deficient design (see _RankRevealingOLS).
    res = shared_importance(_RankRevealingOLS(), X, y, scoring="neg_mean_squared_error", cv=cv)
    imp = res.importances.set_index("feature")["importance"]
    for feature, expected in R_SPEC_SUMMARY.items():
        assert imp[feature] == pytest.approx(expected[0], abs=1e-8)
    assert imp["x1_dup"] == pytest.approx(R_SPEC_SUMMARY["x1"][0], abs=1e-8)
