"""Exact (deterministic) checks of the statistical quantity computed by shared_importance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import KFold

from sharedimp import AnchorResidualizer, shared_importance


def _ols_predict(A_train, y_train, A_test):
    A1 = np.column_stack([np.ones(len(A_train)), A_train])
    beta, *_ = np.linalg.lstsq(A1, y_train, rcond=None)
    return np.column_stack([np.ones(len(A_test)), A_test]) @ beta


# ------------------------------------------- 2 & 3. what the estimator receives
def test_estimator_receives_fold_internal_representations(recorder, reg_data):
    X, y = reg_data
    X, y = X[:30], y[:30]
    cv = KFold(n_splits=3)
    splits = list(cv.split(X))
    shared_importance(recorder(), X, y, cv=cv, scoring="neg_mean_squared_error")

    log = recorder.log
    p = X.shape[1]
    # per (split, anchor) and model: fit, predict (finiteness check), predict (scoring)
    assert len(log) == len(splits) * p * 6
    k = 0
    # tasks run split-major, feature-minor
    for train, test in splits:
        for i in range(p):
            fit_full, pred_full, pred_full2, fit_red, pred_red, pred_red2 = log[k : k + 6]
            k += 6
            np.testing.assert_array_equal(pred_full[1], pred_full2[1])
            np.testing.assert_array_equal(pred_red[1], pred_red2[1])
            assert [fit_full[0], pred_full[0], fit_red[0], pred_red[0]] == [
                "fit",
                "predict",
                "fit",
                "predict",
            ]
            # manual, training-rows-only residualization
            a_tr, a_te = X[train, i], X[test, i]
            exp_tr, exp_te = X[train].copy(), X[test].copy()
            for j in range(p):
                if j == i:
                    continue
                slope = np.cov(a_tr, X[train, j], bias=True)[0, 1] / a_tr.var()
                icpt = X[train, j].mean() - slope * a_tr.mean()
                exp_tr[:, j] = X[train, j] - icpt - slope * a_tr
                exp_te[:, j] = X[test, j] - icpt - slope * a_te
            np.testing.assert_allclose(fit_full[1], exp_tr, atol=1e-12)
            np.testing.assert_allclose(pred_full[1], exp_te, atol=1e-12)
            np.testing.assert_allclose(fit_red[1], np.delete(exp_tr, i, axis=1), atol=1e-12)
            np.testing.assert_allclose(pred_red[1], np.delete(exp_te, i, axis=1), atol=1e-12)
            # Full keeps the raw anchor in place; Reduced has one column fewer
            np.testing.assert_array_equal(fit_full[1][:, i], a_tr)
            assert fit_red[1].shape[1] == p - 1
            # paired: same targets for both models
            np.testing.assert_array_equal(fit_full[2], y[train])
            np.testing.assert_array_equal(fit_red[2], y[train])


def test_assessment_rows_do_not_influence_training(recorder, reg_data):
    """Changing assessment-fold X must not change anything the models are trained on."""
    X, y = reg_data
    X, y = X[:24].copy(), y[:24]
    train, test = np.arange(16), np.arange(16, 24)
    cv = [(train, test)]

    shared_importance(recorder(), X, y, cv=cv, scoring="neg_mean_squared_error")
    fits_a = [rec[1].copy() for rec in recorder.log if rec[0] == "fit"]

    recorder.log = []
    X2 = X.copy()
    X2[test] = X2[test] * 100.0 - 7.0
    shared_importance(recorder(), X2, y, cv=cv, scoring="neg_mean_squared_error")
    fits_b = [rec[1] for rec in recorder.log if rec[0] == "fit"]

    assert len(fits_a) == len(fits_b) == 2 * X.shape[1]
    for a, b in zip(fits_a, fits_b, strict=True):
        np.testing.assert_array_equal(a, b)


# --------------------------------------- 4. linear regression + MSE, closed form
def test_linear_mse_matches_direct_ols_out_of_sample(reg_data):
    X, y = reg_data
    cv = KFold(n_splits=4, shuffle=True, random_state=7)
    res = shared_importance(LinearRegression(), X, y, scoring="neg_mean_squared_error", cv=cv)

    p = X.shape[1]
    expected = np.zeros((4, p))
    full_mse = np.zeros(4)
    for s, (train, test) in enumerate(cv.split(X)):
        # Full model: the representation is an invertible affine map of X, so OLS
        # predictions equal those of OLS on the raw features.
        full_mse[s] = np.mean((y[test] - _ols_predict(X[train], y[train], X[test])) ** 2)
        for i in range(p):
            r = AnchorResidualizer(anchor=i, drop_anchor=True).fit(X[train])
            red_pred = _ols_predict(r.transform(X[train]), y[train], r.transform(X[test]))
            expected[s, i] = np.mean((y[test] - red_pred) ** 2) - full_mse[s]

    got = res.fold_matrix("importance").to_numpy().T  # (split, feature)
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(res.importances_mean, expected.mean(axis=0), rtol=1e-9)
    np.testing.assert_allclose(res.importances_std, expected.std(axis=0, ddof=1), rtol=1e-9)
    # score_full is the (negated) raw-feature OLS test MSE for every anchor
    np.testing.assert_allclose(
        res.fold_matrix("score_full").to_numpy().T, -np.tile(full_mse[:, None], (1, p)), rtol=1e-9
    )


def test_linear_mse_in_sample_closed_form(reg_data):
    """With train == assessment and OLS, importance_i = cov(x_i, y)^2 / var(x_i).

    Derivation (Frisch-Waugh): the residualized features are orthogonal to the
    centered anchor, so adding the anchor reduces the residual sum of squares by
    exactly (x_i_c . y)^2 / (x_i_c . x_i_c).
    """
    X, y = reg_data
    n = len(y)
    idx = np.arange(n)
    res = shared_importance(
        LinearRegression(), X, y, scoring="neg_mean_squared_error", cv=[(idx, idx)]
    )
    yc = y - y.mean()
    for i in range(X.shape[1]):
        xc = X[:, i] - X[:, i].mean()
        expected = (xc @ yc) ** 2 / (xc @ xc) / n
        assert res.importances_mean[i] == pytest.approx(expected, rel=1e-9, abs=1e-12)
    assert np.all(np.isnan(res.importances_std))  # single split
    assert res.n_splits == 1


# --------------------------------------- 5. exact duplicates and independence
def test_exact_duplicate_features_in_sample():
    rng = np.random.default_rng(99)
    n = 120
    a = rng.normal(size=n)
    z = rng.normal(size=n)
    X = np.column_stack([a, a.copy(), z])
    y = 2.0 * a + z + 0.3 * rng.normal(size=n)
    idx = np.arange(n)
    res = shared_importance(
        LinearRegression(), X, y, scoring="neg_mean_squared_error", cv=[(idx, idx)]
    )
    imp = res.importances_mean
    yc = y - y.mean()
    ac = a - a.mean()
    marginal = (ac @ yc) ** 2 / (ac @ ac) / n
    # Both copies get the full (shared) marginal importance, not half and not zero.
    assert imp[0] == pytest.approx(marginal, rel=1e-8)
    assert imp[1] == pytest.approx(marginal, rel=1e-8)
    assert imp[0] == pytest.approx(imp[1], rel=1e-10)


def test_exact_duplicate_does_not_leak_into_reduced_model(recorder):
    rng = np.random.default_rng(5)
    a = rng.normal(size=40)
    X = np.column_stack([a, a.copy(), rng.normal(size=40)])
    y = a + rng.normal(size=40)
    shared_importance(recorder(), X, y, cv=KFold(4), features=[0], scoring="neg_mean_squared_error")
    reduced = [rec[1] for rec in recorder.log if rec[1].shape[1] == 2]
    assert len(reduced) == 12  # 4 fits + 2 x 4 predicts
    for mat in reduced:
        assert np.all(mat[:, 0] == 0.0)  # residual of the duplicate: exactly zero


def test_independent_orthogonal_predictors_in_sample():
    """Orthogonal design, noiseless linear y: importance_i = beta_i^2 * var(x_i)."""
    rng = np.random.default_rng(11)
    n, p = 64, 4
    Z = rng.normal(size=(n, p))
    Z -= Z.mean(axis=0)
    Q, _ = np.linalg.qr(Z)  # orthonormal, and still orthogonal to the constant
    X = Q * np.array([1.0, 2.0, 3.0, 4.0]) * np.sqrt(n)  # var(x_i) = scale_i^2
    beta = np.array([1.0, -0.5, 0.0, 0.25])
    y = X @ beta + 3.0
    idx = np.arange(n)
    res = shared_importance(
        LinearRegression(), X, y, scoring="neg_mean_squared_error", cv=[(idx, idx)]
    )
    expected = beta**2 * X.var(axis=0)
    np.testing.assert_allclose(res.importances_mean, expected, rtol=1e-8, atol=1e-10)
    # the null feature has (numerically) zero importance
    assert abs(res.importances_mean[2]) < 1e-10


def test_independent_predictors_out_of_sample_ranking():
    rng = np.random.default_rng(21)
    n = 2000
    X = rng.normal(size=(n, 3))
    y = 2.0 * X[:, 0] + 1.0 * X[:, 1] + rng.normal(size=n)
    res = shared_importance(
        LinearRegression(), X, y, scoring="neg_mean_squared_error", cv=5, random_state=0
    )
    imp = res.importances_mean
    # population values: beta_i^2 * var(x_i) = 4, 1, 0
    assert imp[0] == pytest.approx(4.0, rel=0.15)
    assert imp[1] == pytest.approx(1.0, rel=0.2)
    assert abs(imp[2]) < 0.02


# ------------------------------------------------------------ scorer sign
def test_scorer_sign_is_not_double_flipped(reg_data):
    X, y = reg_data
    cv = KFold(n_splits=3)
    neg = shared_importance(LinearRegression(), X, y, scoring="neg_mean_squared_error", cv=cv)
    # A greater-is-worse metric wrapped by make_scorer is the same scorer.
    wrapped = shared_importance(
        LinearRegression(),
        X,
        y,
        scoring=make_scorer(mean_squared_error, greater_is_better=False),
        cv=cv,
    )
    np.testing.assert_allclose(neg.importances_mean, wrapped.importances_mean, rtol=1e-12)
    assert (neg.fold_results["score_full"] <= 0).all()  # scores are negated losses
    # informative features: Full has the lower loss => positive importance
    assert neg.importances_mean[0] > 0.5
    assert neg.importances_mean[2] > 0.5
    fr = neg.fold_results
    np.testing.assert_allclose(fr["importance"], fr["score_full"] - fr["score_reduced"])
    # r2 is natively greater-is-better: same sign of importance
    r2 = shared_importance(LinearRegression(), X, y, scoring="r2", cv=cv)
    assert r2.importances_mean[0] > 0
    assert np.all(np.sign(r2.importances_mean[[0, 2]]) == np.sign(neg.importances_mean[[0, 2]]))


def test_dataframe_passes_named_columns_to_estimator(recorder, reg_data):
    X, y = reg_data
    df = pd.DataFrame(X[:20], columns=["a", "b", "c", "d"], index=np.arange(100, 120))
    shared_importance(
        recorder(), df, y[:20], cv=KFold(2), features=["b"], scoring="neg_mean_squared_error"
    )
    mats = [rec[1] for rec in recorder.log]
    assert all(isinstance(m, pd.DataFrame) for m in mats)
    assert list(mats[0].columns) == ["a", "b", "c", "d"]  # full
    assert list(mats[3].columns) == ["a", "c", "d"]  # reduced
    assert set(mats[0].index) <= set(range(100, 120))  # original index is preserved
