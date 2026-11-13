"""Smoke, compatibility, reproducibility and error-handling tests for shared_importance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import get_scorer, log_loss, make_scorer, mean_absolute_error
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    LeaveOneGroupOut,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedKFold,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sharedimp import SharedImportanceResult, shared_importance

SUMMARY = ["feature", "importance", "importance_std", "score_full", "score_reduced", "n_splits"]
FOLD = [
    "feature",
    "split",
    "n_train",
    "n_test",
    "score_full",
    "score_reduced",
    "importance",
    "error",
]


# ------------------------------------------------------------------ 7. smoke
def test_regression_smoke(reg_data):
    X, y = reg_data
    res = shared_importance(
        RandomForestRegressor(n_estimators=30, random_state=0),
        X,
        y,
        scoring="neg_root_mean_squared_error",
        cv=3,
        random_state=42,
    )
    assert isinstance(res, SharedImportanceResult)
    assert list(res.importances.columns) == SUMMARY
    assert list(res.fold_results.columns) == FOLD
    assert res.importances.shape == (4, 6)
    assert res.fold_results.shape == (3 * 4, 8)
    assert res.fold_results["error"].isna().all() and res.n_failed == 0
    assert res.features == ["x0", "x1", "x2", "x3"]
    assert res.scoring == "neg_root_mean_squared_error"
    assert (res.n_splits, res.n_samples, res.n_features) == (3, 240, 4)
    assert np.all(np.isfinite(res.importances_mean))
    assert res.importances_mean[0] > 0.2  # informative
    assert res.importances_mean[2] > 0.2  # informative
    assert abs(res.importances_mean[3]) < 0.1  # noise
    assert "SharedImportanceResult" in repr(res)
    top = res.to_frame(sort=True)
    assert top["importance"].is_monotonic_decreasing


def test_binary_classification_probability_scoring(clf_data):
    X, y, _ = clf_data
    res = shared_importance(
        LogisticRegression(), X, y, scoring="neg_log_loss", cv=4, random_state=0
    )
    assert res.importances_mean[0] > 0.05
    assert res.importances_mean[2] > 0.05
    assert abs(res.importances_mean[3]) < 0.02
    assert "StratifiedKFold" in res.cv
    auc = shared_importance(LogisticRegression(), X, y, scoring="roc_auc", cv=4, random_state=0)
    assert auc.importances_mean[0] > 0.02
    assert (auc.fold_results["score_full"].between(0.5, 1.0)).all()


def test_log_loss_scoring_uses_predicted_probabilities(clf_data):
    """neg_log_loss must be computed from predict_proba on the assessment fold."""
    X, y, _ = clf_data
    cv = StratifiedKFold(3)
    res = shared_importance(LogisticRegression(), X, y, scoring="neg_log_loss", cv=cv)
    train, test = next(iter(cv.split(X, y)))
    # anchor 0, split 0, computed by hand
    from sharedimp import AnchorResidualizer

    r = AnchorResidualizer(anchor=0).fit(X[train])
    ftr, fte = r.transform(X[train]), r.transform(X[test])
    full = LogisticRegression().fit(ftr, y[train])
    red = LogisticRegression().fit(ftr[:, 1:], y[train])
    exp_full = -log_loss(y[test], full.predict_proba(fte))
    exp_red = -log_loss(y[test], red.predict_proba(fte[:, 1:]))
    row = res.fold_results.query("split == 0 and feature == 'x0'").iloc[0]
    assert row["score_full"] == pytest.approx(exp_full, rel=1e-10)
    assert row["score_reduced"] == pytest.approx(exp_red, rel=1e-10)
    assert row["importance"] == pytest.approx(exp_full - exp_red, rel=1e-10)


def test_multiclass_classification_smoke(clf_data):
    X, _, y = clf_data
    assert len(np.unique(y)) == 3
    for scoring in ("neg_log_loss", "accuracy", "roc_auc_ovr", "balanced_accuracy"):
        res = shared_importance(
            LogisticRegression(max_iter=500), X, y, scoring=scoring, cv=3, random_state=1
        )
        assert np.all(np.isfinite(res.importances_mean)), scoring
        assert res.importances_mean[0] > 0, scoring
    rf = shared_importance(
        RandomForestClassifier(n_estimators=25),
        X,
        y,
        scoring="neg_log_loss",
        cv=3,
        random_state=1,
    )
    assert np.all(np.isfinite(rf.importances_mean))


def test_string_class_labels_and_decision_function_scorer(clf_data):
    X, y, _ = clf_data
    labels = np.where(y == 1, "case", "control")
    res = shared_importance(SVC(), X, labels, scoring="roc_auc", cv=3, random_state=0)
    assert res.importances_mean[0] > 0.02
    ser = pd.Series(labels, index=np.arange(1000, 1000 + len(labels)), name="status")
    res2 = shared_importance(SVC(), X, ser, scoring="roc_auc", cv=3, random_state=0)
    np.testing.assert_array_equal(res.importances_mean, res2.importances_mean)


def test_default_scoring_is_mse_for_regressors(reg_data):
    X, y = reg_data
    cv = KFold(3)
    default = shared_importance(LinearRegression(), X, y, cv=cv)
    mse = shared_importance(LinearRegression(), X, y, cv=cv, scoring="neg_mean_squared_error")
    assert default.scoring == "neg_mean_squared_error"
    pd.testing.assert_frame_equal(default.fold_results, mse.fold_results)
    assert (default.fold_results["score_full"] < 0).all()  # negated loss
    # ... not the estimator's own score (R^2)
    r2 = shared_importance(LinearRegression(), X, y, cv=cv, scoring="r2")
    assert not np.allclose(default.importances_mean, r2.importances_mean)
    # the sign convention is the same whatever the metric: Full - Reduced
    assert default.importances_mean[0] > 0 and r2.importances_mean[0] > 0


def test_default_scoring_is_log_loss_for_classifiers(clf_data):
    X, yb, ym = clf_data
    for y in (yb, ym):
        cv = StratifiedKFold(3)
        default = shared_importance(LogisticRegression(max_iter=500), X, y, cv=cv)
        sk = shared_importance(
            LogisticRegression(max_iter=500), X, y, cv=cv, scoring="neg_log_loss"
        )
        assert default.scoring == "neg_log_loss"
        np.testing.assert_allclose(
            default.fold_results[["score_full", "score_reduced", "importance"]],
            sk.fold_results[["score_full", "score_reduced", "importance"]],
            rtol=1e-10,
            atol=1e-12,
        )
        assert (default.fold_results["score_full"] < 0).all()
        assert default.importances_mean[0] > 0  # Full - Reduced, positive = anchor helps
    # pipelines are classified by their final step
    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    assert shared_importance(pipe, X, yb, cv=3, random_state=0).scoring == "neg_log_loss"


def test_default_scoring_needs_probabilities_for_classifiers(clf_data):
    X, y, _ = clf_data
    with pytest.raises(TypeError, match="predict_proba"):
        shared_importance(SVC(), X, y, cv=3)
    res = shared_importance(SVC(), X, y, cv=3, scoring="accuracy", random_state=0)
    assert res.scoring == "accuracy"


# ------------------------------------------------------- 8. pandas == NumPy
def test_pandas_and_numpy_agree(reg_data):
    X, y = reg_data
    df = pd.DataFrame(X, columns=["age", "bmi", "dose", "noise"])
    df.index = np.arange(len(df))[::-1]  # non-trivial index must not matter
    est = RandomForestRegressor(n_estimators=20)
    a = shared_importance(est, X, y, scoring="r2", cv=3, random_state=3)
    b = shared_importance(est, df, pd.Series(y, index=df.index), scoring="r2", cv=3, random_state=3)
    assert b.features == ["age", "bmi", "dose", "noise"]
    assert a.features == ["x0", "x1", "x2", "x3"]
    num = SUMMARY[1:]
    np.testing.assert_array_equal(a.importances[num].to_numpy(), b.importances[num].to_numpy())
    np.testing.assert_array_equal(
        a.fold_results[FOLD[1:]].to_numpy(), b.fold_results[FOLD[1:]].to_numpy()
    )


def test_integer_and_bool_dataframe_columns(reg_data):
    X, y = reg_data
    df = pd.DataFrame(
        {"count": (X[:, 0] * 3).round().astype(int), "flag": X[:, 2] > 0, "z": X[:, 3]}
    )
    res = shared_importance(LinearRegression(), df, y, cv=3, random_state=0)
    assert np.all(np.isfinite(res.importances_mean))


def test_pipeline_estimator(reg_data):
    X, y = reg_data
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    res = shared_importance(pipe, X, y, scoring="neg_mean_absolute_error", cv=3, random_state=0)
    assert res.importances_mean[0] > 0.1


# --------------------------------------------- 9. CV splitters / scoring / n_jobs
def test_cv_splitters(reg_data, clf_data):
    X, y = reg_data
    n = len(y)
    groups = np.arange(n) % 6

    res = shared_importance(LinearRegression(), X, y, cv=GroupKFold(3), groups=groups)
    assert res.n_splits == 3
    # manual check that groups are honoured: n_test sums to n over splits
    per_split = res.fold_results.drop_duplicates("split")
    assert per_split["n_test"].sum() == n
    assert (per_split["n_train"] + per_split["n_test"] == n).all()

    res = shared_importance(LinearRegression(), X, y, cv=LeaveOneGroupOut(), groups=groups)
    assert res.n_splits == 6

    res = shared_importance(
        LinearRegression(), X, y, cv=RepeatedKFold(n_splits=3, n_repeats=2, random_state=0)
    )
    assert res.n_splits == 6
    assert res.fold_results.shape[0] == 6 * 4
    assert (res.importances["n_splits"] == 6).all()

    res = shared_importance(
        LinearRegression(), X, y, cv=ShuffleSplit(n_splits=4, test_size=0.25, random_state=0)
    )
    assert res.n_splits == 4

    Xc, yb, _ = clf_data
    res = shared_importance(
        LogisticRegression(),
        Xc,
        yb,
        scoring="neg_brier_score",
        cv=RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=0),
    )
    assert res.n_splits == 6

    with pytest.raises(ValueError, match="groups"):
        shared_importance(LinearRegression(), X, y, cv=GroupKFold(3))


def test_group_kfold_splits_match_sklearn(recorder, reg_data):
    X, y = reg_data
    X, y = X[:36], y[:36]
    groups = np.repeat(np.arange(6), 6)
    cv = GroupKFold(3)
    shared_importance(
        recorder(), X, y, cv=cv, groups=groups, features=[0], scoring="neg_mean_squared_error"
    )
    fit_targets = [rec[2] for rec in recorder.log if rec[0] == "fit"][::2]
    for (train, _), yt in zip(cv.split(X, y, groups), fit_targets, strict=True):
        np.testing.assert_array_equal(yt, y[train])


def test_integer_cv_is_shuffled_and_stratified(clf_data, reg_data):
    X, y, _ = clf_data
    res = shared_importance(LogisticRegression(), X, y, cv=3, random_state=0)
    assert "StratifiedKFold" in res.cv and "shuffle=True" in res.cv
    Xr, yr = reg_data
    res = shared_importance(LinearRegression(), Xr, yr, cv=3, random_state=0)
    assert res.cv.startswith("KFold") and "shuffle=True" in res.cv
    # explicit iterable of splits
    idx = np.arange(len(yr))
    res = shared_importance(
        LinearRegression(), Xr, yr, cv=[(idx[:160], idx[160:]), (idx[80:], idx[:80])]
    )
    assert res.n_splits == 2
    assert list(res.fold_results.drop_duplicates("split")["n_test"]) == [80, 80]


def test_scoring_variants(reg_data):
    X, y = reg_data
    cv = KFold(3)
    by_name = shared_importance(LinearRegression(), X, y, cv=cv, scoring="neg_mean_absolute_error")
    by_obj = shared_importance(
        LinearRegression(), X, y, cv=cv, scoring=get_scorer("neg_mean_absolute_error")
    )
    by_make = shared_importance(
        LinearRegression(),
        X,
        y,
        cv=cv,
        scoring=make_scorer(mean_absolute_error, greater_is_better=False),
    )

    def custom(estimator, X_, y_):
        return -float(np.mean(np.abs(y_ - estimator.predict(X_))))

    by_callable = shared_importance(LinearRegression(), X, y, cv=cv, scoring=custom)
    for other in (by_obj, by_make, by_callable):
        np.testing.assert_allclose(by_name.importances_mean, other.importances_mean, rtol=1e-12)
    assert by_callable.scoring == "custom"


@pytest.mark.parametrize("n_jobs", [2, -1])
def test_parallel_matches_serial(reg_data, n_jobs):
    X, y = reg_data
    kwargs = dict(scoring="neg_root_mean_squared_error", cv=3, random_state=42)
    est = RandomForestRegressor(n_estimators=15)  # unseeded on purpose
    serial = shared_importance(est, X, y, n_jobs=1, **kwargs)
    par = shared_importance(est, X, y, n_jobs=n_jobs, **kwargs)
    pd.testing.assert_frame_equal(serial.importances, par.importances)
    pd.testing.assert_frame_equal(serial.fold_results, par.fold_results)


# ------------------------------------------------------- 10. reproducibility
def test_reproducible_with_random_state(reg_data):
    X, y = reg_data
    est = GradientBoostingRegressor(n_estimators=20, subsample=0.7)
    a = shared_importance(est, X, y, cv=3, random_state=7)
    b = shared_importance(est, X, y, cv=3, random_state=7)
    c = shared_importance(est, X, y, cv=3, random_state=8)
    pd.testing.assert_frame_equal(a.fold_results, b.fold_results)
    assert not np.allclose(a.importances_mean, c.importances_mean)
    # RandomState instances are accepted too
    d = shared_importance(est, X, y, cv=3, random_state=np.random.RandomState(7))
    pd.testing.assert_frame_equal(a.fold_results, d.fold_results)


def test_estimator_is_cloned_not_mutated(reg_data):
    X, y = reg_data
    est = RandomForestRegressor(n_estimators=10)
    params = est.get_params()
    shared_importance(est, X, y, cv=3, random_state=0)
    assert est.get_params() == params  # random_state still None
    assert not hasattr(est, "estimators_")  # never fitted
    assert clone(est).get_params() == params


def test_user_set_estimator_seed_is_respected(recorder, reg_data, monkeypatch):
    from sharedimp import _fitting

    seen = []
    original = _fitting.clone_with_seed

    def spy(estimator, seed):
        est = original(estimator, seed)
        seen.append(est.get_params())
        return est

    monkeypatch.setattr(_fitting, "clone_with_seed", spy)
    X, y = reg_data
    pipe = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=5, random_state=123))
    shared_importance(pipe, X, y, cv=2, random_state=0, features=[0])
    assert all(p["randomforestregressor__random_state"] == 123 for p in seen)

    seen.clear()
    pipe = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=5))
    shared_importance(pipe, X, y, cv=2, random_state=0, features=[0, 1])
    seeds = [p["randomforestregressor__random_state"] for p in seen]
    assert all(isinstance(s, int) for s in seeds)
    # paired: Full and Reduced of the same (split, anchor) share the seed
    assert seeds[0::2] == seeds[1::2]
    assert len(set(seeds[0::2])) == 4  # but tasks differ

    seen.clear()
    shared_importance(pipe, X, y, cv=KFold(2), random_state=None, features=[0])
    assert all(p["randomforestregressor__random_state"] is None for p in seen)


def test_feature_subset_is_consistent_with_full_run(reg_data):
    X, y = reg_data
    df = pd.DataFrame(X, columns=list("abcd"))
    est = RandomForestRegressor(n_estimators=10)
    full = shared_importance(est, df, y, cv=3, random_state=5)
    sub = shared_importance(est, df, y, cv=3, random_state=5, features=["d", "b"])
    assert sub.features == ["d", "b"]
    assert sub.n_features == 4
    ref = full.importances.set_index("feature").loc[["d", "b"]].reset_index()
    pd.testing.assert_frame_equal(sub.importances, ref)
    by_pos = shared_importance(est, df, y, cv=3, random_state=5, features=[3, -3])
    np.testing.assert_array_equal(by_pos.importances_mean, sub.importances_mean)
    single = shared_importance(est, df, y, cv=3, random_state=5, features="b")
    assert single.features == ["b"]


# -------------------------------------------------- 10. negative importance
def test_negative_importance_is_not_truncated():
    rng = np.random.default_rng(0)
    n = 60
    X = rng.normal(size=(n, 6))
    y = X[:, 0] + rng.normal(size=n) * 2.0  # only x0 matters; small n -> overfitting
    res = shared_importance(LinearRegression(), X, y, scoring="neg_mean_squared_error", cv=KFold(5))
    assert (res.importances_mean < 0).any()
    assert (res.fold_results["importance"] < 0).any()
    fr = res.fold_results
    np.testing.assert_allclose(fr["importance"], fr["score_full"] - fr["score_reduced"])
    np.testing.assert_allclose(
        res.importances_mean, res.fold_matrix("importance").mean(axis=1).to_numpy()
    )


def test_constant_model_gives_exactly_zero(reg_data):
    X, y = reg_data
    res = shared_importance(DummyRegressor(), X, y, cv=3, scoring="neg_mean_squared_error")
    np.testing.assert_array_equal(res.importances_mean, np.zeros(4))


# ------------------------------------------------------------ fit_params
def test_fit_params_are_sliced_per_fold(recorder, reg_data):
    X, y = reg_data
    X, y = X[:30], y[:30]
    w = np.arange(30, dtype=float)
    cv = KFold(3)
    shared_importance(
        recorder(),
        X,
        y,
        cv=cv,
        features=[0],
        scoring="neg_mean_squared_error",
        fit_params={"sample_weight": w, "flag": "keep"},
    )
    fits = [rec for rec in recorder.log if rec[0] == "fit"]
    for (train, _), full, red in zip(cv.split(X), fits[0::2], fits[1::2], strict=True):
        for rec in (full, red):
            np.testing.assert_array_equal(rec[3]["sample_weight"], w[train])
            assert rec[3]["flag"] == "keep"


# ------------------------------------------------------------ error handling
def test_error_handling(reg_data):
    X, y = reg_data
    est = LinearRegression()
    with pytest.raises(ValueError, match="at least 1 feature"):
        shared_importance(est, X[:, :0], y)
    with pytest.raises(ValueError, match="2-dimensional"):
        shared_importance(est, X[:, 0], y)
    with pytest.raises(ValueError, match="inconsistent lengths"):
        shared_importance(est, X, y[:-1])
    Xn = X.copy()
    Xn[3, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        shared_importance(est, Xn, y)
    with pytest.raises(ValueError, match="NaN"):
        shared_importance(est, pd.DataFrame(Xn), y)
    df = pd.DataFrame(X, columns=list("abcd"))
    with pytest.raises(ValueError, match="Non-numeric"):
        shared_importance(est, df.assign(e=["u", "v"] * 120), y)
    with pytest.raises(ValueError, match="duplicated"):
        shared_importance(est, pd.DataFrame(X, columns=["a", "a", "b", "c"]), y)
    with pytest.raises(TypeError, match="Sparse"):
        from scipy import sparse

        shared_importance(est, sparse.csr_matrix(X), y)
    with pytest.raises(TypeError, match="instance"):
        shared_importance(LinearRegression, X, y)
    with pytest.raises(TypeError, match="estimator"):
        shared_importance("rf", X, y)
    with pytest.raises(ValueError, match="multi-metric"):
        shared_importance(est, X, y, scoring=["r2", "neg_mean_squared_error"])
    with pytest.raises(ValueError):
        shared_importance(est, X, y, scoring="not_a_scorer")
    with pytest.raises(ValueError, match="at least 2"):
        shared_importance(est, X, y, cv=1)
    with pytest.raises(TypeError):
        shared_importance(est, X, y, cv=True)
    with pytest.raises(ValueError, match="Unknown feature"):
        shared_importance(est, df, y, features=["zzz"])
    with pytest.raises(ValueError, match="out of bounds"):
        shared_importance(est, X, y, features=[9])
    with pytest.raises(ValueError, match="more than once"):
        shared_importance(est, df, y, features=["a", 0])
    with pytest.raises(ValueError, match="empty"):
        shared_importance(est, X, y, features=[])
    with pytest.raises(TypeError, match="direction"):
        # the direction diagnostic is not part of the v0.1 API
        shared_importance(est, X, y, direction=True)
    with pytest.raises(TypeError, match="fit_params"):
        shared_importance(est, X, y, fit_params=[1, 2])
    with pytest.raises(ValueError, match="y is required"):
        shared_importance(est, X, None)


def test_estimator_errors_propagate_with_on_error_error(reg_data):
    X, y = reg_data
    with pytest.raises(ValueError):
        # a classifier cannot be fitted on a continuous target
        shared_importance(
            LogisticRegression(), X, y, cv=KFold(3), scoring="accuracy", on_error="error"
        )
    with pytest.raises(ValueError, match="on_error"):
        shared_importance(LinearRegression(), X, y, on_error="ignore")


def test_fold_matrix_errors(reg_data):
    X, y = reg_data
    res = shared_importance(LinearRegression(), X, y, cv=3, random_state=0)
    with pytest.raises(ValueError):
        res.fold_matrix("direction")
    with pytest.raises(ValueError):
        res.fold_matrix("feature")
    assert res.fold_matrix("score_reduced").shape == (4, 3)
