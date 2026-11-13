"""Tests of the conventions fixed by the common R/Python specification.

Small deterministic data; every expected value is computed independently.
Section numbers refer to the common specification
(``sharedimp-r/docs/specification.md``).
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeRegressor

import sharedimp
from sharedimp import COMMON_METRICS, AnchorResidualizer, common_scorer, shared_importance
from sharedimp._direction import direction_diagnostic
from sharedimp._fitting import fit_null_model

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


@pytest.fixture
def small():
    """12 rows, 3 features, exactly representable values."""
    a = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
    b = np.array([1.0, 0.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0, 8.0, 11.0, 10.0])
    c = np.array([2.0, -1.0, 0.5, 3.0, -2.0, 1.0, 0.0, 2.5, -1.5, 1.0, 0.5, -0.5])
    X = np.column_stack([a, b, c])
    y = (
        2.0 * a
        - 1.0 * c
        + np.array([0.1, -0.2, 0.0, 0.3, -0.1, 0.2, -0.3, 0.1, 0.0, -0.1, 0.2, -0.2])
    )
    return X, y


# --------------------------------------------- §6 column order of Full / Reduced
@pytest.mark.parametrize("anchor", [0, 1, 2])
def test_full_keeps_anchor_in_original_position(recorder, small, anchor):
    X, y = small
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    shared_importance(
        recorder(), df, y, cv=KFold(2), features=[anchor], scoring="neg_mean_squared_error"
    )
    full, reduced = recorder.log[0][1], recorder.log[2][1]
    assert list(full.columns) == ["a", "b", "c"]  # original order, anchor in place
    expected_reduced = [c for k, c in enumerate(["a", "b", "c"]) if k != anchor]
    assert list(reduced.columns) == expected_reduced
    train = next(iter(KFold(2).split(X)))[0]
    np.testing.assert_array_equal(full.iloc[:, anchor].to_numpy(), X[train, anchor])  # raw anchor
    np.testing.assert_array_equal(reduced.to_numpy(), np.delete(full.to_numpy(), anchor, axis=1))


# ------------------------------------------------------- §5.1 degenerate cases
def test_collinear_columns_are_kept_as_exact_zeros(recorder, small):
    X, y = small
    Xd = np.column_stack([X, X[:, 0], 5.0 - 3.0 * X[:, 0]])  # duplicate and affine copy
    shared_importance(
        recorder(), Xd, y, cv=KFold(3), features=[0], scoring="neg_mean_squared_error"
    )
    for kind, mat in [(rec[0], rec[1]) for rec in recorder.log]:
        width = mat.shape[1]
        assert width in (5, 4), kind  # p and p - 1: nothing is dropped
        dup, aff = (3, 4) if width == 5 else (2, 3)
        assert np.all(mat[:, dup] == 0.0) and np.all(mat[:, aff] == 0.0)


@pytest.mark.parametrize(
    ("scale", "shift"),
    [(1e-12, 0.0), (1.0, 0.0), (1e12, 0.0), (1e-12, 1e-9), (1.0, 1e3), (1e12, -1e15)],
)
def test_collinearity_rule_is_invariant_to_units(small, scale, shift):
    """The rule compares SS_res with SS_tot, so units and offsets do not matter.

    (Offsets: as long as the feature keeps enough significant digits, i.e.
    |mean| / sd is far below 1 / (eps * sqrt(tol)); see the specification.)
    """
    X, _ = small
    a = X[:, 0]
    rng = np.random.default_rng(0)
    near = a + 1e-3 * rng.normal(size=a.size)  # R^2 ~ 1 - 1e-7: NOT exactly collinear
    Z = np.column_stack([a, scale * a + shift, scale * near + shift, scale * X[:, 2] + shift])
    r = AnchorResidualizer(anchor=0).fit(Z)
    np.testing.assert_array_equal(r.collinear_, [False, True, False, False])
    # ... and to the units of the anchor
    Z2 = Z.copy()
    Z2[:, 0] = 1e9 * a - 7.0
    np.testing.assert_array_equal(
        AnchorResidualizer(anchor=0).fit(Z2).collinear_, [False, True, False, False]
    )


def test_collinearity_tolerance_boundary():
    """Flagged iff 1 - R^2 <= 1e-10 (the tolerance fixed by the specification)."""
    a = np.arange(200.0)
    e = np.where(np.arange(200) % 2 == 0, 1.0, -1.0)
    e = e - e.mean()
    e = e - (e @ (a - a.mean())) / ((a - a.mean()) @ (a - a.mean())) * (a - a.mean())  # e ⟂ a
    ss_a, ss_e = (a - a.mean()) @ (a - a.mean()), e @ e
    for target, expected in [(1e-11, True), (1e-9, False)]:
        eps = np.sqrt(target * ss_a / ss_e)  # 1 - R^2 = target (to first order)
        r = AnchorResidualizer(anchor=0).fit(np.column_stack([a, a + eps * e]))
        assert bool(r.collinear_[1]) is expected


def test_constant_anchor_is_detected_exactly():
    """max == min, not a comparison with a (rounded) mean: 0.1 * 3 != 0.3."""
    anchor = np.full(7, 0.1)
    assert anchor.mean() != 0.1  # the mean of a constant is not always that constant
    other = np.array([1.0, 4.0, 2.0, 8.0, 5.0, 7.0, 3.0])
    r = AnchorResidualizer(anchor=0).fit(np.column_stack([anchor, other]))
    assert r.constant_anchor_
    assert r.coef_[1] == 0.0
    new = np.array([[0.1, 4.0], [5.0, 4.0], [-3.0, 6.0]])
    # slope 0: the residual does not depend on the anchor at all
    np.testing.assert_allclose(r.transform(new)[:, 1], new[:, 1] - other.mean(), atol=1e-15)


def test_constant_anchor_warns_once(small):
    X, y = small
    Xc = X.copy()
    Xc[:, 1] = 3.0
    with pytest.warns(UserWarning, match="constant on at least one training set") as rec:
        res = shared_importance(LinearRegression(), Xc, y, cv=KFold(3), scoring="r2")
    assert len([w for w in rec if "constant" in str(w.message)]) == 1
    assert "x1" in str(rec[0].message)
    assert res.importances["importance"].iloc[1] == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------- §3, §8.3 single feature
def test_single_feature_regression_uses_training_mean(small):
    X, y = small
    X1 = X[:, [0]]
    cv = KFold(3)
    res = shared_importance(LinearRegression(), X1, y, scoring="neg_mean_squared_error", cv=cv)
    assert res.importances.shape == (1, 6)
    assert (res.n_features, res.features) == (1, ["x0"])
    for k, (train, test) in enumerate(cv.split(X1)):
        row = res.fold_results.iloc[k]
        assert row["split"] == k
        null_mse = np.mean((y[test] - y[train].mean()) ** 2)  # Reduced = training mean
        A = np.column_stack([np.ones(train.size), X1[train, 0]])
        beta = np.linalg.lstsq(A, y[train], rcond=None)[0]
        full_mse = np.mean((y[test] - (beta[0] + beta[1] * X1[test, 0])) ** 2)
        assert row["score_reduced"] == pytest.approx(-null_mse, rel=1e-12)
        assert row["score_full"] == pytest.approx(-full_mse, rel=1e-10)
        assert row["importance"] == pytest.approx(null_mse - full_mse, rel=1e-10)
    assert res.importances["importance"].iloc[0] > 0


def test_single_feature_classification_uses_training_class_frequencies():
    x = np.array([-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, -0.2, 0.2, -1.2, 1.2])
    y = np.array(["n", "n", "n", "p", "n", "p", "p", "p", "n", "p", "n", "p"])
    cv = StratifiedKFold(3)
    res = shared_importance(LogisticRegression(), x[:, None], y, scoring="neg_log_loss", cv=cv)
    for k, (train, test) in enumerate(cv.split(x[:, None], y)):
        freq_p = np.mean(y[train] == "p")
        p_true = np.where(y[test] == "p", freq_p, 1.0 - freq_p)
        assert res.fold_results["score_reduced"].iloc[k] == pytest.approx(
            np.mean(np.log(p_true)), rel=1e-12
        )
    assert res.importances["importance"].iloc[0] > 0
    acc = shared_importance(LogisticRegression(), x[:, None], y, scoring="accuracy", cv=cv)
    assert np.all(acc.fold_results["score_reduced"] <= 0.5)


def test_single_feature_dataframe_and_null_model_details(small):
    X, y = small
    df = pd.DataFrame({"only": X[:, 0]})
    res = shared_importance(DecisionTreeRegressor(max_depth=2), df, y, cv=KFold(2), random_state=0)
    assert res.features == ["only"]
    # null model: class frequencies in sorted class order, ties -> first class
    null = fit_null_model(LogisticRegression(), 4, np.array(["b", "a", "b", "a"]))
    np.testing.assert_array_equal(null.predict_proba(np.empty((2, 0))), [[0.5, 0.5]] * 2)
    assert list(null.predict(np.empty((2, 0)))) == ["a", "a"]
    # weighted fit -> weighted mean
    null = fit_null_model(
        LinearRegression(), 3, np.array([1.0, 2.0, 6.0]), fit_params={"sample_weight": [1, 1, 2]}
    )
    assert null.predict(np.empty((1, 0)))[0] == pytest.approx(3.75)


# -------------------------------------------------------------- §9 sign, metrics
def test_importance_sign_for_losses_and_gains(small):
    X, y = small
    cv = KFold(3)
    for scoring in ("neg_mean_squared_error", common_scorer("mse"), common_scorer("rmse"), "r2"):
        res = shared_importance(LinearRegression(), X, y, scoring=scoring, cv=cv)
        fr = res.fold_results
        np.testing.assert_allclose(fr["importance"], fr["score_full"] - fr["score_reduced"])
        assert res.importances["importance"].iloc[0] > 0  # the informative anchor helps
    loss = shared_importance(LinearRegression(), X, y, scoring=common_scorer("mse"), cv=cv)
    assert (loss.fold_results["score_full"] <= 0).all()  # losses are reported negated


def test_common_scorer_names_and_errors():
    assert set(COMMON_METRICS) == {
        "mse", "rmse", "mae", "r2", "rsq", "log_loss", "accuracy", "roc_auc",
    }  # fmt: skip
    assert repr(common_scorer("rsq")) == "common_scorer('rsq')"
    with pytest.raises(ValueError, match="Unknown common metric"):
        common_scorer("rsq_trad")
    with pytest.raises(ValueError, match="Unknown common metric"):
        common_scorer(None)


def test_common_scorer_names_are_reported(small):
    X, y = small
    res = shared_importance(LinearRegression(), X, y, scoring=common_scorer("mae"), cv=KFold(2))
    assert res.scoring == "mae"


def test_common_scorers_equal_sklearn_scorers_in_shared_importance(small):
    X, y = small
    cv = KFold(3)
    for sk, common in [
        ("neg_mean_squared_error", "mse"),
        ("neg_root_mean_squared_error", "rmse"),
        ("neg_mean_absolute_error", "mae"),
        ("r2", "r2"),
    ]:
        a = shared_importance(LinearRegression(), X, y, scoring=sk, cv=cv)
        b = shared_importance(LinearRegression(), X, y, scoring=common_scorer(common), cv=cv)
        np.testing.assert_allclose(
            a.fold_results["importance"], b.fold_results["importance"], rtol=1e-10, atol=1e-12
        )


def test_rsq_is_undefined_for_constant_predictions(small):
    X, y = small
    # Reduced model of a single feature is constant -> squared correlation undefined
    res = shared_importance(
        LinearRegression(), X[:, [0]], y, scoring=common_scorer("rsq"), cv=KFold(2)
    )
    assert res.fold_results["score_reduced"].isna().all()
    assert res.fold_results["error"].isna().all()  # undefined metric, not a learner failure
    # no split contributes: the summary is NaN and says so (n_splits == 0)
    assert res.importances["importance"].isna().all()
    assert (res.importances["n_splits"] == 0).all()


def test_classification_scorers_align_classes_missing_from_training():
    """A class absent from the training rows gets probability 0 (clipped in log loss)."""
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y_train = np.array(["a", "a", "b", "b", "a", "b"])
    model = LogisticRegression().fit(X, y_train)
    y_test = np.array(["a", "c"])  # "c" was never seen
    proba = model.predict_proba(X[:2])
    eps = np.finfo(float).eps
    expected = np.mean([np.log(proba[0, 0]), np.log(eps)])
    assert common_scorer("log_loss")(model, X[:2], y_test) == pytest.approx(expected, rel=1e-12)
    assert common_scorer("accuracy")(model, X[:2], y_test) in (0.0, 0.5)

    class NoProba:
        classes_ = np.array(["a", "b"])

        def predict(self, X):
            return np.array(["a"] * len(X))

    with pytest.raises(TypeError, match="predict_proba"):
        common_scorer("log_loss")(NoProba(), X[:2], np.array(["a", "b"]))


# ------------------------------------------------------------ §11 output schema
def test_output_schema(small):
    X, y = small
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    res = shared_importance(LinearRegression(), df, y, scoring=common_scorer("mse"), cv=KFold(3))

    imp = res.importances
    assert list(imp.columns) == SUMMARY
    assert list(imp["feature"]) == ["a", "b", "c"]  # input order, not sorted
    assert all(imp[c].dtype == np.float64 for c in SUMMARY[1:5])
    assert imp["n_splits"].dtype.kind == "i" and (imp["n_splits"] == 3).all()

    fr = res.fold_results
    assert list(fr.columns) == FOLD
    # split-major, feature-minor; split is 0-based in Python (1-based in R)
    assert list(fr["split"]) == [0] * 3 + [1] * 3 + [2] * 3
    assert list(fr["feature"]) == ["a", "b", "c"] * 3
    assert (fr["n_train"] + fr["n_test"] == 12).all()
    assert all(fr[c].dtype.kind == "i" for c in ("split", "n_train", "n_test"))
    assert fr.columns[-1] == "error" and fr["error"].isna().all()  # appended after the common ones

    # the summary is the aggregation of the per-split table
    g = fr.groupby("feature", sort=False)
    np.testing.assert_allclose(imp["importance"], g["importance"].mean())
    np.testing.assert_allclose(imp["importance_std"], g["importance"].std(ddof=1))
    np.testing.assert_allclose(imp["score_full"], g["score_full"].mean())
    np.testing.assert_allclose(imp["score_reduced"], g["score_reduced"].mean())

    wide = res.fold_matrix()
    assert list(wide.index) == ["a", "b", "c"] and list(wide.columns) == [0, 1, 2]
    np.testing.assert_array_equal(wide.loc["b"].to_numpy(), fr["importance"].to_numpy()[1::3])


def test_schema_is_the_same_for_classification_and_feature_subsets(clf_data):
    X, yb, ym = clf_data
    for y in (yb, ym):
        res = shared_importance(
            LogisticRegression(max_iter=500),
            X,
            y,
            scoring=common_scorer("log_loss"),
            cv=3,
            random_state=0,
            features=[2, 0],
        )
        assert list(res.importances.columns) == SUMMARY
        assert list(res.fold_results.columns) == FOLD
        assert list(res.fold_results["feature"]) == ["x2", "x0"] * 3
        assert res.importances["importance"].iloc[1] > 0


# ------------------------------------------------- direction is not in the API
def test_direction_is_not_part_of_the_public_api(small):
    X, y = small
    assert "direction" not in inspect.signature(shared_importance).parameters
    assert not any("direction" in name.lower() for name in sharedimp.__all__)
    res = shared_importance(LinearRegression(), X, y, cv=KFold(2))
    assert "direction" not in res.importances.columns
    assert "direction" not in res.fold_results.columns
    with pytest.raises(TypeError):
        shared_importance(LinearRegression(), X, y, cv=KFold(2), direction=True)


def test_experimental_direction_helper_still_works():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert direction_diagnostic(a, np.array([0.1, 0.4, 0.5, 0.9])) == pytest.approx(1.0)
    assert direction_diagnostic(a, -a) == pytest.approx(-1.0)
    assert np.isnan(direction_diagnostic(a, np.ones(4)))
    assert np.isnan(direction_diagnostic(a, None))


# ------------------------------------------------- parallel == serial, exactly
@pytest.mark.parametrize(
    ("estimator", "scoring", "target"),
    [
        (RandomForestRegressor(n_estimators=10), "neg_mean_squared_error", "reg"),
        (RandomForestClassifier(n_estimators=10), "neg_log_loss", "clf"),
    ],
)
def test_parallel_and_serial_are_identical(reg_data, clf_data, estimator, scoring, target):
    X, y = reg_data if target == "reg" else clf_data[:2]
    kwargs = dict(scoring=scoring, cv=3, random_state=11)
    serial = shared_importance(estimator, X, y, n_jobs=1, **kwargs)
    parallel = shared_importance(estimator, X, y, n_jobs=2, **kwargs)
    again = shared_importance(estimator, X, y, n_jobs=1, **kwargs)
    pd.testing.assert_frame_equal(serial.fold_results, parallel.fold_results)
    pd.testing.assert_frame_equal(serial.importances, parallel.importances)
    pd.testing.assert_frame_equal(serial.fold_results, again.fold_results)
