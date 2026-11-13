"""Learner failures (common specification, "Learner failures").

A failed fit for one (split, feature) pair is recorded as NaN with its error
message, the run continues, one warning is emitted, and each feature is
aggregated over its successful splits. ``on_error="error"`` stops instead.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold

from sharedimp import shared_importance

MARKER = 12345.0


class FailsWithoutMarker(RegressorMixin, BaseEstimator):
    """OLS that refuses to fit unless the marker value is among the training targets."""

    def __init__(self, only_reduced: bool = False, n_features: int = 3):
        self.only_reduced = only_reduced
        self.n_features = n_features

    def fit(self, X, y):
        X = np.asarray(X)
        is_reduced = X.shape[1] == self.n_features - 1
        if MARKER not in np.asarray(y) and (is_reduced or not self.only_reduced):
            raise RuntimeError("marker row is not in the training set")
        self.model_ = LinearRegression().fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X))


@pytest.fixture
def data():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(30, 3))
    y = X[:, 0] - X[:, 2] + 0.1 * rng.normal(size=30)
    y[4] = MARKER  # row 4 is assessed in split 0 of KFold(3) -> split 0 fails
    return X, y


def test_failed_split_is_recorded_and_the_rest_is_aggregated(data):
    X, y = data
    cv = KFold(3)
    with pytest.warns(UserWarning, match=r"3 of 9 \(split, feature\) fits failed") as rec:
        res = shared_importance(FailsWithoutMarker(), X, y, cv=cv, scoring="neg_mean_squared_error")
    assert sum("fits failed" in str(w.message) for w in rec) == 1  # one summary warning
    assert "marker row is not in the training set" in str(rec[0].message)

    fr = res.fold_results
    failed = fr["split"] == 0
    assert failed.sum() == 3 and res.n_failed == 3
    # failed rows: NaN scores and importance, error message kept
    assert fr.loc[failed, ["score_full", "score_reduced", "importance"]].isna().all().all()
    assert (fr.loc[failed, "error"] == "RuntimeError: marker row is not in the training set").all()
    # the other rows are complete and carry no error
    assert fr.loc[~failed, ["score_full", "score_reduced", "importance"]].notna().all().all()
    assert fr.loc[~failed, "error"].isna().all()
    assert (fr["n_train"] == 20).all() and (fr["n_test"] == 10).all()  # sizes always reported

    # aggregation over the successful splits only, and the count says so
    imp = res.importances
    assert (imp["n_splits"] == 2).all()
    ok = fr.loc[~failed]
    g = ok.groupby("feature", sort=False)
    np.testing.assert_allclose(imp["importance"], g["importance"].mean())
    np.testing.assert_allclose(imp["importance_std"], g["importance"].std(ddof=1))
    np.testing.assert_allclose(imp["score_full"], g["score_full"].mean())
    np.testing.assert_allclose(imp["score_reduced"], g["score_reduced"].mean())

    # the successful splits are exactly what a run restricted to them gives
    splits = list(cv.split(X))[1:]
    ref = shared_importance(FailsWithoutMarker(), X, y, cv=splits, scoring="neg_mean_squared_error")
    np.testing.assert_allclose(imp["importance"], ref.importances["importance"], rtol=1e-12)


def test_failure_of_one_model_of_a_pair_discards_the_pair(data):
    X, y = data
    with pytest.warns(UserWarning, match="fits failed"):
        res = shared_importance(
            FailsWithoutMarker(only_reduced=True),
            X,
            y,
            cv=KFold(3),
            scoring="neg_mean_squared_error",
        )
    failed = res.fold_results[res.fold_results["split"] == 0]
    # the Full model was fitted, but a row never mixes a score with a missing one
    assert failed[["score_full", "score_reduced", "importance"]].isna().all().all()
    assert failed["error"].notna().all()


def test_on_error_error_stops_at_the_first_failure(data):
    X, y = data
    with pytest.raises(RuntimeError, match="marker row is not in the training set"):
        shared_importance(
            FailsWithoutMarker(), X, y, cv=KFold(3), scoring="neg_mean_squared_error",
            on_error="error",
        )  # fmt: skip


def test_no_failure_means_no_warning_and_full_counts(data):
    X, y = data
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = shared_importance(LinearRegression(), X, y, cv=KFold(3))
    assert res.n_failed == 0
    assert (res.importances["n_splits"] == 3).all()
    assert res.fold_results["error"].isna().all()


def test_all_splits_failing_gives_nan_summary_with_zero_count(data):
    X, y = data
    with pytest.warns(UserWarning, match=r"9 of 9"):
        # a classifier cannot be fitted on a continuous target: every fit fails
        res = shared_importance(LogisticRegression(), X, y, cv=KFold(3), scoring="accuracy")
    imp = res.importances
    assert (imp["n_splits"] == 0).all()
    assert imp[["importance", "importance_std", "score_full", "score_reduced"]].isna().all().all()
    assert list(imp["feature"]) == ["x0", "x1", "x2"]  # the table keeps its shape
    assert res.fold_results.shape == (9, 8)


def test_single_class_training_set_is_a_failure():
    """Even for a learner that would accept it (a forest predicts the one class)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(24, 2))
    y = np.array(["a"] * 8 + ["b"] * 16)
    idx = np.arange(24)
    cv = [
        (idx[8:], idx[:8]),  # training rows are all "b"
        (np.r_[idx[:4], idx[8:16]], np.r_[idx[4:8], idx[16:]]),
        (np.r_[idx[4:8], idx[16:]], np.r_[idx[:4], idx[8:16]]),
    ]
    with pytest.warns(UserWarning, match=r"2 of 6"):
        res = shared_importance(RandomForestClassifier(n_estimators=5), X, y, cv=cv, random_state=0)
    fr = res.fold_results
    assert (fr.loc[fr["split"] == 0, "error"].str.contains("single class")).all()
    assert fr.loc[fr["split"] > 0, "error"].isna().all()
    assert (res.importances["n_splits"] == 2).all()
    with pytest.raises(ValueError, match="single class"):
        shared_importance(
            RandomForestClassifier(n_estimators=5), X, y, cv=cv, random_state=0, on_error="error"
        )


def test_failures_are_identical_in_parallel(data):
    X, y = data
    kwargs = dict(cv=KFold(3), scoring="neg_mean_squared_error")
    with pytest.warns(UserWarning, match="fits failed"):
        serial = shared_importance(FailsWithoutMarker(), X, y, n_jobs=1, **kwargs)
    with pytest.warns(UserWarning, match="fits failed"):
        parallel = shared_importance(FailsWithoutMarker(), X, y, n_jobs=2, **kwargs)
    pd.testing.assert_frame_equal(serial.fold_results, parallel.fold_results)
    pd.testing.assert_frame_equal(serial.importances, parallel.importances)


def test_fold_matrix_shows_failed_splits_as_nan(data):
    X, y = data
    with pytest.warns(UserWarning):
        res = shared_importance(FailsWithoutMarker(), X, y, cv=KFold(3), scoring="r2")
    wide = res.fold_matrix()
    assert wide[0].isna().all() and wide[[1, 2]].notna().all().all()


# ------------------------------------------------------ non-finite predictions
class PredictsNonFinite(RegressorMixin, BaseEstimator):
    """OLS whose predictions contain a non-finite value when the marker was not in training."""

    def __init__(self, value: float = np.nan):
        self.value = value

    def fit(self, X, y):
        self.poison_ = MARKER not in np.asarray(y)
        self.model_ = LinearRegression().fit(np.asarray(X), y)
        return self

    def predict(self, X):
        pred = self.model_.predict(np.asarray(X))
        if self.poison_:
            pred[0] = self.value  # a single bad prediction is enough
        return pred


def _silent_scorer(estimator, X, y):
    """A scorer that turns non-finite predictions into NaN without raising."""
    with np.errstate(invalid="ignore"):
        return -float(np.mean((np.asarray(y) - estimator.predict(X)) ** 2))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("scoring", ["silent", "common", "sklearn"])
def test_non_finite_predictions_are_recorded_as_failures(data, value, scoring):
    """Whatever the scorer does with them: raise (scikit-learn) or return NaN/inf."""
    from sharedimp import common_scorer

    X, y = data
    scorer = {
        "silent": _silent_scorer,
        "common": common_scorer("mse"),
        "sklearn": "neg_mean_squared_error",
    }[scoring]
    with pytest.warns(UserWarning, match=r"3 of 9 \(split, feature\) fits failed"):
        res = shared_importance(PredictsNonFinite(value), X, y, cv=KFold(3), scoring=scorer)
    fr = res.fold_results
    failed = fr["split"] == 0
    # the error column is filled: the failure is not a silent NaN
    assert (
        fr.loc[failed, "error"] == "ValueError: the learner returned non-finite predictions"
    ).all()
    assert fr.loc[failed, ["score_full", "score_reduced", "importance"]].isna().all().all()
    assert fr.loc[~failed, "error"].isna().all()
    assert np.isfinite(fr.loc[~failed, "importance"]).all()
    assert (res.importances["n_splits"] == 2).all() and res.n_failed == 3
    assert np.isfinite(res.importances["importance"]).all()  # inf never reaches the summary
    with pytest.raises(ValueError, match="non-finite predictions"):
        shared_importance(
            PredictsNonFinite(value), X, y, cv=KFold(3), scoring=scorer, on_error="error"
        )


def test_non_finite_class_probabilities_are_failures():
    from sklearn.base import ClassifierMixin

    class NaNProba(ClassifierMixin, BaseEstimator):
        def fit(self, X, y):
            self.classes_ = np.unique(y)
            return self

        def predict_proba(self, X):
            return np.full((len(X), len(self.classes_)), np.nan)

        def predict(self, X):
            return np.repeat(self.classes_[0], len(X))

    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 2))
    y = np.array(["a", "b"] * 15)
    with pytest.warns(UserWarning, match=r"6 of 6"):
        # accuracy only needs labels, but the probabilities are still inspected
        res = shared_importance(NaNProba(), X, y, cv=3, scoring="accuracy", random_state=0)
    assert res.fold_results["error"].str.contains("non-finite class probabilities").all()
    assert (res.importances["n_splits"] == 0).all()


def test_undefined_metric_with_finite_predictions_is_not_a_failure(data):
    """A NaN score from finite predictions (rsq of a constant prediction) has no error."""
    from sharedimp import common_scorer

    X, y = data
    res = shared_importance(
        LinearRegression(), X[:, [0]], y, cv=KFold(3), scoring=common_scorer("rsq")
    )
    assert res.n_failed == 0 and res.fold_results["error"].isna().all()
    assert res.fold_results["score_reduced"].isna().all()  # null model: constant prediction
    assert (res.importances["n_splits"] == 0).all()
