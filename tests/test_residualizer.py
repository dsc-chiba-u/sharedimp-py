from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sharedimp import AnchorResidualizer


# ---------------------------------------------------------------- 1. hand calc
def test_coefficients_match_hand_calculation(tiny):
    r = AnchorResidualizer(anchor=0).fit(tiny)
    # b on a: Sxy = 8, Sxx = 10 -> slope 0.8, intercept 3 - 0.8 * 2 = 1.4
    # c on a: Sxy = 0          -> slope 0.0, intercept 1.0
    np.testing.assert_allclose(r.coef_, [0.0, 0.8, 0.0], atol=1e-15)
    np.testing.assert_allclose(r.intercept_, [0.0, 1.4, 1.0], atol=1e-15)
    assert r.anchor_index_ == 0
    assert r.n_features_in_ == 3
    assert not r.collinear_.any()


def test_residuals_match_hand_calculation(tiny):
    out = AnchorResidualizer(anchor=0).fit_transform(tiny)
    expected = np.array(
        [
            [0.0, -0.4, 1.0],
            [1.0, 0.8, -1.0],
            [2.0, -1.0, 0.0],
            [3.0, 1.2, -1.0],
            [4.0, -0.6, 1.0],
        ]
    )
    np.testing.assert_allclose(out, expected, atol=1e-14)


def test_matches_numpy_lstsq_for_every_anchor():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 5)) @ rng.normal(size=(5, 5))
    for i in range(5):
        r = AnchorResidualizer(anchor=i).fit(X)
        A = np.column_stack([np.ones(50), X[:, i]])
        for j in range(5):
            if j == i:
                continue
            (a, b), *_ = np.linalg.lstsq(A, X[:, j], rcond=None)
            assert r.intercept_[j] == pytest.approx(a, abs=1e-10)
            assert r.coef_[j] == pytest.approx(b, abs=1e-10)


def test_no_intercept_hand_calculation(tiny):
    r = AnchorResidualizer(anchor=0, fit_intercept=False).fit(tiny)
    # through the origin: sum(a*b)/sum(a*a) = 38/30, sum(a*c)/sum(a*a) = 10/30
    np.testing.assert_allclose(r.coef_, [0.0, 38 / 30, 10 / 30], atol=1e-15)
    np.testing.assert_array_equal(r.intercept_, np.zeros(3))


def test_training_residuals_are_orthogonal_to_anchor():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 4))
    X[:, 2] += 2 * X[:, 1]
    out = AnchorResidualizer(anchor=1).fit_transform(X)
    others = np.delete(out, 1, axis=1)
    np.testing.assert_allclose(others.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(others.T @ (X[:, 1] - X[:, 1].mean()), 0, atol=1e-10)


# ------------------------------------------------ 2. train/assessment separation
def test_transform_uses_training_coefficients_only():
    rng = np.random.default_rng(2)
    X_train = rng.normal(size=(40, 3))
    X_test = rng.normal(size=(15, 3)) * 10 + 5  # very different distribution
    r = AnchorResidualizer(anchor=0).fit(X_train)
    coef, intercept = r.coef_.copy(), r.intercept_.copy()

    out = r.transform(X_test)
    # transform is stateless ...
    np.testing.assert_array_equal(r.coef_, coef)
    np.testing.assert_array_equal(r.intercept_, intercept)
    # ... and equals the manual application of the *training* coefficients
    for j in (1, 2):
        np.testing.assert_allclose(
            out[:, j], X_test[:, j] - intercept[j] - coef[j] * X_test[:, 0], atol=1e-12
        )
    np.testing.assert_array_equal(out[:, 0], X_test[:, 0])
    # and differs from residualizing the assessment rows on themselves
    leaky = AnchorResidualizer(anchor=0).fit_transform(X_test)
    assert not np.allclose(out, leaky)


def test_each_row_is_transformed_independently():
    rng = np.random.default_rng(3)
    X_train = rng.normal(size=(30, 4))
    X_test = rng.normal(size=(10, 4))
    r = AnchorResidualizer(anchor=2).fit(X_train)
    whole = r.transform(X_test)
    rows = np.vstack([r.transform(X_test[k : k + 1]) for k in range(10)])
    np.testing.assert_allclose(whole, rows, atol=1e-14)


# ----------------------------------------------- 3. full / reduced representation
def test_full_and_reduced_representations(tiny):
    full = AnchorResidualizer(anchor=1).fit_transform(tiny)
    reduced = AnchorResidualizer(anchor=1, drop_anchor=True).fit_transform(tiny)
    assert full.shape == (5, 3)
    assert reduced.shape == (5, 2)
    np.testing.assert_array_equal(full[:, 1], tiny[:, 1])  # anchor untouched, same position
    np.testing.assert_array_equal(reduced, np.delete(full, 1, axis=1))


def test_input_is_not_modified(tiny):
    X = tiny.copy()
    AnchorResidualizer(anchor=0).fit_transform(X)
    np.testing.assert_array_equal(X, tiny)
    df = pd.DataFrame(tiny, columns=list("abc"))
    AnchorResidualizer(anchor="a").fit_transform(df)
    np.testing.assert_array_equal(df.to_numpy(), tiny)


# -------------------------------------------------------------- degenerate cases
def test_exact_duplicate_gives_exact_zero_residual():
    rng = np.random.default_rng(4)
    a = rng.normal(size=60)
    X = np.column_stack([a, a, 3.0 - 2.0 * a, rng.normal(size=60)])
    r = AnchorResidualizer(anchor=0).fit(X)
    np.testing.assert_array_equal(r.collinear_, [False, True, True, False])
    out = r.transform(X)
    assert np.all(out[:, 1] == 0.0)
    assert np.all(out[:, 2] == 0.0)
    assert out[:, 3].std() > 0.1
    # also exactly zero on unseen rows
    new = rng.normal(size=(7, 4))
    assert np.all(r.transform(new)[:, [1, 2]] == 0.0)


def test_tol_zero_disables_snapping():
    rng = np.random.default_rng(4)
    a = rng.normal(size=60)
    X = np.column_stack([a, 2 * a + 1])
    r = AnchorResidualizer(anchor=0, tol=0.0).fit(X)
    assert not r.collinear_.any()
    assert np.abs(r.transform(X)[:, 1]).max() < 1e-12


def test_constant_anchor_centers_only():
    X = np.column_stack([np.full(6, 3.0), np.arange(6.0), np.arange(6.0) ** 2])
    r = AnchorResidualizer(anchor=0).fit(X)
    np.testing.assert_array_equal(r.coef_, np.zeros(3))
    assert r.constant_anchor_
    out = r.transform(X)
    np.testing.assert_allclose(out[:, 1], X[:, 1] - X[:, 1].mean())
    assert np.all(np.isfinite(out))


def test_constant_other_feature():
    X = np.column_stack([np.arange(6.0), np.full(6, 7.0)])
    r = AnchorResidualizer(anchor=0).fit(X)
    assert r.coef_[1] == 0.0
    assert not r.collinear_[1]
    assert r.intercept_[1] == 7.0
    assert np.all(r.transform(X)[:, 1] == 0.0)  # exactly, not up to round-off
    # on rows where the feature is not constant it is only centered
    np.testing.assert_array_equal(
        r.transform(np.array([[1.0, 9.0], [2.0, -3.0]]))[:, 1], [2.0, -10.0]
    )


# ------------------------------------------------------------------ pandas / API
def test_dataframe_roundtrip_matches_numpy(tiny):
    df = pd.DataFrame(tiny, columns=["a", "b", "c"], index=list("vwxyz"))
    r = AnchorResidualizer(anchor="b").fit(df)
    out = r.transform(df)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["a", "b", "c"]
    assert list(out.index) == list("vwxyz")
    np.testing.assert_array_equal(out.to_numpy(), AnchorResidualizer(anchor=1).fit_transform(tiny))
    np.testing.assert_array_equal(r.feature_names_in_, ["a", "b", "c"])
    np.testing.assert_array_equal(r.get_feature_names_out(), ["a", "b", "c"])

    red = AnchorResidualizer(anchor="b", drop_anchor=True).fit(df)
    assert list(red.transform(df).columns) == ["a", "c"]
    np.testing.assert_array_equal(red.get_feature_names_out(), ["a", "c"])


def test_feature_names_out_for_arrays(tiny):
    r = AnchorResidualizer(anchor=-1, drop_anchor=True).fit(tiny)
    assert r.anchor_index_ == 2
    np.testing.assert_array_equal(r.get_feature_names_out(), ["x0", "x1"])


def test_clone_and_params():
    r = AnchorResidualizer(anchor=2, fit_intercept=False, drop_anchor=True, tol=1e-8)
    c = clone(r)
    assert c is not r
    assert c.get_params() == r.get_params()
    c.set_params(anchor=0)
    assert r.anchor == 2


def test_works_inside_pipeline(tiny):
    pipe = make_pipeline(AnchorResidualizer(anchor=0, drop_anchor=True), StandardScaler())
    out = pipe.fit_transform(tiny)
    assert out.shape == (5, 2)


@pytest.mark.parametrize(
    ("kwargs", "X", "exc"),
    [
        ({"anchor": 5}, np.zeros((4, 3)), ValueError),
        ({"anchor": -4}, np.zeros((4, 3)), ValueError),
        ({"anchor": "a"}, np.zeros((4, 3)), ValueError),
        ({"anchor": 1.5}, np.zeros((4, 3)), TypeError),
        ({"anchor": True}, np.zeros((4, 3)), TypeError),
        ({"anchor": 0, "tol": -1.0}, np.zeros((4, 3)), ValueError),
        ({"anchor": "zz"}, pd.DataFrame(np.zeros((4, 2)), columns=["a", "b"]), ValueError),
        ({"anchor": 0}, np.array([[1.0, np.nan], [0.0, 1.0]]), ValueError),
    ],
)
def test_fit_errors(kwargs, X, exc):
    with pytest.raises(exc):
        AnchorResidualizer(**kwargs).fit(X)


def test_transform_errors(tiny):
    with pytest.raises(NotFittedError):
        AnchorResidualizer(anchor=0).transform(tiny)
    r = AnchorResidualizer(anchor=0).fit(tiny)
    with pytest.raises(ValueError, match="features"):
        r.transform(tiny[:, :2])
    df = pd.DataFrame(tiny, columns=list("abc"))
    rd = AnchorResidualizer(anchor="a").fit(df)
    with pytest.raises(ValueError, match="Column names"):
        rd.transform(df[["b", "a", "c"]])
