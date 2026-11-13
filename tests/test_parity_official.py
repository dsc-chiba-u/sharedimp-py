"""Parity tests against the official UMFI Python implementation.

Upstream: https://github.com/guanton/umfi_python (linked from the authors' R
repository https://github.com/HydroML/UMFI), pinned to a fixed commit.

The upstream repository ships no license file, so its sources are *not*
vendored here. They are downloaded at test time (pinned by commit SHA and
verified by SHA-256) or read from a local checkout given by the environment
variable ``SHAREDIMP_UMFI_PYTHON_DIR``. If neither is available the tests are
skipped, never faked.

The official implementation and sharedimp differ by design (residualization
gate, in-sample OOB vs. cross-validation, truncation, noise injection), so
the tests are split into two groups; see also ``docs/relation_to_umfi.md``.

``TestExactAgreementExpected``
    Conditions under which the two must agree to floating-point accuracy:
    the linear residualization when the upstream gate (R^2 > 0.01) is passed
    for every feature pair, the column layout of the representation (anchor
    in its original position), and the upstream UMFI-LR score when the
    upstream scoring rule (random-forest OOB R^2 on all rows, truncation at
    0) is applied to sharedimp's Full/Reduced representations.

``TestAgreementNotExpected``
    Conditions under which they must *not* be expected to agree, each pinned
    by a test that asserts the difference: the gate (upstream leaves weakly
    correlated features untouched), truncation (upstream never returns a
    negative importance), and the estimator (5-fold CV with fold-internal
    residualization vs. OOB with full-data residualization), for which only a
    loose sanity bound is checked.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import types
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from sharedimp import AnchorResidualizer, shared_importance

pytestmark = pytest.mark.parity

UPSTREAM_COMMIT = "4db1c23f2355d69fc1471ae8df89c196d7a9a347"
UPSTREAM_RAW = f"https://raw.githubusercontent.com/guanton/umfi_python/{UPSTREAM_COMMIT}/umfi/"
UPSTREAM_SHA256 = {
    "preprocess_LR.py": "07db484f4fb9bb9a638d0a0f16b028b10f1045e1be195683aa8dc217a9919d7f",
    "UMFI.py": "8c90a2d0288303532ea64694d694e90fc94d1975e84427cf67a87848a15e372d",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="session")
def upstream_dir(tmp_path_factory) -> Path:
    local = os.environ.get("SHAREDIMP_UMFI_PYTHON_DIR")
    if local:
        base = Path(local)
        if (base / "umfi").is_dir():
            base = base / "umfi"
    else:
        base = tmp_path_factory.mktemp("umfi_upstream")
        for name in UPSTREAM_SHA256:
            try:
                with urllib.request.urlopen(UPSTREAM_RAW + name, timeout=30) as resp:
                    (base / name).write_bytes(resp.read())
            except Exception as exc:  # network unavailable
                pytest.skip(f"official UMFI sources could not be downloaded: {exc!r}")
    for name, digest in UPSTREAM_SHA256.items():
        path = base / name
        if not path.is_file():
            pytest.skip(f"official UMFI source {name} not found in {base}")
        if _sha256(path) != digest:
            pytest.fail(f"{name} does not match the pinned upstream commit {UPSTREAM_COMMIT}")
    return base


def _load(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def official_preprocess_lr(upstream_dir):
    return _load("_official_preprocess_LR", upstream_dir / "preprocess_LR.py").preprocess_lr


@pytest.fixture
def official_umfi(upstream_dir, official_preprocess_lr, monkeypatch):
    """The upstream ``UMFI`` function.

    ``UMFI.py`` imports ``minepy``, ``tqdm`` and the optimal-transport
    preprocessing at module level. None of them is used on the code path
    exercised here (``preprocessing_methods=["linear regression"]``,
    ``select_optimal_preprocessing=False``), so they are replaced by inert
    stand-ins instead of being installed. The UMFI-LR code path is unmodified.
    """
    pkg = types.ModuleType("umfi")
    pkg.__path__ = []  # mark as package
    lr = types.ModuleType("umfi.preprocess_LR")
    lr.preprocess_lr = official_preprocess_lr
    ot = types.ModuleType("umfi.preprocess_OT")

    def _unused(*args, **kwargs):  # pragma: no cover
        raise AssertionError("optimal-transport path must not be used in the parity test")

    ot.preprocess_ot = _unused
    minepy = types.ModuleType("minepy")
    minepy.MINE = _unused
    tqdm = types.ModuleType("tqdm")
    tqdm.tqdm = lambda iterable, **kwargs: iterable
    for name, mod in {
        "umfi": pkg,
        "umfi.preprocess_LR": lr,
        "umfi.preprocess_OT": ot,
        "minepy": minepy,
        "tqdm": tqdm,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return _load("_official_UMFI", upstream_dir / "UMFI.py").UMFI


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(20230425)
    n = 400
    z = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "a": z + 0.5 * rng.normal(size=n),
            "b": z + 0.5 * rng.normal(size=n),
            "c": 0.6 * z + rng.normal(size=n),
            "d": -0.4 * z + rng.normal(size=n),
        }
    )
    y = pd.Series(2.0 * X["a"] + 1.0 * X["c"] + 0.5 * rng.normal(size=n), name="y")
    return X, y


def _reference_from_sharedimp_representation(X, y, seed, n_trees):
    """Upstream scoring rule applied to sharedimp's Full/Reduced representations.

    Mirrors the upstream sequence of model fits (and therefore of draws from
    NumPy's global RNG, which scikit-learn uses when ``random_state=None``).
    """
    np.random.seed(seed)  # noqa: NPY002 - upstream relies on the global RNG
    out = np.zeros(X.shape[1])
    for i, col in enumerate(X.columns):
        full = AnchorResidualizer(anchor=col).fit_transform(X)
        reduced = AnchorResidualizer(anchor=col, drop_anchor=True).fit_transform(X)
        with_ = RandomForestRegressor(n_estimators=n_trees, oob_score=True).fit(full, y)
        without = RandomForestRegressor(n_estimators=n_trees, oob_score=True).fit(reduced, y)
        out[i] = max(0, with_.oob_score_) - max(0, without.oob_score_)
    out[out < 0] = 0
    return out


def _official_importance(official_umfi, X, y, seed, niter):
    np.random.seed(seed)  # noqa: NPY002
    avg, final = official_umfi(
        X.copy(),
        y,
        preprocessing_methods=["linear regression"],
        niter=niter,
        n_trees=100,
        select_optimal_preprocessing=False,
        RF=True,
    )
    table = final if niter == 1 else avg
    return table.set_index("Feature").loc[list(X.columns), "Importance"].to_numpy(float)


class TestExactAgreementExpected:
    """Same rows, gate passed, upstream scoring rule: agreement to round-off."""

    def test_residualization_when_the_gate_is_passed(self, official_preprocess_lr, data):
        X, _ = data
        for col in X.columns:
            # precondition: every pair passes the upstream gate (R^2 > 0.01)
            r2 = X.corr()[col].drop(col) ** 2
            assert (r2 > 0.01).all()
            official = official_preprocess_lr(X.copy(), col)
            ours = AnchorResidualizer(anchor=col).fit_transform(X)
            np.testing.assert_allclose(ours.to_numpy(), official.to_numpy(), rtol=0, atol=1e-12)

    def test_representation_layout(self, official_preprocess_lr, data):
        """Both keep every column, in the original order, with the raw anchor in place."""
        X, _ = data
        for col in X.columns:
            official = official_preprocess_lr(X.copy(), col)
            ours = AnchorResidualizer(anchor=col).fit_transform(X)
            assert list(official.columns) == list(ours.columns) == list(X.columns)
            np.testing.assert_array_equal(ours[col].to_numpy(), X[col].to_numpy())
            np.testing.assert_array_equal(official[col].to_numpy(), X[col].to_numpy())

    def test_upstream_score_from_sharedimp_representation(self, official_umfi, data):
        X, y = data
        seed = 2023
        official = _official_importance(official_umfi, X, y, seed, niter=1)
        reference = _reference_from_sharedimp_representation(X, y, seed, n_trees=100)
        np.testing.assert_allclose(reference, official, rtol=0, atol=1e-8)
        assert official[0] > 0.1  # the comparison is not trivially 0 == 0


class TestAgreementNotExpected:
    """Deliberate differences; each test asserts that the difference exists."""

    def test_gate_upstream_skips_weakly_correlated_features(self, official_preprocess_lr):
        """Upstream residualizes only if R^2 > 0.01; sharedimp always residualizes."""
        rng = np.random.default_rng(3)
        n = 500
        a = rng.normal(size=n)
        weak = 0.05 * a + rng.normal(size=n)  # R^2 with a is about 0.0025
        X = pd.DataFrame({"a": a, "weak": weak})
        assert np.corrcoef(a, weak)[0, 1] ** 2 <= 0.01
        official = official_preprocess_lr(X.copy(), "a")
        ours = AnchorResidualizer(anchor="a").fit_transform(X)
        np.testing.assert_array_equal(official["weak"].to_numpy(), weak)  # untouched upstream
        slope = np.cov(a, weak, bias=True)[0, 1] / a.var()
        manual = weak - (weak.mean() - slope * a.mean()) - slope * a
        np.testing.assert_allclose(ours["weak"].to_numpy(), manual, atol=1e-12)
        assert abs(np.corrcoef(a, ours["weak"])[0, 1]) < 1e-12
        assert abs(np.corrcoef(a, official["weak"])[0, 1]) > 0.01

    def test_truncation_upstream_never_returns_negative_importance(self, official_umfi):
        """Upstream floors scores and importances at 0; sharedimp reports them as they are.

        With a response that is independent of every feature, the forests'
        out-of-bag R^2 is negative, upstream floors it to 0 for both models
        and reports exactly 0. sharedimp reports the (negative) scores and the
        untruncated differences, some of which are negative.
        """
        rng = np.random.default_rng(8)
        n = 150
        X = pd.DataFrame(rng.normal(size=(n, 5)), columns=list("abcde"))
        y = pd.Series(rng.normal(size=n), name="y")  # pure noise
        official = _official_importance(official_umfi, X, y, seed=1, niter=1)
        ours = shared_importance(
            RandomForestRegressor(n_estimators=100), X, y, scoring="r2", cv=5, random_state=1
        )
        np.testing.assert_array_equal(official, np.zeros(5))
        assert (ours.fold_results["score_full"] < 0).any()  # scores are not floored
        assert (ours.importances_mean != 0).all()
        assert (ours.importances_mean < 0).any()  # negative importance is kept

    def test_cross_validation_vs_oob_agree_only_approximately(self, official_umfi, data):
        """Out-of-sample R^2 gain (sharedimp) vs. in-sample OOB R^2 gain (upstream).

        Same estimand, different estimators (5-fold CV with fold-internal
        residualization vs. OOB with full-data residualization, truncation at
        0). The tolerance is deliberately loose: a sanity check, not a claim of
        numerical equivalence.
        """
        X, y = data
        official = _official_importance(official_umfi, X, y, seed=7, niter=5)
        ours = shared_importance(
            RandomForestRegressor(n_estimators=100), X, y, scoring="r2", cv=5, random_state=7
        ).importances_mean
        assert not np.allclose(ours, official, atol=1e-6)  # they are not the same numbers
        np.testing.assert_allclose(ours, official, atol=0.1)
        assert list(np.argsort(-ours)[:2]) == list(np.argsort(-official)[:2])
