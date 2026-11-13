"""How do least-squares solvers behave on the rank-deficient designs of the parity tests?

Run from the repository root (any environment with sharedimp installed)::

    python tests/fixtures/cross_language/rank_deficiency_study.py

For every design and solver it prints the largest absolute difference between
the Python per-split Full/Reduced scores (negated MSE) and the R reference
(``r_reference.json``: R ``lm.fit`` through sharedimp-r's "linear" learner).
The residualization is the same for every solver; only the downstream
least-squares fit changes. Results are recorded in ``docs/rank_deficiency.md``.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy import linalg as sla
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import PredefinedSplit

from sharedimp import AnchorResidualizer, common_scorer, shared_importance

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
from test_cross_language import AliasingOLS  # noqa: E402

INPUTS = json.loads((HERE / "inputs.json").read_text())
REF = json.loads((HERE / "r_reference.json").read_text())


class _Lstsq(RegressorMixin, BaseEstimator):
    """OLS on ``[1, X]`` through a given least-squares routine."""

    def __init__(self, backend="numpy", rcond=None):
        self.backend = backend
        self.rcond = rcond

    def fit(self, X, y):
        A = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
        y = np.asarray(y, float)
        if self.backend == "numpy":
            self.coef_ = np.linalg.lstsq(A, y, rcond=self.rcond)[0]
        else:
            self.coef_ = sla.lstsq(A, y, cond=self.rcond)[0]
        return self

    def predict(self, X):
        return np.column_stack([np.ones(len(X)), np.asarray(X, float)]) @ self.coef_


SOLVERS = {
    "sklearn LinearRegression()": LinearRegression,
    "numpy lstsq, rcond=None (default)": lambda: _Lstsq("numpy", None),
    "numpy lstsq, rcond=1e-7": lambda: _Lstsq("numpy", 1e-7),
    "scipy lstsq, cond=None (default)": lambda: _Lstsq("scipy", None),
    "numpy lstsq, rcond=1e-20 (no rank decision)": lambda: _Lstsq("numpy", 1e-20),
    "AliasingOLS (R lm.fit convention, tol=1e-7)": AliasingOLS,
}

IM = INPUTS["importance"]
SF = INPUTS["spec_fixture"]
DESIGNS = {
    # name: (X, y, fold labels, reference block, what is rank-deficient)
    "full rank (spec 15.1)": (SF["X"], SF["y"], SF["folds"], ("spec_fixture", "default")),
    "two mutually collinear features (spec 15.1 + x1_dup)": (
        SF["X_collinear"],
        SF["y"],
        SF["folds"],
        ("spec_fixture_collinear", "default"),
    ),
    "duplicate + affine copy (X_degenerate)": (
        IM["X_degenerate"],
        IM["y_regression"],
        IM["folds"],
        ("regression_degenerate", "mse"),
    ),
    "anchor constant on one training set (X_constant)": (
        IM["X_constant"],
        IM["y_regression"],
        IM["folds"],
        ("regression_constant", "mse"),
    ),
}


def singular_value_ratios(X, folds):
    """Smallest/largest singular value of the Full design ``[1, Full]`` per (split, anchor)."""
    X = np.asarray(X, float)
    folds = np.asarray(folds)
    ratios = []
    for f in np.unique(folds):
        train = folds != f
        for i in range(X.shape[1]):
            full = AnchorResidualizer(anchor=i).fit_transform(X[train])
            s = np.linalg.svd(np.column_stack([np.ones(train.sum()), full]), compute_uv=False)
            ratios.append(s[-1] / s[0])
    return np.array(ratios)


def main() -> None:
    print(
        f"Python {sys.version.split()[0]}, NumPy {np.__version__}, SciPy {scipy.__version__}, "
        f"scikit-learn {sklearn.__version__}; R reference: sharedimp-r "
        f"{REF['meta']['sharedimp_r_commit'][:7]} ({REF['meta']['r_version']})"
    )
    for name, (X, y, folds, (block, case)) in DESIGNS.items():
        ref = REF["importance"][block][case]["folds"]
        ratios = singular_value_ratios(X, folds)
        print(f"\n{name}")
        print(
            f"  singular value ratio of the Full design: min {ratios.min():.1e}, "
            f"max {ratios.max():.1e}; cells below 1e-7: {(ratios < 1e-7).sum()} of {ratios.size}"
        )
        for label, make in SOLVERS.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = shared_importance(
                    make(),
                    np.asarray(X, float),
                    np.asarray(y, float),
                    scoring=common_scorer("mse"),
                    cv=PredefinedSplit(np.asarray(folds) - 1),
                    on_error="error",
                )
            d_full = np.abs(res.fold_results["score_full"].to_numpy() - np.array(ref["score_full"]))
            d_red = np.abs(
                res.fold_results["score_reduced"].to_numpy() - np.array(ref["score_reduced"])
            )
            off = int(((d_full > 1e-8) | (d_red > 1e-8)).sum())
            print(
                f"  {label:45s} max|score_full - R| = {d_full.max():.1e}   "
                f"max|score_reduced - R| = {d_red.max():.1e}   "
                f"cells off by > 1e-8: {off} of {d_full.size}"
            )


if __name__ == "__main__":
    main()
