"""Anchor residualizer: the linear dependence-removal step of UMFI-LR."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

__all__ = ["AnchorResidualizer"]


def _as_float_array(X: Any) -> np.ndarray:
    """Validate ``X`` as a dense, finite, 2-D float64 array (always a copy-safe view)."""
    return check_array(X, dtype=np.float64, ensure_2d=True, accept_sparse=False)


def _fit_coefficients(
    X: np.ndarray, anchor_idx: int, fit_intercept: bool, tol: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Simple-regression coefficients of every column on the anchor column.

    Returns ``(coef, intercept, collinear, constant_anchor)``. The first three
    have length ``n_features``; their entries at ``anchor_idx`` are ``0``,
    ``0`` and ``False`` (the anchor itself is never residualized).
    """
    n_features = X.shape[1]
    a = X[:, anchor_idx]
    # "Constant" means exactly constant (max == min). The test uses no
    # arithmetic on the mean, so it cannot be fooled by a rounded mean, and it
    # is unaffected by the location and scale of the column.
    col_min, col_max = X.min(axis=0), X.max(axis=0)
    constant = col_max == col_min
    if fit_intercept:
        a_mean = float(a.mean())
        # For a constant column the mean is that constant, exactly.
        x_mean = np.where(constant, col_min, X.mean(axis=0))
        if constant[anchor_idx]:
            a_mean = float(col_min[anchor_idx])
    else:
        a_mean = 0.0
        x_mean = np.zeros(n_features)
    a_c = a - a_mean
    X_c = X - x_mean
    ss_a = float(a_c @ a_c)
    ss_x = np.einsum("ij,ij->j", X_c, X_c)

    # Constant anchor within the fitting rows: the slope is not identified and
    # the minimum-norm least-squares solution (slope 0) is used.
    constant_anchor = ss_a == 0.0
    coef = np.zeros(n_features) if constant_anchor else (a_c @ X_c) / ss_a
    if fit_intercept:
        coef[constant] = 0.0  # a constant feature has no slope on anything
    intercept = x_mean - coef * a_mean

    resid = X_c - np.outer(a_c, coef)
    ss_res = np.einsum("ij,ij->j", resid, resid)
    # A feature is "collinear" with the anchor when (numerically) all of its
    # variation is explained by the anchor: SS_res <= tol * SS_tot, i.e.
    # 1 - R^2 <= tol. Its residual is pure floating-point noise, which is
    # snapped to exactly zero in ``transform``.
    collinear = (ss_x > 0.0) & (ss_res <= tol * ss_x)

    coef[anchor_idx] = 0.0
    intercept[anchor_idx] = 0.0
    collinear[anchor_idx] = False
    return coef, intercept, collinear, constant_anchor


def _apply_coefficients(
    X: np.ndarray,
    anchor_idx: int,
    coef: np.ndarray,
    intercept: np.ndarray,
    collinear: np.ndarray,
) -> np.ndarray:
    """Apply stored coefficients; the anchor column is passed through unchanged."""
    a = X[:, anchor_idx]
    out = X - intercept - np.outer(a, coef)
    out[:, collinear] = 0.0
    out[:, anchor_idx] = a
    return out


class AnchorResidualizer(TransformerMixin, BaseEstimator):
    r"""Residualize every feature on a single anchor feature by simple linear regression.

    For an anchor feature :math:`x_i` and every other feature :math:`x_j`,
    ``fit`` estimates the ordinary-least-squares simple regression

    .. math:: x_j = a_j + b_j x_i + e_j

    and ``transform`` replaces :math:`x_j` by the residual
    :math:`x_j - \hat a_j - \hat b_j x_i`, using the coefficients learned in
    ``fit``. The anchor column itself is passed through unchanged (or dropped
    when ``drop_anchor=True``). This is the "linear regression" dependence-removal
    step of UMFI-LR (Janssen, Guan & Robeva, AISTATS 2023), implemented as a
    stateful scikit-learn transformer so that coefficients estimated on training
    rows can be applied to held-out rows without information leakage.

    Parameters
    ----------
    anchor : int or str
        The anchor ("protected") feature. An ``int`` is a positional column
        index (negative indices allowed). A ``str`` is a column name and
        requires ``X`` to be a :class:`pandas.DataFrame` at ``fit`` time.
    fit_intercept : bool, default=True
        Whether the simple regressions include an intercept. ``True`` matches
        UMFI-LR. With ``False`` the regressions go through the origin.
    drop_anchor : bool, default=False
        If ``False``, ``transform`` returns the *full* representation: the
        anchor column in its original position plus the residualized other
        features. If ``True``, it returns the *reduced* representation: the
        residualized other features only.
    tol : float, default=1e-10
        Relative tolerance of the exact-collinearity rule: feature ``j`` is
        flagged when ``SS_res_j <= tol * SS_tot_j`` on the fitting rows (that
        is, ``1 - R^2 <= tol``). Residuals of flagged features are set to
        exactly ``0.0`` in ``transform``, for every row, so that
        floating-point round-off cannot leak the anchor into the reduced
        representation. ``1e-10`` is the value fixed by the common R/Python
        specification; set to ``0.0`` to disable the rule.

    Attributes
    ----------
    anchor_index_ : int
        Positional index of the anchor column.
    coef_ : ndarray of shape (n_features,)
        Slope :math:`\hat b_j` of each feature on the anchor. ``0`` at
        ``anchor_index_``.
    intercept_ : ndarray of shape (n_features,)
        Intercept :math:`\hat a_j` of each regression. ``0`` at
        ``anchor_index_`` and everywhere when ``fit_intercept=False``.
    collinear_ : ndarray of bool, shape (n_features,)
        Features flagged by the exact-collinearity rule (see ``tol``).
    constant_anchor_ : bool
        Whether the anchor is exactly constant in the fitting rows. All slopes
        are then ``0``.
    n_features_in_ : int
        Number of features seen during ``fit``.
    feature_names_in_ : ndarray of str, shape (n_features_in_,)
        Column names seen during ``fit``. Only set for DataFrame input with
        all-string column names.

    Notes
    -----
    Degenerate cases are handled by deterministic rules shared with the R
    package (``docs/specification.md``, section 3.1):

    * *Constant anchor* (``max == min`` on the fitting rows): slopes are not
      identified and are set to ``0``; features are only centered.
    * *Constant feature*: slope ``0`` and intercept equal to the constant, so
      its residual is exactly ``0`` on the fitting rows.
    * *Feature exactly collinear with the anchor* (a duplicate or any affine
      copy): its residual is exactly ``0`` for every row, fitting or new.

    All three rules are unaffected by the location and scale (units) of the
    features and of the anchor. Flagged columns are kept (as zeros), never
    dropped, so the number and order of columns do not depend on the data.

    Unlike the reference implementations of UMFI-LR, residualization is
    unconditional (no significance or :math:`R^2` gate) and deterministic (no
    random-noise injection). See ``docs/relation_to_umfi.md``.

    Input must be numeric, dense and finite. A DataFrame in gives a DataFrame
    out (same index, float64 columns).

    Examples
    --------
    >>> import numpy as np
    >>> from sharedimp import AnchorResidualizer
    >>> X = np.array([[0.0, 1.0], [1.0, 3.0], [2.0, 5.0], [3.0, 8.0]])
    >>> r = AnchorResidualizer(anchor=0).fit(X)
    >>> round(float(r.coef_[1]), 6), round(float(r.intercept_[1]), 6)
    (2.3, 0.8)
    """

    def __init__(
        self,
        anchor: int | str = 0,
        *,
        fit_intercept: bool = True,
        drop_anchor: bool = False,
        tol: float = 1e-10,
    ) -> None:
        self.anchor = anchor
        self.fit_intercept = fit_intercept
        self.drop_anchor = drop_anchor
        self.tol = tol

    # ------------------------------------------------------------------ fit
    def _resolve_anchor(self, X: Any, n_features: int) -> int:
        anchor = self.anchor
        if isinstance(anchor, str):
            if not isinstance(X, pd.DataFrame):
                raise ValueError(
                    "A string `anchor` requires X to be a pandas DataFrame; "
                    f"got {type(X).__name__}."
                )
            matches = np.flatnonzero(np.asarray(X.columns == anchor))
            if matches.size == 0:
                raise ValueError(f"anchor {anchor!r} is not a column of X.")
            if matches.size > 1:
                raise ValueError(f"anchor {anchor!r} matches more than one column of X.")
            return int(matches[0])
        if isinstance(anchor, (bool, np.bool_)) or not isinstance(anchor, (int, np.integer)):
            raise TypeError(f"`anchor` must be an int or a str, got {type(anchor).__name__}.")
        idx = int(anchor)
        if not -n_features <= idx < n_features:
            raise ValueError(
                f"anchor index {idx} is out of bounds for X with {n_features} features."
            )
        return idx % n_features

    def fit(self, X: Any, y: Any = None) -> AnchorResidualizer:
        """Estimate the simple-regression coefficients of each feature on the anchor.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Numeric training data.
        y : ignored
            Present for scikit-learn API compatibility.

        Returns
        -------
        self : AnchorResidualizer
        """
        if not isinstance(self.tol, (int, float)) or self.tol < 0:
            raise ValueError(f"`tol` must be a non-negative float, got {self.tol!r}.")
        if hasattr(X, "columns"):
            names = np.asarray(X.columns, dtype=object)
            if all(isinstance(c, str) for c in names):
                self.feature_names_in_ = names
            elif hasattr(self, "feature_names_in_"):
                del self.feature_names_in_
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        Xa = _as_float_array(X)
        self.n_features_in_ = Xa.shape[1]
        self.anchor_index_ = self._resolve_anchor(X, Xa.shape[1])
        self.coef_, self.intercept_, self.collinear_, self.constant_anchor_ = _fit_coefficients(
            Xa, self.anchor_index_, bool(self.fit_intercept), float(self.tol)
        )
        return self

    # ------------------------------------------------------------ transform
    def transform(self, X: Any) -> Any:
        """Residualize ``X`` with the coefficients learned in ``fit``.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Data to transform (training or held-out rows).

        Returns
        -------
        X_out : ndarray or DataFrame
            Shape ``(n_samples, n_features)`` when ``drop_anchor=False`` and
            ``(n_samples, n_features - 1)`` when ``drop_anchor=True``.
        """
        check_is_fitted(self, "coef_")
        Xa = _as_float_array(X)
        if Xa.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {Xa.shape[1]} features, but AnchorResidualizer is expecting "
                f"{self.n_features_in_} features as input."
            )
        if (
            isinstance(X, pd.DataFrame)
            and hasattr(self, "feature_names_in_")
            and list(X.columns) != list(self.feature_names_in_)
        ):
            raise ValueError("Column names of X differ from those seen in fit.")
        out = _apply_coefficients(
            Xa, self.anchor_index_, self.coef_, self.intercept_, self.collinear_
        )
        if self.drop_anchor:
            out = np.delete(out, self.anchor_index_, axis=1)
        if isinstance(X, pd.DataFrame):
            cols = X.columns
            if self.drop_anchor:
                cols = cols.delete(self.anchor_index_)
            return pd.DataFrame(out, index=X.index, columns=cols)
        return out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return output feature names (input names, minus the anchor if dropped)."""
        check_is_fitted(self, "coef_")
        if input_features is None:
            if hasattr(self, "feature_names_in_"):
                names = np.asarray(self.feature_names_in_, dtype=object)
            else:
                names = np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        else:
            names = np.asarray(input_features, dtype=object)
            if names.shape[0] != self.n_features_in_:
                raise ValueError("input_features has the wrong length.")
        if self.drop_anchor:
            names = np.delete(names, self.anchor_index_)
        return names
