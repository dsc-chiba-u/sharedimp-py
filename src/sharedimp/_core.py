"""``shared_importance``: cross-validated UMFI-LR for scikit-learn estimators."""

from __future__ import annotations

import numbers
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.utils import check_random_state

from ._direction import direction_diagnostic, prediction_for_direction
from ._fitting import fit_clone, index_rows, slice_fit_params
from ._resampling import make_splits, resolve_cv
from ._residualizer import AnchorResidualizer
from ._result import FOLD_COLUMNS, SharedImportanceResult, aggregate_fold_results
from ._scoring import Scorer, resolve_scorer, score_model

__all__ = ["shared_importance"]

_MAX_SEED = 2**31 - 1


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
def _validate_X(X: Any) -> tuple[np.ndarray, list[Any], pd.Index | None]:
    """Return ``(float64 array, feature names, DataFrame index or None)``."""
    if isinstance(X, pd.DataFrame):
        if not X.columns.is_unique:
            raise ValueError("X has duplicated column names; feature names must be unique.")
        bad = [
            c
            for c, dt in X.dtypes.items()
            if not (pd.api.types.is_numeric_dtype(dt) or pd.api.types.is_bool_dtype(dt))
        ]
        if bad:
            raise ValueError(
                "All columns of X must be numeric because features are residualized by "
                f"linear regression. Non-numeric columns: {bad!r}. Encode categorical "
                "features (e.g. one-hot) before calling shared_importance()."
            )
        names: list[Any] = list(X.columns)
        index: pd.Index | None = X.index
        values = X.to_numpy(dtype=np.float64, na_value=np.nan)
    else:
        if hasattr(X, "tocsr") or hasattr(X, "toarray"):
            raise TypeError("Sparse X is not supported; pass a dense array or DataFrame.")
        try:
            values = np.asarray(X, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("X must be numeric (convertible to float64).") from exc
        if values.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got {values.ndim} dimension(s).")
        names = [f"x{i}" for i in range(values.shape[1])]
        index = None
    if values.shape[0] < 2:
        raise ValueError("X must contain at least 2 samples.")
    if values.shape[1] < 2:
        raise ValueError(
            "X must contain at least 2 features: the reduced model is fitted on the "
            "residualized *other* features, which is empty with a single feature."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "X contains NaN or infinite values; impute or remove them before calling "
            "shared_importance()."
        )
    return values, names, index


def _validate_y(y: Any, n_samples: int) -> Any:
    if y is None:
        raise ValueError("y is required.")
    if not hasattr(y, "iloc"):
        y = np.asarray(y)
        if y.ndim == 0:
            raise ValueError("y must be array-like with one entry per sample.")
    if len(y) != n_samples:
        raise ValueError(f"X and y have inconsistent lengths: {n_samples} != {len(y)}.")
    return y


def _resolve_features(features: Any, names: list[Any]) -> list[int]:
    """Map a ``features`` specification to positional column indices."""
    p = len(names)
    if features is None:
        return list(range(p))
    if isinstance(features, (str, numbers.Integral)) or not isinstance(
        features, (Sequence, np.ndarray, pd.Index)
    ):
        features = [features]
    if len(features) == 0:
        raise ValueError("`features` must not be empty.")
    out: list[int] = []
    for f in features:
        if isinstance(f, (bool, np.bool_)):
            raise TypeError("`features` entries must be column names or integer positions.")
        if f in names:
            # A label match always wins (covers integer-labelled DataFrame columns).
            idx = names.index(f)
        elif isinstance(f, numbers.Integral):
            idx = int(f)
            if not -p <= idx < p:
                raise ValueError(f"Feature index {idx} is out of bounds for {p} features.")
            idx %= p
        else:
            raise ValueError(f"Unknown feature {f!r}.")
        if idx in out:
            raise ValueError(f"Feature {names[idx]!r} is listed more than once in `features`.")
        out.append(idx)
    return out


# --------------------------------------------------------------------------- #
# One (split, anchor) evaluation
# --------------------------------------------------------------------------- #
def _wrap(values: np.ndarray, columns: list[Any] | None, index: pd.Index | None) -> Any:
    if columns is None:
        return values
    return pd.DataFrame(values, columns=columns, index=index)


def _evaluate_anchor(
    estimator: Any,
    X: np.ndarray,
    y: Any,
    train: np.ndarray,
    test: np.ndarray,
    anchor_idx: int,
    scorer: Scorer,
    *,
    seed: int | None,
    fit_params: Mapping[str, Any] | None,
    columns: list[Any] | None,
    index: pd.Index | None,
    direction: bool,
) -> dict[str, float]:
    """Paired Full/Reduced evaluation of one anchor on one train/assessment split."""
    X_train, X_test = X[train], X[test]
    y_train, y_test = index_rows(y, train), index_rows(y, test)

    # Residualization coefficients are estimated on the training rows only and
    # then applied unchanged to the assessment rows.
    residualizer = AnchorResidualizer(anchor=anchor_idx).fit(X_train)
    full_train = residualizer.transform(X_train)
    full_test = residualizer.transform(X_test)
    reduced_train = np.delete(full_train, anchor_idx, axis=1)
    reduced_test = np.delete(full_test, anchor_idx, axis=1)

    if columns is not None:
        reduced_columns: list[Any] | None = columns[:anchor_idx] + columns[anchor_idx + 1 :]
        train_index = index[train] if index is not None else None
        test_index = index[test] if index is not None else None
    else:
        reduced_columns = train_index = test_index = None

    params = slice_fit_params(fit_params, train, X.shape[0])

    # Paired design: same rows, same hyperparameters, same seed for both models.
    full_model = fit_clone(
        estimator, _wrap(full_train, columns, train_index), y_train, seed=seed, fit_params=params
    )
    full_test_w = _wrap(full_test, columns, test_index)
    score_full = score_model(scorer, full_model, full_test_w, y_test)

    reduced_model = fit_clone(
        estimator,
        _wrap(reduced_train, reduced_columns, train_index),
        y_train,
        seed=seed,
        fit_params=params,
    )
    score_reduced = score_model(
        scorer, reduced_model, _wrap(reduced_test, reduced_columns, test_index), y_test
    )

    out: dict[str, float] = {
        "n_train": int(train.size),
        "n_test": int(test.size),
        "score_full": score_full,
        "score_reduced": score_reduced,
        # Scorers are "greater is better": positive <=> keeping the anchor helps.
        "importance": score_full - score_reduced,
    }
    if direction:
        out["direction"] = direction_diagnostic(
            X_test[:, anchor_idx], prediction_for_direction(full_model, full_test_w)
        )
    return out


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def shared_importance(
    estimator: Any,
    X: Any,
    y: Any,
    *,
    scoring: Any = None,
    cv: Any = 5,
    groups: Any = None,
    features: Any = None,
    direction: bool = False,
    fit_params: Mapping[str, Any] | None = None,
    n_jobs: int | None = None,
    random_state: int | np.random.RandomState | None = None,
    verbose: int = 0,
) -> SharedImportanceResult:
    r"""Cross-validated ultra-marginal feature importance with linear residualization.

    For every anchor feature :math:`x_i` and every train/assessment split:

    1. On the **training** rows, regress every other feature :math:`x_j` on
       :math:`x_i` (simple OLS with intercept) and replace it by its residual.
       The same coefficients are applied to the **assessment** rows.
    2. Fit two clones of ``estimator`` on the training rows:
       *Full* (:math:`x_i` + residualized other features) and
       *Reduced* (residualized other features only).
    3. Score both on the assessment rows with the same scorer and record
       ``importance = score_full - score_reduced``.

    The reported importance is the mean of the per-split differences. Because
    scikit-learn scorers are "greater is better", the importance is positive
    when keeping the anchor improves out-of-sample performance. Negative values
    are reported as they are.

    This is the UMFI-LR quantity of Janssen, Guan & Robeva (AISTATS 2023)
    evaluated out-of-sample; this package does not introduce a new feature
    importance method.

    Parameters
    ----------
    estimator : estimator object
        Any scikit-learn compatible regressor or classifier (including
        ``Pipeline``). It is cloned with :func:`sklearn.base.clone` for every
        fit and never modified; its hyperparameters are used as given (no
        tuning).
    X : pandas.DataFrame or array-like of shape (n_samples, n_features)
        Numeric, dense, finite features; at least two columns. With a
        DataFrame, the models are fitted on DataFrames with the original
        column names.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Target. Continuous, binary or multiclass (labels of any dtype).
    scoring : str, callable or None, default=None
        A single scikit-learn scorer name (e.g. ``"neg_root_mean_squared_error"``,
        ``"neg_log_loss"``, ``"roc_auc_ovr"``) or a scorer callable
        ``scorer(estimator, X, y)``. ``None`` uses ``estimator.score``.
        Probability- and decision-function-based scorers are supported through
        the scikit-learn scorer machinery.
    cv : int, CV splitter or iterable, default=5
        An ``int`` ``k`` gives a *shuffled* ``StratifiedKFold`` (classifier
        with binary/multiclass ``y``) or *shuffled* ``KFold`` (otherwise),
        seeded by ``random_state``. Any scikit-learn splitter
        (``GroupKFold``, ``RepeatedKFold``, ``StratifiedKFold``, ...) or an
        iterable of ``(train, test)`` index pairs is used as given.
    groups : array-like of shape (n_samples,), default=None
        Group labels forwarded to ``cv.split`` (e.g. for ``GroupKFold``).
    features : sequence of str or int, default=None
        Anchor features to evaluate (column names or positions). ``None``
        evaluates every feature. All features are always used as predictors.
    direction : bool, default=False
        If ``True``, also compute the exploratory direction diagnostic: the
        Spearman rank correlation, on the assessment rows, between the anchor
        and the Full model's prediction (``predict`` for regressors, the
        probability of ``classes_[1]`` for binary classifiers; undefined, i.e.
        ``nan``, for multiclass). It is descriptive only and must not be
        interpreted as a causal effect.
    fit_params : dict, default=None
        Extra keyword arguments for ``estimator.fit``. Array-likes with
        ``n_samples`` rows (e.g. ``sample_weight``) are subset to the training
        rows of each split.
    n_jobs : int, default=None
        Number of joblib workers over (split, anchor) tasks. ``-1`` uses all
        cores. Results do not depend on ``n_jobs``.
    random_state : int, RandomState instance or None, default=None
        Controls (a) the shuffling of the folds when ``cv`` is an integer and
        (b) the seeds given to ``random_state`` parameters of the estimator
        that are left as ``None`` (the Full and Reduced models of a pair get
        the same seed). ``random_state`` parameters that are already set on
        the estimator, and user-supplied CV splitters, are never modified.
        With ``None``, nothing is seeded.
    verbose : int, default=0
        Verbosity passed to :class:`joblib.Parallel`.

    Returns
    -------
    result : SharedImportanceResult
        ``result.importances`` has one row per anchor feature and
        ``result.fold_results`` one row per (split, feature).

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.linear_model import LinearRegression
    >>> from sharedimp import shared_importance
    >>> X, y = make_regression(n_samples=200, n_features=4, random_state=0)
    >>> res = shared_importance(LinearRegression(), X, y,
    ...                         scoring="neg_mean_squared_error", cv=5, random_state=0)
    >>> list(res.importances.columns)
    ['feature', 'importance', 'importance_std', 'score_full', 'score_reduced', 'n_splits']
    """
    if not hasattr(estimator, "fit") or not hasattr(estimator, "get_params"):
        raise TypeError(
            "`estimator` must be a scikit-learn compatible estimator instance "
            "(implementing `fit` and `get_params`)."
        )
    if isinstance(estimator, type):
        raise TypeError("`estimator` must be an instance, not a class.")
    if not isinstance(direction, (bool, np.bool_)):
        raise TypeError("`direction` must be a bool.")
    if fit_params is not None and not isinstance(fit_params, Mapping):
        raise TypeError("`fit_params` must be a dict or None.")

    X_values, names, index = _validate_X(X)
    y = _validate_y(y, X_values.shape[0])
    anchor_indices = _resolve_features(features, names)
    scorer, scoring_name = resolve_scorer(estimator, scoring)

    n_samples, n_features = X_values.shape
    if random_state is None:
        cv_seed: int | None = None
        task_seeds: np.ndarray | None = None
    else:
        rng = check_random_state(random_state)
        cv_seed = int(rng.randint(_MAX_SEED))
    cv_resolved = resolve_cv(cv, y, estimator, cv_seed)
    splits = make_splits(cv_resolved, X_values, y, groups)
    if random_state is not None:
        # One seed per (split, column): independent of `features` and `n_jobs`.
        task_seeds = rng.randint(_MAX_SEED, size=(len(splits), n_features))

    columns = names if isinstance(X, pd.DataFrame) else None

    tasks = [(s, j) for s in range(len(splits)) for j in anchor_indices]
    outputs = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_evaluate_anchor)(
            estimator,
            X_values,
            y,
            splits[s][0],
            splits[s][1],
            j,
            scorer,
            seed=None if task_seeds is None else int(task_seeds[s, j]),
            fit_params=fit_params,
            columns=columns,
            index=index,
            direction=bool(direction),
        )
        for s, j in tasks
    )

    pos_of = {j: pos for pos, j in enumerate(anchor_indices)}
    records = [
        {"feature": names[j], "_feature_pos": pos_of[j], "split": s, **out}
        for (s, j), out in zip(tasks, outputs, strict=True)
    ]
    fold_columns = FOLD_COLUMNS + (["direction"] if direction else [])
    fold_df = pd.DataFrame.from_records(records, columns=[*fold_columns, "_feature_pos"])
    feature_labels = [names[j] for j in anchor_indices]
    importances = aggregate_fold_results(fold_df, feature_labels)
    fold_df = fold_df.drop(columns="_feature_pos")

    return SharedImportanceResult(
        importances=importances,
        fold_results=fold_df,
        scoring=scoring_name,
        n_splits=len(splits),
        n_samples=n_samples,
        n_features=n_features,
        feature_names=list(names),
        cv=repr(cv_resolved),
    )
