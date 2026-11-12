"""Result container and aggregation of per-fold results."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["SharedImportanceResult", "aggregate_fold_results"]

FOLD_COLUMNS = [
    "feature",
    "split",
    "n_train",
    "n_test",
    "score_full",
    "score_reduced",
    "importance",
]
SUMMARY_COLUMNS = [
    "feature",
    "importance",
    "importance_std",
    "score_full",
    "score_reduced",
    "n_splits",
]


def aggregate_fold_results(
    fold_results: pd.DataFrame, features: Sequence[Hashable]
) -> pd.DataFrame:
    """Aggregate per-fold results into one row per anchor feature.

    * ``importance``: arithmetic mean over splits of the per-split paired
      difference ``score_full - score_reduced`` (never truncated at zero).
    * ``importance_std``: sample standard deviation (``ddof=1``) of the
      per-split differences; ``nan`` with a single split. This is a descriptive
      measure of between-split variability, not a standard error.
    * ``score_full`` / ``score_reduced``: mean scores over splits.
    * ``direction`` (only if present in ``fold_results``): mean over the splits
      where it is defined; ``nan`` if it is defined in none.

    Rows follow the order of ``features``. Non-finite scores propagate (``nan``
    in, ``nan`` out) instead of being silently dropped.
    """
    has_direction = "direction" in fold_results.columns
    rows: list[dict[str, Any]] = []
    # Positional grouping: feature labels need not be sortable or even unique types.
    codes = fold_results["_feature_pos"].to_numpy()
    for pos, feature in enumerate(features):
        sub = fold_results.loc[codes == pos]
        imp = sub["importance"].to_numpy(dtype=float)
        row: dict[str, Any] = {
            "feature": feature,
            "importance": float(np.mean(imp)),
            "importance_std": float(np.std(imp, ddof=1)) if imp.size > 1 else float("nan"),
            "score_full": float(np.mean(sub["score_full"].to_numpy(dtype=float))),
            "score_reduced": float(np.mean(sub["score_reduced"].to_numpy(dtype=float))),
            "n_splits": int(imp.size),
        }
        if has_direction:
            d = sub["direction"].to_numpy(dtype=float)
            d = d[np.isfinite(d)]
            row["direction"] = float(np.mean(d)) if d.size else float("nan")
        rows.append(row)
    columns = SUMMARY_COLUMNS + (["direction"] if has_direction else [])
    return pd.DataFrame(rows, columns=columns)


@dataclass(frozen=True)
class SharedImportanceResult:
    """Result of :func:`sharedimp.shared_importance`.

    Attributes
    ----------
    importances : pandas.DataFrame
        One row per anchor feature, in the order the features were evaluated,
        with columns ``feature``, ``importance``, ``importance_std``,
        ``score_full``, ``score_reduced``, ``n_splits`` and, when the direction
        diagnostic was requested, ``direction``. ``importance`` is the mean
        over splits of ``score_full - score_reduced``; it is positive when
        keeping the anchor improves out-of-sample performance and may be
        negative (it is never truncated).
    fold_results : pandas.DataFrame
        Long-format per-split results with columns ``feature``, ``split``
        (0-based split index in the order produced by the CV splitter),
        ``n_train``, ``n_test``, ``score_full``, ``score_reduced``,
        ``importance`` and optionally ``direction``.
    scoring : str
        Name of the scorer (``"score"`` for the estimator's default).
    n_splits : int
        Number of train/assessment splits.
    n_samples : int
        Number of samples in ``X``.
    n_features : int
        Number of features in ``X`` (not only the evaluated anchors).
    feature_names : list
        Names of all features in ``X``.
    cv : str
        ``repr`` of the resolved CV splitter.
    """

    importances: pd.DataFrame
    fold_results: pd.DataFrame
    scoring: str
    n_splits: int
    n_samples: int
    n_features: int
    feature_names: list[Any] = field(repr=False)
    cv: str = ""

    @property
    def features(self) -> list[Any]:
        """Evaluated anchor features, in result order."""
        return list(self.importances["feature"])

    @property
    def importances_mean(self) -> np.ndarray:
        """Mean importance per evaluated feature, as a NumPy array."""
        return self.importances["importance"].to_numpy(dtype=float)

    @property
    def importances_std(self) -> np.ndarray:
        """Between-split standard deviation of the importance per feature."""
        return self.importances["importance_std"].to_numpy(dtype=float)

    def to_frame(self, sort: bool = False) -> pd.DataFrame:
        """Return a copy of ``importances``; optionally sorted by decreasing importance."""
        df = self.importances.copy()
        if sort:
            df = df.sort_values("importance", ascending=False, kind="stable").reset_index(drop=True)
        return df

    def fold_matrix(self, value: str = "importance") -> pd.DataFrame:
        """Wide per-split table: one row per feature, one column per split.

        Parameters
        ----------
        value : {"importance", "score_full", "score_reduced", "direction"}
            Which per-split quantity to tabulate.
        """
        if value not in self.fold_results.columns or value in ("feature", "split"):
            raise ValueError(f"Unknown per-fold quantity: {value!r}.")
        fr = self.fold_results
        mat = fr[value].to_numpy(dtype=float).reshape(self.n_splits, -1).T
        return pd.DataFrame(
            mat,
            index=pd.Index(self.features, name="feature"),
            columns=pd.RangeIndex(self.n_splits, name="split"),
        )

    def __repr__(self) -> str:
        head = (
            f"SharedImportanceResult(scoring={self.scoring!r}, n_splits={self.n_splits}, "
            f"n_samples={self.n_samples}, n_features={self.n_features})"
        )
        with pd.option_context("display.max_rows", 20, "display.width", 120):
            body = self.importances.to_string(index=False, max_rows=20)
        return f"{head}\n{body}"
