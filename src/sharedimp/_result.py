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
    "error",
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
    """Aggregate per-split results into one row per anchor feature.

    Only the splits with a defined (non-``nan``) importance contribute: a
    failed fit, or a metric that is undefined on a split, is left out of every
    mean and is not counted.

    * ``importance``: arithmetic mean over the contributing splits of the
      per-split paired difference ``score_full - score_reduced`` (never
      truncated at zero).
    * ``importance_std``: sample standard deviation (``ddof=1``) of those
      differences; ``nan`` with fewer than two contributing splits. This is a
      descriptive measure of between-split variability, not a standard error.
    * ``score_full`` / ``score_reduced``: mean scores over the same splits.
    * ``n_splits``: number of contributing splits (``0`` if none: all summary
      values are then ``nan``).

    Rows follow the order of ``features``.
    """
    rows: list[dict[str, Any]] = []
    # Positional grouping: feature labels need not be sortable or even unique types.
    codes = fold_results["_feature_pos"].to_numpy()
    for pos, feature in enumerate(features):
        sub = fold_results.loc[codes == pos]
        imp = sub["importance"].to_numpy(dtype=float)
        ok = ~np.isnan(imp)
        n_ok = int(ok.sum())
        nan = float("nan")
        rows.append(
            {
                "feature": feature,
                "importance": float(np.mean(imp[ok])) if n_ok else nan,
                "importance_std": float(np.std(imp[ok], ddof=1)) if n_ok > 1 else nan,
                "score_full": float(np.mean(sub["score_full"].to_numpy(dtype=float)[ok]))
                if n_ok
                else nan,
                "score_reduced": float(np.mean(sub["score_reduced"].to_numpy(dtype=float)[ok]))
                if n_ok
                else nan,
                "n_splits": n_ok,
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


@dataclass(frozen=True)
class SharedImportanceResult:
    """Result of :func:`sharedimp.shared_importance`.

    Attributes
    ----------
    importances : pandas.DataFrame
        One row per anchor feature, in the order the features were evaluated,
        with columns ``feature``, ``importance``, ``importance_std``,
        ``score_full``, ``score_reduced``, ``n_splits``. ``importance`` is the
        mean of ``score_full - score_reduced`` over the splits in which it is
        defined; ``n_splits`` is the number of those splits. It is positive
        when keeping the anchor improves out-of-sample performance and may be
        negative (it is never truncated). Scores are "greater is better"
        (losses are negated).
    fold_results : pandas.DataFrame
        Long-format per-split results with columns ``feature``, ``split``
        (0-based index of the split in the order produced by the CV splitter;
        the R package numbers the same splits from 1), ``n_train``,
        ``n_test``, ``score_full``, ``score_reduced``, ``importance`` and
        ``error``. Rows are ordered by split, then by feature. ``error`` is
        ``None`` for a successful fit and the error message for a failed one
        (whose scores and importance are ``nan``).
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

    @property
    def n_failed(self) -> int:
        """Number of (split, feature) fits that failed (see ``fold_results["error"]``)."""
        return int(self.fold_results["error"].notna().sum())

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
        value : {"importance", "score_full", "score_reduced"}
            Which per-split quantity to tabulate.
        """
        if value not in ("importance", "score_full", "score_reduced"):
            raise ValueError(f"Unknown per-split quantity: {value!r}.")
        mat = self.fold_results[value].to_numpy(dtype=float).reshape(self.n_splits, -1).T
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
