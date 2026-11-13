"""Generate the deterministic inputs shared by the R and Python parity checks.

Run from the repository root::

    python tests/fixtures/cross_language/make_inputs.py

The output (``inputs.json``) is committed. Regenerating it requires
regenerating ``r_reference.json`` with ``generate_r_reference.R`` as well.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def main() -> None:
    rng = np.random.default_rng(20261113)

    # ---------------------------------------------------------- metric inputs
    n = 16
    truth = rng.normal(size=n).round(4)
    estimate = (0.7 * truth + 0.5 * rng.normal(size=n) + 0.2).round(4)

    p_b = rng.uniform(0.05, 0.95, size=n).round(4)
    bin_truth = np.where(rng.uniform(size=n) < p_b, "b", "a")
    bin_truth[:2] = ["a", "b"]  # both classes present
    p_b[5] = p_b[6]  # a tie in the scores
    p_b[9] = 0.0  # exercises clipping when the truth is "b"
    bin_truth[9] = "b"
    p_b[10] = 0.5  # argmax tie -> first class

    raw = rng.uniform(0.05, 1.0, size=(n, 3))
    multi_prob = (raw / raw.sum(axis=1, keepdims=True)).round(4)
    multi_prob[:, 2] = (1.0 - multi_prob[:, 0] - multi_prob[:, 1]).round(4)
    multi_truth = np.array(["u", "v", "w"])[rng.integers(0, 3, size=n)]
    multi_truth[:3] = ["u", "v", "w"]

    # ------------------------------------------------- importance inputs
    m = 48
    x0 = rng.normal(size=m)
    x1 = 0.7 * x0 + 0.7 * rng.normal(size=m)
    x2 = rng.normal(size=m)
    x3 = 0.3 * x2 + rng.normal(size=m)
    X = np.column_stack([x0, x1, x2, x3]).round(4)
    # exact duplicate and exact affine copy of column 0 (after rounding)
    X_degenerate = np.column_stack([X, X[:, 0], 3.0 - 2.0 * X[:, 0]])
    y_reg = (1.5 * X[:, 0] - 1.0 * X[:, 2] + 0.5 * rng.normal(size=m)).round(4)
    eta = 1.2 * X[:, 0] - 0.9 * X[:, 2]
    y_bin = np.where(eta + rng.logistic(size=m) > 0, "b", "a")
    s = X[:, 0] + 0.8 * X[:, 2] + 0.9 * rng.normal(size=m)
    y_multi = np.array(["u", "v", "w"])[np.digitize(s, np.quantile(s, [1 / 3, 2 / 3]))]
    folds = (np.arange(m) % 4) + 1  # explicit fold labels, 1..4

    out = {
        "metrics": {
            "regression": {"truth": truth.tolist(), "estimate": estimate.tolist()},
            "binary": {
                "levels": ["a", "b"],
                "truth": bin_truth.tolist(),
                "prob_a": (1.0 - p_b).round(4).tolist(),
                "prob_b": p_b.tolist(),
            },
            "multiclass": {
                "levels": ["u", "v", "w"],
                "truth": multi_truth.tolist(),
                "prob": multi_prob.tolist(),
            },
        },
        "importance": {
            "feature_names": ["x0", "x1", "x2", "x3"],
            "degenerate_feature_names": ["x0", "x1", "x2", "x3", "dup", "affine"],
            "X": X.tolist(),
            "X_degenerate": X_degenerate.tolist(),
            "y_regression": y_reg.tolist(),
            "y_binary": y_bin.tolist(),
            "y_multiclass": y_multi.tolist(),
            "folds": folds.tolist(),
        },
    }
    (HERE / "inputs.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
