"""sharedimp: cross-validated UMFI-LR feature importance for scikit-learn workflows.

The package implements ultra-marginal feature importance with linear
residualization (UMFI-LR; Janssen, Guan & Robeva, AISTATS 2023) as a leakage-free,
cross-validated procedure that works with any scikit-learn compatible estimator,
scorer and CV splitter.
"""

from ._core import shared_importance
from ._metrics import COMMON_METRICS, common_scorer
from ._residualizer import AnchorResidualizer
from ._result import SharedImportanceResult
from ._version import __version__

__all__ = [
    "COMMON_METRICS",
    "AnchorResidualizer",
    "SharedImportanceResult",
    "__version__",
    "common_scorer",
    "shared_importance",
]
