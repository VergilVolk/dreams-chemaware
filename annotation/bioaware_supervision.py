"""Small, dependency-light helpers for BioAware candidate supervision."""
from __future__ import annotations

import numpy as np


def normalise_within_query_logits(values: np.ndarray) -> np.ndarray:
    """Remove arbitrary pairwise-model location and scale within one query."""
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or len(result) < 2 or not np.isfinite(result).all():
        raise ValueError("candidate logits must be a finite one-dimensional group")
    result = result - float(np.mean(result))
    scale = float(np.std(result))
    if scale > 1e-8:
        result = result / scale
    return result
