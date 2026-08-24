"""COSMIC confidence layer (Step-1) -- single-spectrum chemical confidence.

Two-layer chemical-evaluation framework, Layer 1. Produces a *single spectrum*
confidence score on DreaMS's frozen representation, *before* the final cosine
retrieval, combined with the 335 chemical rules, then calibrates it through a
spectrum-space decoy E-value and the self-retrieval ground-truth FDR curve.

See module docstrings in :mod:`annotation.cosmic.score`,
:mod:`annotation.cosmic.decoys`, and :mod:`annotation.cosmic.calibration`.
"""
from .decoys import (
    DECOYS,
    DecoyGenerator,
    ShuffleIntensityDecoy,
    ShuffleMZDecoy,
    StructureSpaceDecoy,
)
from .score import roc_auc, rule_coherence_scores
from .calibration import (
    build_truth_fdr_curve,
    calibrated_fdr,
    decoy_evalue,
    decoy_fraction_at_least,
)

__all__ = [
    "DECOYS",
    "DecoyGenerator",
    "ShuffleIntensityDecoy",
    "ShuffleMZDecoy",
    "StructureSpaceDecoy",
    "roc_auc",
    "rule_coherence_scores",
    "build_truth_fdr_curve",
    "calibrated_fdr",
    "decoy_evalue",
    "decoy_fraction_at_least",
]
