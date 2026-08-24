"""Deployable DreaMS + P2b selective-rerank modules.

This package is the *deployment-clean* subset of the repo: the frozen,
validated pieces that are safe to import into a production annotation
pipeline, kept separate from the experiment scripts under ``tasks/``.

Contents
--------
- :mod:`deploy.p2b_rank_fusion` -- frozen P2b local candidate rank fusion
  (the module validated on the sealed P3 main panel).
- ``deploy/p2b_config.json`` -- the frozen P2b configuration (weights,
  normalization, gates) and its provenance hashes.

The DreaMS encoder and the M0-M9 annotation platform live in ``annotation/``
(embed -> retrieve -> confidence -> FDR -> calibrate -> diff -> pathway ->
darkmatter).  ``deploy`` adds the frozen P2b re-ranking step on top of
DreaMS retrieval.
"""

from .p2b_rank_fusion import (
    FROZEN_CONFIG,
    FusionConfiguration,
    frozen_weights,
    fusion_configuration_from_mapping,
    fuse_one_query,
    grouped_max,
    normalize_pair_features,
    strict_rank,
    unique_top_index,
    validate_ptr,
)

__all__ = [
    "FROZEN_CONFIG",
    "FusionConfiguration",
    "frozen_weights",
    "fusion_configuration_from_mapping",
    "fuse_one_query",
    "grouped_max",
    "normalize_pair_features",
    "strict_rank",
    "unique_top_index",
    "validate_ptr",
]
