from __future__ import annotations

import pandas as pd

from run_contrast_atlas_batch_retrieval import batch_retrieve
from test_contrast_atlas_core import fixtures


def test_batch_retrieval_binds_query_registry_and_deduplicates_patterns() -> None:
    effects, loadings, links = fixtures()
    registry = pd.DataFrame([
        {"query_id": "q", "query_study_id": "Q", "query_contrast_id": "q", "cluster_id": "cq", "source_holdout": "sq", "instrument_holdout": "iq", "data_role": "synthetic"},
        {"query_id": "s", "query_study_id": "S", "query_contrast_id": "s", "cluster_id": "cs", "source_holdout": "ss", "instrument_holdout": "is", "data_role": "synthetic"},
    ])
    result, patterns = batch_retrieve(
        effects, loadings, links, registry,
        min_entities=3, min_patterns=2, min_coverage=0.5,
        min_match_probability=0.8, association_threshold=0.3,
        allow_same_source=False,
    )
    assert set(result["query_id"]) == {"q", "s"}
    assert not result.duplicated(["query_id", "world", "target_study_id", "target_contrast_id"]).any()
    assert not patterns.duplicated(["world", "study_id", "contrast_id", "pattern_id"]).any()
    assert set(result["query_data_role"]) == {"synthetic"}


def test_batch_retrieval_preserves_candidate_universe_for_filtered_world() -> None:
    universe, universe_loadings, universe_links = fixtures()
    evidence = universe.loc[universe["identity_status"].ne("unresolved_entity")].copy()
    keys = set(zip(evidence["study_id"], evidence["contrast_id"], evidence["entity_id"]))
    loadings = universe_loadings.loc[
        [key in keys for key in zip(universe_loadings["study_id"], universe_loadings["contrast_id"], universe_loadings["entity_id"])]
    ].copy()
    links = universe_links.loc[
        universe_links["query_entity_id"].str.endswith(("0", "1"))
        & universe_links["target_entity_id"].str.endswith(("0", "1"))
    ].copy()
    registry = pd.DataFrame([{
        "query_id": "l", "query_study_id": "L", "query_contrast_id": "l",
        "cluster_id": "cl", "source_holdout": "sl", "instrument_holdout": "il",
        "data_role": "synthetic",
    }])
    result, _ = batch_retrieve(
        evidence, loadings, links, registry,
        universe_effects=universe, identity_worlds=("all",),
        min_entities=3, min_patterns=2, min_coverage=0.5,
        min_match_probability=0.8, association_threshold=0.3,
        allow_same_source=False,
    )
    assert len(result) == 3
    assert set(result["target_study_id"]) == {"Q", "S", "O"}
    assert result["study_contrast_association"].eq("abstain_insufficient_comparable_evidence").all()
