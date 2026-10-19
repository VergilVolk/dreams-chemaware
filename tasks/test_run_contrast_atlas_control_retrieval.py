from __future__ import annotations

from pathlib import Path

import pandas as pd

from build_contrast_atlas_control_worlds import filter_world, write_world
from run_contrast_atlas_control_retrieval import run_control_retrievals
from test_contrast_atlas_core import fixtures


def test_control_retrieval_uses_one_candidate_universe(tmp_path: Path) -> None:
    effects, loadings, links = fixtures()
    all_keys = set(zip(effects["study_id"], effects["contrast_id"], effects["entity_id"]))
    known_keys = set(
        zip(
            effects.loc[effects["identity_status"].ne("unresolved_entity"), "study_id"],
            effects.loc[effects["identity_status"].ne("unresolved_entity"), "contrast_id"],
            effects.loc[effects["identity_status"].ne("unresolved_entity"), "entity_id"],
        )
    )
    matched_keys = {
        (row.study_id, row.contrast_id, row.entity_id)
        for row in effects.loc[effects["identity_status"].ne("unresolved_entity")]
        .groupby(["study_id", "contrast_id"], sort=False).head(1).itertuples(index=False)
    }
    tables = {
        "all": filter_world(effects, loadings, links, all_keys),
        "known_only": filter_world(effects, loadings, links, known_keys),
        "measurement_matched_known": filter_world(effects, loadings, links, matched_keys),
        "unresolved_shuffled_ms2": filter_world(effects, loadings, links, all_keys),
    }
    for name, world_tables in tables.items():
        write_world(tmp_path / name, *world_tables)
    registry = pd.DataFrame([
        {
            "query_id": "q", "query_study_id": "Q", "query_contrast_id": "q",
            "cluster_id": "cq", "source_holdout": "sq", "instrument_holdout": "iq",
            "data_role": "synthetic",
        },
        {
            "query_id": "l", "query_study_id": "L", "query_contrast_id": "l",
            "cluster_id": "cl", "source_holdout": "sl", "instrument_holdout": "il",
            "data_role": "synthetic",
        },
    ])
    results, patterns, counts = run_control_retrievals(
        tmp_path, effects, registry,
        min_entities=3, min_patterns=2, min_coverage=0.5,
        min_match_probability=0.8, association_threshold=0.3,
        allow_same_source=False,
    )
    assert set(results["world"]) == set(tables)
    assert results.groupby(["query_id", "world"]).size().nunique() == 1
    assert set(patterns["world"]) == set(tables)
    assert set(counts) == set(tables)
    matched_l = results.loc[
        results["query_id"].eq("l")
        & results["world"].eq("measurement_matched_known")
    ]
    assert matched_l["study_contrast_association"].str.startswith("abstain").all()
