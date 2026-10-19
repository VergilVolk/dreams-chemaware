from __future__ import annotations

import pandas as pd

from build_pan_redu_pretruth_registry import build_registry


def test_same_metadata_stratum_is_frozen_as_unknown_not_truth() -> None:
    topology = pd.DataFrame([
        {"panel_id": "P1", "contrast_id": "C1", "preliminary_source_family": "S1", "comparison_stratum": "sex", "topology_status": "PASS"},
        {"panel_id": "P1", "contrast_id": "C2", "preliminary_source_family": "S1", "comparison_stratum": "age", "topology_status": "PASS"},
        {"panel_id": "P2", "contrast_id": "C3", "preliminary_source_family": "S2", "comparison_stratum": "sex", "topology_status": "PASS"},
    ])
    pairs = pd.DataFrame([{
        "left_panel_id": "P1", "right_panel_id": "P2",
        "left_source_family": "S1", "right_source_family": "S2",
    }])
    registry = build_registry(topology, pairs)
    assert len(registry) == 2
    assert set(registry.truth_class) == {"unknown"}
    assert not registry.eligible_primary.any()
    assert registry.selection_blind_to_transport_scores.all()
