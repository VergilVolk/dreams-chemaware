from __future__ import annotations

import pandas as pd
import pytest

from export_ibd_contrast_atlas_source_artifacts import (
    ADULT_RELATION_CONTRAST,
    ARTIFACT_COLUMNS,
    validate_artifact,
)


def artifact() -> pd.DataFrame:
    rows = []
    for contrast in ("case_vs_control", "subtype_vs_control"):
        for entity in ("e1", "e2"):
            rows.append({
                "study_id": "study",
                "source_family": "family",
                "source_accessions": "A;B",
                "contrast_id": contrast,
                "entity_id": entity,
                "raw_feature_id": entity,
                "ms2_join_id": entity,
                "ms2_mgf_path": "x.mgf",
                "ms2_precursor_mz": 100.0,
                "quant_precursor_mz": 100.0,
                "beta": 0.2,
                "se": 0.1,
                "n_positive": 10,
                "n_negative": 10,
                "identity_status": "unresolved_entity",
                "identity_partition_contract": "test",
                "effect_model": "test",
            })
    return pd.DataFrame(rows, columns=ARTIFACT_COLUMNS)


def test_complete_fixed_entity_universe_passes() -> None:
    validate_artifact(artifact(), "study", {"case_vs_control", "subtype_vs_control"})


def test_export_scope_is_the_registered_near_exact_adult_contrast() -> None:
    assert ADULT_RELATION_CONTRAST == "adult_all_ibd_vs_healthy"


def test_missing_ms2_join_fails_closed() -> None:
    frame = artifact()
    frame.loc[0, "ms2_join_id"] = None
    with pytest.raises(RuntimeError, match="missing MS2 join"):
        validate_artifact(frame, "study", {"case_vs_control", "subtype_vs_control"})


def test_contrast_specific_entity_universe_fails_closed() -> None:
    frame = artifact().loc[lambda x: ~((x.contrast_id == "subtype_vs_control") & (x.entity_id == "e2"))]
    with pytest.raises(RuntimeError, match="candidate entity universe changes"):
        validate_artifact(frame, "study", {"case_vs_control", "subtype_vs_control"})


def test_nonpositive_se_fails_closed() -> None:
    frame = artifact()
    frame.loc[0, "se"] = 0.0
    with pytest.raises(RuntimeError, match="non-finite response"):
        validate_artifact(frame, "study", {"case_vs_control", "subtype_vs_control"})
