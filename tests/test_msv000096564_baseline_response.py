from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


TASKS = Path(__file__).resolve().parents[1] / "tasks"
if str(TASKS) not in sys.path:
    sys.path.insert(0, str(TASKS))

from build_msv000096564_baseline_response import (  # noqa: E402
    baseline_design,
    identity_partition,
    participant_states,
)


def _contract() -> dict:
    return {
        "contrast": {
            "expected_technical_files_per_unit": 3,
            "expected_biological_units": {"control": 1, "ibd": 1},
        }
    }


def test_baseline_design_collapses_only_t1_technical_files() -> None:
    metadata = pd.DataFrame({
        "host_subject_id": ["800805-001", "800805-002", "800805-002"],
        "disease_state": ["control", "ibd_remission", None],
    })
    columns = [
        *(f"15663_800805_1_T1_fecal_{i}.mzML Peak area" for i in range(1, 4)),
        *(f"15663_800805_2_T1_fecal_{i}.mzML Peak area" for i in range(1, 4)),
        "15663_800805_1_T2_fecal_1.mzML Peak area",
    ]
    design = baseline_design(columns, metadata, _contract())
    assert len(design) == 6
    assert set(design.level) == {"control", "ibd"}
    assert not design.filename.str.contains("_T2_").any()
    assert design.groupby("biological_unit_id").size().eq(3).all()


def test_participant_state_ignores_duplicate_empty_metadata_row() -> None:
    metadata = pd.DataFrame({
        "host_subject_id": ["800805-024", "800805-024"],
        "disease_state": ["ibd_remission", None],
    })
    assert participant_states(metadata) == {24: "ibd"}


def test_provisional_lipid_name_does_not_become_known_identity() -> None:
    quant = pd.DataFrame({
        "row ID": ["1", "2"],
        "row m/z": [100.0, 200.0],
        "row retention time": [1.0, 2.0],
    })
    annotations = pd.DataFrame({
        "id": [1], "compound_name": ["LPC 18:1"], "score": [0.8], "method": ["matchedlipid"]
    })
    result = identity_partition(quant, annotations)
    assert set(result.identity_status) == {"unresolved_entity"}
    assert result.source_annotation_status.tolist() == ["provisional_mzmine_matchedlipid", "none"]
