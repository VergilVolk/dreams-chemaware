import pandas as pd

from tasks.build_st001154_hilic_bioaware_external_manifest import (
    candidate_rows_for_mz,
    exclusive_assignments,
)


def test_exclusive_assignment_rejects_shared_scan():
    targets = pd.DataFrame(
        [
            {"ik14": "A", "target_mz": 100.0, "target_rt_sec": 60.0},
            {"ik14": "B", "target_mz": 100.0, "target_rt_sec": 60.0},
        ]
    )
    spectra = pd.DataFrame(
        [{"native_id": "shared", "observed_precursor_mz": 100.0, "observed_rt_sec": 60.0}]
    )
    assert exclusive_assignments(targets, spectra, 10.0, 6.0).empty


def test_candidate_window_is_mass_only():
    approved = pd.DataFrame(
        {"precursor_mz": [99.99, 100.0, 100.0009, 100.01], "ik14": ["A", "B", "C", "D"]}
    )
    result = candidate_rows_for_mz(approved, 100.0, 10.0)
    assert result["ik14"].tolist() == ["B", "C"]
