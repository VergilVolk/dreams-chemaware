import json

import pandas as pd

from tasks.audit_st001154_bioaware_external_readiness import (
    align_targets,
    analysis_record,
    read_concatenated_json,
    truth_rows,
)


def test_concatenated_json_and_truth_selection(tmp_path):
    path = tmp_path / "mwtab.json"
    first = {"METABOLOMICS WORKBENCH": {"ANALYSIS_ID": "A"}}
    second = {
        "METABOLOMICS WORKBENCH": {"ANALYSIS_ID": "B"},
        "MS_METABOLITE_DATA": {
            "Metabolites": [
                {
                    "Metabolite": "x",
                    "Type": "TargetCPD",
                    "Adduct": "[M-H]-",
                    "AnnotationApproach": "MZ_MSMS",
                    "retention times": "1.5",
                    "m/z": "100.0",
                    "InChiKey": "ABCDEFGHIJKLMN-UHFFFAOYSA-N",
                }
            ]
        },
    }
    path.write_text(json.dumps(first) + "\n" + json.dumps(second), encoding="utf-8")
    records = read_concatenated_json(path)
    assert len(records) == 2
    frame = truth_rows(analysis_record(records, "B"))
    assert frame.iloc[0]["ik14"] == "ABCDEFGHIJKLMN"
    assert frame.iloc[0]["target_rt_sec"] == 90.0


def test_alignment_is_ppm_rt_bounded_and_deterministic():
    targets = pd.DataFrame(
        [{"ik14": "ABCDEFGHIJKLMN", "target_mz": 100.0, "target_rt_sec": 60.0}]
    )
    ms2 = pd.DataFrame(
        [
            {"native_id": "far", "observed_precursor_mz": 100.0004, "observed_rt_sec": 80.0},
            {"native_id": "best", "observed_precursor_mz": 100.0001, "observed_rt_sec": 62.0},
            {"native_id": "second", "observed_precursor_mz": 100.0002, "observed_rt_sec": 61.0},
        ]
    )
    aligned = align_targets(targets, ms2, ppm=5.0, rt_seconds=6.0)
    assert len(aligned) == 1
    assert aligned.iloc[0]["native_id"] == "best"
    assert aligned.iloc[0]["matched_scans_in_window"] == 2


def test_source_uses_full_inchikey_before_ik14_fallback():
    from pathlib import Path

    source = Path("tasks/audit_st001154_bioaware_external_readiness.py").read_text(
        encoding="utf-8"
    )
    assert '"exact_full_inchikey"' in source
    assert '"unambiguous_ik14_fallback"' in source
    assert "approved MONA full InChIKeys map to multiple formulas" in source
