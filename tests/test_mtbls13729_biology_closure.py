from __future__ import annotations

import numpy as np
import pandas as pd
import json
import subprocess
import sys
from pathlib import Path

from tasks.audit_mtbls13729_c20_4_ms2 import spectrum_evidence
from tasks.audit_mtbls13729_frozen_ion_families import family_rows


def test_spectrum_evidence_requires_carnitine_pattern() -> None:
    precursor = 448.33946255
    mz = np.asarray([60.0808, 85.0284, 144.1019, precursor - 59.0735, 200.0])
    intensity = np.asarray([10.0, 50.0, 20.0, 30.0, 100.0])
    evidence = spectrum_evidence(mz, intensity, precursor, 0.02, 0.005)
    assert evidence["diagnostic_motif_count"] == 4
    assert evidence["strong_carnitine_motif"] is True


def test_family_rows_collapses_adduct_duplicate_without_relabelling_fdr() -> None:
    frame = pd.DataFrame({
        "panel": ["pos_rp", "pos_rp"],
        "ion_family_id": [7, 7],
        "ion_family_size": [2, 2],
        "feature_id": [1597, 7489],
        "mz": [298.114285, 320.097232],
        "rt_sec": [217.0, 219.96],
        "discovery_priority": [True, True],
        "primary_rmu_fdr10_robust": [True, True],
        "max_rmu_q_across_normalizations": [0.0458, 0.0652],
        "e6_fixed_v2_sw2_name": ["unknown [M+H]+", "Nelarabine [M+Na]+"],
    })
    families = family_rows(frame)
    assert len(families) == 1
    row = families.iloc[0]
    assert row.n_primary_fdr10_features_in_family == 2
    assert row.representative_feature_id == 1597
    assert bool(row.candidate_name_conflict)


def test_closure_summary_preserves_primary_and_secondary_endpoints(tmp_path: Path) -> None:
    family = {
        "status": "mtbls13729_frozen_ion_family_audit_complete",
        "formal_global_peak_graph": True,
        "panels": {"pos_rp": {
            "primary_fdr10_features": 6,
            "primary_fdr10_descriptive_ion_families": 5,
        }},
    }
    anchor = {
        "status": "mtbls13729_c20_4_ms2_audit_complete",
        "samples_with_matching_ms2": 10,
        "samples_with_strong_motif": 9,
        "matching_ms2_spectra": 20,
        "strong_motif_spectra": 18,
    }
    endpoint = {
        "Rmu_vs_RN": {"n_pairs": 10, "mean_class_log2fc": 1.2},
        "Rtu_vs_RN": {"n_pairs": 10, "mean_class_log2fc": 0.2},
        "interaction": {"difference_in_mean_class_log2fc": 1.0},
    }
    class_report = {
        "status": "complete",
        "selection_is_phenotype_blind": True,
        "variants": {"pqn": endpoint},
        "chain_collapsed_sensitivity": {"variants": {"pqn": endpoint}},
    }
    paths = {}
    for name, payload in (("family", family), ("anchor", anchor), ("class", class_report)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = path
    output = tmp_path / "output" / "report.json"
    subprocess.run([
        sys.executable,
        "tasks/summarize_mtbls13729_biology_closure.py",
        "--family-report", str(paths["family"]),
        "--anchor-report", str(paths["anchor"]),
        "--class-report", str(paths["class"]),
        "--output", str(output),
    ], check=True, capture_output=True, text=True)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["feature_to_ion_family_reconciliation"]["features_removed_as_redundant_ion_forms"] == 1
    assert result["gates"]["pass"] is True
