from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from build_lcnec_hsst3n_contrast_atlas_records import build


def synthetic_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    rows = []
    for index in range(9):
        rows.append({"SAMPLE_ID": f"QC{index}", "SAMPLE_CODE": "", "GROUP_CODE": "", "NOTE": "QC sample", "AMOUNT": np.nan, "INJECTION_ID": f"Inj{index + 1:03d}"})
    for index in range(2):
        rows.append({"SAMPLE_ID": f"BL{index}", "SAMPLE_CODE": "", "GROUP_CODE": "", "NOTE": "Method blank", "AMOUNT": np.nan, "INJECTION_ID": f"Inj{index + 10:03d}"})
    for index in range(6):
        rows.append({"SAMPLE_ID": f"SD{index}", "SAMPLE_CODE": "", "GROUP_CODE": "", "NOTE": "Serial dilution", "AMOUNT": np.nan, "INJECTION_ID": f"Inj{index + 12:03d}"})
    for patient in range(34):
        for group in ("TU", "NG"):
            rows.append({
                "SAMPLE_ID": f"S{patient}_{group}", "SAMPLE_CODE": f"P{patient}",
                "GROUP_CODE": group, "NOTE": "Study sample",
                "AMOUNT": 10.0 + patient % 3, "INJECTION_ID": f"Inj{len(rows) + 1:03d}",
            })
    overview = pd.DataFrame(rows)
    families = np.arange(263)
    area = np.zeros((85, 263), dtype=float)
    for row_index, row in overview.iterrows():
        base = 1000.0 + families * 2.0 + row_index
        if row["NOTE"] == "Study sample":
            patient = int(str(row["SAMPLE_CODE"])[1:])
            biological = 1.0 + 0.03 * np.sin(patient + families / 11.0)
            effect = np.where(families % 2 == 0, 1.25 + 0.01 * patient, 0.85 + 0.005 * patient)
            base = base * float(row["AMOUNT"]) * biological
            if row["GROUP_CODE"] == "TU":
                base = base * effect
        area[row_index] = base
    matrix = tmp_path / "matrix.npz"
    np.savez_compressed(
        matrix, area=area, family_id=families,
        sample_id=np.asarray(overview["SAMPLE_ID"].astype(str).tolist(), dtype="U32"),
        sample_code=np.asarray(overview["SAMPLE_CODE"].astype(str).tolist(), dtype="U32"),
        group_code=np.asarray(overview["GROUP_CODE"].astype(str).tolist(), dtype="U8"),
        file_class=np.where(overview["NOTE"].eq("Study sample"), "study", "control"),
        mz=100.0 + families, rt_sec=20.0 + families,
    )
    overview_path = tmp_path / "overview.tsv"
    overview.to_csv(overview_path, sep="\t", index=False)
    ledger_path = tmp_path / "ledger.csv"
    pd.DataFrame({"family_id": families}).to_csv(ledger_path, index=False)
    spectra_path = tmp_path / "spectra.csv"
    pd.DataFrame({"family_id": families, "n_peaks": 10 + families % 5}).to_csv(spectra_path, index=False)
    return matrix, overview_path, ledger_path, spectra_path


def test_lcnec_builder_preserves_patient_denominator_and_avoids_effect_filter(tmp_path: Path) -> None:
    records, unit_effects, covariates, sensitivity, report = build(*synthetic_inputs(tmp_path))
    assert len(records) == 263
    assert records["qc_pass"].all()
    assert records["identity_status"].eq("unresolved_entity").all()
    assert unit_effects["biological_unit_id"].nunique() == 34
    assert len(unit_effects) == 34 * 263
    observed = unit_effects.groupby("entity_id")["effect"].mean()
    expected = records.set_index("entity_id")["beta"]
    assert np.allclose(observed.loc[expected.index], expected)
    assert set(covariates["entity_id"]) == set(records["entity_id"])
    assert len(sensitivity) == 263
    assert report["effect_significance_used_for_selection"] is False
