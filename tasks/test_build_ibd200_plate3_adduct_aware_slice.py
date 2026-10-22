from pathlib import Path

import pandas as pd

from build_ibd200_plate3_adduct_aware_slice import neutral_candidate_tables


def test_neutral_candidate_tables_respects_neutral_mass_and_abstains(tmp_path: Path):
    manifest = tmp_path / "library.csv"
    pd.DataFrame({
        "row": [1, 2],
        "inchikey": ["A", "B"],
        "smiles": ["C", "CC"],
        "formula": ["CH4", "C2H6"],
        "exact_mass": [100.0005, 122.0],
        "precursor_mz": [101.0078, 123.0078],
        "quality_label": ["gold", "gold"],
        "n_peaks": [10, 10],
        "spectrum_id": ["s1", "s2"],
    }).to_csv(manifest, index=False)
    queries = pd.DataFrame({
        "entity_id": ["q1", "q2"],
        "precursor_mz": [123.0, 101.0],
        "best_ion": ["[M+Na]+", ""],
        "neutral_mass": [100.0, float("nan")],
    })
    spectra, structures = neutral_candidate_tables(queries, manifest, 10.0)
    assert spectra["entity_id"].tolist() == ["q1"]
    assert structures["candidate_id"].tolist() == ["A"]
    assert structures["cross_adduct_reference"].tolist() == [True]
