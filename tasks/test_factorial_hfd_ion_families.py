from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


def test_conservative_isotope_family_is_phenotype_blind_and_selects_one_representative(tmp_path: Path) -> None:
    panel = tmp_path / "panel"
    panel.mkdir()
    (panel / "report.json").write_text(json.dumps({
        "status": "development_consumed_abundance_matrix_frozen"
    }), encoding="utf-8")
    anchors = pd.DataFrame({
        "anchor_id": ["A1", "A2", "A3"],
        "precursor_mz": [100.0, 101.003354835, 150.0],
        "rt_seconds": [50.0, 50.4, 80.0],
        "representative_tic": [100.0, 200.0, 300.0],
    })
    anchors.to_csv(panel / "anchor_manifest.tsv", sep="\t", index=False)
    base = np.arange(1, 11, dtype=np.float32)
    values = np.vstack([base, base * 2.0, base[::-1]])
    np.savez_compressed(
        panel / "ms1_peak_height.npz", values=values,
        anchor_id=np.asarray(["A1", "A2", "A3"]), sample_id=np.asarray([f"S{x}" for x in range(10)]),
    )
    registry = tmp_path / "registry.tsv"
    pd.DataFrame([{
        "panel_id": "P1", "dataset_accession": "S1", "source_family": "F1",
        "assay_file": "LC-MS_POS_serum.txt",
    }]).to_csv(registry, sep="\t", index=False)
    output = tmp_path / "families"
    subprocess.run([
        sys.executable, str(Path(__file__).with_name("build_factorial_hfd_ion_families.py")),
        "--panel-object", str(panel), "--panel-registry", str(registry),
        "--panel-id", "P1", "--output-dir", str(output),
    ], check=True, capture_output=True, text=True)
    membership = pd.read_csv(output / "ion_family_membership.tsv", sep="\t")
    a1 = membership.loc[membership.anchor_id.eq("A1")].iloc[0]
    a2 = membership.loc[membership.anchor_id.eq("A2")].iloc[0]
    a3 = membership.loc[membership.anchor_id.eq("A3")].iloc[0]
    assert a1.family_id == a2.family_id
    assert a3.family_id != a1.family_id
    assert not bool(a1.representative)
    assert bool(a2.representative)
    assert bool(a3.representative)
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["denominators"]["families"] == 2
    assert report["leakage_status"] == "no genotype_diet_effect_identity_or_cross_study_truth_read"
