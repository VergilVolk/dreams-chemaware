from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from audit_mtbls13729_contrast_atlas_readiness import audit


def make_fixture(root: Path, complete: bool, formal: bool) -> tuple[Path, Path, Path, Path, Path]:
    mzml = root / "mzml"
    full = root / "full"
    links = root / "links"
    for path in (mzml / "neg", full, links):
        path.mkdir(parents=True, exist_ok=True)
    samples = pd.DataFrame([
        {"panel": "neg", "sample_id": "P1-N", "patient_id": "P1", "tissue": "normal", "histology": "normal", "hdf5_path": str(mzml / "neg" / "P1-N.hdf5")},
        {"panel": "neg", "sample_id": "P1-T", "patient_id": "P1", "tissue": "tumor", "histology": "tumor", "hdf5_path": str(mzml / "neg" / "P1-T.hdf5")},
    ])
    manifest = root / "manifest.csv"
    samples.to_csv(manifest, index=False)
    for sample in (["P1-N", "P1-T"] if complete else ["P1-N"]):
        (mzml / "neg" / f"{sample}.mzML").write_text("x", encoding="utf-8")
    pd.DataFrame({"feature_id": [1, 2], "P1-N": [1, 2], "P1-T": [2, 3]}).to_csv(
        full / "neg__eic_auc_matrix.csv.gz", index=False, compression="gzip",
    )
    pd.DataFrame({"feature_id": [1, 2], "P1-N": [1, 1], "P1-T": [1, 1]}).to_csv(
        full / "neg__eic_detection_matrix.csv.gz", index=False, compression="gzip",
    )
    pd.DataFrame({"feature_id": [1], "query_file": ["P1-N"]}).to_csv(
        links / "neg__linked_ms2.csv.gz", index=False, compression="gzip",
    )
    pilot = root / "pilot.json"
    pilot.write_text(json.dumps({"panels": [{
        "panel": "neg", "formal": formal,
        "matrix_kind": "uniform_requantified_eic" if formal else "discovery_missingness_pilot",
    }]}), encoding="utf-8")
    return mzml, manifest, full, links, pilot


def test_readiness_accepts_complete_reproducible_panel(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path, complete=True, formal=True)
    report = audit(*paths)
    assert report["pass_to_contrast_record"]
    assert report["admissible_panels"] == ["neg"]


def test_readiness_rejects_incomplete_discovery_pilot(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path, complete=False, formal=False)
    report = audit(*paths)
    assert not report["pass_to_contrast_record"]
    assert report["admissible_panels"] == []
