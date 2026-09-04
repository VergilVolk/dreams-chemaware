import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("tasks/audit_metabolic_network_framework_reproducibility.py")


def touch_all(root: Path, relative_paths: tuple[str, ...]) -> None:
    for relative in relative_paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")


def run_audit(tmp_path: Path, complete_metdna3: bool) -> dict:
    metdna3 = tmp_path / "metdna3"
    metdna2 = tmp_path / "metdna2"
    output = tmp_path / "report.json"
    metdna3_code = (
        "R/MRN3main.R",
        "R/MRN3annotation.R",
        "R/MRN3rmRedun.R",
        "R/MRN3preparation.R",
    )
    metdna3_assets = (
        "obj_mrn.rda",
        "info_mrn.rda",
        "md_mrn.rda",
        "obj_mrn_3x1.rda",
        "obj_mrn_3x2.rda",
        "info_mrn_3x.rda",
        "md_mrn_3x.rda",
    )
    metdna2_code = (
        "R/MetDNA2.R",
        "R/AnnotationCredential.R",
        "R/AnnotationCredentialFormula.R",
        "R/AnnotationCredentialPeakGroup.R",
    )
    metdna2_assets = (
        "data/reaction_pair_network.rda",
        "data/md_mrn_emrn.rda",
        "data/lib_adduct_nl.rda",
        "data/lib_formula.rda",
        "data/lib_kegg.rda",
    )
    touch_all(metdna3, metdna3_code)
    if complete_metdna3:
        touch_all(metdna3, metdna3_assets)
    touch_all(metdna2, metdna2_code + metdna2_assets)
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--metdna3",
            str(metdna3),
            "--metdna2",
            str(metdna2),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def test_core_only_selects_reproducible_fallback(tmp_path: Path) -> None:
    report = run_audit(tmp_path, complete_metdna3=False)
    assert report["metdna3"]["core_only"] is True
    assert report["metdna3"]["exact_reproduction_available"] is False
    assert report["kgmn_metdna2"]["reproducible_baseline_ready"] is True
    assert report["decision"] == "kgmn_metdna2_reproducible_fallback"


def test_complete_metdna3_takes_precedence(tmp_path: Path) -> None:
    report = run_audit(tmp_path, complete_metdna3=True)
    assert report["metdna3"]["exact_reproduction_available"] is True
    assert report["decision"] == "exact_metdna3"
