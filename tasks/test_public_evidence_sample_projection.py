import json
import subprocess
import sys

import numpy as np
import pandas as pd


def test_projection_keeps_dustbin_and_rejects_outcomes(tmp_path):
    abundance = pd.DataFrame([
        ["s1", "q1", 3.0, 1.0],
        ["s1", "q2", 1.0, 1.0],
        ["s2", "q1", 0.0, 1.0],
        ["s2", "q2", 3.0, 0.5],
    ], columns=["sample_id", "query_id", "abundance", "reliability"])
    certificates = pd.DataFrame([
        ["q1", "u1", 0.5],
        ["q2", "__NO_MATCH__", 0.0],
    ], columns=["query_id", "assigned_public_unit", "certificate_score"])
    abundance_path = tmp_path / "abundance.csv"
    certificate_path = tmp_path / "certificates.csv"
    output = tmp_path / "out"
    abundance.to_csv(abundance_path, index=False)
    certificates.to_csv(certificate_path, index=False)
    subprocess.run([
        sys.executable, "tasks/project_public_evidence_certificates_to_samples.py",
        "--abundance", str(abundance_path), "--certificates", str(certificate_path),
        "--output", str(output),
    ], check=True)

    signature = pd.read_csv(output / "sample_unit_signature.csv.gz")
    qc = pd.read_csv(output / "sample_coverage_qc.csv").set_index("sample_id")
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert len(signature) == 1
    assert signature.iloc[0]["assigned_public_unit"] == "u1"
    assert np.isclose(qc.loc["s1", "certified_coverage"], np.log(4) / (np.log(4) + np.log(2)))
    assert qc.loc["s2", "certified_coverage"] == 0.0
    assert qc.loc["s2", "dustbin_fraction"] == 1.0
    assert report["contracts"]["phenotype_blind"] is True

    bad = abundance.rename(columns={"reliability": "disease_label"})
    bad_path = tmp_path / "bad.csv"
    bad.to_csv(bad_path, index=False)
    failed = subprocess.run([
        sys.executable, "tasks/project_public_evidence_certificates_to_samples.py",
        "--abundance", str(bad_path), "--certificates", str(certificate_path),
        "--output", str(tmp_path / "bad-out"),
    ], capture_output=True, text=True)
    assert failed.returncode != 0
    assert "forbidden outcome columns" in failed.stderr
