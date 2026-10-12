import json
import subprocess
import sys

import pandas as pd


def test_certificate_assignment_and_dustbin(tmp_path):
    edges = pd.DataFrame([
        ["q1", "s1", "u1", "src1", "i1", "wse", .9],
        ["q1", "s2", "u1", "src2", "i2", "wse", .8],
        ["q1", "s1", "u1", "src1", "i1", "noise", .9],
        ["q1", "s2", "u1", "src2", "i2", "noise", .8],
        ["q2", "s3", "u2", "src1", "i1", "wse", .9],
        ["q2", "s3", "u2", "src1", "i1", "noise", .9],
    ], columns=["query_id", "public_spectrum_id", "public_unit_id", "public_source_id",
                "public_instrument", "evidence_family", "score"])
    edge_path = tmp_path / "edges.csv"
    threshold_path = tmp_path / "thresholds.json"
    universe_path = tmp_path / "queries.csv"
    output = tmp_path / "out"
    edges.to_csv(edge_path, index=False)
    pd.DataFrame({"query_id": ["q1", "q2", "q3"]}).to_csv(universe_path, index=False)
    threshold_path.write_text(json.dumps({"wse": .7, "noise": .7}), encoding="utf-8")
    subprocess.run([sys.executable, "tasks/build_public_spectral_evidence_certificates.py",
                    "--edges", str(edge_path), "--thresholds", str(threshold_path),
                    "--query-universe", str(universe_path), "--output", str(output)], check=True)
    result = pd.read_csv(output / "certificates.csv").set_index("query_id")
    assert result.loc["q1", "assigned_public_unit"] == "u1"
    assert result.loc["q2", "assigned_public_unit"] == "__NO_MATCH__"
    assert result.loc["q3", "assigned_public_unit"] == "__NO_MATCH__"


def test_certificate_rejects_excluded_source(tmp_path):
    edges = pd.DataFrame([
        ["q1", "s1", "u1", "lcnec", "i1", "wse", .9],
    ], columns=["query_id", "public_spectrum_id", "public_unit_id", "public_source_id",
                "public_instrument", "evidence_family", "score"])
    edge_path = tmp_path / "edges.csv"
    threshold_path = tmp_path / "thresholds.json"
    excluded_path = tmp_path / "excluded.csv"
    edges.to_csv(edge_path, index=False)
    threshold_path.write_text(json.dumps({"wse": .7}), encoding="utf-8")
    pd.DataFrame({"public_source_id": ["lcnec"]}).to_csv(excluded_path, index=False)
    failed = subprocess.run([
        sys.executable, "tasks/build_public_spectral_evidence_certificates.py",
        "--edges", str(edge_path), "--thresholds", str(threshold_path),
        "--excluded-public-sources", str(excluded_path), "--output", str(tmp_path / "out"),
    ], capture_output=True, text=True)
    assert failed.returncode != 0
    assert "excluded public sources" in failed.stderr
