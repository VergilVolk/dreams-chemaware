import subprocess
import sys

import pandas as pd


def test_matched_null_preserves_coverage_scores_and_stratum_counts(tmp_path):
    certificates = pd.DataFrame([
        ["q1", "u1", .9], ["q2", "u2", .8], ["q3", "u3", .7],
        ["q4", "__NO_MATCH__", 0.0],
    ], columns=["query_id", "assigned_public_unit", "certificate_score"])
    strata = pd.DataFrame([
        ["q1", "neg", 1], ["q2", "neg", 1], ["q3", "neg", 1], ["q4", "neg", 1],
    ], columns=["query_id", "polarity", "mass_bin"])
    certificate_path = tmp_path / "certificates.csv"
    strata_path = tmp_path / "strata.csv"
    output = tmp_path / "out"
    certificates.to_csv(certificate_path, index=False)
    strata.to_csv(strata_path, index=False)
    subprocess.run([
        sys.executable, "tasks/shuffle_public_evidence_certificates.py",
        "--certificates", str(certificate_path), "--query-strata", str(strata_path),
        "--strata", "polarity,mass_bin", "--repeats", "5", "--output", str(output),
    ], check=True)
    shuffled = pd.read_csv(output / "shuffled_certificates.csv.gz")
    for _, frame in shuffled.groupby("null_repeat"):
        observed = frame.sort_values("query_id").reset_index(drop=True)
        expected = certificates.sort_values("query_id").reset_index(drop=True)
        assert observed["certificate_score"].equals(expected["certificate_score"])
        assert observed.loc[observed["query_id"] == "q4", "assigned_public_unit"].iloc[0] == "__NO_MATCH__"
        assert sorted(observed.loc[observed["query_id"] != "q4", "assigned_public_unit"]) == ["u1", "u2", "u3"]
