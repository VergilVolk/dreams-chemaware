"""Fail-closed validation for the BioAware algorithm-to-biology bridge audit."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/mtbls13729/bioaware_algorithm_biology_bridge_v1"


def main() -> None:
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    benchmark = pd.read_csv(OUT / "bioaware_benchmark_ledger.csv")
    failures = pd.read_csv(OUT / "bioaware_failure_decomposition.csv")
    roles = pd.read_csv(OUT / "mtbls13729_bioaware_role_ledger.csv")
    gaps = pd.read_csv(OUT / "frontier_method_gap_matrix.csv")
    text = (OUT / "REPORT.md").read_text(encoding="utf-8")

    assert report["formal"] is True
    assert report["algorithm_verdict"]["statistically_confirmed_external_gain"] is False
    assert report["algorithm_verdict"]["sota_claim_allowed"] is False
    assert report["mtbls13729_role"]["exact_identity_promotions"] == 0
    assert report["primary_failure_bottleneck"]["queries"] == 11
    assert len(benchmark) == 4
    assert set(benchmark.artifact_version) == {"BioAware v1", "BioAware V3", "BioAware V4", "BioAware V6"}
    assert len(failures) == 6
    assert len(roles) == 4
    assert len(gaps) == 5
    assert (benchmark.loc[benchmark.artifact_version == "BioAware V4", "delta_recall1"].iloc[0]) < 0
    assert (benchmark.loc[benchmark.artifact_version == "BioAware V6", "delta_recall1"].iloc[0]) > 0
    for phrase in (
        "not yet a statistically confirmed annotation-performance upgrade",
        "These version-specific results must be reported together",
        "zero exact identity promotions",
        "Phenotype is forbidden from identity ranking",
        "Do not call BioAware SOTA",
    ):
        assert phrase in text
    print("[validate_bioaware_algorithm_biology_bridge_audit] PASS")


if __name__ == "__main__":
    main()
