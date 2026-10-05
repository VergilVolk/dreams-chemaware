#!/usr/bin/env python
"""Synthetic tests for A1 split and label contracts."""
from __future__ import annotations

from build_reverse_context_joint_a1_manifest import _stable_test, build_manifest


def main() -> None:
    assert _stable_test("x", 0.2, 7, "project") == _stable_test("x", 0.2, 7, "project")
    anchors = [
        {
            "spectrum_id_int": index,
            "ik14": f"IK{index:012d}",
            "classyfire_superclass": f"class-{index % 8}",
        }
        for index in range(256)
    ]
    report = {"pass_to_a1": True, "anchor_metrics": anchors}
    opportunities = [
        {"dataset": f"P{index:04d}", "n_files": "10", "n_ms2": "100"}
        for index in range(600)
    ]
    occurrences = []
    for anchor in anchors[:100]:
        for project in opportunities:
            occurrences.append(
                {
                    "spectrum_id_int": str(anchor["spectrum_id_int"]),
                    "dataset": project["dataset"],
                    "sample_type": "animal" if int(project["dataset"][1:]) % 2 else "plant",
                    "body_part": "missing value",
                    "health_status": "missing value",
                    "polarity": "+ESI",
                    "instrument": "Q Exactive",
                    "matched_files": "1",
                    "matched_spectra": "1",
                    "mean_cosine": "0.9",
                    "max_cosine": "0.9",
                    "mean_matching_peaks": "5",
                }
            )
    ledger, manifest, split_rows = build_manifest(report, occurrences, opportunities, 0.2, 0.2, 7)
    assert ledger and split_rows
    assert set(manifest["quadrants"]) == {
        "development", "project_holdout", "identity_holdout", "joint_holdout"
    }
    assert manifest["quadrants"]["joint_holdout"]["biological_rows"] >= 50
    assert all(row["observation_type"] == "biological" for row in ledger)
    print("[test_build_reverse_context_joint_a1_manifest] PASS")


if __name__ == "__main__":
    main()
