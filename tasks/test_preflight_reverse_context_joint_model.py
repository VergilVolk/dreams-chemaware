#!/usr/bin/env python
"""Local logic tests for reverse-context A0 (no network calls)."""
from __future__ import annotations

from preflight_reverse_context_joint_model import _sql_quote, summarize


def main() -> None:
    assert _sql_quote("O'Brien") == "'O''Brien'"
    anchors = [
        {
            "spectrum_id_int": index,
            "ik14": f"IK{index:012d}",
            "classyfire_superclass": f"class-{index % 6}",
        }
        for index in range(40)
    ]
    denominators = [
        {
            "dataset": f"D{index:04d}",
            "sample_type": "animal",
            "body_part": "blood plasma",
            "health_status": "healthy",
            "polarity": "electrospray ionization (positive)",
            "instrument": "Q Exactive",
            "n_files": 1,
            "n_ms2": 100,
        }
        for index in range(500)
    ]
    occurrences = []
    for index in range(25):
        for project, sample_type in ((f"D{index:04d}", "animal"), (f"X{index:04d}", "food")):
            occurrences.append(
                {
                    "spectrum_id_int": index,
                    "dataset": project,
                    "sample_type": sample_type,
                    "body_part": "missing value",
                    "health_status": "missing value",
                    "polarity": "electrospray ionization (positive)",
                    "instrument": "Q Exactive",
                    "matched_files": 2,
                    "matched_spectra": 3,
                    "mean_cosine": 0.9,
                    "max_cosine": 0.95,
                    "mean_matching_peaks": 5,
                }
            )
    report = summarize(anchors, denominators, occurrences, 0.85, 4)
    assert report["anchors"]["selected"] == 40
    assert report["anchors"]["with_any_public_match"] == 25
    assert report["anchors"]["with_matches_in_ge_2_projects"] == 25
    assert report["anchors"]["with_matches_in_ge_2_biological_sample_types"] == 25
    assert report["pass_to_a1"]
    assert report["anchor_metrics"][0]["sample_type_entropy"] == 1.0
    print("[test_preflight_reverse_context_joint_model] PASS")


if __name__ == "__main__":
    main()
