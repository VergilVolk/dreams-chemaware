"""Contracts for the dense ChemAware query-alignment audit."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from audit_chemaware_dense_boundary_alignment import boundary_alignment_audit  # noqa: E402


def fixture() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    metric_names = np.asarray([
        "action_top_fraction", "action_largest_region_fraction",
        "action_same_neighbor_fraction", "action_best_advantage_over_baseline",
        "global_action_advantage_over_baseline", "candidate_rule_max",
        "candidate_rule_top2_mean", "delta_rule_max", "delta_rule_top2_mean",
    ])
    arm_metric = np.zeros((4, 2, 1, len(metric_names)), dtype=np.float64)
    arm_metric[0, :, 0, :] = 1.0
    evidence = {
        "metric_names": metric_names,
        "arm_names": np.asarray([
            "correct", "null_a", "null_b", "null_c",
        ]),
        "arm_metric": arm_metric,
        "valid": np.ones((2, 1), dtype=bool),
        "benefit": np.ones((2, 1), dtype=bool),
        "proposed_candidate": np.zeros((2, 1), dtype=np.int64),
        "query": np.asarray([0, 2], dtype=np.int64),
        "identity": np.asarray(["A", "B"]),
        "formula": np.asarray(["FA", "FB"]),
        "baseline_rank": np.asarray([2, 2], dtype=np.int16),
        "baseline_candidate": np.asarray([1, 1], dtype=np.int16),
    }
    manifest = {
        "query_ik14": np.asarray(["A", "A", "B"]),
        "query_formula": np.asarray(["FA", "FA", "FB"]),
        "query_ptr": np.asarray([0, 1, 2, 3], dtype=np.int64),
        "molecule_label": np.ones(3, dtype=np.int8),
    }
    triplets = {
        "source_query": np.asarray([0, 1, 2, 1], dtype=np.int64),
        "negative_candidate": np.asarray([1, 1, 0, 1], dtype=np.int16),
        "curriculum_role": np.asarray([31, 31, 32, 33], dtype=np.int8),
    }
    return evidence, triplets, manifest


def main() -> None:
    evidence, triplets, manifest = fixture()
    report = boundary_alignment_audit(evidence, triplets, manifest)
    assert report["correction_events"] == 3
    assert report["direct_query_evidence_events"] == 2
    assert report["identity_broadcast_events"] == 1
    assert report["direct_query_evidence_queries"] == 2
    assert report["identity_broadcast_queries"] == 1
    assert report["exact_query_and_contrasted_candidate_events"] == 1
    assert report["gate"]["all_corrections_have_direct_query_evidence"] is False

    triplets["source_query"] = np.asarray([0, 0, 2, 1], dtype=np.int64)
    triplets["negative_candidate"] = np.asarray([1, 1, 1, 1], dtype=np.int16)
    report = boundary_alignment_audit(evidence, triplets, manifest)
    assert report["gate"]["all_corrections_have_direct_query_evidence"] is True
    assert report["gate"]["all_corrections_match_the_contrasted_candidate"] is True
    print("PASS: ChemAware dense boundary-alignment audit contracts")


if __name__ == "__main__":
    main()
