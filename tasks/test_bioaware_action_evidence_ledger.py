#!/usr/bin/env python
"""Dependency-light unit checks for the BioAware action evidence ledger."""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from audit_bioaware_action_evidence_ledger import (
        action_summary,
        formula_cluster_bootstrap,
        query_metadata,
        validate_candidates,
        validate_transitions,
    )
except ModuleNotFoundError:  # package import mode
    from tasks.audit_bioaware_action_evidence_ledger import (
        action_summary,
        formula_cluster_bootstrap,
        query_metadata,
        validate_candidates,
        validate_transitions,
    )


def candidate_fixture() -> pd.DataFrame:
    rows = []
    for query, truth, wrong, formula, unit, scores in (
        ("q1", "T1", "W1", "F1", "S1__hilic", (0.4, 0.6)),
        ("q2", "T2", "W2", "F2", "S2__rplc", (0.8, 0.7)),
    ):
        for index, (candidate, score) in enumerate(zip((truth, wrong), scores)):
            row = {
                "query_id": query,
                "candidate_id": candidate,
                "spectral_score": score,
                "best_library_row": 2 * len(rows) + index,
                "truth_candidate_id": truth,
                "truth_formula": formula,
                "unit_id": unit,
                "adduct": "[M-H]-",
                "baseline_correct": truth == (wrong if query == "q1" else truth),
                "top_candidate_id": wrong if query == "q1" else truth,
                "is_positive": candidate == truth,
            }
            for offset, feature in enumerate([
                "known_mass_candidate_fraction", "known_path_fraction",
                "known_inverse_depth_mean", "known_log_seed_support_mean",
                "known_log_degree", "edge0_complete_fraction",
                "edge0_bottleneck_mean", "edge1_complete_fraction",
                "edge1_bottleneck_mean", "predicted_edge_increment",
            ]):
                row[feature] = float((candidate == truth) + offset / 100)
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    candidates = validate_candidates(candidate_fixture())
    metadata = query_metadata(candidates)
    transitions = pd.DataFrame([
        {
            "query_id": "q1", "truth_candidate_id": "T1", "truth_formula": "F1",
            "baseline_candidate_id": "W1", "proposed_candidate_id": "T1",
            "final_candidate_id": "T1", "baseline_correct": False,
            "final_correct": True, "corrected": True, "introduced": False,
            "delta": 1, "intervene": True,
        },
        {
            "query_id": "q2", "truth_candidate_id": "T2", "truth_formula": "F2",
            "baseline_candidate_id": "T2", "proposed_candidate_id": "W2",
            "final_candidate_id": "W2", "baseline_correct": True,
            "final_correct": False, "corrected": False, "introduced": True,
            "delta": -1, "intervene": True,
        },
    ])
    validated = validate_transitions(transitions, metadata, "synthetic")
    assert validated["action_outcome"].tolist() == ["corrected", "introduced"]
    bootstrap_a = formula_cluster_bootstrap(validated, repeats=200, seed=7)
    bootstrap_b = formula_cluster_bootstrap(validated, repeats=200, seed=7)
    assert bootstrap_a == bootstrap_b
    assert np.isclose(bootstrap_a["mean"], 0.0)
    summary = action_summary(validated, repeats=200, seed=7)
    assert summary["corrected"] == 1 and summary["introduced"] == 1
    assert summary["risk_weighted_net_lambda2"] == -1
    assert summary["corrected_truth_identities"] == 1

    broken = transitions.copy()
    broken.loc[0, "corrected"] = False
    try:
        validate_transitions(broken, metadata, "broken")
    except RuntimeError as exc:
        assert "corrected flags" in str(exc)
    else:
        raise AssertionError("inconsistent corrected flag was not rejected")
    print("[test_bioaware_action_evidence_ledger] PASS")


if __name__ == "__main__":
    main()
