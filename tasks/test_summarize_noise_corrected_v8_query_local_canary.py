"""Small contract tests for the V8 query-local canary summarizer."""
from __future__ import annotations

from summarize_noise_corrected_v8_query_local_canary import _paired_arm_delta, _role_gate


def test_paired_arm_delta_uses_exact_same_queries_and_counts_risk() -> None:
    baseline = [
        {
            "query_index": index,
            "query_formula": formula,
            "near": near,
            "initial_E8_rank": initial,
            "candidate_rank": candidate,
        }
        for index, formula, near, initial, candidate in zip(
            range(4), ("A", "B", "C", "D"), (True, True, False, False),
            (2, 1, 2, 1), (2, 1, 1, 2),
        )
    ]
    candidate = [dict(row) for row in baseline]
    for row, rank in zip(candidate, (1, 2, 1, 1)):
        row["candidate_rank"] = rank
    result = _paired_arm_delta(baseline, candidate)
    assert result == {
        "queries": 4,
        "delta_recall1_pp": 25.0,
        "corrected": 2,
        "introduced": 1,
        "risk_net_lambda2": 0,
        "near_delta_recall1_pp": 0.0,
        "near_corrected": 1,
        "near_introduced": 1,
    }


def test_role_gate_reads_the_trainer_gradient_calibration_contract() -> None:
    decision = {
        "gradient_calibration": {
            "corrective_embedding_role_energy": {
                "corrective_gradient_locality": "query_action_only",
                "corrective_reference_gradient_exact_zero": True,
                "required_clean_and_action_paths_live": True,
                "query_action_only_locality_gate_passed": True,
                "transfer_reference_dominates_query_energy": False,
                "payload_reference_dominates_action_energy": False,
                "branches": {"transfer": {"reference_energy_fraction": 0.0}},
            }
        }
    }
    result = _role_gate(decision, "query_action_only")
    assert result["reference_gradient_exact_zero"] is True
    assert result["required_clean_and_action_paths_live"] is True
    assert result["query_action_only_gate_passed"] is True
    assert result["branches"]["transfer"]["reference_energy_fraction"] == 0.0


def main() -> None:
    test_paired_arm_delta_uses_exact_same_queries_and_counts_risk()
    test_role_gate_reads_the_trainer_gradient_calibration_contract()
    print("[test_summarize_noise_corrected_v8_query_local_canary] PASS tests=2")


if __name__ == "__main__":
    main()
