"""Pure-Python contracts for V8 corrective gradient locality."""
from __future__ import annotations

from noise_corrected_gradient_locality_v8 import (
    corrective_embedding_role_report,
    corrective_gradient_locality,
)


def _diagnostics(*, reference: float) -> dict[str, float]:
    values: dict[str, float] = {}
    for branch, total, query, action in (
        ("transfer", 5.0, 3.0, 0.0),
        ("payload", 5.0, 0.0, 4.0),
        ("consistency", 5.0, 3.0, 4.0),
    ):
        prefix = f"corrective.embedding_grad_{branch}_"
        values[prefix + "total_norm"] = total
        values[prefix + "query_norm"] = query
        values[prefix + "action_norm"] = action
        values[prefix + "control_norm"] = 0.0
        values[prefix + "reference_norm"] = reference if branch != "consistency" else 0.0
    return values


def test_policies_change_only_the_corrective_reference_path() -> None:
    shared = corrective_gradient_locality("shared")
    local = corrective_gradient_locality("query_action_only")
    assert shared.clean_query_live and local.clean_query_live
    assert shared.action_view_live and local.action_view_live
    assert shared.reference_live is True
    assert local.reference_live is False


def test_query_action_only_requires_zero_reference_and_live_query_action() -> None:
    report = corrective_embedding_role_report(
        _diagnostics(reference=0.0), locality="query_action_only",
    )
    assert report["corrective_reference_gradient_exact_zero"] is True
    assert report["required_clean_and_action_paths_live"] is True
    assert report["query_action_only_locality_gate_passed"] is True
    transfer = report["branches"]["transfer"]
    assert transfer["roles"]["query"]["energy_fraction"] == 0.36


def test_reference_leak_fails_query_local_gate_and_is_visible_as_energy() -> None:
    report = corrective_embedding_role_report(
        _diagnostics(reference=4.0), locality="query_action_only",
    )
    assert report["corrective_reference_gradient_exact_zero"] is False
    assert report["query_action_only_locality_gate_passed"] is False
    assert report["transfer_reference_dominates_query_energy"] is True


def test_shared_mode_reports_bypass_without_applying_query_local_gate() -> None:
    report = corrective_embedding_role_report(
        _diagnostics(reference=4.0), locality="shared",
    )
    assert report["transfer_reference_dominates_query_energy"] is True
    assert report["payload_reference_dominates_action_energy"] is False
    assert report["query_action_only_locality_gate_applicable"] is False
    assert report["query_action_only_locality_gate_passed"] is None


def main() -> None:
    tests = [
        test_policies_change_only_the_corrective_reference_path,
        test_query_action_only_requires_zero_reference_and_live_query_action,
        test_reference_leak_fails_query_local_gate_and_is_visible_as_energy,
        test_shared_mode_reports_bypass_without_applying_query_local_gate,
    ]
    for test in tests:
        test()
    print(f"[test_noise_corrected_gradient_locality_v8] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()
