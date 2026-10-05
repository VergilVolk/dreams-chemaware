#!/usr/bin/env python
"""CPU-only scientific and serialization contracts for grand fusion."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tasks") not in sys.path:
    sys.path.insert(0, str(ROOT / "tasks"))

from grand_fusion_router_core import (  # noqa: E402
    correctness_from_winners,
    feature_names,
    fit_correctness_models,
    predict_correctness,
    query_features,
    risk_ledger,
    route,
    select_threshold,
)
from merge_dreams_task_vectors import merge_state, ties_merge_state  # noqa: E402
from train_grand_fusion_router import registry_gate  # noqa: E402
from adjudicate_grand_fusion_external import primary_recall_block, transition_block  # noqa: E402


def test_query_features_are_label_free_and_ties_fail() -> None:
    methods = ("a", "b", "c")
    scores = {
        "a": np.asarray([0.9, 0.2, 0.1, 0.5, 0.5]),
        "b": np.asarray([0.7, 0.8, 0.1, 0.6, 0.4]),
        "c": np.asarray([0.6, 0.1, 0.9, 0.2, 0.1]),
    }
    ptr = np.asarray([0, 3, 5], dtype=np.int64)
    features, winners = query_features(scores, ptr, methods)
    assert features.shape == (2, len(feature_names(methods)))
    assert winners[0].tolist() == [0, 1, 2]
    assert winners[1].tolist() == [-1, 0, 0]
    assert all("candidate" not in name and "edge" not in name for name in feature_names(methods))


def test_correctness_uses_strict_winner_only() -> None:
    winners = np.asarray([[0, 1], [-1, 0]])
    ptr = np.asarray([0, 2, 4])
    labels = np.asarray([1, 0, 1, 0], dtype=np.int8)
    observed = correctness_from_winners(winners, ptr, labels)
    assert observed.tolist() == [[1, 0], [0, 1]]


def test_risk_threshold_can_choose_full_abstention() -> None:
    methods = ("default", "risky")
    winners = np.zeros((4, 2), dtype=np.int64)
    correctness = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.int8)
    probabilities = np.asarray([[0.4, 0.9], [0.4, 0.9], [0.4, 0.9], [0.4, 0.9]])
    threshold, rows = select_threshold(
        probabilities, winners, correctness, methods, "default", (0.0, 1.1), 2.0,
    )
    assert threshold == 1.1
    chosen = route(probabilities, winners, methods, "default", threshold)
    ledger = risk_ledger(correctness, chosen, methods, "default", 2.0)
    assert ledger["switches"] == 0 and max(row["risk_net"] for row in rows) == 0


def test_correctness_models_fit_and_route_with_expected_shapes() -> None:
    rng = np.random.default_rng(19)
    methods = ("default", "specialist", "constant")
    features = rng.normal(size=(240, len(feature_names(methods))))
    correctness = np.column_stack((
        (features[:, 0] > -0.5).astype(np.int8),
        (features[:, 1] > 0.2).astype(np.int8),
        np.ones(len(features), dtype=np.int8),
    ))
    models = fit_correctness_models(features, correctness, methods)
    probabilities = predict_correctness(models, features[:17], methods)
    assert probabilities.shape == (17, 3) and np.all(np.isfinite(probabilities))
    winners = np.zeros((17, 3), dtype=np.int64)
    chosen = route(probabilities, winners, methods, "default", threshold=0.05)
    assert chosen.shape == (17,) and set(np.unique(chosen)) <= {0, 1, 2}


def test_registry_blocks_retired_rrf() -> None:
    registry = ROOT / "tasks/grand_fusion_components_v1.json"
    try:
        registry_gate(registry, ("weighted_spectral_entropy", "noise_rrf_1_7"), False)
    except RuntimeError as error:
        assert "audit-only" in str(error)
    else:
        raise AssertionError("retired RRF must be blocked from production")
    body = registry_gate(registry, ("weighted_spectral_entropy", "noise_rrf_1_7"), True)
    assert body["schema"] == "dreams_grand_fusion_component_registry_v1"


def test_task_vector_arithmetic_and_nonfloat_guard() -> None:
    base = {"w": torch.tensor([1.0, 2.0]), "counter": torch.tensor([3], dtype=torch.int64)}
    noise = {"w": torch.tensor([2.0, 4.0]), "counter": torch.tensor([3], dtype=torch.int64)}
    chem = {"w": torch.tensor([0.0, 4.0]), "counter": torch.tensor([3], dtype=torch.int64)}
    merged = merge_state(base, noise, chem, 0.5, 0.25)
    assert torch.allclose(merged["w"], torch.tensor([1.25, 3.5]))
    bad = dict(chem)
    bad["counter"] = torch.tensor([4], dtype=torch.int64)
    try:
        merge_state(base, noise, bad, 0.5, 0.25)
    except RuntimeError as error:
        assert "non-floating" in str(error)
    else:
        raise AssertionError("non-floating buffer drift must fail")


def test_ties_resolves_sign_conflict_without_averaging_it() -> None:
    base = {"w": torch.zeros(3)}
    noise = {"w": torch.tensor([2.0, 2.0, 0.1])}
    chem = {"w": torch.tensor([1.0, -1.0, 0.1])}
    merged = ties_merge_state(base, noise, chem, density=1.0, scale=1.0)
    # Coordinate 0 agrees and is averaged. Coordinate 1 elects the positive
    # sign from the larger aggregate and excludes the conflicting Chem delta.
    assert torch.allclose(merged["w"], torch.tensor([1.5, 2.0, 0.1]))


def test_component_registry_is_complete_and_unique() -> None:
    body = json.loads((ROOT / "tasks/grand_fusion_components_v1.json").read_text(encoding="utf-8"))
    names = [row["name"] for row in body["components"]]
    assert len(names) == len(set(names))
    required = {
        "noise_v1", "noise_p2b_v1", "noise_rrf_1_7", "chemaware_v2_reranker",
        "noise_msg_fusion_stack_v1", "chemaware_stage1_encoder", "bioaware_b35",
        "bioaware_b47_exact_event",
    }
    assert required <= set(names)


def test_external_expert_contract_rejects_row_drift() -> None:
    script = (ROOT / "tasks/extend_grand_fusion_pair_evidence.py").read_text(encoding="utf-8")
    assert 'declared != evidence_hash' in script
    assert 'values.shape != (edge_count,)' in script


def test_readiness_audit_is_fail_closed() -> None:
    script = (ROOT / "tasks/audit_grand_fusion_readiness.py").read_text(encoding="utf-8")
    assert '"full_three_layer_claim"' in script
    assert '"PARTIAL_ONLY_FAIL_CLOSED"' in script
    assert 'authorization_after_repair' in script


def test_sbatch_never_enables_rrf_and_keeps_gnps_after_training() -> None:
    script = (ROOT / "tasks/run_grand_fusion_router_1gpu.sbatch").read_text(encoding="utf-8")
    assert "noise_rrf_1_7" not in script
    assert script.index("train_grand_fusion_router.py") < script.index("apply_grand_fusion_router_to_gnps.py")
    assert "#SBATCH --mem" not in script


def test_external_adjudicator_finds_only_overall_blocks() -> None:
    panel = {
        "paired": {
            "corrected": 12,
            "introduced": 3,
            "near_corrected": 7,
            "near_introduced": 2,
            "formula_cluster_paired_ci": {
                "recall@1": {"delta_pp": 0.8, "ci_low_pp": 0.1, "ci_high_pp": 1.4},
            },
            "near_formula_cluster_paired_ci": {
                "recall@1": {"delta_pp": 1.2, "ci_low_pp": 0.0, "ci_high_pp": 2.3},
            },
        }
    }
    recall_path, recall = primary_recall_block(panel)
    transition_path, transition = transition_block(panel)
    assert "near" not in "/".join(recall_path).lower()
    assert recall["delta_pp"] == 0.8
    assert transition_path == ("paired",)
    assert transition["corrected"] == 12 and transition["introduced"] == 3


def test_server_program_has_bounded_arms_and_manual_external_pause() -> None:
    task_vector = (ROOT / "tasks/run_grand_fusion_task_vector_scan_2gpu.sbatch").read_text(encoding="utf-8")
    assert task_vector.count("make_linear ") == 3
    assert task_vector.count("make_ties ") == 4
    submit = (ROOT / "tasks/submit_grand_fusion_program.sh").read_text(encoding="utf-8")
    assert "run_grand_fusion_selected_encoder_gnps_1gpu.sbatch" in submit
    assert "SELECTED_CHECKPOINT=<path>" in submit


def test_chemaware_v2_export_is_truthblind_and_hash_locked() -> None:
    exporter = (ROOT / "tasks/export_chemaware_v2_grand_fusion_scores.py").read_text(
        encoding="utf-8"
    )
    assert 'report.get("frozen_truthblind_policy", {}).get("sha256")' in exporter
    assert '"molecule_label"' not in exporter
    assert 'predict_truthblind_policy(' in exporter
    assert 'np.nextafter(np.max(values)' in exporter
    assert 'expected_official' in exporter and 'official score alignment drifted' in exporter


def test_msg_enters_router_only_through_oof_score_contract() -> None:
    trainer = (ROOT / "tasks/train_noise_msg_fusion_stack.py").read_text(encoding="utf-8")
    assert 'staging / "oof_pair_scores.npz"' in trainer
    assert 'pair_scores=oof_pair_scores' in trainer
    assert 'evidence_sha256=np.asarray' in trainer
    submit = (ROOT / "tasks/submit_grand_fusion_program.sh").read_text(encoding="utf-8")
    assert 'dependency="afterok:${msg_job}"' in submit
    assert 'dependency="afterok:${chem_job}"' in submit
    assert 'method_scores_with_chemaware.npz' in submit


def test_b47_context_layer_is_gated_and_one_shot() -> None:
    freeze = (ROOT / "tasks/freeze_bioaware_b47_repaired_action.py").read_text(
        encoding="utf-8"
    )
    once = (ROOT / "tasks/evaluate_bioaware_b47_repaired_once.py").read_text(
        encoding="utf-8"
    )
    adjudicator = (ROOT / "tasks/adjudicate_bioaware_b47_external.py").read_text(
        encoding="utf-8"
    )
    assert 'report.get("authorization_after_repair") is not True' in freeze
    assert '"truth_fields_used": []' in freeze
    assert 'os.O_EXCL' in once
    assert '"--protocol-scope", "sealed_external"' in once
    assert '"delta_recall_at_1_at_least_3pp"' in adjudicator
    assert '"beats_catalogue_network_and_nulls"' in adjudicator


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"[test_grand_fusion] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()
