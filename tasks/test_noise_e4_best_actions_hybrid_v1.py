"""Golden tests for seven-source actions plus E4 loss and Injector V1."""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

import train_noise_e4_faithful_v1 as historical  # noqa: E402
import train_noise_final_e4a_direct_augmentation as hybrid  # noqa: E402
from evaluate_noise_e4_best_actions_injector_v1_final import (  # noqa: E402
    HYBRID_FORBIDDEN_CONTRACT,
    hybrid_forbidden_contract_drift,
)


class TinyStore:
    def __init__(self) -> None:
        self.values = {
            row: torch.tensor(
                [[100.0 + row, 150.0 + row, 200.0 + row],
                 [0.2 + 0.01 * row, 0.5, 0.3 - 0.005 * row]],
                dtype=torch.float32,
            )
            for row in range(8)
        }

    def one(self, row: int) -> torch.Tensor:
        return self.values[int(row)].clone()

    def get(self, rows) -> list[torch.Tensor]:
        return [self.one(int(row)) for row in rows]


class TinyEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(17)
        self.weight = torch.nn.Parameter(torch.randn(6, 4, dtype=torch.float32) * 0.01)

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        return F.normalize(spectra.flatten(1) @ self.weight, dim=1)


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        action_selection="fixed", policy="curriculum", action_scope="all",
        outer_fold=0, formula_fold_seed=20260825, epochs=4, batch_actions=4,
        views_per_identity=4, error_views_per_identity=0, positive_spectra=4,
        negative_molecules=8, unfreeze_blocks=1, direct_transfer_mode="symmetric",
        rank_reference_mode="shared", guided_noise_policy="none", pmt_arm="none",
        candidate_boundary_loss=False, refresh_hard_negatives=False,
        causal_arm="legacy", backbone_lr=2e-6, head_lr=1e-5,
        weight_decay=1e-4, rank_margin=0.05, temperature=0.10,
        lambda_clean_rank=1.0, lambda_aug_rank=1.0, lambda_consistency=0.25,
        lambda_margin_floor=2.0, lambda_preserve=5.0,
        margin_floor_slack=0.005, safety_ratio=1.0, safety_stream_weight=1.0,
        positive_stream_weight=0.0, grad_clip=1.0,
        injector_target_attributable_fraction=0.25,
        injector_minimum_protective_retention=0.90,
        injector_maximum_update_norm_ratio=1.50,
        initial_student_checkpoint=None, outcome_action_dir=None,
        materialized_action_dir=None, materialized_action_arm="targeted",
        materialized_injection_mode="one_best_e4", pmt_manifest_dir=None,
        guided_action_authorization_dir=None, guided_crossfit_root=None,
        optimizer_boundary_mode="action_injector_v1", amp=False,
        run_suffix="e4_r0_injector_v1_test",
    )


def _examples():
    shared = dict(
        query_index=0, query_row=0, identity="IK", formula="C2H6O",
        positive_rows=(1, 2), negative_rows=(3, 4), official_margin=0.02,
        official_rank=2, sample_weight=1.0, policy="candidate_gradient|step=6",
        target_path=(0, 1), attenuation=0.5,
    )
    return historical.DirectExample(**shared), hybrid.DirectExample(**shared)


def _anchors(store: TinyStore) -> dict[int, np.ndarray]:
    model = TinyEncoder()
    with torch.no_grad():
        return {
            row: model(store.one(row).unsqueeze(0)).squeeze(0).numpy()
            for row in range(5)
        }


def test_strict_hybrid_contract_rejects_every_known_drift() -> None:
    args = _args()
    hybrid.validate_fixed_e4_action_injector_configuration(args)
    for field, bad in (
        ("initial_student_checkpoint", Path("e8.pt")),
        ("materialized_action_dir", Path("later_actions")),
        ("policy", "combined"),
        ("views_per_identity", 2),
        ("direct_transfer_mode", "student_action_stopgrad"),
        ("rank_reference_mode", "official"),
        ("positive_stream_weight", 1.0),
        ("candidate_boundary_loss", True),
        ("backbone_lr", 1e-6),
        ("injector_target_attributable_fraction", 0.20),
        ("amp", True),
    ):
        changed = copy.copy(args)
        setattr(changed, field, bad)
        try:
            hybrid.validate_fixed_e4_action_injector_configuration(changed)
        except ValueError:
            pass
        else:
            raise AssertionError(f"hybrid contract accepted drift: {field}={bad!r}")


def test_final_complete_panel_contract_starts_official_and_rejects_old_v11_path() -> None:
    args = _args()
    args.action_selection = "materialized_routed"
    args.materialized_action_dir = Path("frozen_seven_source_ledger")
    args.materialized_injection_mode = "complete_panel_historical_e4"
    hybrid.validate_materialized_e4_configuration(args)

    warm = copy.copy(args)
    warm.initial_student_checkpoint = Path("e8.pt")
    try:
        hybrid.validate_materialized_e4_configuration(warm)
    except ValueError as error:
        if "official DreaMS" not in str(error):
            raise
    else:
        raise AssertionError("complete-panel hybrid accepted the old E8 warm start")

    compressed = copy.copy(args)
    compressed.materialized_injection_mode = "one_best_e4"
    try:
        hybrid.validate_materialized_e4_configuration(compressed)
    except ValueError:
        pass
    else:
        raise AssertionError("complete-panel hybrid accepted one-best compression")


def test_fixed_action_loss_and_gradients_equal_historical_e4() -> None:
    store = TinyStore()
    old_example, new_example = _examples()
    anchors = _anchors(store)
    args = _args()
    # Isolate the objective from peak-operator implementation details; the
    # frozen historical operator is separately byte-pinned in the SBATCH.
    attenuation = lambda spectrum, path, dose: spectrum * (1.0 - 0.1 * dose)
    historical.attenuate_sequence = attenuation
    hybrid.attenuate_sequence = attenuation
    old_model = TinyEncoder()
    new_model = TinyEncoder()
    old_loss, old_log = historical.direct_action_loss(
        old_model, store, [old_example], anchors, torch.device("cpu"), args,
    )
    new_loss, new_log = hybrid.direct_action_loss(
        new_model, store, [new_example], anchors, torch.device("cpu"), args,
    )
    old_loss.backward()
    new_loss.backward()
    torch.testing.assert_close(new_loss, old_loss, rtol=0, atol=0)
    torch.testing.assert_close(new_model.weight.grad, old_model.weight.grad, rtol=0, atol=0)
    for key in (
        "action_clean_rank", "action_aug_rank", "action_consistency",
        "action_margin_floor", "action_preserve", "action_clean_margin",
        "action_aug_margin", "action_clean_margin_pass", "action_aug_margin_pass",
    ):
        if new_log[key] != old_log[key]:
            raise AssertionError(f"historical action-loss component drifted: {key}")


def test_complete_materialized_panel_uses_the_same_e4_loss_and_gradient() -> None:
    store = TinyStore()
    old_example, new_example = _examples()
    new_example = hybrid.replace(new_example, materialized_action_index=0)
    anchors = _anchors(store)
    args = _args()
    args.action_selection = "materialized_routed"
    args.materialized_injection_mode = "complete_panel_historical_e4"
    attenuation = lambda spectrum, path, dose: spectrum * (1.0 - 0.1 * dose)
    historical.attenuate_sequence = attenuation
    action_tensor = attenuation(
        store.one(new_example.query_row), new_example.target_path, new_example.attenuation,
    ).numpy()[None, ...]
    old_model = TinyEncoder()
    new_model = TinyEncoder()
    old_loss, _ = historical.direct_action_loss(
        old_model, store, [old_example], anchors, torch.device("cpu"), args,
    )
    new_loss, log = hybrid.direct_action_loss(
        new_model, store, [new_example], anchors, torch.device("cpu"), args,
        materialized_action_spectra=action_tensor,
    )
    old_loss.backward()
    new_loss.backward()
    torch.testing.assert_close(new_loss, old_loss, rtol=0, atol=0)
    torch.testing.assert_close(new_model.weight.grad, old_model.weight.grad, rtol=0, atol=0)
    if log["action_aug_rank_active_fraction"] != 1.0:
        raise AssertionError("complete-panel hybrid still applies the failed action-rank gate")


def test_fixed_safety_loss_and_sampler_equal_historical_e4() -> None:
    store = TinyStore()
    old_example, new_example = _examples()
    anchors = _anchors(store)
    args = _args()
    old_model = TinyEncoder()
    new_model = TinyEncoder()
    old_loss, old_log = historical.safety_loss(
        old_model, store, [old_example], anchors, torch.device("cpu"), args,
    )
    new_loss, new_log = hybrid.safety_loss(
        new_model, store, [new_example], anchors, torch.device("cpu"), args,
    )
    old_loss.backward()
    new_loss.backward()
    torch.testing.assert_close(new_loss, old_loss, rtol=0, atol=0)
    torch.testing.assert_close(new_model.weight.grad, old_model.weight.grad, rtol=0, atol=0)
    if new_log != old_log:
        raise AssertionError("historical safety-loss components drifted")

    old_pool = [copy.copy(old_example) for _ in range(3)]
    new_pool = [copy.copy(new_example) for _ in range(3)]
    old_schedule = historical.identity_balanced_epoch(
        old_pool, np.random.default_rng(23), 4,
    )
    new_schedule = hybrid.identity_balanced_epoch(
        new_pool, np.random.default_rng(23), 4,
    )
    if [item.policy for item in new_schedule] != [item.policy for item in old_schedule]:
        raise AssertionError("historical identity-balanced schedule drifted")


def test_source_declares_the_only_allowed_boundary_change() -> None:
    source = (ROOT / "tasks/train_noise_final_e4a_direct_augmentation.py").read_text(
        encoding="utf-8",
    )
    required = (
        "objective_reference_by_row is official_by_row",
        "e4_action_injector.capture_corrective_()",
        "e4_action_injector.step_and_inject_",
        '"injector_is_only_optimizer_boundary_change"',
        '"pure_e4_control_retrained": False',
    )
    missing = [token for token in required if token not in source]
    if missing:
        raise AssertionError(f"hybrid source lost its explicit boundary contract: {missing}")
    if tuple(hybrid.FIXED_POLICY["curriculum"]) != tuple(historical.FIXED_POLICY["curriculum"]):
        raise AssertionError("E4 R0 nine-cell curriculum drifted")
    for token in (
        '"complete_panel_historical_e4"',
        '"hybrid_changes_only_action_supplier_and_optimizer_boundary"',
        '"complete_seven_source_action_panel_used_without_one_best_compression"',
        '"complete_panel_unit_action_weights_without_identity_or_source_reweighting"',
        'materialized_injection_mode != "complete_panel_historical_e4"',
        "MATERIALIZED_MULTI_ACTION_PRECLIP_LOSS_SCALE",
        "audit_historical_e4_action_specific_alignment",
        '"shared_loss_terms_between_compared_objectives": False',
        '"positive_alignment_required"',
        '"action_specific_to_clean_corrective_gradient_gate_passed"',
        '"materialized_injection_mode": args.materialized_injection_mode',
        '"data_sha256": sha256_file(args.data)',
        '"architecture_checkpoint_sha256"',
        "audit_official_reencoding_zero_change",
        "official_checkpoint_edge_scores_plus_candidate_comparison_tie_aware_ranks",
    ):
        if token not in source:
            raise AssertionError(f"complete-panel hybrid contract is missing: {token}")


def test_evaluator_distinguishes_outer_train_routing_from_teacher_target() -> None:
    if hybrid_forbidden_contract_drift(dict(HYBRID_FORBIDDEN_CONTRACT)):
        raise AssertionError("valid outer-train routing provenance was rejected as a teacher")
    leaked = dict(HYBRID_FORBIDDEN_CONTRACT)
    leaked["teacher"] = "absolute_embedding_target"
    drift = hybrid_forbidden_contract_drift(leaked)
    if drift != {
        "teacher": {
            "expected": "outer_train_action_routing_only_no_teacher_target",
            "observed": "absolute_embedding_target",
        }
    }:
        raise AssertionError(f"teacher-target drift was not isolated: {drift}")


def main() -> None:
    tests = (
        test_strict_hybrid_contract_rejects_every_known_drift,
        test_final_complete_panel_contract_starts_official_and_rejects_old_v11_path,
        test_fixed_action_loss_and_gradients_equal_historical_e4,
        test_complete_materialized_panel_uses_the_same_e4_loss_and_gradient,
        test_fixed_safety_loss_and_sampler_equal_historical_e4,
        test_source_declares_the_only_allowed_boundary_change,
        test_evaluator_distinguishes_outer_train_routing_from_teacher_target,
    )
    for test in tests:
        test()
    print(f"[test_noise_e4_best_actions_hybrid_v1] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()
