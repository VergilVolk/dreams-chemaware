"""Static fail-closed checks for the E4-native two-GPU restoration job."""
from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "tasks/run_noise_e4_native_best_actions_2gpu.sbatch"
TRAINER = ROOT / "tasks/train_noise_final_e4a_direct_augmentation.py"
sys.path.insert(0, str(ROOT / "tasks"))
from noise_final_e4_pmt_core import (  # noqa: E402
    coverage_first_identity_balanced_schedules,
    select_materialized_best_action_union,
)
import pandas as pd  # noqa: E402


def test_resource_contract() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert re.findall(r"^#SBATCH --gpus=.*$", text, flags=re.MULTILINE) == [
        "#SBATCH --gpus=2",
    ]
    assert not re.search(
        r"^#SBATCH --(?:mem|mem-per-cpu|mem-per-gpu)(?:=|\s)",
        text, flags=re.MULTILINE,
    )


def test_embedded_preflight_python_compiles() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    marker = "<<'PY'\n"
    assert marker in text
    embedded = text.split(marker, 1)[1].split("\nPY\n", 1)[0]
    ast.parse(embedded)


def test_old_kernel_and_complete_action_bank_are_joined() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    required = (
        "noise_corrected_best_v5_canary_fold_0_run_2332784",
        "--action-selection materialized_routed",
        "--direct-transfer-mode symmetric --rank-reference-mode shared",
        "--epochs 4 --batch-actions 4 --views-per-identity 4",
        "--lambda-clean-rank 1.0 --lambda-aug-rank 1.0",
        "--lambda-consistency 0.25 --lambda-margin-floor 2.0",
        "--lambda-preserve 5.0",
        '"N_mature", "P_guided_original", "E10B", "E11", "E12B"',
        '"A4_exact", "V4_gradient_path"',
        "strict_replay_margin_floor = 5e-6",
        '"numerical_boundary_actions_excluded"',
        "summarize_noise_e4_native_best_actions.py",
    )
    for value in required:
        assert value in text
    assert "train_noise_corrected_routed_direct.py" not in text
    assert "audit_noise_corrected_v4_action_router.py" not in text


def test_two_gpu_action_view_only_control() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert 'run_arm "$GPU_ZERO" targeted &' in text
    assert 'run_arm "$GPU_ONE" shuffled &' in text
    assert '--materialized-action-arm "$ARM"' in text
    assert "STATUS_TARGETED == 137" in text
    assert "STATUS_SHUFFLED == 137" in text
    assert "retrying alone on allocated GPU" in text
    assert "E4-native fail-fast worker failure" in text
    assert 'kill "$PID_TARGETED"' in text
    assert 'kill "$PID_SHUFFLED"' in text
    assert not re.search(r"^\s*rm(?:\s|$)", text, flags=re.MULTILINE)
    assert 'mv "$RUN_ROOT" "$FINAL_ROOT"' in text


def test_trainer_keeps_historical_e4_loss() -> None:
    source = TRAINER.read_text(encoding="utf-8")
    ast.parse(source)
    for token in (
        "args.lambda_clean_rank * clean_rank",
        "args.lambda_aug_rank * aug_rank",
        "args.lambda_consistency * consistency",
        "args.lambda_margin_floor * floor",
        "args.lambda_preserve * preserve",
        "materialized_action_spectra=materialized_action_spectra",
        "materialized_all_actions_exposed_before_recycling",
        "one_maximum_margin_action_per_query",
        "coverage_first_identity_balanced_schedules",
        "historical_identity_exposure_budget_restored",
        "causal_control_donors_restricted_to_selected_union",
        "selected_action_rows_preserved",
        "selected_actions_equal_unit_weight",
        "routing_scores_not_used_as_loss_targets",
        "qualifying_sources",
        "select_materialized_direct_action_panel(",
        "corrected_full_metrics(",
        "score_candidate_boundary(",
        "streaming_evaluate_embeddings(",
        '"exact_router_scoring_used": True',
        '"unexplained_rank_mismatches"',
        "del cache_embeddings, cache_index",
        "del initial_package",
        "initial_reference_encoded = initial_encoded",
        '"all_actions_encoded_exactly_once": True',
        '"candidate_reference_embeddings_reused_from_current_E8": True',
        '"requested_action_batch_size"',
        "args.eval_batch_size,",
        "MATERIALIZED_ACTION_REPLAY_MARGIN_FLOOR = 5e-6",
        '"numerical_boundary_actions_excluded"',
        '"clean_and_action_active_candidate_row_union_preserved": True',
    ):
        assert token in source


def test_complete_coverage_keeps_identity_dose() -> None:
    action_ids = [f"action-{index}" for index in range(21)]
    identities = ["two-actions"] * 2 + ["nineteen-actions"] * 19
    policies = [f"policy-{index % 5}" for index in range(21)]
    schedules = coverage_first_identity_balanced_schedules(
        action_ids, identities, policies, epochs=4, views_per_identity=4, seed=17,
    )
    flat = [index for epoch in schedules for index in epoch]
    assert len(set(flat[:len(action_ids)])) == len(action_ids)
    assert Counter(identities[index] for index in flat) == Counter({
        "two-actions": 16,
        "nineteen-actions": 19,
    })
    assert schedules == coverage_first_identity_balanced_schedules(
        action_ids, identities, policies, epochs=4, views_per_identity=4, seed=17,
    )


def test_best_union_rejects_weaker_and_noncorrective_rows() -> None:
    rows = pd.DataFrame({
        "action_id": ["q1-low", "q1-best", "q2-not-top1", "q3-harmful"],
        "query_index": [1, 1, 2, 3],
        "source": ["N_mature", "E12B", "E11", "A4_exact"],
        "family": ["n", "e12", "e11", "a4"],
        "recipe_id": ["r1", "r2", "r3", "r4"],
        "supervision_kind": ["corrective", "corrective", "corrective", "harmful"],
        "clean_rank": [2, 2, 3, 2],
        "action_rank": [1, 1, 2, 1],
        "action_margin": [0.02, 0.09, -0.01, 0.20],
    })
    best, qualifying = select_materialized_best_action_union(rows)
    assert qualifying["action_id"].tolist() == ["q1-best", "q1-low"]
    assert best["action_id"].tolist() == ["q1-best"]


if __name__ == "__main__":
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_")
    ]
    for test in tests:
        test()
    print(f"[test_noise_e4_native_best_actions_sbatch] PASS tests={len(tests)}")
