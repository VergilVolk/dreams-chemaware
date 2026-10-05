"""Fail-closed CPU checks for the two-GPU E4 multi-action direct job."""
from __future__ import annotations

import ast
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "tasks/run_noise_e4_multiaction_balanced_2gpu.sbatch"
TRAINER = ROOT / "tasks/train_noise_final_e4a_direct_augmentation.py"
SUMMARY = ROOT / "tasks/summarize_noise_e4_native_best_actions.py"
sys.path.insert(0, str(ROOT / "tasks"))
from noise_final_e4_pmt_core import (  # noqa: E402
    coverage_first_identity_balanced_schedules,
    identity_family_balanced_action_weights,
    identity_family_balanced_exposure_weights,
    select_materialized_direct_action_panel,
)


def action_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "action_id": ["a1", "a2", "a3", "a4", "a5", "bad"],
        "query_index": [1, 1, 2, 3, 3, 4],
        "query_ik14": ["i1", "i1", "i1", "i2", "i2", "i3"],
        "source": ["N", "N", "E12", "A4", "P", "N"],
        "family": ["n", "n", "e12", "a4", "p", "n"],
        "recipe_id": ["r1", "r2", "r3", "r4", "r5", "r6"],
        "supervision_kind": ["corrective"] * 6,
        "clean_rank": [2, 2, 3, 4, 4, 2],
        "action_rank": [1, 1, 1, 1, 1, 2],
        "action_margin": [0.02, 0.08, 0.04, 0.03, 0.06, -0.01],
    })


def test_resource_contract() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert re.findall(r"^#SBATCH --gpus=.*$", text, flags=re.MULTILINE) == [
        "#SBATCH --gpus=2",
    ]
    assert not re.search(
        r"^#SBATCH --(?:mem|mem-per-cpu|mem-per-gpu)(?:=|\s)",
        text, flags=re.MULTILINE,
    )
    assert 'run_arm "$GPU_ZERO" targeted &' in text
    assert 'run_arm "$GPU_ONE" shuffled &' in text
    assert not re.search(r"^\s*rm(?:\s|$)", text, flags=re.MULTILINE)


def test_embedded_preflight_python_compiles() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    embedded = text.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    ast.parse(embedded)


def test_job_joins_full_action_panel_to_direct_e4() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    for token in (
        "noise_corrected_best_v5_canary_fold_0_run_2332784",
        "--materialized-injection-mode multi_action_balanced",
        "--direct-transfer-mode symmetric --rank-reference-mode shared",
        "--epochs 4 --batch-actions 4 --views-per-identity 4",
        "select_materialized_direct_action_panel(",
        "identity_family_balanced_exposure_weights(",
        'mode="multi_action_balanced"',
        '"all_strict_actions_exposed_before_recycling"',
        '"physical_recycled_exposures"',
        '"effective_weight_per_identity": 16.0',
        '"preclip_loss_scale": 0.16',
        "summarize_noise_e4_native_best_actions.py",
    ):
        assert token in text


def test_multi_panel_keeps_all_strict_actions_not_one_best() -> None:
    frame = action_frame()
    multi, strict = select_materialized_direct_action_panel(
        frame, mode="multi_action_balanced", margin_floor=5e-6,
    )
    one, _ = select_materialized_direct_action_panel(
        frame, mode="one_best_e4", margin_floor=5e-6,
    )
    assert multi.action_id.tolist() == ["a2", "a1", "a3", "a5", "a4"]
    assert one.action_id.tolist() == ["a2", "a3", "a5"]
    assert len(strict) == 5
    assert multi.query_index.duplicated().any()


def test_identity_and_source_family_effective_dose() -> None:
    multi, _ = select_materialized_direct_action_panel(
        action_frame(), mode="multi_action_balanced", margin_floor=5e-6,
    )
    weights = identity_family_balanced_action_weights(
        multi, total_weight_per_identity=16.0,
    )
    weighted = multi.assign(weight=weights)
    identity = weighted.groupby("query_ik14").weight.sum()
    assert np.allclose(identity.to_numpy(float), 16.0, rtol=0.0, atol=2e-5)
    i1 = weighted.loc[weighted.query_ik14.eq("i1")]
    family = i1.groupby(["source", "family"]).weight.sum()
    assert np.allclose(family.to_numpy(float), [8.0, 8.0], rtol=0.0, atol=2e-5)
    n_rows = i1.loc[i1.source.eq("N"), "weight"].to_numpy(float)
    assert np.allclose(n_rows, [4.0, 4.0], rtol=0.0, atol=2e-5)


def test_full_physical_coverage_precedes_identity_filler() -> None:
    action_ids = [f"a-{index}" for index in range(30)]
    identities = [f"i-{index % 3}" for index in range(30)]
    policies = [f"p-{index % 7}" for index in range(30)]
    schedules = coverage_first_identity_balanced_schedules(
        action_ids, identities, policies, epochs=4, views_per_identity=4, seed=9,
    )
    flat = [index for epoch in schedules for index in epoch]
    assert len(set(flat[:len(action_ids)])) == len(action_ids)
    assert len(flat) == 48
    frame = pd.DataFrame({
        "action_id": action_ids,
        "query_ik14": identities,
        "source": [f"s-{index % 2}" for index in range(30)],
        "family": [f"f-{index % 2}" for index in range(30)],
    })
    weights = identity_family_balanced_exposure_weights(
        frame, flat, total_weight_per_identity=16.0,
    )
    effective = pd.Series([weights[index] for index in flat]).groupby(
        pd.Series([identities[index] for index in flat]),
    ).sum()
    assert np.allclose(effective.to_numpy(float), 16.0, rtol=0.0, atol=2e-5)


def test_trainer_and_summary_enforce_repaired_contract() -> None:
    source = TRAINER.read_text(encoding="utf-8")
    summary = SUMMARY.read_text(encoding="utf-8")
    ast.parse(source)
    ast.parse(summary)
    for token in (
        "gated_materialized_action_rank(",
        "materialized_direct_mean(",
        "MATERIALIZED_MULTI_ACTION_PRECLIP_LOSS_SCALE = 0.16",
        '"all_strict_actions_exposed_before_recycling": True',
        "identity_family_balanced_exposure_weights(",
        "corrected_score_embedding_query_subset(",
        '"materialized_action_loss_is_balanced_multi_action_direct"',
        '"corrected_metrics_use_exact_query_matvec"',
        '"candidate_reference_embeddings_reused_from_current_E8": True',
    ):
        assert token in source
    for token in (
        'target_mode == "multi_action_balanced"',
        '"held_table_and_registered_recall1_exactly_identical": True',
        "held-table and registered exact Recall@1 disagree",
    ):
        assert token in summary


if __name__ == "__main__":
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_")
    ]
    for test in tests:
        test()
    print(f"[test_noise_e4_multiaction_balanced_sbatch] PASS tests={len(tests)}")
