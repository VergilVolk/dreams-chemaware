"""Fast contract tests for the Noise native residual Stage-2 route."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))


def source(name: str) -> str:
    return (ROOT / "tasks" / name).read_text(encoding="utf-8")


def test_builder_is_current_geometry_and_outer_blind() -> None:
    body = source("build_noise_dreams_native_residual_stage2.py")
    assert "stage1_target_margin" in body
    assert "target_minus_control_margin" in body
    assert "target_minus_clean_margin" in body
    assert "one_residual_action_per_query" in body
    assert "outer-held formula reached" in body
    assert "held_per_query.csv.gz" not in body
    assert "candidate_rank" not in body


def test_training_is_native_and_warm_started() -> None:
    body = source("train_noise_dreams_native_residual_stage2.py")
    assert "load_base_model(\n        args.warm_start_checkpoint" in body
    assert "ContrastiveHead(" in body
    assert "native_dataset(" in body
    assert "native_query_disjoint_one_pass_batches(" in body
    assert '"lr": 1e-6' in body
    assert '"max_epochs": 1' in body
    assert "torch.optim.Adam" in body
    assert "custom_loss\": False" in body
    assert "action-attributable" not in body
    assert "teacher" not in body.lower()


def test_sbatch_uses_exactly_two_gpus_without_memory_request() -> None:
    body = source("run_noise_dreams_native_residual_stage2_2gpu.sbatch")
    assert "#SBATCH --gpus=2" in body
    assert "#SBATCH --mem" not in body
    assert "run_2344820" in body
    assert 'run_arm "${ALLOCATED_GPUS[0]}" targeted &' in body
    assert 'run_arm "${ALLOCATED_GPUS[1]}" control &' in body
    assert "--absolute-target-pp 5.0" in body
    assert "sbatch " not in body
    assert '--checkpoint "$ARMS/${arm}_slim.pt"' in body
    assert "--no-write-held-metric-evidence" in body
    assert "NOISE_STAGE2_TRIPLETS" in body
    assert 'SOURCE_SNAPSHOT="$LOCAL_ROOT/source_snapshot"' in body
    assert 'cp "$ARMS/control/final.ckpt"' not in body
    assert 'cp "$ARMS/targeted/final.ckpt"' not in body
    assert "NOISE_DREAMS_NATIVE_RESIDUAL_STAGE2_PROMOTION_PASS" in body


def test_all_python_sources_parse() -> None:
    for name in (
        "build_noise_dreams_native_residual_stage2.py",
        "train_noise_dreams_native_residual_stage2.py",
        "summarize_noise_dreams_native_residual_stage2.py",
    ):
        ast.parse(source(name), filename=name)


def test_residual_pool_and_query_disjoint_schedule_when_runtime_is_available() -> None:
    try:
        from build_noise_dreams_native_residual_stage2 import make_pool
        from summarize_noise_dreams_native_residual_stage2 import (
            auc_metric_comparison,
            query_auc_comparison,
            rank_comparison,
        )
        from train_noise_dreams_native import native_query_disjoint_one_pass_batches
    except ModuleNotFoundError as error:
        if error.name == "pytorch_lightning":
            return
        raise
    selected = pd.DataFrame({
        "query_index": [10, 11, 12],
        "query_row": [100, 110, 120],
        "query_formula": ["A", "B", "C"],
        "action_positive_row": [101, 111, 121],
        "action_negative_row": [102, 112, 122],
        "clean_positive_row": [103, 113, 123],
        "clean_negative_row": [104, 114, 124],
        "clean_boundary_active": [True, True, False],
    })
    sentinels = pd.DataFrame({
        "query_index": [20, 21, 22, 23],
        "query_row": [200, 210, 220, 230],
        "query_formula": ["D", "E", "F", "G"],
        "clean_positive_row": [201, 211, 221, 231],
        "clean_negative_row": [202, 212, 222, 232],
    })
    pool = make_pool(selected, sentinels)
    action = pool["event_action_index"]
    assert np.array_equal(action[action >= 0], np.arange(3, dtype=np.int64))
    action_events = np.flatnonzero(pool["event_kind"] == 2)
    clean_events = np.flatnonzero(pool["event_kind"] == 0)
    assert len(action_events) == 3
    assert np.all(pool["registry_kind"][pool["anchor_idx"][action_events]] == 1)
    assert np.all(pool["registry_kind"][pool["anchor_idx"][clean_events]] == 0)
    assert np.all(pool["registry_kind"][pool["positive_idx"]] == 0)
    assert np.all(pool["registry_kind"][pool["negative_idx"]] == 0)
    indices = np.arange(500, 500 + len(action), dtype=np.int64)
    batches, _, report = native_query_disjoint_one_pass_batches(
        pool, indices, batch_size=4, seed=3407,
    )
    position = {int(value): index for index, value in enumerate(indices)}
    queries = pool["event_query"]
    for batch in batches:
        batch_queries = [int(queries[position[int(value)]]) for value in batch]
        assert len(batch_queries) == len(set(batch_queries))
    assert report["maximum_action_events_per_semantic_unit"] == 1
    assert report["every_base_event_exposed_exactly_once"] is True
    comparison = rank_comparison(
        np.asarray([1, 1, 2, 1, 3, 1, 2, 1]),
        np.asarray([2, 1, 2, 3, 3, 2, 2, 4]),
        np.asarray(["A", "B", "C", "D", "E", "F", "G", "H"]),
        np.asarray([True, True, False, True, False, True, False, True]),
        repeats=100, seed=7,
    )
    assert comparison["corrected"] == 4
    assert comparison["introduced"] == 0
    assert comparison["risk_net_lambda2"] == 4
    candidate = {
        "retrieval": {"macro_query_auroc": 0.91, "macro_query_auprc": 0.81},
        "near_subset": {"macro_query_auroc": 0.82, "macro_query_auprc": 0.72},
        "micro_candidate": {"auroc": 0.93, "auprc": 0.83},
        "massspecgym_10ppm_pooled_pairwise": {"auroc": 0.84, "auprc": 0.74},
        "massspecgym_mh_10ppm_pooled_pairwise": {"auroc": 0.85, "auprc": 0.75},
    }
    reference = {
        panel: {metric: value - 0.01 for metric, value in metrics.items()}
        for panel, metrics in candidate.items()
    }
    auc = auc_metric_comparison(candidate, reference)
    assert set(auc) == set(candidate)
    assert all(
        body[metric]["improved"] is True
        and np.isclose(body[metric]["delta_pp"], 1.0)
        for panel, body in auc.items()
        for metric in candidate[panel]
    )
    candidate_table = pd.DataFrame({
        "candidate_macro_query_auc": [0.8, 0.9, 0.7, 0.95],
        "candidate_macro_query_auprc": [0.7, 0.8, 0.6, 0.85],
    })
    reference_table = candidate_table - 0.02
    query_auc = query_auc_comparison(
        candidate_table,
        reference_table,
        np.asarray(["A", "B", "C", "D"]),
        np.asarray([True, True, False, True]),
        repeats=100,
        seed=11,
    )
    assert np.isclose(query_auc["macro_query_auroc"]["delta_pp"], 2.0)
    assert np.isclose(query_auc["macro_query_auprc"]["near_delta_pp"], 2.0)


def main() -> None:
    tests = [value for key, value in globals().items() if key.startswith("test_")]
    for test in sorted(tests, key=lambda item: item.__name__):
        test()
    print(f"[test_noise_dreams_native_residual_stage2] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()
