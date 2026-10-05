"""CPU/static tests for the Stage-3 frozen zero-update audit."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "tasks/audit_noise_dreams_native_stage3_zero_update.py"
SBATCH = ROOT / "tasks/run_noise_dreams_native_stage3_zero_update.sbatch"


def _function(name: str, namespace: dict):
    tree = ast.parse(AUDIT.read_text(encoding="utf-8"))
    node = next(
        body for body in tree.body
        if isinstance(body, ast.FunctionDef) and body.name == name
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(AUDIT), "exec"), namespace)
    return namespace[name]


def test_coverage_reports_independent_units_and_extra_clean_queries() -> None:
    distribution = _function("distribution", {"np": np, "pd": pd})
    coverage_audit = _function(
        "coverage_audit", {"np": np, "pd": pd, "distribution": distribution},
    )
    selected = pd.DataFrame([
        {
            "query_index": 1, "query_ik14": "A", "query_formula": "F1",
            "source": "S1", "family": "X", "difficulty_tier": "easy",
            "action_positive_row": 10, "action_negative_row": 20,
            "stage1_action_index": 100, "action_id": "a", "positive_pool_size": 3,
        },
        {
            "query_index": 1, "query_ik14": "A", "query_formula": "F1",
            "source": "S2", "family": "Y", "difficulty_tier": "hard",
            "action_positive_row": 11, "action_negative_row": 20,
            "stage1_action_index": 101, "action_id": "b", "positive_pool_size": 2,
        },
        {
            "query_index": 2, "query_ik14": "B", "query_formula": "F2",
            "source": "S1", "family": "X", "difficulty_tier": "medium",
            "action_positive_row": 12, "action_negative_row": 21,
            "stage1_action_index": 102, "action_id": "c", "positive_pool_size": 1,
        },
    ])
    pool = {
        "event_kind": np.asarray([2, 2, 2, 0, 0, 0], dtype=np.int8),
        "event_action_index": np.asarray([0, 1, 2, -1, -1, -1]),
        "event_query": np.asarray([1, 1, 2, 1, 2, 3]),
    }
    report = coverage_audit(selected, pool)
    assert report["action_rows"] == 3
    assert report["unique_queries"] == 2
    assert report["unique_identities"] == 2
    assert report["unique_query_positive_negative_boundaries"] == 3
    assert report["extra_clean_only_queries"] == 1
    assert report["all_action_queries_have_clean_event"] is True
    assert report["rows_with_at_least_two_measured_positives"] == 2
    assert report["rows_with_at_least_three_measured_positives"] == 1
    assert report["source_by_tier"]["S1"] == {
        "easy": 1, "hard": 0, "medium": 1,
    }


def test_gradient_comparison_is_exact_for_small_vectors() -> None:
    compare = _function("compare_gradients", {"torch": torch})
    left = [torch.tensor([1.0, 0.0]), torch.tensor([1.0])]
    right = [torch.tensor([0.0, 1.0]), torch.tensor([1.0])]
    result = compare(left, right, {"backbone": [0], "head": [1], "all": [0, 1]})
    assert abs(result["backbone"]["cosine"]) < 1e-12
    assert abs(result["head"]["cosine"] - 1.0) < 1e-12
    assert abs(result["all"]["cosine"] - 0.5) < 1e-12


def test_audit_is_zero_update_and_submission_is_bounded() -> None:
    source = AUDIT.read_text(encoding="utf-8")
    script = SBATCH.read_text(encoding="utf-8")
    assert "optimizer.step" not in source
    assert "torch.optim" not in source
    assert '"weights_updated": False' in source
    assert "targeted_vs_control_parameter_gradient" in source
    assert "targeted_vs_clean_parameter_gradient" in source
    assert "role_gradient_norms" in source
    assert "#SBATCH --gpus=1" in script
    assert "#SBATCH --mem" not in script
    assert "--gradient-events-per-stratum 8" in script


def main() -> None:
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
    print(f"[test_noise_dreams_native_stage3_zero_update] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()
