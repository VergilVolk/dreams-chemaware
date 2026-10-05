"""CPU/static gates for the calibrated in-place Stage-3 repair."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "tasks"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
REPAIR = ROOT / "tasks/repair_noise_dreams_native_multidifficulty_stage3.py"
TRAINER = ROOT / "tasks/train_noise_dreams_native_residual_stage2.py"
SUMMARY = ROOT / "tasks/summarize_noise_dreams_native_residual_stage2.py"
SBATCH = ROOT / "tasks/run_noise_dreams_native_stage3_calibrated_repair_2gpu.sbatch"


def _function(path: Path, name: str, namespace: dict) -> object:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        body for body in tree.body
        if isinstance(body, ast.FunctionDef) and body.name == name
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def test_selector_keeps_original_relations_and_one_event_per_query() -> None:
    select = _function(
        REPAIR, "select_calibrated_events",
        {
            "np": np, "pd": pd, "CALIBRATED_GAIN_FLOOR": 1e-4,
            "__name__": "repair_test",
        },
    )
    rows = []
    for index, (query, tier, pos_gain, margin_gain) in enumerate((
        (10, "hard", 0.20, 0.15),
        (10, "easy", 0.10, 0.05),
        (11, "medium", -0.02, 0.20),
        (11, "easy", 0.04, 0.03),
        (12, "hard", 0.08, -0.01),
        (12, "medium", 0.06, 0.02),
    )):
        control_positive = 0.4
        control_margin = -0.2
        rows.append({
            "query_index": query,
            "query_formula": f"C{query}H{query}",
            "query_ik14": f"IK{query:012d}",
            "source": "E10B",
            "family": "candidate_gradient",
            "difficulty_tier": tier,
            "action_id": f"a{index}",
            "stage1_action_index": 100 + index,
            "action_positive_row": 1000 + index,
            "action_negative_row": 2000 + index,
            "targeted_positive_similarity": control_positive + pos_gain,
            "control_positive_similarity": control_positive,
            "targeted_margin": control_margin + margin_gain,
            "control_margin": control_margin,
            "clean_same_boundary_margin": control_margin - 0.01,
            "targeted_hinge_active": True,
        })
    selected = select(pd.DataFrame(rows))
    assert list(selected["query_index"]) == [10, 11, 12]
    assert list(selected["difficulty_tier"]) == ["hard", "easy", "medium"]
    assert selected["query_index"].is_unique
    assert (selected["targeted_positive_gain"] > 0).all()
    assert (selected["targeted_margin_advantage"] > 0).all()
    assert (selected["targeted_clean_margin_advantage"] > 0).all()


def test_pool_compaction_preserves_exact_memberships_and_balances_query_dose() -> None:
    compact = _function(
        REPAIR, "compact_native_pool",
        {
            "np": np,
            "pd": pd,
            "ACTION": 1,
            "fixed_unicode": lambda values: np.asarray(list(values), dtype="U32"),
            "__name__": "repair_test",
        },
    )
    # Four action events (two for query 10), one clean event per action query,
    # and four action-free clean protection candidates.
    kinds = np.asarray([2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
    queries = np.asarray([10, 10, 11, 12, 10, 11, 12, 20, 21, 22, 23])
    actions = np.asarray([0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1])
    registry_kind = np.asarray([1, 1, 1, 1] + [0] * 22, dtype=np.int8)
    registry_source = np.asarray([0, 1, 2, 3] + list(range(22)), dtype=np.int64)
    count = len(kinds)
    anchors = np.asarray(list(range(4)) + list(range(4, 11)), dtype=np.int64)
    positive = np.asarray(list(range(11, 22)), dtype=np.int64)
    negative = np.asarray(list(range(15, 26)), dtype=np.int64)
    pool = {
        "registry_kind": registry_kind,
        "registry_source_index": registry_source,
        "anchor_idx": anchors,
        "positive_ptr": np.arange(count + 1, dtype=np.int64),
        "positive_idx": positive,
        "negative_ptr": np.arange(count + 1, dtype=np.int64),
        "negative_idx": negative,
        "event_kind": kinds,
        "event_query": queries,
        "event_action_index": actions,
        "event_formula": np.asarray([f"F{q}" for q in queries], dtype="U8"),
    }
    selected = pd.DataFrame({
        "stage3_action_index": [1, 2, 3],
        "query_index": [10, 11, 12],
    })
    output, report = compact(pool, selected, protection_events=4)
    action = output["event_kind"] == 2
    assert list(output["event_query"][action]) == [10, 11, 12]
    assert list(output["event_action_index"][action]) == [0, 1, 2]
    assert report == {
        "action_events": 3,
        "action_queries": 3,
        "clean_events_for_action_queries": 3,
        "action_free_protection_events": 4,
        "total_events": 10,
    }
    assert set(output["registry_kind"]) <= {0, 1}
    assert set(output["registry_source_index"][output["registry_kind"] == 1]) == {0, 1, 2}


def test_runtime_changes_only_dose_selection_and_safe_continuation_scale() -> None:
    repair = REPAIR.read_text(encoding="utf-8")
    trainer = TRAINER.read_text(encoding="utf-8")
    summary = SUMMARY.read_text(encoding="utf-8")
    script = SBATCH.read_text(encoding="utf-8")
    for forbidden in (
        "outer_rank", "held_rank", "corrected_query", "introduced_query",
        "CircleLoss", "SupCon", "distill",
    ):
        assert forbidden not in repair
    assert "targeted_positive_gain" in repair
    assert "targeted_margin_advantage" in repair
    assert "drop_duplicates(\"query_index\"" in repair
    assert '"multidifficulty_stage3_repair"' in trainer
    assert '"lr": 1e-6' in trainer
    assert "NOISE_DREAMS_NATIVE_MULTIDIFFICULTY_STAGE3_REPAIR" in summary
    assert "#SBATCH --gpus=2" in script
    assert "#SBATCH --mem" not in script
    assert "--curriculum multidifficulty_stage3_repair" in script
    assert "--lr 1e-6" in script
    assert "--max-epochs 1" in script
    assert "--no-write-held-metric-evidence" in script
    assert "repair_noise_dreams_native_multidifficulty_stage3.py" in script


def main() -> None:
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
    print(
        "[test_noise_dreams_native_stage3_calibrated_repair] "
        f"PASS tests={len(tests)}"
    )


if __name__ == "__main__":
    main()
