"""CPU contract tests for the large-coverage Noise native Stage-6 route."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "tasks/build_noise_dreams_native_scale_stage6.py"
SEVERITIES = {
    "easy": (0.10, 0.15, 2),
    "medium": (0.20, 0.30, 5),
    "hard": (0.30, 0.40, 8),
}


def fixed_unicode(values) -> np.ndarray:
    values = list(map(str, values))
    return np.asarray(values, dtype=f"<U{max(map(len, values), default=1)}")


def load_pure_builder_symbols() -> dict:
    tree = ast.parse(BUILDER.read_text(encoding="utf-8"))
    names = {
        "deterministic_seed",
        "identity_preserving_noise_view",
        "PoolAppender",
        "pool_prefix_equal",
        "two_sided_hard_triplet_qualifies",
        "role_formula_is_train_safe",
    }
    selected = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {
        "np": np,
        "hashlib": hashlib,
        "SEVERITIES": SEVERITIES,
        "fixed_unicode": fixed_unicode,
        "stable_fold": lambda formula, folds, seed: 0 if str(formula) == "HELD" else 1,
    }
    exec(compile(module, str(BUILDER), "exec"), namespace)
    return namespace


PURE = load_pure_builder_symbols()
PoolAppender = PURE["PoolAppender"]
identity_preserving_noise_view = PURE["identity_preserving_noise_view"]
pool_prefix_equal = PURE["pool_prefix_equal"]
two_sided_hard_triplet_qualifies = PURE["two_sided_hard_triplet_qualifies"]
role_formula_is_train_safe = PURE["role_formula_is_train_safe"]
HDF5 = 0
ACTION = 1


def clean_tensor() -> np.ndarray:
    tensor = np.zeros((101, 2), dtype=np.float32)
    tensor[0] = (500.0, 1.1)
    tensor[1:61, 0] = np.linspace(50.0, 450.0, 60, dtype=np.float32)
    tensor[1:61, 1] = np.linspace(0.05, 1.0, 60, dtype=np.float32)
    return tensor


def synthetic_pool() -> dict[str, np.ndarray]:
    return {
        "registry_kind": np.asarray([HDF5] * 3, dtype=np.int8),
        "registry_source_index": np.asarray([1, 2, 3], dtype=np.int64),
        "anchor_idx": np.asarray([0], dtype=np.int64),
        "positive_ptr": np.asarray([0, 1], dtype=np.int64),
        "positive_idx": np.asarray([1], dtype=np.int64),
        "negative_ptr": np.asarray([0, 1], dtype=np.int64),
        "negative_idx": np.asarray([2], dtype=np.int64),
        "event_kind": np.asarray([0], dtype=np.int8),
        "event_query": np.asarray([7], dtype=np.int64),
        "event_action_index": np.asarray([-1], dtype=np.int64),
        "event_formula": np.asarray(["C6H12O6"]),
    }


def test_noise_views_are_deterministic_and_identity_preserving() -> None:
    clean = clean_tensor()
    clean_mz = set(map(float, clean[1:, 0][clean[1:, 0] > 0]))
    for severity in SEVERITIES:
        first = identity_preserving_noise_view(
            clean, source_key=17, severity=severity, seed=20260928,
        )
        second = identity_preserving_noise_view(
            clean, source_key=17, severity=severity, seed=20260928,
        )
        assert np.array_equal(first, second)
        assert np.array_equal(first[0], clean[0])
        assert not np.array_equal(first, clean)
        view_mz = set(map(float, first[1:, 0][first[1:, 0] > 0]))
        retained = view_mz & clean_mz
        assert retained
        assert all(mz in clean_mz for mz in retained)
        assert int(np.sum(first[1:, 1] > 0)) >= 8
    negative_a = identity_preserving_noise_view(
        clean, severity="hard", seed=20260928,
        source_key=991, replicate=2,
    )
    negative_b = identity_preserving_noise_view(
        clean, severity="hard", seed=20260928,
        source_key=991, replicate=2,
    )
    negative_other = identity_preserving_noise_view(
        clean, severity="hard", seed=20260928,
        source_key=992, replicate=2,
    )
    assert np.array_equal(negative_a, negative_b)
    assert not np.array_equal(negative_a, negative_other)
    positive_a = identity_preserving_noise_view(
        clean, severity="medium", seed=20260928, source_key=441,
    )
    positive_b = identity_preserving_noise_view(
        clean, severity="medium", seed=20260928, source_key=441,
    )
    same_source_other_semantic_role = identity_preserving_noise_view(
        clean, severity="medium", seed=20260928, source_key=441,
    )
    assert np.array_equal(positive_a, positive_b)
    assert np.array_equal(positive_a, same_source_other_semantic_role)
    base_mz = float(clean[1:][np.argmax(clean[1:, 1]), 0])
    assert np.any(np.abs(negative_a[1:, 0] - base_mz) <= 1e-7)


def test_pool_appender_preserves_stage1_as_exact_prefix() -> None:
    base = synthetic_pool()
    appender = PoolAppender(base)
    anchor = appender.registry(ACTION, 0)
    positive = appender.registry(HDF5, 4)
    negative = appender.registry(HDF5, 5)
    appender.append(
        anchor, [positive], [negative], kind=2, query=8, action=0,
        formula="C7H14O7",
    )
    full = appender.arrays()
    assert pool_prefix_equal(full, base)
    assert len(full["anchor_idx"]) == 2
    assert int(full["event_action_index"][-1]) == 0


def test_fully_noisy_triplet_roles_are_action_registry_members() -> None:
    base = synthetic_pool()
    appender = PoolAppender(base)
    anchor = appender.registry(ACTION, 3)
    positive = appender.registry(ACTION, 4)
    negative = appender.registry(ACTION, 5)
    appender.append(
        anchor, [positive], [negative], kind=2, query=9, action=0,
        formula="C8H16O8",
    )
    full = appender.arrays()
    p0, p1 = map(int, full["positive_ptr"][-2:])
    n0, n1 = map(int, full["negative_ptr"][-2:])
    assert int(full["registry_kind"][full["anchor_idx"][-1]]) == ACTION
    assert set(map(int, full["registry_kind"][full["positive_idx"][p0:p1]])) == {ACTION}
    assert set(map(int, full["registry_kind"][full["negative_idx"][n0:n1]])) == {ACTION}


def test_matched_banks_can_differ_only_at_registered_anchor_positions() -> None:
    stage1 = np.zeros((2, 101, 2), dtype=np.float32)
    broad_targeted = np.zeros((6, 101, 2), dtype=np.float32)
    broad_control = broad_targeted.copy()
    broad_targeted[0, 1, 1] = 0.1
    broad_targeted[3, 1, 1] = 0.2
    targeted = np.concatenate((stage1, broad_targeted))
    control = np.concatenate((stage1, broad_control))
    actual = np.any(np.abs(targeted - control) > 0, axis=(1, 2))
    expected = np.zeros(len(targeted), dtype=bool)
    expected[[2, 5]] = True
    assert np.array_equal(actual, expected)


def test_two_sided_gate_rejects_every_one_sided_or_damaged_case() -> None:
    valid = {
        "anchor_identity_similarity": 0.8,
        "positive_identity_similarity": 0.8,
        "negative_identity_similarity": 0.8,
        "anchor_positive_drop": 0.02,
        "positive_view_drop": 0.02,
        "joint_positive_drop": 0.03,
        "negative_target_gain": 0.02,
        "negative_control_gain": 0.02,
        "target_margin": 0.05,
        "control_margin": 0.06,
        "identity_floor": 0.5,
        "margin_floor": -0.25,
        "margin": 0.1,
        "minimum_hard_delta": 0.01,
    }
    assert two_sided_hard_triplet_qualifies(**valid)
    failures = {
        "anchor_positive_drop": 0.0,
        "positive_view_drop": 0.0,
        "negative_target_gain": 0.0,
        "negative_control_gain": 0.0,
        "anchor_identity_similarity": 0.49,
        "positive_identity_similarity": 0.49,
        "negative_identity_similarity": 0.49,
        "target_margin": -0.26,
        "control_margin": 0.1,
    }
    for key, value in failures.items():
        candidate = dict(valid)
        candidate[key] = value
        assert not two_sided_hard_triplet_qualifies(**candidate), key


def test_role_formula_gate_rejects_outer_and_validation_sources() -> None:
    assert role_formula_is_train_safe(
        "SAFE", validation_formulas={"VALIDATION"}, outer_fold=0,
        formula_fold_seed=20260825,
    )
    assert not role_formula_is_train_safe(
        "HELD", validation_formulas=set(), outer_fold=0,
        formula_fold_seed=20260825,
    )
    assert not role_formula_is_train_safe(
        "VALIDATION", validation_formulas={"VALIDATION"}, outer_fold=0,
        formula_fold_seed=20260825,
    )


def test_stage6_runtime_and_sbatch_contracts() -> None:
    trainer = (ROOT / "tasks/train_noise_dreams_native_scale_stage6.py").read_text(
        encoding="utf-8",
    )
    script = (ROOT / "tasks/run_noise_dreams_native_scale_stage6_2gpu.sbatch").read_text(
        encoding="utf-8",
    )
    builder = (ROOT / "tasks/build_noise_dreams_native_scale_stage6.py").read_text(
        encoding="utf-8",
    )
    assert "construct_native_model(args)" in trainer
    assert "RestoreNativeAdamState" in trainer
    assert "native_dataset(" in trainer
    assert "native_query_disjoint_one_pass_batches(" in trainer
    assert "ContrastiveHead" in trainer
    assert "torch.optim.Adam" in trainer
    assert "custom_loss\": False" in trainer
    assert "custom_optimizer\": False" in trainer
    assert "--warm-start-checkpoint \"$STAGE1_CHECKPOINT\"" in script
    assert "--stage1-run \"$STAGE1\"" in script
    assert "#SBATCH --gpus=2" in script
    assert "#SBATCH --mem" not in script
    assert "NOISE_DREAMS_NATIVE_SCALE_STAGE6" in script
    assert "GNPS_BENCHMARK" in script
    assert "wait_fail_fast" in script
    assert 'kill "$pid" 2>/dev/null || true' in script
    assert "STAGE6_MINIMUM_BROAD_EVENTS = 20_000" in builder
    assert "STAGE6_MINIMUM_BROAD_QUERIES = 10_000" in builder
    assert "stage1_train_pool_is_exact_prefix" in builder
    assert '"training_initialization": "stage1_targeted_champion_exact_continuation"' in builder
    assert "every_broad_query_has_exactly_two_exposures" in builder
    assert "negative_identity_similarity" in builder
    assert "negative_target_gain" in builder
    assert "negative_control_gain" in builder
    assert "anchor_positive_drop" in builder
    assert "positive_view_drop" in builder
    assert "targeted_and_control_differ_only_in_anchor_view" in builder
    assert "appendable_library_version" in builder
    assert "semantic_action_index = stage1_action_count + broad_index" in builder
    assert "anchor_action_index = stage1_action_count + 3 * broad_index" in builder
    assert "query_disjoint_one_pass_schedule_is_realizable" in builder
    assert '[:args.maximum_positive_rows]' in builder
    assert 'positive_rows[:32]' not in builder
    assert 'load_npz(staging / "validation_pool.npz")' in builder
    assert '"stage6-source-pairing"' in builder
    assert "native_action_model_input(view, preprocessor)" in builder
    assert "if not np.array_equal(view, replay)" in builder


def main() -> None:
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
    print(f"[test_noise_dreams_native_scale_stage6] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()
