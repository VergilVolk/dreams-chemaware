"""Direct scientific-contract tests for Noise native Stage-5."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "tasks"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

BUILDER = ROOT / "tasks/build_noise_dreams_native_negative_residual_stage5.py"


class _Registry:
    HDF5 = 0
    ACTION = 1
    HDF5_CLONE = 2

    def __init__(self) -> None:
        self.positions: dict[tuple[int, int], int] = {}
        self.kind: list[int] = []
        self.source: list[int] = []

    def add(self, kind: int, source: int) -> int:
        key = (int(kind), int(source))
        if key not in self.positions:
            self.positions[key] = len(self.kind)
            self.kind.append(key[0])
            self.source.append(key[1])
        return self.positions[key]

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.kind, dtype=np.int8), np.asarray(self.source, dtype=np.int64)


def _function(name: str, namespace: dict) -> object:
    tree = ast.parse(BUILDER.read_text(encoding="utf-8"))
    node = next(
        body for body in tree.body
        if isinstance(body, ast.FunctionDef) and body.name == name
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(BUILDER), "exec"), namespace)
    return namespace[name]


_event_members = _function("_event_members", {"np": np})
_registry_sources = _function("_registry_sources", {"np": np})
aggregate_action_row_scores = _function(
    "aggregate_action_row_scores", {"np": np},
)
select_hinge_active_negative_rows = _function(
    "select_hinge_active_negative_rows", {"np": np},
)
build_arm_pool = _function(
    "build_arm_pool",
    {
        "np": np,
        "Registry": _Registry,
        "fixed_unicode": lambda values: np.asarray(list(values), dtype="U16"),
        "_event_members": _event_members,
        "_registry_sources": _registry_sources,
    },
)


def _stage1_pool() -> dict[str, np.ndarray]:
    # Three clean queries. Query 0 will replace one negative pool despite
    # having many source actions; it must still contribute exactly one event.
    return {
        "registry_kind": np.zeros(12, dtype=np.int8),
        "registry_source_index": np.asarray(
            [10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33], dtype=np.int64,
        ),
        "anchor_idx": np.asarray([0, 4, 8], dtype=np.int64),
        "positive_ptr": np.asarray([0, 2, 4, 6], dtype=np.int64),
        "positive_idx": np.asarray([1, 2, 5, 6, 9, 10], dtype=np.int64),
        "negative_ptr": np.asarray([0, 1, 2, 3], dtype=np.int64),
        "negative_idx": np.asarray([3, 7, 11], dtype=np.int64),
        "event_kind": np.zeros(3, dtype=np.int8),
        "event_query": np.asarray([0, 1, 2], dtype=np.int64),
        "event_action_index": np.full(3, -1, dtype=np.int64),
        "event_formula": np.asarray(["A", "B", "C"]),
    }


def _members(pool: dict[str, np.ndarray], prefix: str, event: int) -> tuple[int, ...]:
    ptr = pool[f"{prefix}_ptr"]
    idx = pool[f"{prefix}_idx"][int(ptr[event]):int(ptr[event + 1])]
    return tuple(map(int, pool["registry_source_index"][idx]))


def test_query_dose_is_one_and_only_negative_membership_changes() -> None:
    stage1 = _stage1_pool()
    clean = {0: 0, 1: 1, 2: 2}
    target = build_arm_pool(stage1, clean, {0: [40, 41]}, {0: 0})
    control = build_arm_pool(stage1, clean, {0: [50, 51]}, {0: 0})
    assert list(target["event_query"]) == [0, 1, 2]
    assert len(np.unique(target["event_query"])) == len(target["event_query"])
    assert list(target["event_action_index"]) == [0, -1, -1]
    assert list(target["event_kind"]) == [1, 0, 0]
    assert not np.any(target["registry_kind"] == 1)
    assert not np.any(control["registry_kind"] == 1)
    for event in range(3):
        assert int(target["registry_source_index"][target["anchor_idx"][event]]) == int(
            control["registry_source_index"][control["anchor_idx"][event]]
        )
        assert _members(target, "positive", event) == _members(control, "positive", event)
    assert _members(target, "negative", 0) == (40, 41)
    assert _members(control, "negative", 0) == (50, 51)
    assert _members(target, "negative", 1) == _members(control, "negative", 1) == (23,)


def test_source_balanced_action_aggregation_does_not_reward_multiplicity() -> None:
    # Source A contributes two identical actions while source B contributes
    # one. Source-balanced max-then-mean must give both sources equal weight.
    actions = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    sources = np.asarray(["A", "A", "B"])
    candidates = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    observed = aggregate_action_row_scores(actions, sources, candidates)
    assert np.allclose(observed, np.asarray([0.5, 0.5], dtype=np.float32))


def test_negative_molecule_is_action_ranked_but_every_dynamic_draw_is_active() -> None:
    measured = np.asarray([
        [1.00, 0.00],  # clean anchor row 0
        [0.99, 0.01],  # easiest positive row 1
        [0.96, 0.04],  # second positive row 2
        [0.91, 0.09],  # molecule 0, hinge active
        [0.70, 0.30],  # molecule 1, inactive despite high action score
        [0.92, 0.08],  # molecule 2, hinge active and action preferred
    ], dtype=np.float32)
    rows, molecules, audit = select_hinge_active_negative_rows(
        [np.asarray([3]), np.asarray([4]), np.asarray([5])],
        {index: index for index in range(len(measured))}, measured, measured[0],
        np.asarray([1, 2]), np.asarray([0.2, 1.0, 0.9], dtype=np.float32),
        np.asarray([3, 4, 5]), margin=0.1, maximum_molecules=2,
    )
    assert rows == [5, 3]
    assert molecules == [2, 0]
    assert audit["negative_similarity_min"] > audit["hinge_threshold"]


def test_trainer_keeps_native_stage1_runtime_and_forbids_action_model_input() -> None:
    trainer = (ROOT / "tasks/train_noise_dreams_native_residual_stage2.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(trainer)
    assert '"negative_residual_stage5"' in trainer
    assert '"lr": 5e-6' in trainer
    assert '"triplet_loss_margin": 0.1' in trainer
    assert '"batch_size": 4' in trainer
    assert '"max_epochs": 1' in trainer
    assert "Stage-5 action tensor entered the native model-input registry" in trainer
    assert any(isinstance(node, ast.ClassDef) for node in tree.body)


def test_sbatch_requires_two_gpus_and_runs_internal_plus_gnps_evaluation() -> None:
    script = (ROOT / "tasks/run_noise_dreams_native_negative_residual_stage5_2gpu.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --gpus=2" in script
    assert "--mem" not in script
    assert "--curriculum negative_residual_stage5" in script
    assert "--lr 5e-6 --weight-decay 0 --triplet-loss-margin 0.1" in script
    assert "evaluate_noise_dreams_native.py" in script
    assert "evaluate_gnps_gold_silver_10ppm_embeddings.py" in script
    assert "--triplet-dir \"$TRIPLETS/$arm\"" in script


def test() -> None:
    test_query_dose_is_one_and_only_negative_membership_changes()
    test_source_balanced_action_aggregation_does_not_reward_multiplicity()
    test_negative_molecule_is_action_ranked_but_every_dynamic_draw_is_active()
    test_trainer_keeps_native_stage1_runtime_and_forbids_action_model_input()
    test_sbatch_requires_two_gpus_and_runs_internal_plus_gnps_evaluation()


def main() -> None:
    test()
    print("[test_noise_dreams_native_negative_residual_stage5] PASS tests=5")


if __name__ == "__main__":
    main()
