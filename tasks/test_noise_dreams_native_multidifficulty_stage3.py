"""CPU/static gates for the native Noise multi-difficulty Stage-3 route."""
from __future__ import annotations

import ast
import heapq
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "tasks"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
BUILDER = ROOT / "tasks/build_noise_dreams_native_multidifficulty_stage3.py"
TRAINER = ROOT / "tasks/train_noise_dreams_native_residual_stage2.py"
SUMMARY = ROOT / "tasks/summarize_noise_dreams_native_residual_stage2.py"
SBATCH = ROOT / "tasks/run_noise_dreams_native_multidifficulty_stage3_2gpu.sbatch"


def _function(path: Path, name: str, namespace: dict) -> object:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        body for body in tree.body
        if isinstance(body, (ast.FunctionDef, ast.AsyncFunctionDef))
        and body.name == name
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


class _Registry:
    HDF5 = 0
    ACTION = 1

    def __init__(self) -> None:
        self.entries: list[tuple[int, int]] = []
        self.positions: dict[tuple[int, int], int] = {}

    def add(self, kind: int, source: int) -> int:
        key = (int(kind), int(source))
        if key not in self.positions:
            self.positions[key] = len(self.entries)
            self.entries.append(key)
        return self.positions[key]

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray([entry[0] for entry in self.entries], dtype=np.int8),
            np.asarray([entry[1] for entry in self.entries], dtype=np.int64),
        )


def _schedule_function() -> object:
    validate_pool = _function(
        ROOT / "tasks/train_noise_dreams_native.py",
        "validate_pool",
        {"np": np, "__name__": "stage3_test"},
    )
    return _function(
        ROOT / "tasks/train_noise_dreams_native.py",
        "native_query_disjoint_one_pass_batches",
        {
            "np": np,
            "heapq": heapq,
            "validate_pool": validate_pool,
            "__name__": "stage3_test",
        },
    )


def test_scientific_contract_is_explicit() -> None:
    source = BUILDER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert tree is not None
    for token in (
        "positive_order[0]",
        "positive_order[len(positive_order) // 2]",
        "positive_order[-1]",
        "np.argmax(negative_scores)",
        '"--minimum-action-events"',
        '"--maximum-action-events-per-query"',
        '"every_positive_is_same_identity"',
        '"every_negative_is_different_identity"',
        '"registered_source_closure"',
        '"hard_tier_is_empirically_harder_than_easy_tier"',
    ):
        assert token in source, token
    assert "default=10000" in source
    assert "default=4" in source
    assert 'tier_counts["hard"] >= 4000' in source
    assert 'tier_counts["hard"] / len(selected) >= 0.35' in source
    forbidden = ("outer_rank", "held_rank", "corrected_query", "introduced_query")
    assert not any(token in source for token in forbidden)


def _record(index: int, similarity: float) -> dict[str, object]:
    relation = lambda offset: {
        "action_positive_row": 1000 + 10 * index + offset,
        "action_negative_row": 9000 + index,
        "positive_similarity": 0.7 - 0.1 * offset,
        "negative_similarity": 0.65,
        "triplet_margin": 0.05 - 0.1 * offset,
        "positive_pool_size": 3,
        "positive_identity_verified": True,
        "negative_identity_verified": True,
    }
    return {
        "stage1_action_index": index,
        "action_id": f"a{index}",
        "query_index": 7,
        "query_row": 77,
        "query_formula": "C7H7NO2",
        "query_ik14": "ABCDEFGHIJKLMN",
        "source": "A4_exact",
        "family": "candidate_gradient",
        "action_clean_similarity": similarity,
        "easy_margin": 0.05,
        "medium_margin": -0.05,
        "hard_margin": -0.15,
        "easy": relation(0),
        "medium": relation(1),
        "hard": relation(2),
    }


def test_query_selection_spans_tiers_without_reusing_actions() -> None:
    choose_query_events = _function(
        BUILDER, "choose_query_events", {"np": np, "__name__": "stage3_test"},
    )

    chosen = choose_query_events(
        [_record(index, 0.99 - 0.1 * index) for index in range(6)], 4, 0.1,
    )
    assert len(chosen) == 4
    assert [row["difficulty_tier"] for row in chosen].count("easy") == 1
    assert [row["difficulty_tier"] for row in chosen].count("medium") == 1
    assert [row["difficulty_tier"] for row in chosen].count("hard") == 2
    assert len({row["stage1_action_index"] for row in chosen}) == 4
    assert all(float(row["triplet_margin"]) < 0.1 for row in chosen)
    assert all("easy" not in row and "medium" not in row and "hard" not in row for row in chosen)


def test_pool_preserves_native_dynamic_membership_contract() -> None:
    make_pool = _function(
        BUILDER,
        "make_pool",
        {
            "np": np,
            "pd": pd,
            "Registry": _Registry,
            "fixed_unicode": lambda values: np.asarray(list(values), dtype="U64"),
            "__name__": "stage3_test",
        },
    )
    validate_pool = _function(
        ROOT / "tasks/train_noise_dreams_native.py",
        "validate_pool",
        {"np": np, "__name__": "stage3_test"},
    )

    selected = pd.DataFrame([
        {
            "query_index": query,
            "query_formula": f"C{query}H{query}O2",
            "difficulty_tier": tier,
            "action_positive_row": 100 + event,
            "action_negative_row": 200 + event,
        }
        for event, (query, tier) in enumerate(
            (
                (query, tier)
                for query in range(7, 11)
                for tier in ("easy", "medium", "hard")
            ),
            start=1,
        )
    ])
    clean = {
        query: {
            "query_row": query,
            "query_formula": f"C{query}H{query}O2",
            "positive_rows": [300 + query, 400 + query],
            "negative_rows": [500 + query, 600 + query],
        }
        for query in range(7, 11)
    }
    sentinels = [
        {
            "query_index": query, "query_row": query,
            "query_formula": f"C{query}H{query}",
            "positive_rows": [300 + query, 400 + query],
            "negative_rows": [500 + query, 600 + query],
        }
        for query in range(20, 24)
    ]
    pool = make_pool(selected, clean, sentinels)
    validate_pool(pool)
    action = pool["event_kind"] == 2
    clean_mask = ~action
    assert int(action.sum()) == 12
    assert np.array_equal(
        pool["event_action_index"][action], np.arange(12, dtype=np.int64),
    )
    assert np.all(np.diff(pool["positive_ptr"])[action] == 1)
    assert np.all(np.diff(pool["negative_ptr"])[action] == 1)
    assert np.all(np.diff(pool["positive_ptr"])[clean_mask] == 2)
    assert np.all(np.diff(pool["negative_ptr"])[clean_mask] == 2)
    native_query_disjoint_one_pass_batches = _schedule_function()
    indices = np.arange(1000, 1000 + len(pool["event_kind"]), dtype=np.int64)
    batches, _, audit = native_query_disjoint_one_pass_batches(
        pool, indices, batch_size=4, seed=3407,
    )
    position = {int(value): index for index, value in enumerate(indices)}
    for batch in batches:
        queries = [
            int(pool["event_query"][position[int(dataset_index)]])
            for dataset_index in batch
        ]
        assert len(queries) == len(set(queries))
    assert audit["every_base_event_exposed_exactly_once"] is True
    assert audit["same_query_events_never_share_an_optimizer_batch"] is True


def test_production_scale_schedule_needs_only_final_batch_padding() -> None:
    schedule = _schedule_function()
    queries: list[int] = []
    kinds: list[int] = []
    actions: list[int] = []
    action = 0
    for query in range(3000):
        for _ in range(1 + query % 4):
            queries.append(query)
            kinds.append(2)
            actions.append(action)
            action += 1
        queries.append(query)
        kinds.append(0)
        actions.append(-1)
    for query in range(3000, 3256):
        queries.append(query)
        kinds.append(0)
        actions.append(-1)
    count = len(queries)
    pool = {
        "registry_kind": np.asarray([0], dtype=np.int8),
        "registry_source_index": np.asarray([0], dtype=np.int64),
        "anchor_idx": np.zeros(count, dtype=np.int64),
        "positive_ptr": np.arange(count + 1, dtype=np.int64),
        "positive_idx": np.zeros(count, dtype=np.int64),
        "negative_ptr": np.arange(count + 1, dtype=np.int64),
        "negative_idx": np.zeros(count, dtype=np.int64),
        "event_kind": np.asarray(kinds, dtype=np.int8),
        "event_query": np.asarray(queries, dtype=np.int64),
        "event_action_index": np.asarray(actions, dtype=np.int64),
        "event_formula": np.asarray([f"F{query}" for query in queries], dtype="U16"),
    }
    indices = np.arange(count, dtype=np.int64)
    batches, fillers, audit = schedule(pool, indices, batch_size=4, seed=3407)
    assert len(fillers) < 4
    assert audit["action_events"] == action
    assert audit["every_base_event_exposed_exactly_once"] is True
    assert audit["same_query_events_never_share_an_optimizer_batch"] is True
    assert sum(map(len, batches)) == count + len(fillers)


def test_training_and_submission_reuse_only_native_runtime() -> None:
    trainer = TRAINER.read_text(encoding="utf-8")
    summary = SUMMARY.read_text(encoding="utf-8")
    script = SBATCH.read_text(encoding="utf-8")
    assert '"multidifficulty_stage3"' in trainer
    assert '"lr": 5e-6' in trainer
    assert '"maximum_steps": STAGE3_MAX_OPTIMIZER_STEPS' in trainer
    assert "ContrastiveHead" in trainer
    assert "native_dataset" in trainer
    assert "torch.optim.Adam" in trainer
    assert "NOISE_DREAMS_NATIVE_MULTIDIFFICULTY_STAGE3" in summary
    assert "#SBATCH --gpus=2" in script
    assert "#SBATCH --mem" not in script
    assert "--curriculum multidifficulty_stage3" in script
    assert "--lr 5e-6" in script
    assert "--no-write-held-metric-evidence" in script
    assert "--minimum-action-events 10000" in script
    assert "--absolute-target-pp 5.0" in script
    assert "max-epochs 1" in script


def main() -> None:
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
    print(f"[test_noise_dreams_native_multidifficulty_stage3] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()
