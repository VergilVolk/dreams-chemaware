"""Contracts for direct Noise triplets on the proven native runtime."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from build_noise_action_hard_native_triplets import (
    NOISE_ACTION_HARD,
    action_row_audit,
    build_arm_pool,
    locate_row,
)
from test_noise_dreams_native import fixture


ROOT = Path(__file__).resolve().parents[1]


def test_one_native_event_per_unique_noise_relation() -> None:
    graph, cache, actions = fixture()
    duplicate = actions.iloc[[0]].copy()
    duplicate["action_id"] = "duplicate-source-discovery"
    actions = pd.concat([actions, duplicate], ignore_index=True)
    action_row_audit(actions, graph)
    pool, audit, aliases = build_arm_pool(
        actions, graph, cache, negative_column="action_hard_negative_row",
    )
    required = {
        "anchor_idx", "positive_ptr", "positive_idx", "negative_ptr",
        "negative_idx", "source_query", "negative_molecule", "source_tag",
    }
    assert set(pool) == required
    assert aliases[-1] == aliases[0]
    assert audit["raw_action_multiplicity_is_optimizer_dose"] is False
    assert audit["actions_collapsed_by_query_negative_deduplication"] >= 1
    assert audit["qualified_action_aliases"] == len(actions)
    assert audit["triplet_events"] == audit["unique_query_negative_pairs"]
    assert np.any((pool["source_tag"] & NOISE_ACTION_HARD) > 0)
    for event, query_value in enumerate(pool["source_query"]):
        query = int(query_value)
        assert int(pool["anchor_idx"][event]) == int(graph["query_row"][query])
        p0, p1 = map(int, pool["positive_ptr"][event:event + 2])
        n0, n1 = map(int, pool["negative_ptr"][event:event + 2])
        positives = pool["positive_idx"][p0:p1]
        negatives = pool["negative_idx"][n0:n1]
        assert len(positives) and len(negatives)
        assert int(graph["query_row"][query]) not in set(map(int, positives))
        assert all(locate_row(graph, query, int(row))[1] for row in positives)
        assert not any(locate_row(graph, query, int(row))[1] for row in negatives)
        assert {
            locate_row(graph, query, int(row))[0] for row in negatives
        } == {int(pool["negative_molecule"][event])}


def test_sbatch_reuses_native_runtime_directly() -> None:
    script = (ROOT / "tasks/run_noise_dreams_native_2gpu.sbatch").read_text(
        encoding="utf-8"
    )
    assert "tasks/train_noise_reference_native.py" in script
    assert "tasks/train_chemaware_dreams_native.py" not in script
    assert '--train-pool "$TRIPLETS/train_pool_targeted.npz"' in script
    assert '--max-steps "$TRAIN_STEPS"' in script
    assert "--checkpoint-mode fixed_steps" in script
    assert "--keep-final-partial-batch" in script
    assert "best.ckpt" not in script
    assert "--batch-size 4 --num-workers 0" in script
    assert '--official-checkpoint "$OFFICIAL_SLIM"' in script
    assert "train_noise_dreams_native.py" not in script
    assert "targeted_action_spectra" not in script


def test_every_conjunctive_gate_is_a_positive_success_assertion() -> None:
    source = (
        ROOT / "tasks/build_noise_action_hard_native_triplets.py"
    ).read_text(encoding="utf-8")
    assert '"outer_held_formulas_used_for_training": False' not in source
    assert '"outer_held_formulas_excluded_from_training": bool(' in source


def main() -> None:
    test_one_native_event_per_unique_noise_relation()
    test_sbatch_reuses_native_runtime_directly()
    test_every_conjunctive_gate_is_a_positive_success_assertion()
    print("[test_noise_action_hard_native_triplets] PASS tests=3", flush=True)


if __name__ == "__main__":
    main()
