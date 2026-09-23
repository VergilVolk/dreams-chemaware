"""CPU contracts for ChemAware counterfactual-specific safety replay pools."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from build_chemaware_reference_aligned_native_triplets import CHEMICAL_HARD, PRIMARY_HARD
from build_chemaware_specific_replay_native_triplets import (
    MATCHED_NULL,
    SAFETY_REPLAY,
    SPECIFIC_CHEMISTRY,
    compose_pool,
    matched_null_schedule,
    matched_geometry_audit,
    prune_to_matched_null_capacity,
    select_safety,
    select_specific_schedule,
    validate_source_pool,
)


def pool(events: list[tuple[int, int, int, int, float]]) -> dict[str, np.ndarray]:
    """query, candidate, tag, row, hinge -> one-edge native events."""
    n = len(events)
    query = np.asarray([event[0] for event in events], dtype=np.int64)
    row = np.asarray([event[3] for event in events], dtype=np.int64)
    return {
        "anchor_idx": query + 1000,
        "positive_ptr": np.arange(n + 1, dtype=np.int64),
        "positive_idx": row + 100,
        "negative_ptr": np.arange(n + 1, dtype=np.int64),
        "negative_idx": row + 200,
        "source_query": query,
        "negative_candidate": np.asarray([event[1] for event in events], dtype=np.int16),
        "positive_reference_row": row + 100,
        "negative_reference_row": row + 200,
        "source_tag": np.asarray([event[2] for event in events], dtype=np.int8),
        "activation_probability": np.ones(n, dtype=np.float32),
        "mean_hinge_at_mining": np.asarray([event[4] for event in events], dtype=np.float32),
    }


def main() -> None:
    correct = pool([
        (10, 1, PRIMARY_HARD, 1, .01),
        (10, 2, CHEMICAL_HARD, 2, .20),
        (10, 2, CHEMICAL_HARD, 3, .10),
        (11, 1, PRIMARY_HARD, 4, .02),
        (11, 3, CHEMICAL_HARD, 5, .30),
    ])
    null_a = pool([
        (10, 1, PRIMARY_HARD, 1, .01), (10, 2, CHEMICAL_HARD, 6, .15),
        (10, 4, CHEMICAL_HARD, 7, .12), (10, 5, CHEMICAL_HARD, 18, .11),
        (11, 1, PRIMARY_HARD, 4, .02),
        (11, 3, CHEMICAL_HARD, 8, .25), (11, 5, CHEMICAL_HARD, 9, .10),
    ])
    null_b = pool([
        (10, 1, PRIMARY_HARD, 1, .01), (10, 4, CHEMICAL_HARD, 10, .13),
        (10, 5, CHEMICAL_HARD, 11, .11), (11, 1, PRIMARY_HARD, 4, .02),
        (11, 3, CHEMICAL_HARD, 12, .24), (11, 6, CHEMICAL_HARD, 13, .09),
    ])
    null_c = pool([
        (10, 1, PRIMARY_HARD, 1, .01), (10, 6, CHEMICAL_HARD, 14, .14),
        (10, 7, CHEMICAL_HARD, 15, .12), (11, 1, PRIMARY_HARD, 4, .02),
        (11, 7, CHEMICAL_HARD, 16, .23), (11, 8, CHEMICAL_HARD, 17, .08),
    ])
    nulls = {"a": null_a, "b": null_b, "c": null_c}
    validate_source_pool(correct, "synthetic:correct")
    safety = select_safety(correct)
    assert set(safety) == {10, 11}
    schedule = select_specific_schedule(
        correct, nulls, safety, cap=3, maximum_null_agreement=1,
    )
    # Query 10 candidate 2 occurs in one null and is retained. Query 11
    # candidate 3 occurs in two nulls and is rejected.
    assert len(schedule[10]) == 2
    assert schedule[11] == []
    schedule, removed = prune_to_matched_null_capacity(
        correct, nulls, safety, schedule,
    )
    assert removed == 0
    correct_output = compose_pool(
        correct, safety, correct, schedule, SPECIFIC_CHEMISTRY,
    )
    assert np.array_equal(correct_output["curriculum_role"], [1, 2, 2, 1])
    matched = matched_null_schedule(null_a, schedule, correct, safety)
    # Safety must be sourced from the correct pool even for a null arm.
    null_output = compose_pool(correct, safety, null_a, matched, MATCHED_NULL)
    assert np.array_equal(null_output["source_query"], correct_output["source_query"])
    assert np.array_equal(null_output["curriculum_role"], [1, 3, 3, 1])
    assert 2 not in null_output["negative_candidate"][[1, 2]]
    assert np.array_equal(
        null_output["negative_reference_row"][[0, 3]],
        correct_output["negative_reference_row"][[0, 3]],
    )
    geometry = matched_geometry_audit(correct_output, null_output)
    assert geometry["matched_chemical_events"] == 2
    assert geometry["mean_absolute_hinge_gap"] < 0.1

    sbatch = (
        Path(__file__).resolve().parent / "run_chemaware_specific_replay_native.sbatch"
    )
    if sbatch.exists():
        text = sbatch.read_text(encoding="utf-8")
        assert "#SBATCH --gpus=1" in text
        assert "#SBATCH --mem" not in text
        assert "--official-checkpoint \"$BASE_CHECKPOINT\"" in text
        assert "--formula-role 2" in text and "--formula-role 3" in text
        assert "--require-positive-formula-ci" in text
    print("PASS: ChemAware specific-replay native-triplet contracts")


if __name__ == "__main__":
    main()
