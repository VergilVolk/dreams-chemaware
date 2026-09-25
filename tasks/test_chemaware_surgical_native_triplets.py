"""CPU contracts for fixed-budget ChemAware surgical curriculum injection."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from build_chemaware_action_hard_native_triplets import (
    ACTION_HARD,
    OFFICIAL_HARD,
    SPECIFIC_HARD,
)
from build_chemaware_specific_replay_native_triplets import (
    MATCHED_NULL,
    SPECIFIC_CHEMISTRY,
)
from build_chemaware_surgical_native_triplets import (
    SURGICAL_CHEMICAL,
    common_schedule,
    surgical_pool,
    validate_native_pool,
)


ARMS = ("correct", "content_permuted", "content_permuted_b", "content_permuted_c")


def native_pool(events: list[tuple[int, int, int, int]]) -> dict[str, np.ndarray]:
    """query, candidate, source_tag, row; one edge on each ragged side."""
    count = len(events)
    return {
        "anchor_idx": np.asarray([1000 + e[0] for e in events], dtype=np.int64),
        "positive_ptr": np.arange(count + 1, dtype=np.int64),
        "positive_idx": np.asarray([100 + e[3] for e in events], dtype=np.int64),
        "negative_ptr": np.arange(count + 1, dtype=np.int64),
        "negative_idx": np.asarray([200 + e[3] for e in events], dtype=np.int64),
        "source_query": np.asarray([e[0] for e in events], dtype=np.int64),
        "negative_candidate": np.asarray([e[1] for e in events], dtype=np.int16),
        "source_tag": np.asarray([e[2] for e in events], dtype=np.int8),
    }


def specific_pool(
    events: list[tuple[int, int, int, int]], role: int,
) -> dict[str, np.ndarray]:
    result = native_pool(events)
    result.update({
        "curriculum_role": np.full(len(events), role, dtype=np.int8),
        "positive_reference_row": result["positive_idx"].copy(),
        "negative_reference_row": result["negative_idx"].copy(),
        "activation_probability": np.ones(len(events), dtype=np.float32),
        "mean_hinge_at_mining": np.full(len(events), 0.2, dtype=np.float32),
    })
    return result


def main() -> None:
    # Every query retains its official event. Query 10 has two replaceable
    # events, query 11 has one, and query 12 has no capacity and must be absent
    # from the common schedule.
    base_template = [
        (10, 1, OFFICIAL_HARD, 1),
        (10, 2, ACTION_HARD | SPECIFIC_HARD, 2),
        (10, 3, ACTION_HARD, 3),
        (11, 1, OFFICIAL_HARD | ACTION_HARD, 4),
        (11, 4, SPECIFIC_HARD, 5),
        (12, 1, OFFICIAL_HARD, 6),
    ]
    base = {arm: native_pool(base_template) for arm in ARMS}
    specific = {}
    for offset, arm in enumerate(ARMS):
        role = SPECIFIC_CHEMISTRY if arm == "correct" else MATCHED_NULL
        # Candidate 2 is an exact refinement in correct and candidate 20+ is a
        # swap in nulls. The schedule, not candidate identity, stays matched.
        candidate = 2 if arm == "correct" else 20 + offset
        specific[arm] = specific_pool([
            (10, candidate, SPECIFIC_HARD, 20 + offset),
            (10, 30 + offset, SPECIFIC_HARD, 30 + offset),
            (11, 40 + offset, SPECIFIC_HARD, 40 + offset),
            (12, 50 + offset, SPECIFIC_HARD, 50 + offset),
        ], role)
    for arm in ARMS:
        validate_native_pool(base[arm], f"base:{arm}")
        validate_native_pool(specific[arm], f"specific:{arm}")
    counts, schedule = common_schedule(base, specific, cap=1)
    assert counts == {10: 1, 11: 1}
    outputs = {}
    audits = {}
    for arm in ARMS:
        outputs[arm], audits[arm] = surgical_pool(
            base[arm], specific[arm], schedule[arm],
        )
        assert len(outputs[arm]["anchor_idx"]) == len(base[arm]["anchor_idx"])
        assert audits[arm]["replacements"] == 2
        assert audits[arm]["replacement_queries"] == 2
        assert np.sum(outputs[arm]["source_tag"] & SURGICAL_CHEMICAL > 0) == 2
        official = (base[arm]["source_tag"] & OFFICIAL_HARD) > 0
        assert np.array_equal(
            outputs[arm]["negative_candidate"][official],
            base[arm]["negative_candidate"][official],
        )
    assert audits["correct"]["exact_candidate_reference_refinements"] == 1
    assert audits["correct"]["candidate_swaps"] == 1
    assert all(audits[arm]["events"] == audits[arm]["base_events"] for arm in ARMS)

    sbatch = Path(__file__).resolve().parent / "run_chemaware_surgical_native.sbatch"
    if sbatch.exists():
        text = sbatch.read_text(encoding="utf-8")
        assert "#SBATCH --gpus=1" in text
        assert "#SBATCH --mem" not in text
        assert "train_chemaware_specific_replay_native.py" in text
        assert "train_chemaware_dreams_native.py" not in text
        assert "--official-checkpoint data/e1/official_embedding_slim.pt" in text
        assert "--paired-reference stage1_base" in text
        assert "--formula-role 2" in text and "--formula-role 3" in text
        assert "--require-positive-formula-ci" in text
        assert '"$OUT/training_dose${dose}" 3000 1000' in text
        assert '--max-steps "$maximum_steps"' in text
        assert '--save-every-n-steps "$save_steps"' in text
        assert '--chemical-events-per-query "$dose"' in text
        assert "correct_checkpoint_sha256.tsv" in text
        assert "CHEMAWARE_SURGICAL_ROLE2_STOP" in text
        assert "train_native()" in text
        assert "for dose in 1 2" in text
        assert "for null_name in content_permuted content_permuted_b content_permuted_c" in text
    print("PASS: ChemAware surgical native-triplet contracts")


if __name__ == "__main__":
    main()
