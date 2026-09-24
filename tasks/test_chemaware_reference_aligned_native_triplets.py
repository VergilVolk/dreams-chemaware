"""CPU contracts for reference-aligned ChemAware native triplets."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from build_chemaware_reference_aligned_native_triplets import (
    CHEMICAL_HARD,
    PRIMARY_HARD,
    SAFE_SENTINEL,
    SECONDARY_HARD,
    STRICT_SPECIFIC,
    QueryGeometry,
    chemical_rejection_order_key,
    retain_query_events,
    select_candidate_slots,
)


METRICS = (
    "action_top_fraction", "action_largest_region_fraction",
    "action_same_neighbor_fraction", "action_best_advantage_over_baseline",
    "candidate_rule_rank_fraction", "candidate_rule_max",
    "candidate_rule_top2_mean", "delta_rule_max", "delta_rule_top2_mean",
)


def main() -> None:
    arms, queries, slots = 4, 1, 3
    metrics = {name: np.zeros((arms, queries, slots), dtype=np.float32) for name in METRICS}
    metrics["action_top_fraction"][0, 0, 2] = 0.9
    metrics["candidate_rule_rank_fraction"][0, 0] = (0.5, 0.4, 0.1)
    metrics["candidate_rule_max"][0, 0, 2] = 0.8
    evidence = {
        "query": np.asarray([0]),
        "valid": np.asarray([[True, True, True]]),
        "proposed_candidate": np.asarray([[1, 2, 3]], dtype=np.int16),
        "baseline_candidate": np.asarray([1], dtype=np.int16),
    }
    manifest = {
        "query_ptr": np.asarray([0, 4]),
        "molecule_label": np.asarray([True, False, False, False]),
    }
    geometry = QueryGeometry(
        query_row=100,
        positive_rows=np.asarray([101, 102]),
        positive_scores=np.asarray([0.80, 0.70], dtype=np.float32),
        molecule_scores=np.asarray([0.80, 0.78, 0.76, 0.74], dtype=np.float32),
        negative_rows=(
            np.asarray([101]), np.asarray([201, 202]),
            np.asarray([301, 302]), np.asarray([401, 402]),
        ),
        negative_scores=(
            np.asarray([0.80]), np.asarray([0.78, 0.77]),
            np.asarray([0.76, 0.75]), np.asarray([0.74, 0.73]),
        ),
        activation_probability=np.asarray([0.0, 1.0, 1.0, 0.5], dtype=np.float32),
        mean_hinge=np.asarray([0.0, 0.18, 0.16, 0.08], dtype=np.float32),
    )
    correction = np.zeros((1, 3), dtype=bool)
    protection = np.asarray([[False, False, True]])
    chosen = dict(select_candidate_slots(
        evidence, manifest, metrics, 0, 0, correction, protection, geometry,
        candidates_per_query=3, chemical_hardness_window=0.5,
        min_chemical_activation_probability=0.05,
        chemical_candidates_per_query=1,
    ))
    assert set(chosen) == {1, 2, 3}
    assert chosen[1] & PRIMARY_HARD
    assert chosen[2] & SECONDARY_HARD
    assert chosen[3] & CHEMICAL_HARD
    assert chosen[3] & STRICT_SPECIFIC

    # A chemically justified candidate must retain its chemical role when it
    # is already the checkpoint-hardest false molecule.  The historical code
    # silently left such candidates tagged as generic safety events.
    hardest_protection = np.asarray([[True, False, False]])
    chosen = dict(select_candidate_slots(
        evidence, manifest, metrics, 0, 0, correction, hardest_protection, geometry,
        candidates_per_query=3, chemical_hardness_window=0.5,
        min_chemical_activation_probability=0.05,
        chemical_candidates_per_query=1,
    ))
    assert chosen[1] & PRIMARY_HARD
    assert chosen[1] & CHEMICAL_HARD
    assert chosen[1] & STRICT_SPECIFIC

    # Direction matters: prefer a false candidate rejected by the center arm,
    # not one promoted by it with an equally large absolute contrast.
    signed = {name: value.copy() for name, value in metrics.items()}
    signed["candidate_rule_max"][:, 0, 1] = (0.0, 0.8, 0.8, 0.8)
    signed["candidate_rule_max"][:, 0, 2] = (0.8, 0.0, 0.0, 0.0)
    rejected_key = chemical_rejection_order_key(1, 0, 0, signed, 0.76, False)
    promoted_key = chemical_rejection_order_key(2, 0, 0, signed, 0.74, False)
    assert rejected_key > promoted_key

    # A chemically suggested candidate with no active native hinge cannot
    # displace the arm-independent hard fallback.
    inactive = QueryGeometry(
        **{
            **geometry.__dict__,
            "activation_probability": np.asarray([0.0, 1.0, 1.0, 0.0], dtype=np.float32),
        }
    )
    chosen = dict(select_candidate_slots(
        evidence, manifest, metrics, 0, 0, correction, protection, inactive,
        candidates_per_query=3, chemical_hardness_window=0.5,
        min_chemical_activation_probability=0.05,
        chemical_candidates_per_query=1,
    ))
    assert not chosen[3] & CHEMICAL_HARD

    events = [
        {"candidate": 1, "negative_row": 20, "negative_score": 0.7,
         "tag": PRIMARY_HARD, "activation": 0.5, "hinge": 0.1},
        {"candidate": 1, "negative_row": 21, "negative_score": 0.6,
         "tag": PRIMARY_HARD, "activation": 0.4, "hinge": 0.08},
        {"candidate": 2, "negative_row": 30, "negative_score": 0.5,
         "tag": CHEMICAL_HARD, "activation": 0.2, "hinge": 0.04},
    ]
    retained, sentinel = retain_query_events(events, 2)
    assert not sentinel and {int(event["candidate"]) for event in retained} == {1, 2}
    safe, sentinel = retain_query_events([
        {"candidate": 1, "negative_row": 20, "negative_score": 0.7,
         "tag": PRIMARY_HARD, "activation": 0.0, "hinge": 0.0},
        {"candidate": 2, "negative_row": 30, "negative_score": 0.6,
         "tag": SECONDARY_HARD, "activation": 0.0, "hinge": 0.0},
    ], 2)
    assert sentinel and len(safe) == 1 and int(safe[0]["tag"]) & SAFE_SENTINEL

    sbatch = (
        Path(__file__).resolve().parent / "run_chemaware_reference_aligned_native.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --gpus=1" in sbatch
    assert "#SBATCH --mem" not in sbatch
    assert "srun --export=ALL --preserve-env" in sbatch
    assert "--negative-references-per-candidate 2" in sbatch
    assert "--candidates-per-query 3" in sbatch
    assert "validate_chemaware_triplet_activity_gain.py" in sbatch
    assert "--min-active-event-ratio 1.25" in sbatch
    assert "--max-steps 2000 --checkpoint-mode fixed_steps" in sbatch
    assert "--formula-role 2" in sbatch and "--formula-role 3" in sbatch
    assert "role_4" not in sbatch.lower()
    print("PASS: ChemAware reference-aligned native-triplet contracts")


if __name__ == "__main__":
    main()
