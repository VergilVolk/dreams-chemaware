"""CPU contracts for high-capacity ChemAware pair-expanded triplets."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from build_chemaware_reference_aligned_native_triplets import (
    CHEMICAL_HARD,
    HARD_FALLBACK,
    PRIMARY_HARD,
    SECONDARY_HARD,
    QueryGeometry,
    explicit_positive_indices,
    select_candidate_slots,
)


METRICS = (
    "action_top_fraction", "action_largest_region_fraction",
    "action_same_neighbor_fraction", "action_best_advantage_over_baseline",
    "candidate_rule_rank_fraction", "candidate_rule_max",
    "candidate_rule_top2_mean", "delta_rule_max", "delta_rule_top2_mean",
)


def main() -> None:
    arms, queries, slots = 4, 1, 5
    metrics = {name: np.zeros((arms, queries, slots), dtype=np.float32) for name in METRICS}
    metrics["candidate_rule_rank_fraction"][0, 0] = (0.5, 0.4, 0.1, 0.2, 0.3)
    metrics["candidate_rule_max"][0, 0] = (0.0, 0.0, 0.9, 0.8, 0.1)
    evidence = {
        "query": np.asarray([0]),
        "valid": np.ones((1, 5), dtype=bool),
        "proposed_candidate": np.asarray([[1, 2, 3, 4, 5]], dtype=np.int16),
        "baseline_candidate": np.asarray([1], dtype=np.int16),
    }
    manifest = {
        "query_ptr": np.asarray([0, 6]),
        "molecule_label": np.asarray([True, False, False, False, False, False]),
    }
    geometry = QueryGeometry(
        query_row=100,
        positive_rows=np.asarray([101, 102, 103]),
        positive_scores=np.asarray([0.80, 0.70, 0.60], dtype=np.float32),
        molecule_scores=np.asarray([0.80, 0.79, 0.78, 0.77, 0.76, 0.75], dtype=np.float32),
        negative_rows=tuple(np.asarray([200 + 2*i, 201 + 2*i]) for i in range(6)),
        negative_scores=tuple(np.asarray([0.79 - .01*i, 0.78 - .01*i]) for i in range(6)),
        activation_probability=np.asarray([0.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        mean_hinge=np.asarray([0.0, .2, .19, .18, .17, .16], dtype=np.float32),
    )
    correction = np.zeros((1, 5), dtype=bool)
    protection = np.asarray([[False, False, True, True, False]])
    selected = dict(select_candidate_slots(
        evidence, manifest, metrics, 0, 0, correction, protection, geometry,
        candidates_per_query=5, chemical_hardness_window=0.5,
        min_chemical_activation_probability=0.05,
        chemical_candidates_per_query=2,
    ))
    assert len(selected) == 5
    assert selected[1] & PRIMARY_HARD
    assert selected[2] & SECONDARY_HARD
    assert selected[3] & CHEMICAL_HARD
    assert selected[4] & CHEMICAL_HARD
    assert selected[5] & HARD_FALLBACK

    scores = np.asarray([0.90, 0.80, 0.70, 0.60])
    hinge = np.asarray([0.0, 0.01, 0.10, 0.20])
    # Boundary positive is the highest-scoring active view; hard positive is
    # the lowest-scoring active view.
    assert explicit_positive_indices(scores, hinge, 2) == [1, 3]
    assert explicit_positive_indices(scores, np.zeros(4), 2) == [0]

    sbatch = (
        Path(__file__).resolve().parent / "run_chemaware_pair_expanded_native.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --gpus=1" in sbatch
    assert "#SBATCH --mem" not in sbatch
    assert "srun --export=ALL --preserve-env" in sbatch
    assert "--candidates-per-query 5" in sbatch
    assert "--chemical-candidates-per-query 2" in sbatch
    assert "--explicit-positive-references-per-negative 2" in sbatch
    assert "--max-active-spectrum-events-per-query 12" in sbatch
    assert "--max-steps 3000 --checkpoint-mode fixed_steps" in sbatch
    assert "--formula-role 2" in sbatch and "--formula-role 3" in sbatch
    assert "role_4" not in sbatch.lower()
    print("PASS: ChemAware pair-expanded native-triplet contracts")


if __name__ == "__main__":
    main()
