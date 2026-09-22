"""CPU contracts for checkpoint-adaptive ChemAware native triplets."""
from __future__ import annotations

import numpy as np
from pathlib import Path

from build_chemaware_residual_native_triplets import (
    ADAPTIVE_HARD,
    CHEMICAL_HARD,
    FALLBACK_HARD,
    MARGIN_VIOLATION,
    STRICT_SPECIFIC,
    positions,
    select_query_negatives,
)
from select_chemaware_residual_checkpoint import step_from_checkpoint


METRICS = (
    "action_top_fraction", "action_largest_region_fraction",
    "action_same_neighbor_fraction", "action_best_advantage_over_baseline",
    "candidate_rule_rank_fraction", "candidate_rule_max",
    "candidate_rule_top2_mean", "delta_rule_max", "delta_rule_top2_mean",
)


def evidence_and_metrics():
    arms, queries, slots = 4, 1, 2
    cache = {name: np.zeros((arms, queries, slots), dtype=np.float32) for name in METRICS}
    # Candidate 2 is a correct-specific chemical boundary, while candidate 1
    # is the current embedding's hardest false molecule.
    cache["action_top_fraction"][0, 0] = (0.1, 0.8)
    cache["action_top_fraction"][1:, 0, 1] = 0.1
    cache["action_largest_region_fraction"][0, 0, 1] = 0.8
    cache["action_same_neighbor_fraction"][0, 0, 1] = 0.8
    cache["action_best_advantage_over_baseline"][0, 0, 1] = 0.5
    cache["candidate_rule_rank_fraction"][0, 0] = (0.5, 0.1)
    cache["candidate_rule_max"][0, 0, 1] = 0.7
    cache["candidate_rule_top2_mean"][0, 0, 1] = 0.6
    cache["delta_rule_max"][0, 0, 1] = 0.5
    cache["delta_rule_top2_mean"][0, 0, 1] = 0.4
    evidence = {
        "query": np.asarray([0]),
        "valid": np.asarray([[True, True]]),
        "proposed_candidate": np.asarray([[1, 2]], dtype=np.int16),
        "baseline_candidate": np.asarray([1], dtype=np.int16),
    }
    manifest = {
        "query_ptr": np.asarray([0, 3]),
        "molecule_label": np.asarray([True, False, False]),
    }
    return evidence, manifest, cache


def main() -> None:
    cache_rows = np.asarray([1, 3, 8], dtype=np.int64)
    assert positions(cache_rows, np.asarray([8, 1])).tolist() == [2, 0]
    try:
        positions(cache_rows, np.asarray([7]))
    except RuntimeError:
        pass
    else:
        raise AssertionError("missing cache row was silently accepted")

    evidence, manifest, metrics = evidence_and_metrics()
    correction = np.zeros((1, 2), dtype=bool)
    protection = np.asarray([[False, True]])
    selected = select_query_negatives(
        evidence, manifest, metrics, 0, 0,
        np.asarray([0.80, 0.75, 0.72]), correction, protection,
        margin=0.1, hardness_window=0.10, events_per_query=2,
    )
    selected = dict(selected)
    assert set(selected) == {1, 2}
    assert selected[1] & ADAPTIVE_HARD
    assert selected[2] & CHEMICAL_HARD
    assert selected[2] & STRICT_SPECIFIC
    assert selected[1] & MARGIN_VIOLATION
    assert selected[2] & MARGIN_VIOLATION

    # Outside the declared hardness window, the second candidate is a generic
    # current-hardness fallback and cannot be called chemical evidence.
    selected = dict(select_query_negatives(
        evidence, manifest, metrics, 0, 0,
        np.asarray([0.80, 0.75, 0.20]), correction, protection,
        margin=0.1, hardness_window=0.10, events_per_query=2,
    ))
    assert selected[2] & FALLBACK_HARD
    assert not selected[2] & CHEMICAL_HARD

    assert step_from_checkpoint("/tmp/step-000750.ckpt") == 750

    sbatch = (Path(__file__).resolve().parent / "run_chemaware_residual_native_stage2.sbatch").read_text(
        encoding="utf-8",
    )
    assert "#SBATCH --gpus=1" in sbatch
    assert "#SBATCH --mem" not in sbatch
    assert "srun --export=ALL --preserve-env" in sbatch
    assert '--evidence-dir "$EVIDENCE"' in sbatch
    assert "--max-steps 1000 --checkpoint-mode fixed_steps" in sbatch
    assert "--formula-role 2" in sbatch and "--formula-role 3" in sbatch
    assert "role_4" not in sbatch.lower()
    print("PASS: ChemAware residual native-triplet contracts")


if __name__ == "__main__":
    main()
