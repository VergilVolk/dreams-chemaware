"""CPU contracts for ChemAware high-coverage native triplet mining."""
from __future__ import annotations

import numpy as np

from build_chemaware_high_coverage_native_triplets import build_pool
from chemaware_native_triplet_mining_core import (
    MiningRecipe,
    directional_masks,
    select_recipe,
    summarize_masks,
    validate_evidence,
)


def synthetic_evidence() -> dict[str, np.ndarray]:
    queries, candidates, arms = 300, 2, 4
    names = np.asarray((
        "action_top_fraction", "action_largest_region_fraction",
        "action_same_neighbor_fraction", "action_best_advantage_over_baseline",
        "candidate_official_rank_fraction",
    ))
    metric = np.zeros((arms, queries, candidates, len(names)), dtype=np.float32)
    valid = np.ones((queries, candidates), dtype=bool)
    benefit = np.zeros_like(valid)
    harmful = np.zeros_like(valid)
    # 180 correction queries: only correct chemistry robustly promotes truth.
    benefit[:180, 0] = True
    metric[0, :180, 0] = (0.30, 0.25, 0.85, 0.12, 0.05)
    metric[1:, :180, 0] = (0.05, 0.04, 0.35, 0.01, 0.05)
    # 120 protection queries: correct chemistry rejects both hard false
    # candidates while every semantic null promotes them.
    harmful[180:, :] = True
    metric[0, 180:, :] = (0.01, 0.01, 0.20, -0.05, 0.10)
    metric[1:, 180:, :] = (0.25, 0.20, 0.80, 0.08, 0.10)
    # Fill invariant official-rank metric across arms.
    metric[:, :, :, 4] = metric[0, :, :, 4]
    return {
        "schema": np.asarray("chemaware_multinull_triplet_evidence_v1"),
        "query": np.arange(queries, dtype=np.int64),
        "formula": np.asarray([f"F{i // 2}" for i in range(queries)]),
        "identity": np.asarray([f"I{i}" for i in range(queries)]),
        "valid": valid,
        "proposed_candidate": np.vstack((
            np.tile(np.asarray((0, 2), dtype=np.int16), (180, 1)),
            np.tile(np.asarray((1, 2), dtype=np.int16), (120, 1)),
        )),
        "baseline_candidate": np.r_[np.ones(180), np.zeros(120)].astype(np.int16),
        "baseline_rank": np.r_[np.full(180, 2), np.ones(120)].astype(np.int16),
        "benefit": benefit,
        "harmful": harmful,
        "arm_names": np.asarray(("correct", "null_a", "null_b", "null_c")),
        "metric_names": names,
        "action_count": np.asarray(63, dtype=np.int16),
        "arm_metric": metric,
    }


def synthetic_manifest() -> dict[str, np.ndarray]:
    query_count = 300
    query_ptr = np.arange(0, 3 * query_count + 1, 3, dtype=np.int64)
    molecule_ptr = [0]
    pair_rows: list[int] = []
    query_rows: list[int] = []
    molecule_label: list[bool] = []
    for query in range(query_count):
        base = 4 * query
        query_rows.append(base)
        for local, rows in enumerate(((base, base + 1), (base + 2,), (base + 3,))):
            pair_rows.extend(rows)
            molecule_ptr.append(len(pair_rows))
            molecule_label.append(local == 0)
    return {
        "query_row": np.asarray(query_rows, dtype=np.int64),
        "query_formula": np.asarray([f"F{i // 2}" for i in range(query_count)]),
        "query_ptr": query_ptr,
        "molecule_ptr": np.asarray(molecule_ptr, dtype=np.int64),
        "molecule_label": np.asarray(molecule_label, dtype=bool),
        "pair_candidate_row": np.asarray(pair_rows, dtype=np.int64),
    }


def main() -> None:
    evidence = synthetic_evidence()
    audit = validate_evidence(evidence)
    assert audit == {"queries": 300, "candidate_events": 600, "arms": 4, "metrics": 5}
    recipe = MiningRecipe(
        min_support=0.05, min_support_delta=0.05, min_region=0.04,
        min_neighbor=0.50, min_advantage_delta=0.02,
        max_official_rank_fraction=0.25,
    )
    correction, protection, _ = directional_masks(evidence, recipe, 0)
    summary = summarize_masks(evidence, correction, protection)
    assert summary["correction_directions"] == 180
    assert summary["protection_directions"] == 240
    assert summary["anchor_queries"] == 300
    selected_recipe, selected, rows = select_recipe(
        evidence, min_anchor_queries=128, min_formulas=64,
        min_specificity_ratio=1.2, recipes=[recipe],
    )
    assert selected_recipe == recipe
    assert selected["admissible"] and selected["specific_candidate_surplus"] > 0
    assert len(rows) == 1
    pool, pool_audit = build_pool(evidence, synthetic_manifest(), correction, protection)
    assert pool_audit["anchor_queries"] == 300
    assert pool_audit["candidate_directions"] == 420
    assert pool_audit["positive_reference_edges"] == 300
    assert pool_audit["negative_reference_edges"] == 420
    assert len(np.unique(pool["anchor_idx"])) == 300

    # A harmful candidate promoted by correct chemistry is not mislabeled as
    # protection: protection requires correct-specific rejection.
    promoted = synthetic_evidence()
    promoted["arm_metric"][0, 180:, :, 0] = 0.50
    _, promoted_protection, _ = directional_masks(promoted, recipe, 0)
    assert not promoted_protection[180:].any()

    # Candidate semantics are exchangeable: making a null arm carry the same
    # signal causes recipe selection to fail rather than invent specificity.
    nonspecific = synthetic_evidence()
    nonspecific["arm_metric"][1] = nonspecific["arm_metric"][0]
    nonspecific["arm_metric"][2] = nonspecific["arm_metric"][0]
    try:
        select_recipe(
            nonspecific, min_anchor_queries=128, min_formulas=64,
            min_specificity_ratio=1.2, recipes=[recipe],
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("non-specific chemistry unexpectedly passed")
    print("PASS: ChemAware high-coverage triplet mining contracts")


if __name__ == "__main__":
    main()
