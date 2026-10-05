"""Small contract tests for the corrected positive-action headroom audit."""
from __future__ import annotations

import numpy as np

from audit_noise_corrected_positive_action_headroom import (
    recipe_id, registered_recipes, select_queries, select_reference_rows,
)


def test_registered_recipe_ids_are_unique_and_cover_late_families() -> None:
    recipes = registered_recipes()
    ids = [recipe_id(*item) for item in recipes]
    assert len(ids) == len(set(ids))
    families = {item[1] for item in recipes}
    assert {
        "consensus_projection", "prevalence_attenuation", "recurrent_union_mix",
        "balanced_peak_exchange",
    } <= families
    assert {
        item[2] for item in recipes
        if item[0] == "top3" and item[1] == "prevalence_attenuation"
    } == {0.25, 0.50, 0.75, 1.00}
    assert {"transport_then_union", "consensus_then_union"} <= families
    assert any(item[4] == 0.50 and item[5] == 10 for item in recipes)
    shortlist = registered_recipes("mature_shortlist")
    assert len(shortlist) == 10
    assert set(shortlist) <= set(recipes)
    fixed = registered_recipes("fixed_p1")
    assert fixed == [("top3", "transport_then_union", 1.0, 0.5, 0.67, 5, False)]


def test_reference_selection_is_deterministic() -> None:
    rows = np.asarray([10, 11, 12, 13], dtype=np.int64)
    scores = np.asarray([0.8, 0.2, 0.6, 0.4], dtype=np.float32)
    vectors = np.eye(4, dtype=np.float32)
    assert select_reference_rows(rows, scores, vectors, "top3").tolist() == [10, 12, 13]
    assert select_reference_rows(rows, scores, vectors, "farthest3").tolist() == [11, 13, 12]
    first = select_reference_rows(rows, scores, vectors, "maxmin6")
    second = select_reference_rows(rows, scores, vectors, "maxmin6")
    assert first.tolist() == second.tolist()
    assert set(first.tolist()) == set(rows.tolist())


def test_query_scope_excludes_replay_boundary_flips() -> None:
    class Graph:
        n_queries = 5

    frozen = np.asarray([2, 2, 1, 1, 3], dtype=np.int16)
    replayed = np.asarray([2, 1, 1, 2, 3], dtype=np.int16)
    assert select_queries(Graph(), "official_errors", 0, 1, frozen, replayed).tolist() == [0, 4]
    assert select_queries(Graph(), "official_correct", 0, 1, frozen, replayed).tolist() == [2]


if __name__ == "__main__":
    test_registered_recipe_ids_are_unique_and_cover_late_families()
    test_reference_selection_is_deterministic()
    test_query_scope_excludes_replay_boundary_flips()
    print("PASS=3")
