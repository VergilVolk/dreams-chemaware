"""Contracts for reference-aligned Noise native triplets."""
from __future__ import annotations

import numpy as np
import pandas as pd

from build_noise_reference_aligned_native_triplets import (
    ACTION_EXACT,
    _split_queries,
    build_pools,
)
from test_noise_dreams_native import fixture


def active_fixture():
    graph, cache, actions = fixture()
    embedding = np.zeros_like(cache["embeddings"])
    for query_row in graph["query_row"]:
        row = int(query_row)
        embedding[row, :2] = [1.0, 0.0]
        embedding[row + 1, :2] = [0.8, 0.6]
        embedding[row + 2, :2] = [0.98, np.sqrt(1.0 - 0.98 ** 2)]
        embedding[row + 3, :2] = [0.90, np.sqrt(1.0 - 0.90 ** 2)]
        embedding[row + 4, :2] = [0.70, np.sqrt(1.0 - 0.70 ** 2)]
    cache = {**cache, "embeddings": embedding.astype(np.float32)}
    duplicate = actions.iloc[[0]].copy()
    duplicate["action_id"] = "duplicate-exact-boundary"
    alternate = actions.iloc[[0]].copy()
    alternate["action_id"] = "same-relation-alternate-boundary"
    alternate["action_hard_negative_row"] = int(alternate.iloc[0]["action_hard_negative_row"]) + 1
    actions = pd.concat([actions, duplicate, alternate], ignore_index=True)
    return graph, cache, actions


def test_formula_diverse_base_and_exact_boundaries_survive() -> None:
    graph, cache, actions = active_fixture()
    train_queries, validation_queries, split = _split_queries(
        graph, actions, outer_fold=0, formula_fold_seed=20260825,
        validation_folds=10, validation_fold=0, validation_seed=20260920,
        base_queries_per_formula=1,
    )
    train, validation, aliases, audit = build_pools(
        graph, cache, actions, train_queries, validation_queries,
        margin=0.1, negative_reference_cap=2, positive_reference_cap=2,
    )
    assert split["formula_coverage_fraction"] == 1.0
    assert split["selected_train_formulas"] > split["action_formulas"]
    assert len(validation["anchor_idx"]) == len(validation_queries)
    assert audit["all_actions_mapped"] is True
    assert audit["one_dynamic_action_pool_per_query"] is True
    assert audit["every_action_query_has_exactly_two_native_events"] is True
    assert audit["every_action_query_has_one_base_event"] is True
    assert audit["non_action_queries_receive_no_base_dose"] is True
    assert audit["all_action_aliases_participate_in_dynamic_pool"] is True
    assert audit["action_query_events"] == split["action_queries"]
    assert audit["maximum_events_per_query"] == 2
    assert audit["unique_exact_action_boundaries"] == len(actions) - 1
    assert audit["actions_with_exact_positive_in_dynamic_pool"] == len(actions)
    assert audit["actions_with_exact_negative_in_dynamic_pool"] == len(actions)
    assert aliases.loc[0, "optimization_event_index"] == aliases.loc[
        len(actions) - 2, "optimization_event_index"
    ]
    assert aliases.loc[0, "optimization_event_index"] == aliases.loc[
        len(actions) - 1, "optimization_event_index"
    ]
    selected = int(aliases.loc[0, "optimization_event_index"])
    p0, p1 = map(int, train["positive_ptr"][selected:selected + 2])
    n0, n1 = map(int, train["negative_ptr"][selected:selected + 2])
    selected_positive = set(map(int, train["positive_idx"][p0:p1]))
    selected_negative = set(map(int, train["negative_idx"][n0:n1]))
    selected_aliases = [0, len(actions) - 2, len(actions) - 1]
    assert set(actions.loc[selected_aliases, "action_positive_row"]) <= selected_positive
    assert set(actions.loc[selected_aliases, "action_hard_negative_row"]) <= selected_negative
    tags = np.asarray(train["source_tag"], dtype=np.int64)
    exact_indices = np.flatnonzero((tags & ACTION_EXACT) > 0)
    assert len(exact_indices) == split["action_queries"]
    for event in exact_indices:
        p0, p1 = map(int, train["positive_ptr"][event:event + 2])
        n0, n1 = map(int, train["negative_ptr"][event:event + 2])
        assert p1 - p0 >= 1
        assert n1 - n0 >= 1
        assert float(train["activation_probability"][event]) > 0.0


def main() -> None:
    test_formula_diverse_base_and_exact_boundaries_survive()
    print("[test_noise_reference_aligned_native_triplets] PASS", flush=True)


if __name__ == "__main__":
    main()
