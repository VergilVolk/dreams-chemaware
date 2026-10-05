"""Fast structural tests for full-graph shared-map reachability."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from audit_noise_corrected_fullgraph_shared_map import nested_delta, representative_positions


def test_representatives_use_official_molecule_max_and_hard_negative_order() -> None:
    graph = SimpleNamespace(
        n_queries=1,
        query_ptr=np.asarray([0, 3]),
        molecule_ptr=np.asarray([0, 2, 4, 5]),
        molecule_label=np.asarray([1, 0, 0]),
        pair_candidate_row=np.asarray([10, 11, 12, 13, 14]),
        features=np.asarray([[.2], [.9], [.7], [.8], [.6]], dtype=np.float32),
        dreams_column=0,
    )
    row_index = {row: index for index, row in enumerate([10, 11, 12, 13, 14])}
    positive, negative = representative_positions(graph, row_index, top_negatives=3)
    assert positive.tolist() == [1]
    assert negative.tolist() == [[3, 4, 4]]


def test_nested_delta_keeps_only_numeric_shared_leaves() -> None:
    candidate = {"a": 0.8, "nested": {"x": 4, "label": "new"}}
    official = {"a": 0.5, "nested": {"x": 1, "label": "old"}}
    delta = nested_delta(candidate, official)
    assert np.isclose(delta["a"], 0.3)
    assert delta["nested"] == {"x": 3.0}


def main() -> None:
    test_representatives_use_official_molecule_max_and_hard_negative_order()
    test_nested_delta_keeps_only_numeric_shared_leaves()
    print("[test_noise_corrected_fullgraph_shared_map] PASS tests=2")


if __name__ == "__main__":
    main()
