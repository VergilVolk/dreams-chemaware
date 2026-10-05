"""CPU contracts for topology-persistent ChemAware candidate regions."""
from __future__ import annotations

import numpy as np

from audit_chemaware_persistent_candidate_policy import (
    action_topology,
    connected_candidate_regions,
    region_table,
)


def main() -> None:
    actions = [(0.0, 0.1), (0.0, 0.2), (0.1, 0.0), (0.1, 0.1), (0.1, 0.2)]
    neighbors = action_topology(actions)
    proposed = np.asarray([1, 1, 2, 1, 1], dtype=np.int16)
    regions = connected_candidate_regions(proposed, np.ones(5, dtype=bool), neighbors)
    assert [set(map(int, region)) for region in regions] == [{0, 1, 3, 4}, {2}]

    # Removing both grid bridges must split candidate 1 into two regions.
    changed = np.asarray([True, False, True, False, True])
    regions = connected_candidate_regions(proposed, changed, neighbors)
    assert [set(map(int, region)) for region in regions] == [{0}, {2}, {4}]

    feature = np.zeros((1, 5, 23), dtype=np.float32)
    feature[0, :, 5] = 1.0
    feature[0, :, 4] = np.asarray([0.1, 0.2, 0.3, 0.8, 0.4])
    table = {
        "feature": feature,
        "rank": np.asarray([[1, 1, 2, 1, 1]], dtype=np.int16),
        "proposed": proposed[None, :],
        "baseline_rank": np.asarray([2], dtype=np.int16),
    }
    collapsed = region_table(table, actions, global_action=4)
    assert int(collapsed["valid"].sum()) == 2
    # The broad candidate-1 region uses action 3 because it has max margin.
    assert int(collapsed["representative"][0, 0]) == 3
    assert int(collapsed["region_size"][0, 0]) == 4
    assert bool(collapsed["benefit"][0, 0])
    assert not bool(collapsed["benefit"][0, 1])
    # The candidate-1 region contains the chosen global action (index 4).
    assert float(collapsed["feature"][0, 0, -2]) == 1.0
    print("PASS: 4 persistent-candidate ChemAware topology contracts")


if __name__ == "__main__":
    main()
