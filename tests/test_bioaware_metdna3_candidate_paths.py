from __future__ import annotations

from tasks.build_bioaware_metdna3_candidate_path_table import shortest_seed_evidence


def test_candidate_path_requires_observed_intermediates_and_uses_shortest_depth() -> None:
    graph = {
        "C": {"I", "S2"},
        "I": {"C", "S1"},
        "S1": {"I"},
        "S2": {"C"},
    }
    direct = shortest_seed_evidence("C", graph, {"S1", "S2"}, {"C", "I"}, 3)
    assert direct["minimum_depth"] == 1
    assert direct["shortest_seed_count"] == 1
    blocked = shortest_seed_evidence("C", graph, {"S1"}, {"C"}, 3)
    assert not blocked["path_available"]
