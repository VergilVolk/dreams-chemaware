from __future__ import annotations

from tasks.audit_bioaware_metdna3_recursive_headroom import UnionFind, bounded_distances


def test_bounded_recursive_distance_requires_observed_intermediates() -> None:
    graph = {"S": {"A"}, "A": {"S", "B"}, "B": {"A"}}
    assert bounded_distances(graph, {"S"}, {"A", "B"}, 1) == {"S": 0, "A": 1}
    assert bounded_distances(graph, {"S"}, {"A", "B"}, 2)["B"] == 2
    assert "B" not in bounded_distances(graph, {"S"}, {"B"}, 3)


def test_cross_window_union_refuses_transitive_duplicate_window() -> None:
    union = UnionFind(3, ["window_a", "window_b", "window_a"])
    assert union.union(0, 1)
    assert not union.union(1, 2)
    assert union.find(0) == union.find(1)
    assert union.find(2) != union.find(0)
