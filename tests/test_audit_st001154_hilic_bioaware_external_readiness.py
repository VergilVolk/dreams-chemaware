import pandas as pd

from tasks.audit_st001154_hilic_bioaware_external_readiness import (
    reachable_to_other_seed,
    rhea_graph,
    undirected_edge_graph,
)


def test_graphs_and_leave_truth_out_reachability():
    edges = pd.DataFrame([{"ik14_a": "A", "ik14_b": "B"}, {"ik14_a": "B", "ik14_b": "C"}])
    graph = undirected_edge_graph(edges)
    assert not reachable_to_other_seed("A", {"A", "C"}, graph, 1)
    assert reachable_to_other_seed("A", {"A", "C"}, graph, 2)
    assert not reachable_to_other_seed("A", {"A"}, graph, 3)


def test_rhea_excludes_currency_and_large_reactions():
    frame = pd.DataFrame(
        [
            {"reaction_id": 1, "compound_id": "A", "is_currency": False},
            {"reaction_id": 1, "compound_id": "B", "is_currency": False},
            {"reaction_id": 1, "compound_id": "W", "is_currency": True},
        ]
    )
    graph = rhea_graph(frame)
    assert graph == {"A": {"B"}, "B": {"A"}}
