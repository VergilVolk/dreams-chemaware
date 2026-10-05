"""Regression tests for complete routed-union residual selection."""
from __future__ import annotations

import pandas as pd

from select_noise_corrected_action_residual_v4 import residual_union_tables


def test_complete_union_and_selected_union_are_distinct() -> None:
    rows = []
    for query, rank in ((1, 2), (2, 3), (3, 2), (4, 1)):
        for action, route, action_rank, selected, source in (
            ("a", "corrective", 1 if query in {1, 2} else 2, query == 1, "N_mature"),
            ("b", "uncertain", 2, False, "E10B"),
        ):
            rows.append({
                "query_index": query,
                "query_row": query + 10,
                "query_ik14": f"IK{query}",
                "query_formula": f"F{query}",
                "formula_fold": 1,
                "clean_rank": rank,
                "clean_margin": -0.1 if rank > 1 else 0.1,
                "source": source,
                "route": route,
                "action_rank": action_rank,
                "selected_corrective": selected,
            })
    residual, report = residual_union_tables(pd.DataFrame(rows))
    assert set(residual["query_index"]) == {3}
    assert report["initial_E8_error_queries_in_ledger"] == 3
    assert report["complete_union_strict_top1_covered_queries"] == 2
    assert report["selected_union_strict_top1_covered_queries"] == 1
    assert report["complete_union_residual_queries"] == 1
    assert report["source_coverage"]["N_mature"] == {
        "strict_top1_queries": 2,
        "unique_strict_top1_queries": 2,
    }


def test_query_metadata_must_be_invariant() -> None:
    frame = pd.DataFrame({
        "query_index": [1, 1], "query_row": [2, 3],
        "query_ik14": ["IK", "IK"], "query_formula": ["F", "F"],
        "formula_fold": [1, 1], "clean_rank": [2, 2],
        "clean_margin": [-0.1, -0.1], "source": ["N_mature", "E10B"],
        "route": ["uncertain", "uncertain"], "action_rank": [2, 2],
        "selected_corrective": [False, False],
    })
    try:
        residual_union_tables(frame)
    except RuntimeError as error:
        assert "query_row" in str(error)
    else:
        raise AssertionError("non-invariant metadata must fail")


def test_complete_error_table_keeps_queries_absent_from_route_frontiers() -> None:
    ledger = pd.DataFrame({
        "query_index": [1], "query_row": [11], "query_ik14": ["IK1"],
        "query_formula": ["F1"], "formula_fold": [1], "clean_rank": [2],
        "clean_margin": [-0.1], "source": ["N_mature"],
        "route": ["corrective"], "action_rank": [1],
        "selected_corrective": [True],
    })
    all_errors = pd.DataFrame({
        "query_index": [1, 2], "query_row": [11, 12],
        "query_ik14": ["IK1", "IK2"], "query_formula": ["F1", "F2"],
        "formula_fold": [1, 2], "clean_rank": [2, 3],
        "clean_margin": [-0.1, float("nan")],
    })
    residual, report = residual_union_tables(ledger, all_errors)
    assert list(residual["query_index"]) == [2]
    assert report["initial_E8_error_queries_in_ledger"] == 2
    assert report["complete_union_residual_queries"] == 1


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(
        f"[test_select_noise_corrected_action_residual_v4] PASS tests={len(tests)}",
        flush=True,
    )
