from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluate_calms_closed_set_selective_ranking import (
    assert_same_axis,
    fixed_coverage_points,
    formula_cluster_bootstrap,
    global_metrics,
    load_query_table,
    paired_functional_deltas,
)


def fixture_table(gaps: tuple[float, ...], ranks: tuple[int, ...]) -> pd.DataFrame:
    n = len(gaps)
    return pd.DataFrame({
        "query_index": np.arange(n),
        "query_row": np.arange(100, 100 + n),
        "query_formula": ["A", "A", "B", "C", "C", "D"][:n],
        "rank": ranks,
        "top1_top2_gap": gaps,
    })


def test_selective_prefix_and_metrics_are_closed_set() -> None:
    table = fixture_table((6, 5, 4, 3, 2, 1), (1, 1, 1, 2, 1, 2))
    points = fixed_coverage_points(
        table, "calms_v132", "identity_disjoint", (0.5, 1.0)
    )
    assert points[0]["retained_queries"] == 3
    assert points[0]["selective_recall_at_1"] == 1.0
    assert points[1]["selective_recall_at_1"] == pytest.approx(4 / 6)
    metrics = global_metrics(table)
    assert metrics["full_coverage_recall_at_1"] == pytest.approx(4 / 6)
    assert metrics["aurc"] >= metrics["optimal_aurc_at_observed_error_rate"]
    assert metrics["excess_aurc"] >= -1e-15


def test_formula_bootstrap_pairs_method_functionals() -> None:
    calms = fixture_table((6, 5, 4, 3, 2, 1), (1, 1, 1, 2, 1, 2))
    wse = fixture_table((6, 5, 4, 3, 2, 1), (1, 2, 1, 2, 1, 2))
    assert_same_axis(calms, wse, "identity_disjoint")
    tables = {"calms_v132": calms, "weighted_spectral_entropy": wse}
    points = []
    for method, table in tables.items():
        points.extend(
            fixed_coverage_points(table, method, "identity_disjoint", (0.5, 1.0))
        )
    bootstrap, clusters = formula_cluster_bootstrap(
        tables, (0.5, 1.0), resamples=100, seed=7
    )
    assert clusters == 4
    delta = paired_functional_deltas(
        "identity_disjoint", points, bootstrap, (0.5, 1.0)
    )
    assert len(delta) == 2
    assert delta[-1]["candidate_minus_comparator"] == pytest.approx(1 / 6)


def test_loader_rejects_duplicate_axis(tmp_path) -> None:
    table = fixture_table((3, 2, 1), (1, 2, 1))
    table.loc[2, "query_index"] = 1
    path = tmp_path / "bad.csv"
    table.to_csv(path, index=False)
    with pytest.raises(RuntimeError, match="duplicate query_index"):
        load_query_table(path, "calms_v132", "identity_disjoint")


def test_axis_mismatch_fails_closed() -> None:
    left = fixture_table((3, 2, 1), (1, 2, 1))
    right = left.copy()
    right.loc[1, "query_formula"] = "X"
    with pytest.raises(RuntimeError, match="query_formula axis mismatch"):
        assert_same_axis(left, right, "formula_disjoint")
