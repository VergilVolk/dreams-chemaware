"""Contract tests for the GLM composition axis-curriculum builder.

Layer 1 exercises the pure selection logic in GLM_compose_axis_core (no torch,
no data, no project artifacts).  Layer 2 checks static contracts on the heavy
builder source by reading it, so this test runs on a login node without the
training environment.
"""
from __future__ import annotations

import ast
from pathlib import Path

import GLM_compose_axis_core as core

BUILDER = Path(__file__).with_name("GLM_build_compose_axis_curriculum.py")


def entry(axis: str, rank: float) -> dict:
    return {"axis": axis, "candidates": [{"rank_key": (rank, rank, "a")}]}


def test_axis_classification_is_deterministic_and_tie_safe() -> None:
    assert core.classify_axis(0.30, 0.10) == core.AXIS_PD
    assert core.classify_axis(0.10, 0.30) == core.AXIS_NE
    assert core.classify_axis(0.20, 0.20) == core.AXIS_PD, "ties must go to PD"


def test_axis_interleaving_preserves_both_axes_and_order() -> None:
    assert core.interleave_axes([1, 3, 5], [2, 4]) == [1, 2, 3, 4, 5]
    assert core.interleave_axes([], [7, 8]) == [7, 8]
    assert core.interleave_axes([9], []) == [9]
    assert core.interleave_axes([1, 2, 3], [4]) == [1, 4, 2, 3]


def test_ordering_modes_select_every_query_exactly_once() -> None:
    per_query = {
        0: entry(core.AXIS_PD, 0.9),
        1: entry(core.AXIS_NE, 0.8),
        2: entry(core.AXIS_PD, 0.7),
        3: entry(core.AXIS_NE, 0.6),
    }
    balanced = core.order_queries("balanced", per_query)
    maximum = core.order_queries("advantage-max", per_query)
    for ordered in (balanced, maximum):
        assert sorted(ordered) == [0, 1, 2, 3], "no query may be dropped or duplicated"
    assert balanced == [0, 1, 2, 3]
    assert maximum == [0, 1, 2, 3]
    skewed = {
        0: entry(core.AXIS_PD, 0.9),
        1: entry(core.AXIS_PD, 0.8),
        2: entry(core.AXIS_NE, 0.1),
    }
    assert core.order_queries("balanced", skewed) == [0, 2, 1]
    assert core.order_queries("advantage-max", skewed) == [0, 1, 2]


def test_unknown_mode_is_rejected() -> None:
    try:
        core.order_queries("axis-magic", {})
    except ValueError:
        return
    raise AssertionError("unknown mode must be rejected")


def test_builder_reuses_validated_helpers_read_only() -> None:
    tree = ast.parse(BUILDER.read_text(encoding="utf-8"))
    imported: set[str] = set()
    defined: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            for alias in node.names:
                imported.add(alias.name)
        if isinstance(node, ast.FunctionDef):
            defined.add(node.name)
    assert "build_noise_dreams_native_residual_stage2" in imported
    for helper in ("graph_boundary", "make_pool",
                   "assert_exact_checkpoint_reconstruction", "sha256_file"):
        assert helper in imported, f"{helper} must be reused, not reimplemented"
        assert helper not in defined, f"{helper} must not be redefined locally"
    assert "GLM_compose_axis_core" in imported
    for helper in ("classify_axis", "order_queries"):
        assert helper in imported


def test_builder_pins_the_warm_start_and_keeps_one_relation_per_query() -> None:
    text = BUILDER.read_text(encoding="utf-8")
    assert "--expected-warm-start-sha256" in text
    assert "warm-start checkpoint does not match the pinned hash" in text
    assert "composition did not produce one unique relation per query" in text
    assert "one_residual_relation_per_query" in text
    assert "each_query_contributes_at_most_action_plus_clean" in text
    assert "compose_axis" in text
    assert "GLM_COMPOSE_AXIS_CURRICULUM_BUILD_COMPLETE" in text
    assert "outer_performance_claimed" in text


def test_builder_records_both_axis_provenance_blocks() -> None:
    text = BUILDER.read_text(encoding="utf-8")
    for key in ("axis_definitions", "axis_selected_queries", "axis_qualified_queries",
                "axis_margin_summary", "population_medians",
                "axis_floor_declared_before_training",
                "both_axes_present_when_balanced"):
        assert key in text, key


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("GLM composition builder contracts passed")
