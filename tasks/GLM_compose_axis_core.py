"""Lightweight pure logic for the GLM composition axis curriculum.

Kept free of torch/dreams imports so the selection logic is unit-testable
locally, and imported by GLM_build_compose_axis_curriculum.py (read-only use).
"""
from __future__ import annotations

AXIS_PD = "positive_deficit"
AXIS_NE = "negative_excess"


def classify_axis(deficit: float, excess: float) -> str:
    """Assign one query to the axis with the larger deviation from the
    processed-population median geometry.  Deterministic; ties go to PD."""
    return AXIS_PD if deficit >= excess else AXIS_NE


def interleave_axes(pd_queue: list, ne_queue: list) -> list:
    """Alternate PD/NE so both axes are represented as far as availability
    allows; leftover queries keep their within-axis ranking."""
    ordered: list = []
    pd_work = list(pd_queue)
    ne_work = list(ne_queue)
    while pd_work or ne_work:
        if pd_work:
            ordered.append(pd_work.pop(0))
        if ne_work:
            ordered.append(ne_work.pop(0))
    return ordered


def order_queries(mode: str, per_query: dict) -> list:
    """Return the selected-query order for a mode.  Both modes select exactly
    one relation per query; they differ only in which queries are preferred."""
    if mode == "advantage-max":
        return sorted(
            per_query,
            key=lambda q: per_query[q]["candidates"][0]["rank_key"],
            reverse=True,
        )
    if mode != "balanced":
        raise ValueError(f"unknown axis-balance mode: {mode}")
    pd_queue = sorted(
        (q for q, entry in per_query.items() if entry["axis"] == AXIS_PD),
        key=lambda q: per_query[q]["candidates"][0]["rank_key"], reverse=True)
    ne_queue = sorted(
        (q for q, entry in per_query.items() if entry["axis"] == AXIS_NE),
        key=lambda q: per_query[q]["candidates"][0]["rank_key"], reverse=True)
    return interleave_axes(pd_queue, ne_queue)
