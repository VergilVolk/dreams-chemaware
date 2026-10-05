#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from audit_bioaware_b47_u1c_seed_denominator import (  # noqa: E402
    NESTED_STAGE_ORDER, first_stage_below, fixed_route, parse_bool,
)


def test_independent_feature_stage_is_not_in_nested_order() -> None:
    assert "primary_feature_consensus" not in NESTED_STAGE_ORDER


def test_first_stage_below_uses_only_nested_funnel() -> None:
    counts = [1000, 800, 300, 250, 190, 100]
    stages = {
        name: {"candidate_identities": value}
        for name, value in zip(NESTED_STAGE_ORDER, counts)
    }
    stages["primary_feature_consensus"] = {"candidate_identities": 1}
    assert first_stage_below(stages, 200) == (
        "reaction_graph_eligible_after_spectral_consensus"
    )


def test_bool_parser_is_dependency_free_and_strict() -> None:
    assert parse_bool(True)
    assert parse_bool("YES")
    assert not parse_bool(False)
    assert not parse_bool("0")


def test_route_is_frozen_by_the_first_failed_stage() -> None:
    route = fixed_route("primary_absolute_gate", 500)
    assert route["code"] == "SPECTRAL_CONFIDENCE_BOTTLENECK"
    assert route["pass_denominator_to_exact_event"] is False
    assert fixed_route("none", 500)["pass_denominator_to_exact_event"] is True
    assert fixed_route("none", 199)["code"] == "STOP_GRAPH_UNIVERSE_UNDERPOWERED"


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
    print("[test_bioaware_b47_u1c_seed_denominator] PASS", flush=True)
