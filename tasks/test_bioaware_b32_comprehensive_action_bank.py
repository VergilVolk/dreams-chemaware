#!/usr/bin/env python
"""Dependency-light source checks for the BioAware B32 bank."""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    source = (ROOT / "tasks/build_bioaware_b32_comprehensive_action_bank.py").read_text(
        encoding="utf-8"
    )
    ast.parse(source)
    lowered = source.lower()
    for token in (
        "current_v4_unique_corrective",
        "current_v4_unique_safety",
        "all_b17_and_current_v4_harms_are_safety_constraints",
        "sink_conflicting_corrections_have_zero_weight",
        "query_reference_encoder_must_be_shared",
        "p2b_used\": false",
    ):
        assert token in lowered, token
    print("[test_bioaware_b32_comprehensive_action_bank] PASS", flush=True)


if __name__ == "__main__":
    main()
