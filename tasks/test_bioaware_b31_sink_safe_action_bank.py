#!/usr/bin/env python
"""Dependency-light contract checks for BioAware B31."""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    source = (ROOT / "tasks/build_bioaware_b31_sink_safe_action_bank.py").read_text(encoding="utf-8")
    ast.parse(source)
    for token in (
        "corrective supervision contains only the 54", "all seven original b17",
        "uncertain_sink_conflict", "action_candidate_reference_retained_for_safety",
        "sink_conflicting_b17_corrections_have_zero_corrective_weight",
        "exact_b20_tensors_reused",
    ):
        assert token in source.lower(), token
    print("[test_bioaware_b31_sink_safe_action_bank] PASS", flush=True)


if __name__ == "__main__":
    main()
