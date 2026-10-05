#!/usr/bin/env python
"""Cheap contract tests for the candidate-level BioAware teacher ledger."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    path = ROOT / "tasks/build_bioaware_b5_candidate_teacher.py"
    text = path.read_text(encoding="utf-8")
    ast.parse(text, filename=str(path))
    required = (
        "teacher_model_score", "proposed_candidate_id",
        "proposal_probability", "assert_candidate_replay",
        "corrected replay mismatch", "introduced replay mismatch",
        "outer_fold_never_teacher_scored\": True",
        "candidate_level_teacher_scores_retained\": True",
        "teacher_action_replayed_exactly\": True",
    )
    for token in required:
        if token not in text:
            raise AssertionError(f"B5 teacher builder lacks contract token: {token}")
    forbidden = (
        "training_example_router_only", "safe_corrected_by_outer][",
    )
    for token in forbidden:
        if token in text:
            raise AssertionError(f"B5 teacher builder contains forbidden B4 shortcut: {token}")
    print("[BioAware B5 candidate-teacher source checks] PASS")


if __name__ == "__main__":
    main()
