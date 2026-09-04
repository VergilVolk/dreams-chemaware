"""Static and pure-math tests for the CPG0 mature N/P teacher."""
from __future__ import annotations

import ast
from pathlib import Path

from build_noise_final_cpg0_residual_teacher import (
    ACTION_FIELDS, N_GRID, P_INTENSITY_DOSES, P_INTENSITY_FAMILIES,
    P_TRANSFER_DOSES, P_TRANSFER_FAMILIES, advantage_label,
    require_positive_gates, transition,
)


def main() -> None:
    assert len(N_GRID) == 9
    assert len(P_INTENSITY_FAMILIES) * len(P_INTENSITY_DOSES) == 12
    assert len(P_TRANSFER_FAMILIES) * len(P_TRANSFER_DOSES) == 9
    assert len(ACTION_FIELDS) == len(set(ACTION_FIELDS))
    assert transition(2, 1) == "corrected" and transition(1, 2) == "introduced"
    assert advantage_label(0.02, 0.01) == "positive"
    assert advantage_label(-0.02, 0.01) == "harmful"
    assert advantage_label(0.0, 0.01) == "neutral"
    require_positive_gates({"P3_not_consumed": True, "P2b_forbidden": True})
    try:
        require_positive_gates({"P3_consumed": False})
    except RuntimeError:
        pass
    else:
        raise AssertionError("negative-valued gate was incorrectly accepted")
    source_path = Path(__file__).with_name("build_noise_final_cpg0_residual_teacher.py")
    source = source_path.read_text(encoding="utf-8")
    ast.parse(source)
    forbidden = (
        "passing_cells", "best_fixed_cell", "new_beyond_pn", "oracle_recoverable",
        "corrected_queries", "p2b_score",
    )
    lowered = source.lower()
    for token in forbidden:
        assert token not in lowered, f"outcome-selected token leaked into CPG0 builder: {token}"
    for token in (
        "paired_candidate_residual", "candidate_residuals.h5", "formula_fold",
        "matched_control_paths", "hardest_wrong_reference_rows", "P3_not_consumed",
        "failed_complete", "failed_partial", "compute_complete.json",
    ):
        assert token in source
    print("[test_noise_final_cpg0_residual_teacher] PASS")


if __name__ == "__main__":
    main()
