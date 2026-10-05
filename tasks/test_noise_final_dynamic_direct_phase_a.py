"""Static regression tests for dynamic-direct Phase-A training."""
from __future__ import annotations
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    trainer = (ROOT / "tasks/train_noise_final_dynamic_direct_phase_a.py").read_text(encoding="utf-8")
    summary = (ROOT / "tasks/summarize_noise_final_dynamic_direct_phase_a.py").read_text(encoding="utf-8")
    for source in (trainer, summary):
        ast.parse(source)
    required = (
        'ARMS = ("clean_continuation", "matched_random", "static_target", "dynamic_np")',
        "materialize_pair", "full_candidate_molecule_list_training", "initial_checkpoint_sha256",
        "dynamic_selected_no_op_weight", "P2b_used\": False",
        "formula_identity_query_equal_weights", 'schedule["epoch"]',
    )
    missing = [token for token in required if token not in trainer]
    if missing:
        raise RuntimeError(f"Phase-A trainer misses contracts: {missing}")
    forbidden = ("p2b_score", "best_fixed_cell", "passing_cells")
    present = [token for token in forbidden if token in trainer.lower()]
    if present:
        raise RuntimeError(f"Phase-A trainer contains forbidden outcome/expert selectors: {present}")
    if ("holm_adjusted_p" not in summary or "formula_cluster_ci" not in summary
            or "dynamic_beats_matched_random_formula_ci_positive" not in summary
            or "dynamic_not_systematically_clipped" not in summary):
        raise RuntimeError("Phase-A summary lacks paired multiplicity/statistical audit")
    print("[test_noise_final_dynamic_direct_phase_a] PASS")


if __name__ == "__main__":
    main()
