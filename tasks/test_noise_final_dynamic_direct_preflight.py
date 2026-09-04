"""Static regressions for the model-free dynamic-direct preflight."""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    path = ROOT / "tasks/audit_noise_final_dynamic_direct_preflight.py"
    source = path.read_text(encoding="utf-8")
    ast.parse(source)
    required = (
        "dynamic-direct preflight missing exact files",
        "validate_n_cells",
        "outer_held_outcome_used_for_training",
        "full_candidate_graph_preserved",
        "historical_outcome_columns_quarantined",
        "initial checkpoint and requested outer fold do not match",
        "exact clean geometry used to define L0/L1 action labels",
        "clean_duplicate",
        '"P2b": "forbidden"',
        '"P3_consumed": False',
    )
    missing = [token for token in required if token not in source]
    if missing:
        raise RuntimeError(f"preflight implementation contract drifted: {missing}")
    forbidden = ("load_base_model(", ".cuda(", "optimizer.step(", "P2b_score")
    present = [token for token in forbidden if token in source]
    if present:
        raise RuntimeError(f"model-free preflight contains forbidden operations: {present}")
    print("[test_noise_final_dynamic_direct_preflight] PASS")


if __name__ == "__main__":
    main()
