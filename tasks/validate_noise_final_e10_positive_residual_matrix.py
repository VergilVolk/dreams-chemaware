"""Fail-closed validator for the formal E10 positive residual matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cell-set", choices=("core", "expanded"), default="core")
    args = parser.parse_args()
    required = [
        args.output_dir / "report.json",
        args.output_dir / "cell_summary.csv",
        args.output_dir / "oracle_per_query.csv.gz",
        args.output_dir / "matrix.npz",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    report = json.loads(required[0].read_text(encoding="utf-8"))
    expected_status = (
        "noise_final_e10_positive_residual_matrix_complete" if args.cell_set == "core"
        else "noise_final_e10b_positive_action_expansion_complete"
    )
    expected_cells = 7 if args.cell_set == "core" else 19
    if report.get("status") != expected_status or report.get("cell_set") != args.cell_set or not report.get("formal"):
        raise RuntimeError("E10 report status/formal contract failed")
    if report.get("held_queries") != 5923 or report.get("mature_e8_rank_reproduction_mismatches") != 0:
        raise RuntimeError("E10 held-task size or mature-rank reproduction failed")
    if report.get("cells") != expected_cells or report.get("direction_controls") != expected_cells:
        raise RuntimeError("E10 cell/control count mismatch")
    contracts = report.get("contracts", {})
    required_contracts = {
        "real_same_identity_positive_references": True,
        "wrong_candidate_direction_control": True,
        "one_shared_mature_embedding_geometry": True,
        "outcome_used_only_for_union_headroom": True,
        "P2b": "forbidden",
        "P3_consumed": False,
    }
    if any(contracts.get(key) != value for key, value in required_contracts.items()):
        raise RuntimeError("E10 scientific contract failed")
    cells = pd.read_csv(required[1])
    oracle = pd.read_csv(required[2])
    matrix = np.load(required[3], allow_pickle=True)
    if len(cells) != expected_cells or cells["cell_id"].duplicated().any():
        raise RuntimeError("E10 cell summary is malformed")
    if len(oracle) != 5923 or oracle["query_index"].duplicated().any():
        raise RuntimeError("E10 oracle table is malformed")
    expected_matrix_cells = expected_cells * 2
    if (matrix["result_rank"].shape != (5923, expected_matrix_cells)
            or matrix["result_margin"].shape != (5923, expected_matrix_cells)):
        raise RuntimeError("E10 action matrix shape mismatch")
    union = report.get("no_op_aware_union_headroom", {})
    if union.get("incremental_introduced") != 0:
        raise RuntimeError("a no-op-aware oracle may not introduce errors")
    expected_pass = bool(report.get("passing_fixed_cells")) and bool(union.get("reaches_five_total_points"))
    if report.get("pass_to_conditional_noise_training") is not expected_pass:
        raise RuntimeError("E10 pass decision is inconsistent with preregistered gates")
    print(f"[validate_noise_final_e10_positive_residual_matrix] PASS; fixed={len(cells):,}", flush=True)


if __name__ == "__main__":
    main()
