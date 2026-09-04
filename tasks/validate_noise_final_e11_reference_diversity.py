"""Fail-closed validator for E11."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = {name: args.output_dir / name for name in (
        "report.json", "cell_summary.csv", "oracle_per_query.csv.gz", "matrix.npz",
    )}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    report = json.loads(paths["report.json"].read_text(encoding="utf-8"))
    if report.get("status") != "noise_final_e11_reference_diversity_complete" or not report.get("formal"):
        raise RuntimeError("E11 report status/formal failed")
    if report.get("held_queries") != 5923 or report.get("cells") != 16 or report.get("direction_controls") != 16:
        raise RuntimeError("E11 task/cell counts failed")
    if report.get("e10b_union_reproduction_mismatches") != 0:
        raise RuntimeError("E11 did not reproduce E10-B union")
    contracts = report.get("contracts", {})
    expected = {"only_reference_selection_changed": True,
                "real_same_identity_positive_references": True,
                "wrong_candidate_direction_control": True,
                "one_shared_mature_embedding_geometry": True,
                "outcome_used_only_for_union_headroom": True,
                "P2b": "forbidden", "P3_consumed": False}
    if any(contracts.get(key) != value for key, value in expected.items()):
        raise RuntimeError("E11 scientific contract failed")
    cells = pd.read_csv(paths["cell_summary.csv"])
    oracle = pd.read_csv(paths["oracle_per_query.csv.gz"])
    matrix = np.load(paths["matrix.npz"], allow_pickle=True)
    if len(cells) != 16 or cells["cell_id"].duplicated().any():
        raise RuntimeError("E11 cell summary malformed")
    if len(oracle) != 5923 or oracle["query_index"].duplicated().any():
        raise RuntimeError("E11 oracle table malformed")
    if matrix["result_rank"].shape != (5923, 32):
        raise RuntimeError("E11 matrix shape failed")
    union = report["union_headroom"]
    if union["newly_lost_vs_e10b"] != 0:
        raise RuntimeError("no-op-aware E11 union lost E10-B-correct queries")
    expected_pass = bool(report["passing_fixed_cells"]) and bool(union["reaches_five_total_points"])
    if report["pass_to_conditional_noise_training"] is not expected_pass:
        raise RuntimeError("E11 pass decision inconsistent")
    print("[validate_noise_final_e11_reference_diversity] PASS", flush=True)


if __name__ == "__main__":
    main()
