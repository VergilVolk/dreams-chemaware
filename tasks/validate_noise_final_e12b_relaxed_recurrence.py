"""Fail-closed validator for E12-B."""
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
    if report.get("status") != "noise_final_e12b_relaxed_recurrence_complete" or not report.get("formal"):
        raise RuntimeError("E12-B status/formal failed")
    if report.get("held_queries") != 5923 or report.get("cells") != 25 or report.get("direction_controls") != 25:
        raise RuntimeError("E12-B task/cell counts failed")
    if report.get("e11_union_reproduction_mismatches") != 0:
        raise RuntimeError("E12-B failed to reproduce E11")
    cells = pd.read_csv(paths["cell_summary.csv"])
    oracle = pd.read_csv(paths["oracle_per_query.csv.gz"])
    matrix = np.load(paths["matrix.npz"], allow_pickle=True)
    if len(cells) != 25 or cells["cell_id"].duplicated().any():
        raise RuntimeError("E12-B cell summary malformed")
    if len(oracle) != 5923 or oracle["query_index"].duplicated().any():
        raise RuntimeError("E12-B oracle table malformed")
    if matrix["result_rank"].shape != (5923, 50):
        raise RuntimeError("E12-B matrix shape failed")
    union = report["union_headroom"]
    if union["newly_lost_vs_e11"] != 0:
        raise RuntimeError("no-op-aware E12-B union lost E11-correct queries")
    contracts = report.get("contracts", {})
    if contracts.get("e12a_authorized") is not True or contracts.get("prevalence_fixed_before_outcomes") != 0.5:
        raise RuntimeError("E12-B preregistration contract failed")
    if contracts.get("P2b") != "forbidden" or contracts.get("P3_consumed") is not False:
        raise RuntimeError("E12-B isolation contract failed")
    expected_pass = bool(report["passing_fixed_cells"]) and bool(union["reaches_five_total_points"])
    if report["pass_to_conditional_noise_training"] is not expected_pass:
        raise RuntimeError("E12-B pass decision inconsistent")
    print("[validate_noise_final_e12b_relaxed_recurrence] PASS", flush=True)


if __name__ == "__main__":
    main()
