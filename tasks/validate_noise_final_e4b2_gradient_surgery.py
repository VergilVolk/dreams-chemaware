"""Validate the immutable E4-B2 zero-update gradient-surgery result."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

from noise_final_core import sha256_file


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/validation/g8r_noise_final_e4b2_gradient_surgery_expanded",
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    required = {
        "report": args.output_dir / "report.json",
        "panel": args.output_dir / "panel.csv.gz",
        "formula": args.output_dir / "per_formula.csv.gz",
        "summary": args.output_dir / "screen_summary.csv.gz",
        "ranking": args.output_dir / "screen_ranking.csv.gz",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    report = json.loads(required["report"].read_text(encoding="utf-8"))
    if (
        report.get("status") != "noise_final_e4b2_gradient_surgery_complete"
        or not report.get("formal")
        or report.get("pass_to_training") is not False
        or report.get("contracts", {}).get("optimizer_steps") != 0
        or report.get("contracts", {}).get("weights_changed") is not False
        or report.get("contracts", {}).get("P2b") != "forbidden"
        or report.get("contracts", {}).get("P3_consumed") is not False
    ):
        raise RuntimeError("E4-B2 report violates the zero-update contract")
    panel = pd.read_csv(required["panel"], low_memory=False)
    per_formula = pd.read_csv(required["formula"], low_memory=False)
    summary = pd.read_csv(required["summary"], low_memory=False)
    ranking = pd.read_csv(required["ranking"], low_memory=False)
    forbidden = {"corrected", "introduced", "target_rank", "target_margin", "random_margin"}
    if forbidden.intersection(panel.columns):
        raise RuntimeError("action outcomes leaked into E4-B2 panel")
    if panel["cell_id"].nunique() != 9 or set(panel["panel_split"].astype(str)) != {"screen", "confirm"}:
        raise RuntimeError("E4-B2 panel lost its 9-cell/two-split structure")
    screen = set(panel.loc[panel["panel_split"].eq("screen"), "query_formula"].astype(str))
    confirm = set(panel.loc[panel["panel_split"].eq("confirm"), "query_formula"].astype(str))
    if screen.intersection(confirm):
        raise RuntimeError("E4-B2 screen/confirm formula leakage")
    selection_requirements = report.get("selection_requirements")
    if not isinstance(selection_requirements, dict):
        raise RuntimeError("E4-B2 report lacks frozen formula-selection requirements")
    screen_formulas = int(selection_requirements.get("screen_formulas_per_stratum", -1))
    maximum_confirm = int(
        selection_requirements.get("maximum_confirm_formulas_per_stratum", -1)
    )
    minimum_confirm = int(
        selection_requirements.get("minimum_confirm_formulas_per_stratum", -1)
    )
    if screen_formulas != 32 or maximum_confirm != 64 or minimum_confirm != 1:
        raise RuntimeError("E4-B2 report violates the discovery/confirmation formula contract")
    grouped = panel.groupby(["cell_id", "panel_split", "baseline_state"])[
        "query_formula"
    ].nunique()
    if len(grouped) != 36:
        raise RuntimeError("E4-B2 panel lacks a cell/split/state stratum")
    for (cell_id, split, state), count in grouped.items():
        if split == "screen" and int(count) != screen_formulas:
            raise RuntimeError(f"invalid B1 discovery stratum: {cell_id}|{state}")
        if split == "confirm" and not minimum_confirm <= int(count) <= maximum_confirm:
            raise RuntimeError(f"invalid independent confirmation stratum: {cell_id}|{state}")
    required_formula_columns = {
        "panel_split", "cell_id", "baseline_state", "configuration",
        "query_formula", "query_index", "forward_margin", "clean_alignment",
        "gradient_consensus", "action_descent_retention", "action_fidelity",
        "norm_ratio",
    }
    if required_formula_columns - set(per_formula.columns):
        raise RuntimeError("E4-B2 per-formula artifact is incomplete")
    if (
        per_formula["configuration"].nunique() != 18
        or per_formula.duplicated([
            "panel_split", "cell_id", "baseline_state", "configuration", "query_formula",
        ]).any()
        or not np.isfinite(
            per_formula.select_dtypes(include=[np.number]).to_numpy(np.float64)
        ).all()
    ):
        raise RuntimeError("E4-B2 per-formula matrix is invalid")
    if len(summary) != 2 * 9 * 2 * 18:
        raise RuntimeError("E4-B2 summary did not report the complete fixed matrix")
    if len(ranking) != 9 * 18 or ranking["screen_gate_pass"].dtype != bool:
        raise RuntimeError("E4-B2 screen ranking is incomplete")
    selected = report.get("screen", {}).get("selected_candidates", [])
    confirmation = report.get("confirmation", {})
    results = confirmation.get("results", [])
    if len(selected) > 3 or len(results) != len(selected):
        raise RuntimeError("E4-B2 screen/confirm selection count mismatch")
    selected_keys = {(str(row["cell_id"]), str(row["configuration"])) for row in selected}
    result_keys = {(str(row["cell_id"]), str(row["configuration"])) for row in results}
    if selected_keys != result_keys:
        raise RuntimeError("confirm results do not match screen-locked candidates")
    confirmed = sorted(
        f"{row['cell_id']}|{row['configuration']}"
        for row in results if row.get("confirm_pass")
    )
    if confirmed != sorted(confirmation.get("confirmed_configurations", [])):
        raise RuntimeError("confirmed configuration ledger mismatch")
    artifacts = report.get("artifacts", {})
    observed = {
        "panel_sha256": sha256_file(required["panel"]),
        "per_formula_sha256": sha256_file(required["formula"]),
        "screen_summary_sha256": sha256_file(required["summary"]),
        "screen_ranking_sha256": sha256_file(required["ranking"]),
    }
    if artifacts != observed or not all(report.get("gates", {}).values()):
        raise RuntimeError("E4-B2 artifact hashes or structural gates failed")
    print(
        "[validate_noise_final_e4b2_gradient_surgery] PASS "
        f"actions={len(panel)} configs=18 selected={len(selected)} confirmed={len(confirmed)}"
    )


if __name__ == "__main__":
    main()
