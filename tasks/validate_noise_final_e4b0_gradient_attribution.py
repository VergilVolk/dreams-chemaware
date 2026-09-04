"""Fail-closed validation of a completed E4-B0 gradient audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from noise_final_core import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/validation/g8r_noise_final_e4b0_gradient_attribution",
    )
    return parser.parse_args()


def validate_ci(value: object, label: str) -> None:
    if not isinstance(value, dict) or set(value) != {"mean", "ci_low", "ci_high"}:
        raise RuntimeError(f"invalid CI object: {label}")
    numbers = [float(value[key]) for key in ("ci_low", "mean", "ci_high")]
    if not np.all(np.isfinite(numbers)) or not numbers[0] <= numbers[1] <= numbers[2]:
        raise RuntimeError(f"non-finite or non-monotone CI: {label}")


def main() -> None:
    args = arguments()
    required = {
        "report": args.output_dir / "report.json",
        "panel": args.output_dir / "panel.csv.gz",
        "per_formula": args.output_dir / "per_formula_gradient.csv.gz",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"E4-B0 output is incomplete: {missing}")
    report = json.loads(required["report"].read_text(encoding="utf-8"))
    panel = pd.read_csv(required["panel"], low_memory=False)
    per_formula = pd.read_csv(required["per_formula"], low_memory=False)
    if report.get("status") != "noise_final_e4b0_gradient_attribution_complete":
        raise RuntimeError("wrong E4-B0 report status")
    if not report.get("formal"):
        raise RuntimeError("E4-B0 report is not formal")
    contracts = report.get("contracts", {})
    if (
        contracts.get("optimizer_steps") != 0
        or contracts.get("weights_changed") is not False
        or contracts.get("outcomes_used_for_panel_selection") is not False
        or contracts.get("P2b") != "forbidden"
        or contracts.get("P3_consumed") is not False
    ):
        raise RuntimeError("E4-B0 no-update or leakage contract failed")
    forbidden = {"corrected", "introduced", "target_rank", "target_margin", "random_margin"}
    if forbidden.intersection(panel.columns):
        raise RuntimeError("outcome columns are present in the frozen E4-B0 panel")
    if (
        len(panel) != 128
        or panel["query_formula"].nunique() != 32
        or panel.groupby("query_formula")["query_index"].nunique().ne(4).any()
        or panel.groupby(["query_formula", "selector"]).size().ne(2).any()
        or len(panel[["selector", "attenuation", "step"]].drop_duplicates()) != 9
    ):
        raise RuntimeError("E4-B0 panel structure is not the frozen 32x4 design")
    if (
        len(per_formula) != 64
        or set(per_formula["checkpoint"].astype(str))
        != {"official_initialization", "clean_continuation"}
        or per_formula.groupby("checkpoint")["formula"].nunique().ne(32).any()
        or per_formula.duplicated(["checkpoint", "formula"]).any()
        or per_formula["actions"].astype(int).ne(4).any()
    ):
        raise RuntimeError("E4-B0 per-formula gradient table is incomplete")
    numeric = per_formula.select_dtypes(include=[np.number])
    if numeric.empty or not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("E4-B0 gradient table contains non-finite values")
    checkpoints = report.get("checkpoints", {})
    if set(checkpoints) != {"official_initialization", "clean_continuation"}:
        raise RuntimeError("E4-B0 checkpoint comparison is incomplete")
    ci_names = (
        "margin_advantage_selected_random_formula_ci",
        "margin_advantage_two_control_formula_ci",
        "current_specific_norm_ratio_formula_ci",
        "current_target_branch_norm_ratio_formula_ci",
        "current_target_branch_clean_alignment_formula_ci",
        "paired_advantage_norm_ratio_formula_ci",
        "current_specific_clean_alignment_formula_ci",
        "paired_advantage_clean_alignment_formula_ci",
        "paired_advantage_gradient_consensus_formula_ci",
    )
    for checkpoint, values in checkpoints.items():
        if int(values.get("formulas", -1)) != 32 or int(values.get("actions", -1)) != 128:
            raise RuntimeError(f"incomplete checkpoint audit: {checkpoint}")
        for name in ci_names:
            validate_ci(values.get(name), f"{checkpoint}.{name}")
    artifacts = report.get("artifacts", {})
    if (
        artifacts.get("panel_sha256") != sha256_file(required["panel"])
        or artifacts.get("per_formula_gradient_sha256")
        != sha256_file(required["per_formula"])
    ):
        raise RuntimeError("E4-B0 artifact hash mismatch")
    gates = report.get("gates", {})
    if not gates or not all(isinstance(value, bool) for value in gates.values()):
        raise RuntimeError("E4-B0 gates are missing or non-boolean")
    expected_pass = bool(all(gates.values()))
    if report.get("pass_to_paired_advantage_pilot") is not expected_pass:
        raise RuntimeError("E4-B0 pilot decision disagrees with its frozen gates")
    print(
        "[validate_noise_final_e4b0_gradient_attribution] PASS "
        f"pilot={expected_pass} formulas=32 actions=128"
    )


if __name__ == "__main__":
    main()
