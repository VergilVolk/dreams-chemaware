"""Fail-closed validation for a completed E4-B1 stratified gradient audit."""
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
        default=ROOT / "data/validation/g8r_noise_final_e4b1_stratified_gradient",
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    required = {
        "report": args.output_dir / "report.json",
        "panel": args.output_dir / "panel.csv.gz",
        "per_action": args.output_dir / "per_action_gradient.csv.gz",
        "groups": args.output_dir / "group_summary.csv.gz",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"E4-B1 output is incomplete: {missing}")
    report = json.loads(required["report"].read_text(encoding="utf-8"))
    panel = pd.read_csv(required["panel"], low_memory=False)
    per_action = pd.read_csv(required["per_action"], low_memory=False)
    groups = pd.read_csv(required["groups"], low_memory=False)
    if (
        report.get("status") != "noise_final_e4b1_stratified_gradient_complete"
        or not report.get("formal")
        or report.get("pass_to_training") is not False
    ):
        raise RuntimeError("wrong or unsafe E4-B1 report state")
    contracts = report.get("contracts", {})
    if (
        contracts.get("action_outcomes_used_for_panel_selection") is not False
        or contracts.get("optimizer_steps") != 0
        or contracts.get("weights_changed") is not False
        or contracts.get("P2b") != "forbidden"
        or contracts.get("P3_consumed") is not False
    ):
        raise RuntimeError("E4-B1 no-update/leakage contract failed")
    forbidden = {"corrected", "introduced", "target_rank", "target_margin", "random_margin"}
    if forbidden.intersection(panel.columns):
        raise RuntimeError("action outcomes leaked into E4-B1 panel")
    primary_counts = panel.groupby("primary_stratum")["query_formula"].nunique()
    if (
        len(panel) != 576
        or panel["primary_stratum"].nunique() != 18
        or primary_counts.ne(32).any()
        or panel.duplicated(["primary_stratum", "query_formula"]).any()
        or panel.groupby(["cell_id", "baseline_state"]).ngroups != 18
        or set(panel["baseline_state"].astype(str))
        != {"official_error", "official_correct"}
        or panel["baseline_state"].value_counts().ne(288).any()
        or panel["formula_fold"].astype(int).eq(0).any()
    ):
        raise RuntimeError("E4-B1 panel is not the frozen balanced 18x32 design")
    required_action_columns = {
        "query_index", "identity", "query_formula", "cell_id", "primary_stratum",
        "baseline_state", "score_error_family", "near_state",
        "target_minus_selected_random", "paired_advantage_clean_alignment",
        "paired_advantage_clean_alignment_head",
        "paired_advantage_clean_alignment_backbone",
        "primary_gradient_consensus", "primary_gradient_consensus_head",
        "primary_gradient_consensus_backbone", "current_specific_clean_alignment",
    }
    if required_action_columns - set(per_action.columns):
        raise RuntimeError("E4-B1 per-action diagnostics are incomplete")
    if (
        len(per_action) != 576
        or per_action.duplicated(["primary_stratum", "query_formula"]).any()
        or set(per_action["primary_stratum"].astype(str))
        != set(panel["primary_stratum"].astype(str))
    ):
        raise RuntimeError("E4-B1 per-action ledger does not match the panel")
    numeric = per_action.select_dtypes(include=[np.number])
    if numeric.empty or not np.isfinite(numeric.to_numpy(np.float64)).all():
        raise RuntimeError("E4-B1 per-action diagnostics contain non-finite values")
    primary = groups.loc[groups["group_type"].eq("primary_cell_state")].copy()
    required_summary_columns = {
        "target_minus_selected_random_adjusted_low",
        "primary_gradient_consensus_adjusted_low",
        "paired_advantage_clean_alignment_adjusted_low",
        "paired_advantage_clean_alignment_head_ci_low",
        "paired_advantage_clean_alignment_backbone_ci_low",
        "primary_gradient_consensus_head_ci_low",
        "primary_gradient_consensus_backbone_ci_low",
        "eligible_error_cell",
    }
    if required_summary_columns - set(primary.columns):
        raise RuntimeError("E4-B1 primary summary lacks adjusted/layer diagnostics")
    if (
        len(primary) != 18
        or primary["group_id"].duplicated().any()
        or primary["formulas"].astype(int).ne(32).any()
        or not np.isfinite(
            primary.select_dtypes(include=[np.number]).to_numpy(np.float64)
        ).all()
    ):
        raise RuntimeError("E4-B1 primary group summary is malformed")
    expected_eligible = (
        primary["group_id"].astype(str).str.endswith("|official_error")
        & primary["target_minus_selected_random_adjusted_low"].gt(0)
        & primary["primary_gradient_consensus_adjusted_low"].gt(0)
        & primary["paired_advantage_clean_alignment_adjusted_low"].gt(0)
    )
    observed_eligible = primary["eligible_error_cell"].astype(str).str.lower().isin(
        {"true", "1"}
    )
    if not np.array_equal(expected_eligible.to_numpy(bool), observed_eligible.to_numpy(bool)):
        raise RuntimeError("E4-B1 eligibility is not reproduced by frozen adjusted gates")
    eligible = sorted(primary.loc[expected_eligible, "group_id"].astype(str))
    if (
        eligible != sorted(map(str, report.get("eligible_error_cells", [])))
        or len(eligible) != int(report.get("eligible_error_cell_count", -1))
        or bool(eligible) is not report.get("has_conditionally_coherent_error_cells")
    ):
        raise RuntimeError("E4-B1 report eligibility disagrees with group summary")
    multiplicity = report.get("multiplicity", {})
    if (
        multiplicity.get("primary_groups") != 18
        or multiplicity.get("primary_endpoints") != 3
        or not np.isclose(
            float(multiplicity.get("one_sided_bonferroni_alpha", -1)), 0.05 / 54,
        )
        or multiplicity.get("bootstrap_resamples") != 50000
    ):
        raise RuntimeError("E4-B1 multiplicity contract drifted")
    artifacts = report.get("artifacts", {})
    if (
        artifacts.get("panel_sha256") != sha256_file(required["panel"])
        or artifacts.get("per_action_gradient_sha256")
        != sha256_file(required["per_action"])
        or artifacts.get("group_summary_sha256") != sha256_file(required["groups"])
    ):
        raise RuntimeError("E4-B1 artifact hash mismatch")
    gates = report.get("gates", {})
    if not gates or not all(value is True for value in gates.values()):
        raise RuntimeError(f"E4-B1 structural gates failed: {gates}")
    print(
        "[validate_noise_final_e4b1_stratified_gradient] PASS "
        f"eligible={len(eligible)} actions=576 primary_groups=18"
    )


if __name__ == "__main__":
    main()
