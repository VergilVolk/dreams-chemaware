"""Fail-closed structural validator for the E4-A causal attribution summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.output_dir / "report.json"
    ledger_path = args.output_dir / "paired_per_query.csv.gz"
    if not report_path.is_file() or not ledger_path.is_file():
        raise FileNotFoundError("causal attribution summary is incomplete")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "noise_final_e4a_causal_attribution_complete" or not report.get("formal"):
        raise RuntimeError("unexpected causal attribution status")
    arms = report.get("arms", {})
    if set(arms) != {"clean_duplicate", "matched_random", "targeted"}:
        raise RuntimeError("causal attribution does not contain exactly three arms")
    contracts = report.get("contracts", {})
    expected = {
        "one_shared_clean_spectrum_encoder_per_arm": True,
        "only_action_view_differs_between_arms": True,
        "matched_controls_frozen_before_training": True,
        "outcomes_used_for_control_assignment": False,
        "full_candidate_graph_used_for_primary_clean_evaluation": True,
        "P2b": "forbidden",
        "P3_consumed": False,
    }
    for key, value in expected.items():
        if contracts.get(key) != value:
            raise RuntimeError(f"causal attribution contract failed: {key}")
    comparisons = report.get("comparisons", {})
    required_comparisons = {
        "targeted_vs_matched_random", "targeted_vs_clean_duplicate",
        "matched_random_vs_clean_duplicate",
    }
    if set(comparisons) != required_comparisons:
        raise RuntimeError("causal attribution paired comparisons are incomplete")
    ledger = pd.read_csv(ledger_path)
    required_columns = {
        "query_index", "query_formula", "has_near", "baseline_rank",
        "clean_duplicate_rank", "matched_random_rank", "targeted_rank",
        "clean_duplicate_top_molecule_local", "matched_random_top_molecule_local",
        "targeted_top_molecule_local", "clean_duplicate_full_margin",
        "matched_random_full_margin", "targeted_full_margin",
    }
    if required_columns - set(ledger.columns):
        raise RuntimeError("causal attribution paired ledger lacks required columns")
    if ledger["query_index"].duplicated().any() or len(ledger) < 4000:
        raise RuntimeError("causal attribution paired ledger has invalid held-query coverage")
    primary = comparisons["targeted_vs_matched_random"]
    print(
        "[validate_noise_final_e4a_causal_attribution] PASS "
        f"dR1={primary['delta_recall1']:+.6f} "
        f"C/I={primary['corrected']}/{primary['introduced']} "
        f"pass_to_learnability={report['pass_to_action_learnability']}"
    )


if __name__ == "__main__":
    main()
