"""Fail-closed comparison of P/N/S weight scan against the fixed N-only run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/validation/g8r_noise_final_e4a_direct"))
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("data/validation/g8r_noise_final_pn_weight_scan_summary.json"))
    args = parser.parse_args()
    baseline_path = (
        args.root / "curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_highlr_multifold"
        / f"seed_{args.seed}" / f"fold_{args.fold}" / "decision.json"
    )
    if not baseline_path.is_file():
        raise FileNotFoundError(f"fixed N-only comparator is missing: {baseline_path}")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if not baseline.get("pass_to_multifold"):
        raise RuntimeError("fixed N-only comparator did not pass its registered gates")
    n_held = baseline["held_clean"]

    rows = []
    for weight in (0.125, 0.25, 0.5, 1.0):
        tag = f"curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_pnw_{weight:g}_pv2_pn_scan"
        path = args.root / tag / f"seed_{args.seed}" / f"fold_{args.fold}" / "decision.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        report = json.loads(path.read_text(encoding="utf-8"))
        held = report["held_clean"]
        cross = held["held_cross_condition_positive"]
        cross_query = held["held_positive_clean"]
        eligible = bool(
            report.get("pass_to_multifold")
            and held["delta_recall1"] >= n_held["delta_recall1"]
            and held["risk_net"] >= n_held["risk_net"]
            and cross["delta_cosine"] > 0
            and cross_query["delta_accuracy"] >= 0
        )
        rows.append({
            "positive_stream_weight": weight,
            "delta_recall1": held["delta_recall1"],
            "incremental_delta_vs_n_only": held["delta_recall1"] - n_held["delta_recall1"],
            "delta_near_recall1": held["delta_near_recall1"],
            "corrected": held["corrected"],
            "introduced": held["introduced"],
            "risk_net": held["risk_net"],
            "preservation_mean": held["preservation_mean"],
            "cross_condition_pair_delta_cosine": cross["delta_cosine"],
            "cross_condition_query_delta_accuracy": cross_query["delta_accuracy"],
            "all_gates_pass": report.get("pass_to_multifold", False),
            "eligible_vs_fixed_n_only": eligible,
            "path": str(path),
        })
    eligible = [row for row in rows if row["eligible_vs_fixed_n_only"]]
    selected = max(
        eligible,
        key=lambda row: (row["risk_net"], row["delta_recall1"], row["cross_condition_pair_delta_cosine"]),
        default=None,
    )
    output = {
        "status": "noise_final_pn_weight_scan_complete",
        "fixed_n_only": {
            "path": str(baseline_path),
            "delta_recall1": n_held["delta_recall1"],
            "delta_near_recall1": n_held["delta_near_recall1"],
            "corrected": n_held["corrected"],
            "introduced": n_held["introduced"],
            "risk_net": n_held["risk_net"],
        },
        "pn_scan": rows,
        "selection_rule": (
            "all registered gates; overall gain and risk-net no worse than the exact N-only comparator; "
            "cross-condition pair cosine improves; cross-condition query Recall@1 does not degrade; "
            "then maximize risk-net, overall gain, pair-cosine gain"
        ),
        "selected": selected,
        "pass_to_multifold": selected is not None,
        "claim_limit": "single held-formula development fold; selected recipe requires 5-fold x 3-seed confirmation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
