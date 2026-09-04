"""Summarize the preregistered E4-A optimizer scan without touching P3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=ROOT / "data/validation/g8r_noise_final_e4a_direct",
    )
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/validation/g8r_noise_final_e4a_optimizer_scan.json",
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    expected = {
        "low_lr": "opt_low_lr_ep4",
        "high_lr": "opt_high_lr_ep4",
        "clip2": "opt_clip2_ep4",
        "clip4": "opt_clip4_ep4",
        "epoch6": "opt_epoch6_ep6",
        "high_lr_clip2": "opt_high_lr_clip2_ep4",
    }
    rows = []
    missing = []
    for name, suffix in expected.items():
        matches = list(args.root.glob(
            f"curriculum_all_views4_blocks1_blr_*_hlr_*_{suffix}/"
            f"seed_{args.seed}/fold_{args.fold}/decision.json"
        ))
        if len(matches) != 1:
            missing.append({"name": name, "matches": [str(path) for path in matches]})
            continue
        decision = json.loads(matches[0].read_text(encoding="utf-8"))
        held = decision["held_clean"]
        history = decision["history"]
        rows.append({
            "name": name,
            "decision": str(matches[0]),
            "delta_recall1": held["delta_recall1"],
            "delta_near_recall1": held["delta_near_recall1"],
            "delta_mrr": held["delta_mrr"],
            "corrected": held["corrected"],
            "introduced": held["introduced"],
            "risk_net": held["risk_net"],
            "preservation_mean": held["preservation_mean"],
            "formula_ci_low": held["formula_cluster_delta_recall1"]["ci_low"],
            "mean_clip_fraction": sum(x["gradient_clip_applied"] for x in history) / len(history),
            "mean_clip_scale": sum(x["gradient_clip_scale"] for x in history) / len(history),
            "pass_to_multifold": decision["pass_to_multifold"],
        })
    if missing:
        raise RuntimeError(f"optimizer scan incomplete or ambiguous: {missing}")

    eligible = [row for row in rows if row["pass_to_multifold"]]
    selected = None
    if eligible:
        # Safety-first selection fixed before reading the results.
        selected = max(
            eligible,
            key=lambda row: (row["risk_net"], row["delta_recall1"], row["preservation_mean"]),
        )["name"]
    report = {
        "status": "noise_final_e4a_optimizer_scan_complete",
        "protocol": "same fold, seed, data and action curriculum; optimizer-only scan",
        "rows": rows,
        "selection_rule": (
            "among runs passing every existing E4-A gate, maximize corrected-2*introduced, "
            "then Recall@1 delta, then preservation"
        ),
        "selected_for_multifold": selected,
        "claim_limit": (
            "fold-0 optimizer development only; the selected configuration requires formula-fold "
            "and seed replication before any general performance claim"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
