"""Select a P/N/S capacity configuration without weakening safety gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


CONFIGS = (
    (0, 1, 2, "2e-06", "1e-05"),
    (1, 1, 4, "2e-06", "1e-05"),
    (2, 2, 0, "1e-06", "1e-05"),
    (3, 2, 2, "1e-06", "1e-05"),
    (4, 2, 4, "1e-06", "1e-05"),
    (5, 3, 2, "5e-07", "1e-05"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/validation/g8r_noise_final_e4a_direct"))
    parser.add_argument("--weight-summary", type=Path, default=Path("data/validation/g8r_noise_final_pn_weight_scan_summary.json"))
    parser.add_argument("--output", type=Path, default=Path("data/validation/g8r_noise_final_pn_capacity_scan_summary.json"))
    args = parser.parse_args()
    weight_body = json.loads(args.weight_summary.read_text(encoding="utf-8"))
    selected_weight = weight_body.get("selected")
    if selected_weight is None:
        raise RuntimeError("P-weight scan did not select a recipe")
    weight = float(selected_weight["positive_stream_weight"])
    comparator_path = Path(selected_weight["path"])
    comparator = json.loads(comparator_path.read_text(encoding="utf-8"))["held_clean"]
    rows = []
    for index, blocks, error_views, blr, hlr in CONFIGS:
        tag = (
            f"curriculum_all_views4_blocks{blocks}_blr_{blr}_hlr_{hlr}"
            f"_pnw_{weight:g}_pv2"
            + (f"_ev{error_views}" if error_views else "")
            + f"_pn_capacity_c{index}"
        )
        path = args.root / tag / "seed_20260828/fold_0/decision.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        report = json.loads(path.read_text(encoding="utf-8"))
        held = report["held_clean"]
        eligible = bool(
            report.get("pass_to_multifold")
            and held["delta_recall1"] > comparator["delta_recall1"]
            and held["risk_net"] >= comparator["risk_net"]
            and held["introduced"] <= max(comparator["introduced"], 8)
        )
        rows.append({
            "configuration": index,
            "unfreeze_blocks": blocks,
            "error_views_per_identity": error_views,
            "backbone_lr": blr,
            "head_lr": hlr,
            "delta_recall1": held["delta_recall1"],
            "incremental_vs_pn_weight_baseline": held["delta_recall1"] - comparator["delta_recall1"],
            "delta_near_recall1": held["delta_near_recall1"],
            "corrected": held["corrected"],
            "introduced": held["introduced"],
            "risk_net": held["risk_net"],
            "preservation_mean": held["preservation_mean"],
            "preservation_p01": held["preservation_p01"],
            "eligible": eligible,
            "path": str(path),
        })
    eligible = [row for row in rows if row["eligible"]]
    selected = max(eligible, key=lambda row: (row["risk_net"], row["delta_recall1"]), default=None)
    body = {
        "status": "noise_final_pn_capacity_scan_complete",
        "positive_stream_weight": weight,
        "comparator": {"path": str(comparator_path), **comparator},
        "configurations": rows,
        "selected": selected,
        "pass_to_multifold": selected is not None,
        "selection_rule": (
            "all P/N/S gates pass; gain strictly exceeds selected one-block P/N comparator; "
            "risk-net no worse; introduced no larger than max(comparator,8); then maximize risk-net and gain"
        ),
        "claim_limit": "single held-formula capacity screen; no five-point model claim.",
    }
    args.output.write_text(json.dumps(body, indent=2), encoding="utf-8")
    print(json.dumps(body, indent=2), flush=True)


if __name__ == "__main__":
    main()
