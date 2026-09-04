"""Aggregate the preregistered 5-formula-fold x 3-seed E4-A confirmation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", type=Path,
        default=(ROOT / "data/validation/g8r_noise_final_e4a_direct/"
                 "curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_highlr_multifold"),
    )
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/validation/g8r_noise_final_e4a_highlr_multifold.json")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    seeds = (20260828, 20260829, 20260830)
    runs = []
    for seed in seeds:
        for fold in range(5):
            path = args.run_root / f"seed_{seed}" / f"fold_{fold}" / "decision.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            decision = json.loads(path.read_text(encoding="utf-8"))
            held = decision["held_clean"]
            runs.append({
                "seed": seed, "fold": fold, "path": str(path),
                "n": held["n_queries"], "errors": held["errors"],
                "delta_recall1": held["delta_recall1"],
                "delta_mrr": held["delta_mrr"],
                "near_n": held["near_n"],
                "delta_near_recall1": held["delta_near_recall1"],
                "corrected": held["corrected"], "introduced": held["introduced"],
                "risk_net": held["risk_net"],
                "preservation_mean": held["preservation_mean"],
                "formula_ci_low": held["formula_cluster_delta_recall1"]["ci_low"],
                "pass": decision["pass_to_multifold"],
            })

    by_seed = []
    for seed in seeds:
        local = [row for row in runs if row["seed"] == seed]
        n = sum(row["n"] for row in local)
        near_n = sum(row["near_n"] for row in local)
        corrected = sum(row["corrected"] for row in local)
        introduced = sum(row["introduced"] for row in local)
        by_seed.append({
            "seed": seed, "n_queries": n,
            "delta_recall1": (corrected - introduced) / n,
            "corrected": corrected, "introduced": introduced,
            "risk_net": corrected - 2 * introduced,
            "delta_mrr": sum(row["delta_mrr"] * row["n"] for row in local) / n,
            "delta_near_recall1": sum(
                row["delta_near_recall1"] * row["near_n"] for row in local
            ) / near_n,
            "minimum_preservation": min(row["preservation_mean"] for row in local),
            "all_fold_gates_pass": all(row["pass"] for row in local),
        })

    report = {
        "status": "noise_final_e4a_highlr_multifold_complete",
        "runs": runs,
        "by_seed": by_seed,
        "gates": {
            "all_15_runs_positive_recall": all(row["delta_recall1"] > 0 for row in runs),
            "all_3_seed_pooled_recall_positive": all(row["delta_recall1"] > 0 for row in by_seed),
            "all_3_seed_risk_net_positive": all(row["risk_net"] > 0 for row in by_seed),
            "all_3_seed_near_nonnegative": all(row["delta_near_recall1"] >= 0 for row in by_seed),
            "all_3_seed_fold_gates_pass": all(row["all_fold_gates_pass"] for row in by_seed),
        },
        "claim_limit": "development confirmation only; sealed P3 remains untouched",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
