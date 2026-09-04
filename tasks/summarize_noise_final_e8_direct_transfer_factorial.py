"""Paired formula-cluster decision for the preregistered E8 factorial."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ARMS = {
    "baseline_symmetric_shared": (
        "curriculum", "e8_baseline_symmetric_shared",
    ),
    "stopgrad_shared": (
        "curriculum", "e8_stopgrad_shared",
    ),
    "officialaction_shared": (
        "curriculum", "e8_officialaction_shared",
    ),
    "symmetric_officialref": (
        "curriculum", "e8_symmetric_officialref",
    ),
    "officialaction_officialref": (
        "curriculum", "e8_officialaction_officialref",
    ),
    "candidate_terminal": (
        "candidate", "e8_candidate_terminal",
    ),
    "combined_terminal": (
        "combined", "e8_combined_terminal",
    ),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path("data/validation/g8r_noise_final_e8_direct_transfer"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/validation/g8r_noise_final_e8_direct_transfer_summary.json"),
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260830)
    return parser.parse_args()


def arm_path(root: Path, policy: str, suffix: str) -> Path:
    return (
        root
        / f"{policy}_all_views4_blocks1_blr_2e-06_hlr_1e-05_{suffix}"
        / "seed_20260830/fold_0"
    )


def formula_bootstrap(effect: np.ndarray, formulas: np.ndarray,
                      resamples: int, seed: int) -> dict[str, float]:
    unique, inverse = np.unique(formulas.astype(str), return_inverse=True)
    sums = np.bincount(inverse, weights=effect.astype(float))
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = rng.integers(0, len(unique), len(unique))
        draws[index] = sums[sampled].sum() / counts[sampled].sum()
    return {
        "mean": float(np.mean(effect)),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite E8 summary: {args.output}")
    reports: dict[str, dict] = {}
    queries: dict[str, pd.DataFrame] = {}
    for name, (policy, suffix) in ARMS.items():
        directory = arm_path(args.root, policy, suffix)
        report_path = directory / "decision.json"
        query_path = directory / "held_per_query.csv.gz"
        if not report_path.is_file() or not query_path.is_file():
            raise FileNotFoundError(f"incomplete E8 arm {name}: {directory}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "noise_final_e4a_direct_augmentation_complete":
            raise RuntimeError(f"unexpected E8 status for {name}")
        reports[name] = report
        queries[name] = pd.read_csv(query_path).sort_values("query_index").reset_index(drop=True)

    reference = queries["baseline_symmetric_shared"]
    identity_columns = ["query_index", "query_row", "query_ik14", "query_formula", "has_near"]
    for name, frame in queries.items():
        if not reference[identity_columns].equals(frame[identity_columns]):
            raise RuntimeError(f"E8 held-query alignment differs for {name}")
        if not np.array_equal(reference["baseline_rank"], frame["baseline_rank"]):
            raise RuntimeError(f"E8 official rank differs for {name}")

    formulas = reference["query_formula"].astype(str).to_numpy()
    near = reference["has_near"].astype(bool).to_numpy()
    comparisons: dict[str, dict] = {}
    base_rank = reference["final_rank"].to_numpy(np.int16)
    for position, (name, frame) in enumerate(queries.items()):
        if name == "baseline_symmetric_shared":
            continue
        rank = frame["final_rank"].to_numpy(np.int16)
        effect = (rank == 1).astype(np.int8) - (base_rank == 1).astype(np.int8)
        comparisons[f"{name}_vs_baseline"] = {
            "delta_recall1": float(np.mean(effect)),
            "corrected": int(np.sum((base_rank != 1) & (rank == 1))),
            "introduced": int(np.sum((base_rank == 1) & (rank != 1))),
            "near_delta_recall1": float(np.mean(effect[near])),
            "formula_cluster_ci": formula_bootstrap(
                effect, formulas, args.bootstrap, args.seed + position,
            ),
        }

    arms: dict[str, dict] = {}
    eligible: list[str] = []
    for name, report in reports.items():
        held = report["held_clean"]
        item = {
            "delta_recall1_vs_official": float(held["delta_recall1"]),
            "delta_near_recall1_vs_official": float(held["delta_near_recall1"]),
            "delta_mrr_vs_official": float(held["delta_mrr"]),
            "corrected_vs_official": int(held["corrected"]),
            "introduced_vs_official": int(held["introduced"]),
            "preservation_mean": float(held["preservation_mean"]),
            "official_gate_pass": bool(all(report["gates"].values())),
            "configuration": {
                "policy": report["configuration"]["policy"],
                "direct_transfer_mode": report["configuration"].get(
                    "direct_transfer_mode", "symmetric"
                ),
                "rank_reference_mode": report["configuration"].get(
                    "rank_reference_mode", "shared"
                ),
            },
        }
        if name != "baseline_symmetric_shared":
            paired = comparisons[f"{name}_vs_baseline"]
            item["paired_vs_baseline"] = paired
            item["incremental_gate"] = bool(
                item["official_gate_pass"]
                and paired["formula_cluster_ci"]["ci_low"] > 0
                and paired["near_delta_recall1"] >= 0
                and paired["corrected"] > paired["introduced"]
            )
            if item["incremental_gate"]:
                eligible.append(name)
        arms[name] = item

    selected = (
        max(eligible, key=lambda name: arms[name]["delta_recall1_vs_official"])
        if eligible else None
    )
    output = {
        "status": "noise_final_e8_direct_transfer_factorial_summary_complete",
        "formal": True,
        "outer_formula_fold": 0,
        "seed": args.seed,
        "arms": arms,
        "paired_comparisons": comparisons,
        "eligible_arms": eligible,
        "selected_for_multifold": selected,
        "decision": (
            "advance the selected transfer mechanism to multifold"
            if selected else
            "no transfer mechanism significantly exceeds the mature symmetric direct-noise baseline"
        ),
        "claim_limit": (
            "One consumed development formula fold. Every arm saves one shared clean-spectrum "
            "DreaMS encoder; P2b and sealed P3 are not used."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
