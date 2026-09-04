"""Paired decision for the preregistered E13 relaxed-transfer pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ARMS = {
    "reproduction_control": (None, 0.0, "e13_repro_control"),
    "stopgrad_w050": ("stopgrad", 0.5, "e13_relax_stopgrad_w050"),
    "symmetric_w025": ("symmetric", 0.25, "e13_relax_symmetric_w025"),
    "symmetric_w050": ("symmetric", 0.5, "e13_relax_symmetric_w050"),
    "symmetric_w100": ("symmetric", 1.0, "e13_relax_symmetric_w100"),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path("data/validation/g8r_noise_final_e13_relaxed_transfer"),
    )
    parser.add_argument(
        "--archived-e8", type=Path,
        default=Path(
            "data/validation/g8r_noise_final_e8_direct_transfer/"
            "curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_"
            "e8_baseline_symmetric_shared/seed_20260830/fold_0"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/validation/g8r_noise_final_e13_relaxed_transfer_summary.json"),
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser.parse_args()


def arm_path(root: Path, mode: str | None, weight: float, suffix: str) -> Path:
    tag = "curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05"
    if mode is not None:
        tag += (
            f"_gpn_transfer_gw{weight:g}_gv4_gtm_{mode}"
            f"_grp0.5_gmax10_gscope_all"
        )
    tag += f"_{suffix}"
    return root / tag / "seed_20260830/fold_0"


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
        raise RuntimeError(f"refusing to overwrite E13 summary: {args.output}")
    reports: dict[str, dict] = {}
    tables: dict[str, pd.DataFrame] = {}
    for name, (mode, weight, suffix) in ARMS.items():
        directory = arm_path(args.root, mode, weight, suffix)
        report_path, table_path = directory / "decision.json", directory / "held_per_query.csv.gz"
        if not report_path.is_file() or not table_path.is_file():
            raise FileNotFoundError(f"incomplete E13 arm {name}: {directory}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "noise_final_e4a_direct_augmentation_complete" or not report.get("formal"):
            raise RuntimeError(f"E13 arm is not formal: {name}")
        if report.get("contracts", {}).get("P2b") != "forbidden" or report.get("contracts", {}).get("P3_consumed"):
            raise RuntimeError(f"E13 contract violation: {name}")
        reports[name] = report
        tables[name] = pd.read_csv(table_path).sort_values("query_index", kind="stable").reset_index(drop=True)

    control = tables["reproduction_control"]
    identity_columns = ["query_index", "query_row", "query_ik14", "query_formula", "has_near"]
    for name, frame in tables.items():
        if not control[identity_columns].equals(frame[identity_columns]):
            raise RuntimeError(f"held-query alignment differs for {name}")
        if not np.array_equal(control["baseline_rank"], frame["baseline_rank"]):
            raise RuntimeError(f"official ranks differ for {name}")

    archived_path = args.archived_e8 / "held_per_query.csv.gz"
    if not archived_path.is_file():
        raise FileNotFoundError(f"missing archived E8 per-query table: {archived_path}")
    archived = pd.read_csv(archived_path).sort_values("query_index", kind="stable").reset_index(drop=True)
    if not control[identity_columns].equals(archived[identity_columns]):
        raise RuntimeError("E13 control and archived E8 query ledgers differ")
    reproduction_mismatches = int(np.sum(
        control["final_rank"].to_numpy(np.int16)
        != archived["final_rank"].to_numpy(np.int16)
    ))
    if reproduction_mismatches:
        raise RuntimeError(
            f"same-executable E13 control failed archived E8 replay for {reproduction_mismatches} queries"
        )

    base_rank = control["final_rank"].to_numpy(np.int16)
    formulas = control["query_formula"].astype(str).to_numpy()
    near = control["has_near"].astype(bool).to_numpy()
    arms: dict[str, dict] = {}
    eligible: list[str] = []
    for position, (name, report) in enumerate(reports.items()):
        held = report["held_clean"]
        item = {
            "delta_recall1_vs_official": float(held["delta_recall1"]),
            "delta_near_recall1_vs_official": float(held["delta_near_recall1"]),
            "delta_mrr_vs_official": float(held["delta_mrr"]),
            "corrected_vs_official": int(held["corrected"]),
            "introduced_vs_official": int(held["introduced"]),
            "preservation_mean": float(held["preservation_mean"]),
            "official_gate_pass": bool(all(report["gates"].values())),
        }
        if name != "reproduction_control":
            rank = tables[name]["final_rank"].to_numpy(np.int16)
            effect = (rank == 1).astype(np.int8) - (base_rank == 1).astype(np.int8)
            paired = {
                "delta_recall1": float(np.mean(effect)),
                "corrected": int(np.sum((base_rank != 1) & (rank == 1))),
                "introduced": int(np.sum((base_rank == 1) & (rank != 1))),
                "risk_net_lambda2": int(
                    np.sum((base_rank != 1) & (rank == 1))
                    - 2 * np.sum((base_rank == 1) & (rank != 1))
                ),
                "near_delta_recall1": float(np.mean(effect[near])),
                "formula_cluster_ci": formula_bootstrap(
                    effect, formulas, args.bootstrap, args.seed + position,
                ),
            }
            item["paired_vs_mature_e8_control"] = paired
            item["incremental_gate"] = bool(
                paired["formula_cluster_ci"]["ci_low"] > 0
                and paired["corrected"] > paired["introduced"]
                and paired["risk_net_lambda2"] > 0
                and paired["near_delta_recall1"] >= 0
                and item["preservation_mean"] >= 0.995
            )
            if item["incremental_gate"]:
                eligible.append(name)
        arms[name] = item

    selected = (
        max(
            eligible,
            key=lambda name: (
                arms[name]["paired_vs_mature_e8_control"]["risk_net_lambda2"],
                arms[name]["paired_vs_mature_e8_control"]["delta_recall1"],
            ),
        )
        if eligible else None
    )
    output = {
        "status": "noise_final_e13_relaxed_transfer_summary_complete",
        "formal": True,
        "outer_formula_fold": 0,
        "archived_e8_reproduction_rank_mismatches": reproduction_mismatches,
        "fixed_action_teacher_context": {
            "cell": "top3|standard|max=10|dose=0.50",
            "delta_recall1_vs_mature_e8": 0.01789633631605605,
            "corrected": 137,
            "introduced": 31,
            "not_a_weight_result": True,
        },
        "arms": arms,
        "eligible_arms": eligible,
        "selected_for_multifold": selected,
        "decision": (
            "advance selected relaxed-transfer shared encoder to formula-multifold confirmation"
            if selected else
            "relaxed fixed action did not transfer significantly into the shared embedding"
        ),
        "claim_limit": (
            "One consumed development formula fold. All candidates are clean-spectrum shared "
            "DreaMS encoders; P2b and sealed P3 are not used."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
