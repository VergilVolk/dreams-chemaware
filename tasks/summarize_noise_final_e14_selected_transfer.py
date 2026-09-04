"""Paired held-formula decision for the E14 conditional action teacher pilot."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from noise_final_core import json_dump, sha256_file  # noqa: E402
from train_noise_final_r2_shared_encoder import formula_bootstrap_delta  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--selected-no-margin", type=Path, required=True)
    parser.add_argument("--selected-delta", type=Path, required=True)
    parser.add_argument("--selected-delta-risk", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260906)
    return parser.parse_args()


def load_run(path: Path, outer_fold: int) -> tuple[dict, pd.DataFrame]:
    decision_path = path / "decision.json"
    per_query_path = path / "held_per_query.csv.gz"
    if not decision_path.is_file() or not per_query_path.is_file():
        raise FileNotFoundError(path)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if (
        decision.get("status") != "noise_final_e4a_direct_augmentation_complete"
        or not decision.get("formal")
        or int(decision.get("configuration", {}).get("outer_fold", -1)) != outer_fold
    ):
        raise RuntimeError(f"invalid E14 run: {path}")
    table = pd.read_csv(per_query_path).sort_values("query_index", kind="stable")
    required = {
        "query_index", "query_formula", "has_near", "baseline_rank",
        "initialization_rank", "final_rank",
    }
    if required - set(table.columns) or table["query_index"].duplicated().any():
        raise RuntimeError(f"malformed E14 held table: {path}")
    return decision, table.reset_index(drop=True)


def comparison(
    reference: np.ndarray,
    candidate: np.ndarray,
    formula: np.ndarray,
    near: np.ndarray,
    resamples: int,
    seed: int,
) -> dict:
    reference_correct = reference == 1
    candidate_correct = candidate == 1
    ci = formula_bootstrap_delta(reference, candidate, formula, resamples, seed)
    return {
        "delta_recall1": float(np.mean(candidate_correct) - np.mean(reference_correct)),
        "corrected": int(np.sum(~reference_correct & candidate_correct)),
        "introduced": int(np.sum(reference_correct & ~candidate_correct)),
        "risk_net_lambda2": int(
            np.sum(~reference_correct & candidate_correct)
            - 2 * np.sum(reference_correct & ~candidate_correct)
        ),
        "near_delta_recall1": float(
            np.mean(candidate_correct[near]) - np.mean(reference_correct[near])
        ),
        "formula_cluster_ci": ci,
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite E14 summary: {args.output}")
    runs = {
        "control": load_run(args.control, args.outer_fold),
        "selected_no_margin": load_run(args.selected_no_margin, args.outer_fold),
        "selected_delta": load_run(args.selected_delta, args.outer_fold),
        "selected_delta_risk": load_run(args.selected_delta_risk, args.outer_fold),
    }
    tables = {name: item[1] for name, item in runs.items()}
    reference = tables["control"]
    for name, table in tables.items():
        for column in ("query_index", "query_formula", "baseline_rank", "initialization_rank"):
            if not np.array_equal(table[column].to_numpy(), reference[column].to_numpy()):
                raise RuntimeError(f"E14 paired ledger mismatch: {name}.{column}")
    formula = reference["query_formula"].astype(str).to_numpy()
    near = reference["has_near"].astype(bool).to_numpy()
    official = reference["baseline_rank"].to_numpy(np.int16)
    initialization = reference["initialization_rank"].to_numpy(np.int16)
    control = reference["final_rank"].to_numpy(np.int16)
    no_margin = tables["selected_no_margin"]["final_rank"].to_numpy(np.int16)
    delta = tables["selected_delta"]["final_rank"].to_numpy(np.int16)
    delta_risk = tables["selected_delta_risk"]["final_rank"].to_numpy(np.int16)
    comparisons = {
        "control_vs_initialization": comparison(
            initialization, control, formula, near, args.bootstrap_resamples, args.seed,
        ),
        "selected_no_margin_vs_control": comparison(
            control, no_margin, formula, near, args.bootstrap_resamples, args.seed + 1,
        ),
        "selected_delta_vs_control": comparison(
            control, delta, formula, near, args.bootstrap_resamples, args.seed + 2,
        ),
        "selected_delta_risk_vs_control": comparison(
            control, delta_risk, formula, near, args.bootstrap_resamples, args.seed + 3,
        ),
        "risk_control_increment_vs_delta": comparison(
            delta, delta_risk, formula, near, args.bootstrap_resamples, args.seed + 4,
        ),
        "selected_delta_risk_vs_official": comparison(
            official, delta_risk, formula, near, args.bootstrap_resamples, args.seed + 5,
        ),
    }
    selected_decision = runs["selected_delta_risk"][0]
    selected_increment = comparisons["selected_delta_risk_vs_control"]
    delta_increment = comparisons["selected_delta_vs_control"]
    report = {
        "status": "noise_final_e14_selected_transfer_summary_complete",
        "formal": True,
        "outer_formula_fold": int(args.outer_fold),
        "queries": int(len(reference)),
        "formulas": int(len(set(formula))),
        "comparisons": comparisons,
        "teacher_replay": selected_decision.get("guided_teacher_replay", {}),
        "gates": {
            "selected_delta_risk_beats_continuation_formula_ci": bool(
                selected_increment["formula_cluster_ci"]["ci_low"] > 0
            ),
            "selected_delta_risk_corrected_gt_introduced": bool(
                selected_increment["corrected"] > selected_increment["introduced"]
            ),
            "selected_delta_risk_near_nonnegative": bool(
                selected_increment["near_delta_recall1"] >= 0
            ),
            "risk_controls_do_not_increase_incremental_introduced": bool(
                selected_increment["introduced"] <= delta_increment["introduced"]
            ),
            "risk_controls_do_not_reduce_risk_weighted_net": bool(
                selected_increment["risk_net_lambda2"]
                >= delta_increment["risk_net_lambda2"]
            ),
            "teacher_replay_exact": bool(
                selected_decision.get("guided_teacher_replay", {}).get(
                    "action_margin_max_abs_error", 1.0
                ) <= 2e-4
            ),
            "preservation_vs_mature_ge_0_995": bool(
                selected_decision["held_clean"]["preservation_vs_initialization_mean"] >= 0.995
            ),
        },
        "provenance": {
            name: {
                "decision_sha256": sha256_file(path / "decision.json"),
                "per_query_sha256": sha256_file(path / "held_per_query.csv.gz"),
            }
            for name, path in {
                "control": args.control,
                "selected_no_margin": args.selected_no_margin,
                "selected_delta": args.selected_delta,
                "selected_delta_risk": args.selected_delta_risk,
            }.items()
        },
        "claim_limit": (
            "One held formula fold. Only a positive paired increment over the warm "
            "continuation control establishes transfer of selected action capacity."
        ),
    }
    report["pass_to_multifold"] = bool(all(report["gates"].values()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_dump(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
