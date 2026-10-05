"""Compare a bounded three-arm direct-v3 canary on one held query panel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from noise_final_core import sha256_file
from summarize_noise_corrected_direct_v3_arms import formula_cluster_ci


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routed-dir", type=Path, required=True)
    parser.add_argument("--shuffled-dir", type=Path, required=True)
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path, arm: str) -> tuple[dict[str, object], pd.DataFrame]:
    decision_path = path / "decision.json"
    table_path = path / "held_per_query.csv.gz"
    checkpoint_path = path / "final_shared_encoder.pt"
    if missing := [
        str(value) for value in (decision_path, table_path, checkpoint_path)
        if not value.is_file()
    ]:
        raise FileNotFoundError(missing)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if (
        decision.get("status") != "noise_corrected_routed_direct_complete"
        or decision.get("formal") is not False
        or decision.get("arm") != arm
        or decision.get("corrective_objective_mode") != "v3_direct"
        or "development_held_graph" not in decision.get("evaluation", {})
    ):
        raise RuntimeError(f"development canary contract failed: {path}")
    if decision.get("provenance", {}).get(
        "final_shared_encoder_sha256"
    ) != sha256_file(checkpoint_path):
        raise RuntimeError(f"canary checkpoint hash differs: {path}")
    table = pd.read_csv(table_path, low_memory=False).sort_values(
        "query_index", kind="stable",
    ).reset_index(drop=True)
    required = {
        "query_index", "query_formula", "near", "initial_E8_rank",
        "candidate_rank", "initial_E8_reciprocal_rank",
        "candidate_reciprocal_rank", "initial_E8_macro_query_auc",
        "candidate_macro_query_auc", "initial_E8_macro_query_auprc",
        "candidate_macro_query_auprc",
        "initial_E8_positive_vs_best_negative_margin",
        "candidate_positive_vs_best_negative_margin",
        "initial_E8_signed_top1_top2_gap", "candidate_signed_top1_top2_gap",
    }
    if missing := required - set(table.columns):
        raise RuntimeError(f"canary held table misses {sorted(missing)}")
    if table["query_index"].duplicated().any():
        raise RuntimeError("canary held table contains duplicate queries")
    return decision, table


def _recall_delta(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    return {
        f"recall@{cutoff}_delta_pp": float(
            100.0 * (np.mean(right <= cutoff) - np.mean(left <= cutoff))
        )
        for cutoff in (1, 2, 3, 5, 10, 20)
    }


def _paired_arm_delta(
    left: pd.DataFrame, right: pd.DataFrame, formulas: np.ndarray,
    *, repeats: int, seed: int,
) -> dict[str, object]:
    left_rank = left["candidate_rank"].to_numpy(np.int64)
    right_rank = right["candidate_rank"].to_numpy(np.int64)
    result: dict[str, object] = _recall_delta(left_rank, right_rank)
    result.update({
        "mrr_delta_pp": float(100.0 * np.mean(
            right["candidate_reciprocal_rank"]
            - left["candidate_reciprocal_rank"]
        )),
        "macro_query_auroc_delta_pp": float(100.0 * np.mean(
            right["candidate_macro_query_auc"]
            - left["candidate_macro_query_auc"]
        )),
        "macro_query_auprc_delta_pp": float(100.0 * np.mean(
            right["candidate_macro_query_auprc"]
            - left["candidate_macro_query_auprc"]
        )),
        "mean_margin_delta": float(np.mean(
            right["candidate_positive_vs_best_negative_margin"]
            - left["candidate_positive_vs_best_negative_margin"]
        )),
        "mean_signed_top1_top2_gap_delta": float(np.mean(
            right["candidate_signed_top1_top2_gap"]
            - left["candidate_signed_top1_top2_gap"]
        )),
        "formula_cluster_recall1_delta": formula_cluster_ci(
            formulas,
            (right_rank == 1).astype(float) - (left_rank == 1).astype(float),
            repeats, seed, familywise_hypotheses=3,
        ),
    })
    return result


def _top1_outcomes(
    left: pd.DataFrame, right: pd.DataFrame, near: np.ndarray,
) -> dict[str, object]:
    left_rank = left["candidate_rank"].to_numpy(np.int64)
    right_rank = right["candidate_rank"].to_numpy(np.int64)
    near = np.asarray(near, dtype=bool)
    if left_rank.shape != right_rank.shape or near.shape != left_rank.shape:
        raise RuntimeError("canary Top1 outcome arrays are not aligned")
    corrected = (left_rank > 1) & (right_rank == 1)
    introduced = (left_rank == 1) & (right_rank > 1)

    def counts(mask: np.ndarray) -> dict[str, int]:
        fixed = int(np.sum(corrected & mask))
        broken = int(np.sum(introduced & mask))
        return {
            "corrected": fixed,
            "introduced": broken,
            "risk_net_lambda2": fixed - 2 * broken,
        }

    return {
        **counts(np.ones(len(near), dtype=bool)),
        "near": counts(near),
    }


def _complete_metric_delta(
    left: dict[str, object], right: dict[str, object],
) -> dict[str, float]:
    """Compare every registered held metric with positive meaning better.

    Count fields are alignment checks, not performance metrics.  Rank deltas
    are reversed so positive always means that the routed arm lowered rank.
    """
    output: dict[str, float] = {}
    for panel in ("retrieval", "near_subset"):
        left_panel = left[panel]
        right_panel = right[panel]
        if int(left_panel["queries"]) != int(right_panel["queries"]):
            raise RuntimeError(f"{panel} query counts differ across canary arms")
        for cutoff in (1, 2, 3, 5, 10, 20):
            output[f"{panel}.recall@{cutoff}_delta_pp"] = 100.0 * (
                float(right_panel[f"recall@{cutoff}"])
                - float(left_panel[f"recall@{cutoff}"])
            )
        for metric in ("mrr", "macro_query_auroc", "macro_query_auprc"):
            output[f"{panel}.{metric}_delta_pp"] = 100.0 * (
                float(right_panel[metric]) - float(left_panel[metric])
            )
        for metric in (
            "mean_positive_vs_best_negative_margin",
            "mean_top1_top2_gap",
            "mean_signed_top1_top2_gap",
        ):
            output[f"{panel}.{metric}_delta"] = (
                float(right_panel[metric]) - float(left_panel[metric])
            )
        for metric in ("mean_rank", "median_rank"):
            output[f"{panel}.{metric}_reduction"] = (
                float(left_panel[metric]) - float(right_panel[metric])
            )
    for panel in (
        "micro_candidate", "massspecgym_10ppm_pooled_pairwise",
        "massspecgym_mh_10ppm_pooled_pairwise",
    ):
        if panel not in left or panel not in right:
            if panel in left or panel in right:
                raise RuntimeError(f"{panel} exists in only one canary arm")
            continue
        for count in (
            "molecules", "spectrum_pairs", "positive_pairs", "negative_pairs",
        ):
            if count in left[panel] or count in right[panel]:
                if int(left[panel][count]) != int(right[panel][count]):
                    raise RuntimeError(f"{panel} {count} differs across canary arms")
        for metric in ("auroc", "auprc"):
            output[f"{panel}.{metric}_delta_pp"] = 100.0 * (
                float(right[panel][metric]) - float(left[panel][metric])
            )
    if not output or not all(np.isfinite(value) for value in output.values()):
        raise RuntimeError("complete canary metric delta is empty or non-finite")
    return output


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {args.output_dir}")
    loaded = {
        arm: _load(path, arm)
        for arm, path in (
            ("routed_direct", args.routed_dir),
            ("shuffled_action_control", args.shuffled_dir),
            ("clean_control", args.clean_dir),
        )
    }
    decisions = {name: value[0] for name, value in loaded.items()}
    tables = {name: value[1] for name, value in loaded.items()}
    identity = ["query_index", "query_formula", "near"]
    baseline = [
        "initial_E8_rank", "initial_E8_reciprocal_rank",
        "initial_E8_macro_query_auc", "initial_E8_macro_query_auprc",
        "initial_E8_positive_vs_best_negative_margin",
        "initial_E8_signed_top1_top2_gap",
    ]
    routed = tables["routed_direct"]
    for name, table in tables.items():
        if not routed[identity].equals(table[identity]):
            raise RuntimeError(f"canary held queries differ for {name}")
        if not np.allclose(
            routed[baseline].to_numpy(float), table[baseline].to_numpy(float),
            atol=1e-7, rtol=0,
        ):
            raise RuntimeError(f"canary initial E8 baseline differs for {name}")
    provenance_keys = (
        "candidate_graph_sha256", "source_manifest_sha256",
        "routed_ledger_report_sha256", "training_actions_sha256",
        "action_spectra_sha256", "initial_student_checkpoint_sha256",
        "official_checkpoint_sha256",
    )
    for key in provenance_keys:
        if len({
            decision["provenance"].get(key) for decision in decisions.values()
        }) != 1:
            raise RuntimeError(f"canary arms do not share {key}")
    formulas = routed["query_formula"].astype(str).to_numpy()
    initial_table = routed.copy()
    initial_table["candidate_rank"] = routed["initial_E8_rank"].to_numpy()
    initial_table["candidate_reciprocal_rank"] = routed[
        "initial_E8_reciprocal_rank"
    ].to_numpy()
    initial_table["candidate_macro_query_auc"] = routed[
        "initial_E8_macro_query_auc"
    ].to_numpy()
    initial_table["candidate_macro_query_auprc"] = routed[
        "initial_E8_macro_query_auprc"
    ].to_numpy()
    initial_table["candidate_positive_vs_best_negative_margin"] = routed[
        "initial_E8_positive_vs_best_negative_margin"
    ].to_numpy()
    initial_table["candidate_signed_top1_top2_gap"] = routed[
        "initial_E8_signed_top1_top2_gap"
    ].to_numpy()
    comparisons = {
        "routed_vs_initial_E8": _paired_arm_delta(
            initial_table,
            routed, formulas, repeats=args.bootstrap_resamples, seed=args.seed,
        ),
        "routed_vs_shuffled": _paired_arm_delta(
            tables["shuffled_action_control"], routed, formulas,
            repeats=args.bootstrap_resamples, seed=args.seed + 1,
        ),
        "routed_vs_clean": _paired_arm_delta(
            tables["clean_control"], routed, formulas,
            repeats=args.bootstrap_resamples, seed=args.seed + 2,
        ),
    }
    comparison_left = {
        "routed_vs_initial_E8": initial_table,
        "routed_vs_shuffled": tables["shuffled_action_control"],
        "routed_vs_clean": tables["clean_control"],
    }
    for name, left in comparison_left.items():
        comparisons[name]["top1_outcomes"] = _top1_outcomes(
            left, routed, routed["near"].to_numpy(bool),
        )
    arm_metrics = {
        name: decision["evaluation"]["development_held_graph"]["candidate"]
        for name, decision in decisions.items()
    }
    initial_metrics = decisions["routed_direct"]["evaluation"][
        "development_held_graph"
    ]["initial_E8"]
    complete_metric_deltas = {
        "routed_vs_initial_E8": _complete_metric_delta(
            initial_metrics, arm_metrics["routed_direct"],
        ),
        "routed_vs_shuffled": _complete_metric_delta(
            arm_metrics["shuffled_action_control"], arm_metrics["routed_direct"],
        ),
        "routed_vs_clean": _complete_metric_delta(
            arm_metrics["clean_control"], arm_metrics["routed_direct"],
        ),
    }
    initial_rank = routed["initial_E8_rank"].to_numpy(np.int64)
    routed_rank = routed["candidate_rank"].to_numpy(np.int64)
    corrected = (initial_rank > 1) & (routed_rank == 1)
    introduced = (initial_rank == 1) & (routed_rank > 1)
    checks = {
        "routed_recall1_above_initial": comparisons[
            "routed_vs_initial_E8"
        ]["recall@1_delta_pp"] > 0,
        "routed_recall1_above_shuffled": comparisons[
            "routed_vs_shuffled"
        ]["recall@1_delta_pp"] > 0,
        "routed_recall1_above_clean": comparisons[
            "routed_vs_clean"
        ]["recall@1_delta_pp"] > 0,
        "routed_risk_net_positive": int(corrected.sum() - 2 * introduced.sum()) > 0,
        "routed_risk_net_above_shuffled_positive": comparisons[
            "routed_vs_shuffled"
        ]["top1_outcomes"]["risk_net_lambda2"] > 0,
        "routed_risk_net_above_clean_positive": comparisons[
            "routed_vs_clean"
        ]["top1_outcomes"]["risk_net_lambda2"] > 0,
        "routed_near_risk_net_nonnegative_for_all_comparators": all(
            comparisons[name]["top1_outcomes"]["near"]["risk_net_lambda2"] >= 0
            for name in comparisons
        ),
    }
    complete_checks = {
        name: {
            "all_registered_metric_directions_nonnegative": bool(
                all(value >= -1e-12 for value in deltas.values())
            ),
            "strictly_improved_metric_count": int(sum(
                value > 1e-12 for value in deltas.values()
            )),
            "registered_metric_count": int(len(deltas)),
            "formula_cluster_recall1_ci_low_strict_positive": bool(
                comparisons[name]["formula_cluster_recall1_delta"]["ci_low_pp"] > 0
            ),
        }
        for name, deltas in complete_metric_deltas.items()
    }
    report = {
        "status": "noise_corrected_direct_v3_bounded_canary_summary_complete",
        "formal": False,
        "promotion_authorized": False,
        "training_seed": args.seed,
        "held_queries": int(len(routed)),
        "routed_corrected": int(corrected.sum()),
        "routed_introduced": int(introduced.sum()),
        "routed_risk_net_lambda2": int(corrected.sum() - 2 * introduced.sum()),
        "comparisons": comparisons,
        "complete_metric_deltas": complete_metric_deltas,
        "arm_metrics": arm_metrics,
        "directional_checks": checks,
        "complete_metric_directional_checks": complete_checks,
        "strict_all_metric_followup_gate_passed": bool(
            all(checks.values())
            and all(
                item["all_registered_metric_directions_nonnegative"]
                and item["formula_cluster_recall1_ci_low_strict_positive"]
                for item in complete_checks.values()
            )
        ),
        "promising_for_formal_followup": bool(all(checks.values())),
        "contracts": {
            "same_bounded_outer_formula_held_queries": True,
            "same_exact_initial_E8_baseline": True,
            "routed_compared_to_shuffled_and_clean_controls": True,
            "complete_registered_metric_family_reported_per_arm": True,
            "complete_registered_metric_deltas_reported_for_all_comparators": True,
            "mean_and_median_rank_are_oriented_as_reductions": True,
            "formula_cluster_CI_is_descriptive_not_promotional": True,
            "teacher_or_distillation_target_used": False,
        },
        "provenance": {
            name: {
                "decision_sha256": sha256_file(path / "decision.json"),
                "held_table_sha256": sha256_file(path / "held_per_query.csv.gz"),
            }
            for name, path in (
                ("routed_direct", args.routed_dir),
                ("shuffled_action_control", args.shuffled_dir),
                ("clean_control", args.clean_dir),
            )
        },
        "claim_limit": (
            "One-seed bounded one-epoch canary.  It can reject a weak route or "
            "justify a formal run, but cannot establish a 4 pp full-graph gain."
        ),
    }
    combined = routed[identity + baseline].copy()
    for name, table in tables.items():
        for column in (
            "candidate_rank", "candidate_reciprocal_rank",
            "candidate_macro_query_auc", "candidate_macro_query_auprc",
            "candidate_positive_vs_best_negative_margin",
            "candidate_signed_top1_top2_gap",
        ):
            combined[f"{name}_{column}"] = table[column].to_numpy()
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent,
    ))
    try:
        combined.to_csv(staging / "paired_canary_per_query.csv.gz", index=False)
        (staging / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8",
        )
        staging.rename(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
