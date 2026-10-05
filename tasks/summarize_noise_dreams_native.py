"""Paired decision for Noise-selected and matched-control native triplets."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_noise_dreams_native import formula_cluster_ci


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targeted", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-delta-recall1-pp", type=float, default=4.0)
    parser.add_argument(
        "--minimum-action-curriculum-delta-recall1-pp", type=float, default=0.5
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260920)
    return parser.parse_args()


def load_arm(path: Path, arm: str) -> tuple[dict, pd.DataFrame]:
    report_path = path / "report.json"
    table_path = path / "held_per_query.csv.gz"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    table = pd.read_csv(table_path, low_memory=False)
    if (
        report.get("status") != "NOISE_DREAMS_NATIVE_HELD_EVALUATION_COMPLETE"
        or report.get("arm") != arm
        or len(table) != 18333
        or table["query_index"].duplicated().any()
    ):
        raise RuntimeError(f"native held arm is incomplete: {path}")
    return report, table


def directional_checks(candidate: dict, reference: dict, prefix: str) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    cand_retrieval = candidate["retrieval"]
    ref_retrieval = reference["retrieval"]
    for name in ("recall@1", "recall@2", "recall@3", "recall@5", "recall@10", "recall@20", "mrr", "macro_query_auroc", "macro_query_auprc", "mean_positive_vs_best_negative_margin", "mean_signed_top1_top2_gap"):
        checks[f"{prefix}.retrieval.{name}"] = float(cand_retrieval[name]) >= float(ref_retrieval[name])
    for name in ("mean_rank", "median_rank"):
        checks[f"{prefix}.retrieval.{name}"] = float(cand_retrieval[name]) <= float(ref_retrieval[name])
    cand_near = candidate["near_subset"]
    ref_near = reference["near_subset"]
    for name in ("recall@1", "recall@2", "recall@3", "recall@5", "recall@10", "recall@20", "mrr", "macro_query_auroc", "macro_query_auprc", "mean_positive_vs_best_negative_margin", "mean_signed_top1_top2_gap"):
        checks[f"{prefix}.near.{name}"] = float(cand_near[name]) >= float(ref_near[name])
    for name in ("mean_rank", "median_rank"):
        checks[f"{prefix}.near.{name}"] = float(cand_near[name]) <= float(ref_near[name])
    for panel in (
        "micro_candidate",
        "massspecgym_10ppm_pooled_pairwise",
        "massspecgym_mh_10ppm_pooled_pairwise",
    ):
        for metric in ("auroc", "auprc"):
            checks[f"{prefix}.{panel}.{metric}"] = float(candidate[panel][metric]) >= float(reference[panel][metric])
    return checks


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    targeted, targeted_table = load_arm(args.targeted, "targeted")
    control, control_table = load_arm(args.control, "control")
    keys = ["query_index", "query_row", "query_ik14", "query_formula", "near", "official_rank"]
    if targeted_table[keys].to_dict("list") != control_table[keys].to_dict("list"):
        raise RuntimeError("targeted and control held ledgers are not paired")
    formulas = targeted_table["query_formula"].astype(str).to_numpy()
    near = targeted_table["near"].astype(bool).to_numpy()
    targeted_rank = targeted_table["candidate_rank"].to_numpy(int)
    control_rank = control_table["candidate_rank"].to_numpy(int)
    official_rank = targeted_table["official_rank"].to_numpy(int)
    r1_vs_control = (targeted_rank == 1).astype(float) - (
        control_rank == 1
    ).astype(float)
    mrr_vs_control = 1.0 / targeted_rank - 1.0 / control_rank
    paired_ci = {
        "targeted_vs_control_recall1": formula_cluster_ci(
            formulas, r1_vs_control, repeats=args.bootstrap_resamples,
            seed=args.seed, hypotheses=12,
        ),
        "targeted_vs_control_near_recall1": formula_cluster_ci(
            formulas[near], r1_vs_control[near], repeats=args.bootstrap_resamples,
            seed=args.seed + 1, hypotheses=12,
        ),
        "targeted_vs_control_mrr": formula_cluster_ci(
            formulas, mrr_vs_control, repeats=args.bootstrap_resamples,
            seed=args.seed + 2, hypotheses=12,
        ),
        "targeted_vs_control_near_mrr": formula_cluster_ci(
            formulas[near], mrr_vs_control[near], repeats=args.bootstrap_resamples,
            seed=args.seed + 3, hypotheses=12,
        ),
    }
    official_metrics = targeted["official"]
    targeted_metrics = targeted["candidate"]
    control_metrics = control["candidate"]
    mature_e8_metrics = targeted.get("mature_e8")
    if (
        mature_e8_metrics is None
        or targeted.get("candidate_vs_mature_e8") is None
        or targeted.get("mature_e8_formula_cluster_paired_ci") is None
        or "mature_e8_rank" not in targeted_table
    ):
        raise RuntimeError("targeted arm lacks the registered mature-E8 comparator")
    directional = {
        **directional_checks(targeted_metrics, official_metrics, "targeted_vs_official"),
        **directional_checks(targeted_metrics, control_metrics, "targeted_vs_control"),
        **directional_checks(targeted_metrics, mature_e8_metrics, "targeted_vs_mature_e8"),
    }
    delta_recall1_pp = 100.0 * (
        float(targeted_metrics["retrieval"]["recall@1"])
        - float(official_metrics["retrieval"]["recall@1"])
    )
    targeted_risk = targeted["candidate_vs_official"]
    targeted_vs_mature_e8_risk = targeted["candidate_vs_mature_e8"]
    mature_e8_ci = targeted["mature_e8_formula_cluster_paired_ci"]
    delta_recall1_vs_mature_e8_pp = 100.0 * (
        float(targeted_metrics["retrieval"]["recall@1"])
        - float(mature_e8_metrics["retrieval"]["recall@1"])
    )
    targeted_corrected_vs_control = (control_rank > 1) & (targeted_rank == 1)
    targeted_introduced_vs_control = (control_rank == 1) & (targeted_rank > 1)
    targeted_vs_control_risk = {
        "corrected": int(np.sum(targeted_corrected_vs_control)),
        "introduced": int(np.sum(targeted_introduced_vs_control)),
        "risk_net_lambda2": int(
            np.sum(targeted_corrected_vs_control)
            - 2 * np.sum(targeted_introduced_vs_control)
        ),
        "near_corrected": int(np.sum(targeted_corrected_vs_control & near)),
        "near_introduced": int(np.sum(targeted_introduced_vs_control & near)),
        "near_risk_net_lambda2": int(
            np.sum(targeted_corrected_vs_control & near)
            - 2 * np.sum(targeted_introduced_vs_control & near)
        ),
    }
    delta_recall1_vs_control_pp = 100.0 * float(np.mean(r1_vs_control))
    official_ci = targeted["formula_cluster_paired_ci"]
    gates = {
        "targeted_recall1_gain_at_least_4pp": delta_recall1_pp >= args.minimum_delta_recall1_pp,
        "targeted_recall1_gain_vs_mature_e8_at_least_4pp": (
            delta_recall1_vs_mature_e8_pp >= args.minimum_delta_recall1_pp
        ),
        "targeted_action_curriculum_gain_vs_control_at_least_0_5pp": (
            delta_recall1_vs_control_pp
            >= args.minimum_action_curriculum_delta_recall1_pp
        ),
        "all_targeted_vs_official_formula_cis_strict_positive": all(
            float(official_ci[name]["ci_low_pp"]) > 0
            for name in ("recall1", "near_recall1", "mrr", "near_mrr")
        ),
        "all_targeted_vs_control_formula_cis_strict_positive": all(
            float(body["ci_low_pp"]) > 0 for body in paired_ci.values()
        ),
        "all_targeted_vs_mature_e8_formula_cis_strict_positive": all(
            float(mature_e8_ci[name]["ci_low_pp"]) > 0
            for name in ("recall1", "near_recall1", "mrr", "near_mrr")
        ),
        "targeted_risk_net_positive": int(targeted_risk["risk_net_lambda2"]) > 0,
        "targeted_near_risk_net_positive": int(
            targeted_risk["near_risk_net_lambda2"]
        ) > 0,
        "targeted_vs_control_risk_net_positive": int(
            targeted_vs_control_risk["risk_net_lambda2"]
        ) > 0,
        "targeted_vs_control_near_risk_net_positive": int(
            targeted_vs_control_risk["near_risk_net_lambda2"]
        ) > 0,
        "targeted_vs_mature_e8_risk_net_positive": int(
            targeted_vs_mature_e8_risk["risk_net_lambda2"]
        ) > 0,
        "targeted_vs_mature_e8_near_risk_net_positive": int(
            targeted_vs_mature_e8_risk["near_risk_net_lambda2"]
        ) > 0,
        "all_registered_metrics_nonregressing_vs_official_control_and_mature_e8": all(
            directional.values()
        ),
    }
    report = {
        "status": "NOISE_DREAMS_NATIVE_PROMOTION_PASS"
        if all(gates.values()) else "NOISE_DREAMS_NATIVE_PROMOTION_FAIL",
        "delta_recall1_pp": delta_recall1_pp,
        "delta_recall1_vs_mature_e8_pp": delta_recall1_vs_mature_e8_pp,
        "delta_recall1_vs_control_pp": delta_recall1_vs_control_pp,
        "minimum_delta_recall1_pp": args.minimum_delta_recall1_pp,
        "minimum_action_curriculum_delta_recall1_pp": (
            args.minimum_action_curriculum_delta_recall1_pp
        ),
        "targeted": targeted_metrics,
        "control": control_metrics,
        "official": official_metrics,
        "mature_e8": mature_e8_metrics,
        "targeted_vs_official_risk": targeted_risk,
        "targeted_vs_control_risk": targeted_vs_control_risk,
        "targeted_vs_mature_e8_risk": targeted_vs_mature_e8_risk,
        "targeted_vs_mature_e8_formula_cluster_paired_ci": mature_e8_ci,
        "targeted_vs_control_formula_cluster_paired_ci": paired_ci,
        "directional_metric_checks": directional,
        "gates": gates,
        "claim_limit": (
            "A pass is corrected MassSpecGym formula-held evidence, not an external dataset claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{args.output.name}.", dir=args.output.parent
    ))
    try:
        (staging / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
