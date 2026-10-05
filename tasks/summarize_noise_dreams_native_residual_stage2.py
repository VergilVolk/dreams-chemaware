"""Promote a Stage-2 Noise continuation only if it improves the frozen champion."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_noise_dreams_native import formula_cluster_ci
from summarize_noise_dreams_native_replicates import directional_checks


AUC_METRICS = {
    "retrieval": ("macro_query_auroc", "macro_query_auprc"),
    "near_subset": ("macro_query_auroc", "macro_query_auprc"),
    "micro_candidate": ("auroc", "auprc"),
    "massspecgym_10ppm_pooled_pairwise": ("auroc", "auprc"),
    "massspecgym_mh_10ppm_pooled_pairwise": ("auroc", "auprc"),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--targeted", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--result-prefix", default="NOISE_DREAMS_NATIVE_RESIDUAL_STAGE2",
    )
    parser.add_argument("--absolute-target-pp", type=float, default=5.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260926)
    return parser.parse_args()


def load_arm(path: Path, expected_arm: str | None) -> tuple[dict, pd.DataFrame]:
    report = json.loads((path / "report.json").read_text(encoding="utf-8"))
    table = pd.read_csv(path / "held_per_query.csv.gz", low_memory=False)
    if (
        report.get("status") != "NOISE_DREAMS_NATIVE_HELD_EVALUATION_COMPLETE"
        or (expected_arm is not None and report.get("arm") != expected_arm)
        or len(table) != 18333
        or table["query_index"].duplicated().any()
        or report.get("mature_e8") is None
    ):
        raise RuntimeError(f"incomplete residual evaluation: {path}")
    return report, table


def rank_comparison(
    candidate: np.ndarray,
    reference: np.ndarray,
    formulas: np.ndarray,
    near: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    candidate = np.asarray(candidate, dtype=np.int64)
    reference = np.asarray(reference, dtype=np.int64)
    delta_r1 = (candidate == 1).astype(float) - (reference == 1).astype(float)
    delta_mrr = 1.0 / candidate.astype(float) - 1.0 / reference.astype(float)
    corrected = (candidate == 1) & (reference != 1)
    introduced = (candidate != 1) & (reference == 1)
    return {
        "delta_recall1_pp": float(100.0 * np.mean(delta_r1)),
        "delta_mrr": float(np.mean(delta_mrr)),
        "recall1_formula_cluster_ci": formula_cluster_ci(
            formulas, delta_r1, repeats=repeats, seed=seed, hypotheses=4,
        ),
        "mrr_formula_cluster_ci": formula_cluster_ci(
            formulas, delta_mrr, repeats=repeats, seed=seed + 1, hypotheses=4,
        ),
        "near_recall1_formula_cluster_ci": formula_cluster_ci(
            formulas[near], delta_r1[near], repeats=repeats,
            seed=seed + 2, hypotheses=4,
        ),
        "near_mrr_formula_cluster_ci": formula_cluster_ci(
            formulas[near], delta_mrr[near], repeats=repeats,
            seed=seed + 3, hypotheses=4,
        ),
        "corrected": int(np.sum(corrected)),
        "introduced": int(np.sum(introduced)),
        "risk_net_lambda2": int(np.sum(corrected) - 2 * np.sum(introduced)),
        "near_corrected": int(np.sum(corrected & near)),
        "near_introduced": int(np.sum(introduced & near)),
        "near_risk_net_lambda2": int(
            np.sum(corrected & near) - 2 * np.sum(introduced & near)
        ),
    }


def query_auc_comparison(
    candidate_table: pd.DataFrame,
    reference_table: pd.DataFrame,
    formulas: np.ndarray,
    near: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    """Paired formula-cluster inference for per-query AUROC and AUPRC."""
    result: dict[str, object] = {}
    columns = {
        "macro_query_auroc": "candidate_macro_query_auc",
        "macro_query_auprc": "candidate_macro_query_auprc",
    }
    for offset, (metric, column) in enumerate(columns.items()):
        if column not in candidate_table or column not in reference_table:
            raise RuntimeError(f"held query ledger lacks {column}")
        delta = (
            candidate_table[column].to_numpy(np.float64)
            - reference_table[column].to_numpy(np.float64)
        )
        result[metric] = {
            "delta": float(np.mean(delta)),
            "delta_pp": float(100.0 * np.mean(delta)),
            "formula_cluster_ci": formula_cluster_ci(
                formulas, delta, repeats=repeats, seed=seed + 2 * offset,
                hypotheses=8,
            ),
            "near_delta": float(np.mean(delta[near])),
            "near_delta_pp": float(100.0 * np.mean(delta[near])),
            "near_formula_cluster_ci": formula_cluster_ci(
                formulas[near], delta[near], repeats=repeats,
                seed=seed + 2 * offset + 1, hypotheses=8,
            ),
        }
    return result


def absolute_gain(report: dict) -> dict[str, float]:
    candidate = report["candidate"]["retrieval"]["recall@1"]
    official = report["official"]["retrieval"]["recall@1"]
    mature = report["mature_e8"]["retrieval"]["recall@1"]
    return {
        "vs_official_pp": float(100.0 * (candidate - official)),
        "vs_mature_e8_pp": float(100.0 * (candidate - mature)),
    }


def auc_metric_comparison(candidate: dict, reference: dict) -> dict[str, dict]:
    """Return numerical AUC/AUPRC values instead of only direction booleans."""
    comparison: dict[str, dict] = {}
    for panel, metrics in AUC_METRICS.items():
        comparison[panel] = {}
        for metric in metrics:
            candidate_value = float(candidate[panel][metric])
            reference_value = float(reference[panel][metric])
            delta = candidate_value - reference_value
            comparison[panel][metric] = {
                "candidate": candidate_value,
                "reference": reference_value,
                "delta": delta,
                "delta_pp": 100.0 * delta,
                "improved": delta > 0.0,
            }
    return comparison


def main() -> None:
    args = arguments()
    if args.result_prefix not in {
        "NOISE_DREAMS_NATIVE_RESIDUAL_STAGE2",
        "NOISE_DREAMS_NATIVE_MULTIDIFFICULTY_STAGE3",
        "NOISE_DREAMS_NATIVE_MULTIDIFFICULTY_STAGE3_REPAIR",
        "NOISE_DREAMS_NATIVE_BROAD_POSITIVE_STAGE4",
        "NOISE_DREAMS_NATIVE_NEGATIVE_RESIDUAL_STAGE5",
        "NOISE_DREAMS_NATIVE_SCALE_STAGE6",
    }:
        raise RuntimeError("unregistered Noise native result prefix")
    candidate_label = (
        "stage6"
        if args.result_prefix.endswith("STAGE6")
        else "stage5"
        if args.result_prefix.endswith("STAGE5")
        else "stage4"
        if args.result_prefix.endswith("STAGE4")
        else "stage3_repair"
        if args.result_prefix.endswith("STAGE3_REPAIR")
        else "stage3" if args.result_prefix.endswith("STAGE3") else "stage2"
    )
    if args.output.exists():
        raise FileExistsError(args.output)
    stage1, stage1_table = load_arm(args.stage1, None)
    targeted, targeted_table = load_arm(args.targeted, "targeted")
    control, control_table = load_arm(args.control, "control")
    keys = [
        "query_index", "query_row", "query_ik14", "query_formula", "near",
        "official_rank", "mature_e8_rank",
    ]
    if not (
        stage1_table[keys].to_dict("list")
        == targeted_table[keys].to_dict("list")
        == control_table[keys].to_dict("list")
    ):
        raise RuntimeError("Stage-1, targeted and control held ledgers are not paired")
    formulas = targeted_table["query_formula"].astype(str).to_numpy()
    near = targeted_table["near"].to_numpy(bool)
    stage1_rank = stage1_table["candidate_rank"].to_numpy(np.int64)
    targeted_rank = targeted_table["candidate_rank"].to_numpy(np.int64)
    control_rank = control_table["candidate_rank"].to_numpy(np.int64)
    target_vs_stage1 = rank_comparison(
        targeted_rank, stage1_rank, formulas, near,
        repeats=args.bootstrap_resamples, seed=args.seed,
    )
    target_vs_control = rank_comparison(
        targeted_rank, control_rank, formulas, near,
        repeats=args.bootstrap_resamples, seed=args.seed + 10,
    )
    gains = absolute_gain(targeted)
    directional = {
        **directional_checks(
            targeted["candidate"], stage1["candidate"],
            f"{candidate_label}_vs_stage1",
        ),
        **directional_checks(
            targeted["candidate"], targeted["official"],
            f"{candidate_label}_vs_official",
        ),
        **directional_checks(
            targeted["candidate"], targeted["mature_e8"],
            f"{candidate_label}_vs_mature_e8",
        ),
    }
    auc_comparisons = {
        f"{candidate_label}_vs_stage1": auc_metric_comparison(
            targeted["candidate"], stage1["candidate"],
        ),
        "targeted_vs_control": auc_metric_comparison(
            targeted["candidate"], control["candidate"],
        ),
        f"{candidate_label}_vs_official": auc_metric_comparison(
            targeted["candidate"], targeted["official"],
        ),
        f"{candidate_label}_vs_mature_e8": auc_metric_comparison(
            targeted["candidate"], targeted["mature_e8"],
        ),
    }
    query_auc_comparisons = {
        f"{candidate_label}_vs_stage1": query_auc_comparison(
            targeted_table, stage1_table, formulas, near,
            repeats=args.bootstrap_resamples, seed=args.seed + 20,
        ),
        "targeted_vs_control": query_auc_comparison(
            targeted_table, control_table, formulas, near,
            repeats=args.bootstrap_resamples, seed=args.seed + 30,
        ),
    }
    gates = {
        f"{candidate_label}_recall1_exceeds_stage1": (
            target_vs_stage1["delta_recall1_pp"] > 0
        ),
        f"{candidate_label}_mrr_exceeds_stage1": target_vs_stage1["delta_mrr"] > 0,
        f"{candidate_label}_vs_stage1_recall1_formula_ci_positive": (
            target_vs_stage1["recall1_formula_cluster_ci"]["ci_low_pp"] > 0
        ),
        f"{candidate_label}_vs_stage1_risk_net_positive": (
            target_vs_stage1["risk_net_lambda2"] > 0
        ),
        f"{candidate_label}_vs_stage1_near_risk_net_nonnegative": (
            target_vs_stage1["near_risk_net_lambda2"] >= 0
        ),
        f"{candidate_label}_vs_stage1_near_formula_cis_positive": all(
            target_vs_stage1[key]["ci_low_pp"] > 0
            for key in (
                "near_recall1_formula_cluster_ci", "near_mrr_formula_cluster_ci",
            )
        ),
        "targeted_beats_matched_control": target_vs_control["delta_recall1_pp"] > 0,
        "targeted_vs_control_formula_ci_positive": (
            target_vs_control["recall1_formula_cluster_ci"]["ci_low_pp"] > 0
        ),
        "targeted_vs_control_risk_net_positive": target_vs_control["risk_net_lambda2"] > 0,
        "targeted_vs_control_near_risk_net_nonnegative": (
            target_vs_control["near_risk_net_lambda2"] >= 0
        ),
        "targeted_vs_control_near_formula_cis_positive": all(
            target_vs_control[key]["ci_low_pp"] > 0
            for key in (
                "near_recall1_formula_cluster_ci", "near_mrr_formula_cluster_ci",
            )
        ),
        "all_registered_directions_noninferior": all(directional.values()),
        "absolute_gain_vs_official_reaches_target": (
            gains["vs_official_pp"] >= args.absolute_target_pp
        ),
        "absolute_gain_vs_mature_e8_reaches_target": (
            gains["vs_mature_e8_pp"] >= args.absolute_target_pp
        ),
        "baseline_formula_cis_positive": all(
            float(panel[metric]["ci_low_pp"]) > 0
            for panel in (
                targeted["formula_cluster_paired_ci"],
                targeted["mature_e8_formula_cluster_paired_ci"],
            )
            for metric in ("recall1", "near_recall1", "mrr", "near_mrr")
        ),
        "baseline_risk_nets_positive": all(
            int(panel[metric]) > 0
            for panel in (
                targeted["candidate_vs_official"],
                targeted["candidate_vs_mature_e8"],
            )
            for metric in ("risk_net_lambda2", "near_risk_net_lambda2")
        ),
    }
    if candidate_label in {"stage4", "stage6"}:
        for comparison in (
            f"{candidate_label}_vs_stage1",
            "targeted_vs_control",
            f"{candidate_label}_vs_official",
            f"{candidate_label}_vs_mature_e8",
        ):
            gates[f"all_auc_auprc_improve_{comparison}"] = all(
                bool(metric["improved"])
                for panel in auc_comparisons[comparison].values()
                for metric in panel.values()
            )
        for comparison in (f"{candidate_label}_vs_stage1", "targeted_vs_control"):
            gates[f"macro_auc_formula_cis_positive_{comparison}"] = all(
                float(metric[ci_name]["ci_low_pp"]) > 0
                for metric in query_auc_comparisons[comparison].values()
                for ci_name in ("formula_cluster_ci", "near_formula_cluster_ci")
            )
    champion_retention = {
        "selected_checkpoint": (
            f"{candidate_label}_targeted" if all(gates.values()) else "stage1_targeted"
        ),
        "stage1_is_never_overwritten": True,
        "five_pp_is_a_promotion_gate_not_a_claim": True,
    }
    report = {
        "status": (
            f"{args.result_prefix}_PROMOTION_PASS"
            if all(gates.values())
            else f"{args.result_prefix}_RETAIN_STAGE1"
        ),
        "absolute_target_pp": args.absolute_target_pp,
        "absolute_candidate_gain": gains,
        "targeted_vs_stage1": target_vs_stage1,
        "targeted_vs_control": target_vs_control,
        "auc_metric_comparisons": auc_comparisons,
        "query_auc_formula_cluster_comparisons": query_auc_comparisons,
        "directional_metric_checks": directional,
        "gates": gates,
        "champion_retention": champion_retention,
        "claim_limit": (
            "Corrected MassSpecGym formula-held internal evidence; no external claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        (staging / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8",
        )
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
