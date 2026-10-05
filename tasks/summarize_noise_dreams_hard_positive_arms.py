"""Causal promotion decision for targeted versus same-query control actions."""
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
from summarize_noise_dreams_native_residual_stage2 import auc_metric_comparison


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targeted", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-delta-recall1-pp", type=float, default=2.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260925)
    return parser.parse_args()


def load_arm(path: Path, arm: str) -> tuple[dict, pd.DataFrame]:
    report = json.loads((path / "report.json").read_text(encoding="utf-8"))
    table = pd.read_csv(path / "held_per_query.csv.gz", low_memory=False)
    if (
        report.get("status") != "NOISE_DREAMS_NATIVE_HELD_EVALUATION_COMPLETE"
        or report.get("arm") != arm
        or len(table) != 18333
        or table["query_index"].duplicated().any()
    ):
        raise RuntimeError(f"incomplete hard-positive arm: {path}")
    return report, table


def gain_summary(report: dict) -> dict[str, float]:
    candidate = report["candidate"]
    official = report["official"]
    mature = report["mature_e8"]
    return {
        "delta_recall1_vs_official_pp": 100.0 * (
            float(candidate["retrieval"]["recall@1"])
            - float(official["retrieval"]["recall@1"])
        ),
        "delta_recall1_vs_mature_e8_pp": 100.0 * (
            float(candidate["retrieval"]["recall@1"])
            - float(mature["retrieval"]["recall@1"])
        ),
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    targeted, targeted_table = load_arm(args.targeted, "targeted")
    control, control_table = load_arm(args.control, "control")
    keys = [
        "query_index", "query_row", "query_ik14", "query_formula", "near",
        "official_rank",
    ]
    if targeted_table[keys].to_dict("list") != control_table[keys].to_dict("list"):
        raise RuntimeError("targeted and control evaluations are not paired")

    targeted_gain = gain_summary(targeted)
    control_gain = gain_summary(control)
    targeted_rank = targeted_table["candidate_rank"].to_numpy(np.int64)
    control_rank = control_table["candidate_rank"].to_numpy(np.int64)
    formulas = targeted_table["query_formula"].astype(str).to_numpy()
    delta = (targeted_rank == 1).astype(float) - (control_rank == 1).astype(float)
    targeted_vs_control_pp = float(100.0 * np.mean(delta))
    causal_ci = formula_cluster_ci(
        formulas, delta, repeats=args.bootstrap_resamples,
        seed=args.seed, hypotheses=1,
    )
    corrected = int(np.sum((targeted_rank == 1) & (control_rank != 1)))
    introduced = int(np.sum((targeted_rank != 1) & (control_rank == 1)))
    targeted_directional = {
        **directional_checks(
            targeted["candidate"], targeted["official"], "targeted_vs_official"
        ),
        **directional_checks(
            targeted["candidate"], targeted["mature_e8"], "targeted_vs_mature_e8"
        ),
    }
    auc_comparisons = {
        "targeted_vs_official": auc_metric_comparison(
            targeted["candidate"], targeted["official"],
        ),
        "targeted_vs_mature_e8": auc_metric_comparison(
            targeted["candidate"], targeted["mature_e8"],
        ),
        "targeted_vs_control": auc_metric_comparison(
            targeted["candidate"], control["candidate"],
        ),
    }
    gates = {
        "targeted_gain_vs_official_at_least_registered_pp": (
            targeted_gain["delta_recall1_vs_official_pp"]
            >= args.minimum_delta_recall1_pp
        ),
        "targeted_gain_vs_mature_e8_at_least_registered_pp": (
            targeted_gain["delta_recall1_vs_mature_e8_pp"]
            >= args.minimum_delta_recall1_pp
        ),
        "targeted_formula_cluster_cis_vs_baselines_strict_positive": all(
            float(panel["recall1"]["ci_low_pp"]) > 0
            for panel in (
                targeted["formula_cluster_paired_ci"],
                targeted["mature_e8_formula_cluster_paired_ci"],
            )
        ),
        "targeted_risk_net_lambda2_vs_baselines_positive": all(
            int(panel["risk_net_lambda2"]) > 0
            for panel in (
                targeted["candidate_vs_official"],
                targeted["candidate_vs_mature_e8"],
            )
        ),
        "targeted_beats_same_query_control": targeted_vs_control_pp > 0,
        "targeted_vs_control_formula_ci_strict_positive": (
            float(causal_ci["ci_low_pp"]) > 0
        ),
        "targeted_vs_control_corrected_exceeds_introduced": corrected > introduced,
    }
    report = {
        "status": (
            "NOISE_DREAMS_HARD_POSITIVE_PROMOTION_PASS"
            if all(gates.values()) else "NOISE_DREAMS_HARD_POSITIVE_PROMOTION_FAIL"
        ),
        "design": (
            "native DreaMS hard-positive-only continuation; targeted and "
            "registered same-query control share q/p/n, schedule, seed and "
            "optimizer dose; clean-to-action training is forbidden"
        ),
        "targeted": targeted_gain,
        "control": control_gain,
        "targeted_vs_control_recall1_pp": targeted_vs_control_pp,
        "targeted_vs_control_formula_cluster_ci": causal_ci,
        "targeted_vs_control_corrected": corrected,
        "targeted_vs_control_introduced": introduced,
        "auc_metric_comparisons": auc_comparisons,
        "targeted_directional_metric_checks": targeted_directional,
        "minimum_delta_recall1_pp": args.minimum_delta_recall1_pp,
        "gates": gates,
        "claim_limit": (
            "Corrected MassSpecGym formula-held internal evidence; no external-dataset claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
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
