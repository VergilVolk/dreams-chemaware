#!/usr/bin/env python
"""Join internal Stage-4 adjudication with the sealed GNPS transfer panels."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path


PANELS = ("identity_disjoint", "formula_disjoint")
COMPARISONS = (
    "stage1_vs_official",
    "targeted_vs_official",
    "targeted_vs_stage1",
    "targeted_vs_control",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal-summary", type=Path, required=True)
    parser.add_argument("--gnps-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--absolute-target-pp", type=float, default=5.0)
    parser.add_argument(
        "--result-prefix",
        choices=(
            "NOISE_DREAMS_NATIVE_STAGE4",
            "NOISE_DREAMS_NATIVE_STAGE5",
            "NOISE_DREAMS_NATIVE_STAGE6",
        ),
        default="NOISE_DREAMS_NATIVE_STAGE4",
    )
    return parser.parse_args()


def read_report(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def delta(candidate: dict, baseline: dict, *keys: str) -> float:
    for key in keys:
        candidate = candidate[key]
        baseline = baseline[key]
    return float(candidate) - float(baseline)


def panel_deltas(panel: dict) -> dict[str, float]:
    candidate = panel["candidate"]
    baseline = panel["baseline"]
    return {
        "recall@1": delta(candidate, baseline, "retrieval", "recall@1"),
        "mrr": delta(candidate, baseline, "retrieval", "mrr"),
        "macro_query_auroc": delta(
            candidate, baseline, "retrieval", "macro_query_auroc",
        ),
        "macro_query_auprc": delta(
            candidate, baseline, "retrieval", "macro_query_auprc",
        ),
        "near_recall@1": delta(candidate, baseline, "near_subset", "recall@1"),
        "near_mrr": delta(candidate, baseline, "near_subset", "mrr"),
        "near_macro_query_auroc": delta(
            candidate, baseline, "near_subset", "macro_query_auroc",
        ),
        "near_macro_query_auprc": delta(
            candidate, baseline, "near_subset", "macro_query_auprc",
        ),
        "micro_candidate_auroc": delta(
            candidate, baseline, "micro_candidate", "auroc",
        ),
        "micro_candidate_auprc": delta(
            candidate, baseline, "micro_candidate", "auprc",
        ),
        "pooled_pairwise_auroc": delta(
            candidate, baseline, "gnps_10ppm_pooled_pairwise", "auroc",
        ),
        "pooled_pairwise_auprc": delta(
            candidate, baseline, "gnps_10ppm_pooled_pairwise", "auprc",
        ),
    }


def comparison_summary(report: dict) -> dict:
    if report.get("status") != "gnps_gold_silver_10ppm_embedding_evaluation_complete":
        raise RuntimeError("GNPS comparison is incomplete")
    output: dict[str, dict] = {}
    for name in PANELS:
        panel = report["panels"][name]
        deltas = panel_deltas(panel)
        paired = panel["paired"]
        ci = paired["formula_cluster_paired_ci"]
        near_ci = paired["near_formula_cluster_paired_ci"]
        output[name] = {
            "deltas": deltas,
            "delta_pp": {key: 100.0 * value for key, value in deltas.items()},
            "all_registered_metrics_improve": all(value > 0.0 for value in deltas.values()),
            "recall1_formula_ci_low_pp": float(ci["recall@1"]["ci_low_pp"]),
            "mrr_formula_ci_low_pp": float(ci["mrr"]["ci_low_pp"]),
            "macro_auroc_formula_ci_low_pp": float(
                ci["macro_query_auroc"]["ci_low_pp"]
            ),
            "macro_auprc_formula_ci_low_pp": float(
                ci["macro_query_auprc"]["ci_low_pp"]
            ),
            "near_recall1_formula_ci_low_pp": float(
                near_ci["recall@1"]["ci_low_pp"]
            ),
            "near_mrr_formula_ci_low_pp": float(near_ci["mrr"]["ci_low_pp"]),
            "near_macro_auroc_formula_ci_low_pp": float(
                near_ci["macro_query_auroc"]["ci_low_pp"]
            ),
            "near_macro_auprc_formula_ci_low_pp": float(
                near_ci["macro_query_auprc"]["ci_low_pp"]
            ),
            "all_registered_formula_cis_positive": all(
                float(source[metric]["ci_low_pp"]) > 0.0
                for source in (ci, near_ci)
                for metric in (
                    "recall@1", "mrr", "macro_query_auroc", "macro_query_auprc",
                )
            ),
            "risk_net_lambda2": int(paired["risk_net_lambda2"]),
            "near_risk_net_lambda2": int(paired["near_risk_net_lambda2"]),
        }
    return output


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    internal = read_report(args.internal_summary)
    comparisons = {
        name: comparison_summary(read_report(args.gnps_root / name / "report.json"))
        for name in COMPARISONS
    }
    gates: dict[str, bool] = {
        "internal_massspecgym_promotion_pass": str(internal.get("status", "")).endswith(
            "_PROMOTION_PASS"
        ),
    }
    for comparison in ("targeted_vs_stage1", "targeted_vs_control"):
        for panel in PANELS:
            body = comparisons[comparison][panel]
            gates[f"{comparison}_{panel}_all_metrics_improve"] = bool(
                body["all_registered_metrics_improve"]
            )
            gates[f"{comparison}_{panel}_all_formula_cis_positive"] = bool(
                body["all_registered_formula_cis_positive"]
            )
            gates[f"{comparison}_{panel}_risk_net_positive"] = (
                int(body["risk_net_lambda2"]) > 0
            )
            gates[f"{comparison}_{panel}_near_risk_net_nonnegative"] = (
                int(body["near_risk_net_lambda2"]) >= 0
            )
    for panel in PANELS:
        recall_gain_pp = comparisons["targeted_vs_official"][panel]["delta_pp"][
            "recall@1"
        ]
        gates[f"targeted_vs_official_{panel}_recall1_reaches_absolute_target"] = (
            recall_gain_pp >= args.absolute_target_pp
        )
    passed = all(gates.values())
    report = {
        "status": (
            f"{args.result_prefix}_INTERNAL_AND_GNPS_PROMOTION_PASS"
            if passed else f"{args.result_prefix}_RETAIN_STAGE1"
        ),
        "absolute_target_pp": args.absolute_target_pp,
        "internal_summary": internal,
        "gnps_gold_silver": comparisons,
        "gates": gates,
        "selected_checkpoint": (
            f"{args.result_prefix.rsplit('_', 1)[-1].lower()}_targeted"
            if passed else "stage1_targeted"
        ),
        "claim_limit": (
            "MassSpecGym internal held graph plus GNPS Gold/Silver identity- and "
            "formula-disjoint transfer; not an exact NIST20 replication."
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
