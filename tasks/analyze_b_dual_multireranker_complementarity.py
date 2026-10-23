#!/usr/bin/env python
"""Diagnose information, observability and harm in the B integration run.

This is a truth-aware development audit, not a deployable router.  It never
fits a model.  Oracle unions and label-conditioned summaries quantify whether
complementarity exists; they must not be reported as attainable performance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from grand_unified_evidence_core import strict_ranks
from train_unified_candidate_energy_crossfit import formula_fold
from train_wse_conditional_residual import sha256


RAW = {
    "stage1_raw": "chemaware_stage1_encoder",
    "phasea_raw": "chemaware_phase_a_encoder",
    "msg_raw": "noise_msg_fusion_stack_v1",
    "rrf_raw": "noise_rrf_1_7",
    "p2b_raw": "p2b_noise_v1_frozen",
}
ARMS = (
    "stage1_only", "phasea_only", "dual_encoder", "complete_spectral",
    "complete_spectral_additive", "audit_plus_rrf", "complete_without_v2_refit",
)


def top_indices(scores: np.ndarray, ptr: np.ndarray) -> np.ndarray:
    return np.asarray([
        int(left) + int(np.argmax(scores[int(left):int(right)]))
        for left, right in zip(ptr[:-1], ptr[1:])
    ], dtype=np.int64)


def margins(scores: np.ndarray, ptr: np.ndarray) -> np.ndarray:
    result = np.empty(len(ptr) - 1, dtype=np.float64)
    for query, (left, right) in enumerate(zip(ptr[:-1], ptr[1:])):
        block = np.sort(np.asarray(scores[int(left):int(right)], dtype=np.float64))
        result[query] = block[-1] - block[-2]
    return result


def quantiles(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"n": 0, "q25": None, "median": None, "q75": None}
    return {
        "n": int(len(values)), "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)), "q75": float(np.quantile(values, 0.75)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--identity-bundle", type=Path, required=True)
    parser.add_argument("--formula-bundle", type=Path, required=True)
    parser.add_argument("--model-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    model_report = json.loads(args.model_report.read_text(encoding="utf-8"))
    seed = int(model_report["optimization_replayed_from_B_parent"]["seed"])
    folds = len(model_report["folds"])
    result = {
        "status": "B_DUAL_MULTIRERANKER_COMPLEMENTARITY_AUDIT_COMPLETE",
        "dataset": "GNPS Gold/Silver 10-ppm",
        "dataset_role": "development_consumed",
        "truth_aware_diagnostic_only": True,
        "deployable_router_or_performance_claim": False,
        "enveda_accessed": False,
        "panels": {},
    }
    with np.load(args.predictions, allow_pickle=False) as predictions:
        for panel_name, bundle_path in (
            ("identity", args.identity_bundle), ("formula", args.formula_bundle),
        ):
            with np.load(bundle_path, allow_pickle=False) as bundle:
                ptr = np.asarray(bundle["query_ptr"], dtype=np.int64)
                labels = np.asarray(bundle["labels"], dtype=np.int8)
                names = tuple(bundle["module_names"].astype("U"))
                scores = {"B": np.asarray(predictions[f"{panel_name}__B"], dtype=np.float32)}
                scores.update({
                    arm: np.asarray(predictions[f"{panel_name}__{arm}"], dtype=np.float32)
                    for arm in ARMS
                })
                scores.update({
                    short: np.asarray(bundle["scores"][names.index(full)], dtype=np.float32)
                    for short, full in RAW.items()
                })
                ranks = {name: strict_ranks(value, ptr, labels) for name, value in scores.items()}
                tops = {name: top_indices(value, ptr) for name, value in scores.items()}
                margin = {name: margins(value, ptr) for name, value in scores.items()}
                b_correct = ranks["B"] == 1
                panel = {
                    "queries": int(len(b_correct)), "B_correct": int(b_correct.sum()),
                    "B_errors": int((~b_correct).sum()), "raw_experts": {},
                    "integrated_arms": {},
                }
                correction_masks = {}
                for name in ("stage1_raw", "phasea_raw", "msg_raw", "rrf_raw"):
                    corrected = (~b_correct) & (ranks[name] == 1)
                    harms = b_correct & (ranks[name] != 1)
                    decisive = corrected | harms
                    correction_masks[name] = corrected
                    observable_auc = None
                    inverse_b_margin_auc = None
                    expert_margin_auc = None
                    if corrected.any() and harms.any():
                        outcome = corrected[decisive].astype(np.int8)
                        observable_auc = float(roc_auc_score(
                            outcome, (margin[name] - margin["B"])[decisive],
                        ))
                        inverse_b_margin_auc = float(roc_auc_score(
                            outcome, -margin["B"][decisive],
                        ))
                        expert_margin_auc = float(roc_auc_score(
                            outcome, margin[name][decisive],
                        ))
                    decisions = int(corrected.sum() + harms.sum())
                    panel["raw_experts"][name] = {
                        "recall@1": float(np.mean(ranks[name] == 1)),
                        "corrected_B_errors": int(corrected.sum()),
                        "misleads_B_correct": int(harms.sum()),
                        "truth_oracle_correction_precision": (
                            float(corrected.sum() / decisions) if decisions else None
                        ),
                        "top1_disagreement_queries": int(np.sum(tops[name] != tops["B"])),
                        "margin_advantage_auroc_correction_vs_harm": observable_auc,
                        "negative_B_margin_auroc_correction_vs_harm": inverse_b_margin_auc,
                        "expert_margin_auroc_correction_vs_harm": expert_margin_auc,
                        "B_margin_when_corrects": quantiles(margin["B"][corrected]),
                        "B_margin_when_harms": quantiles(margin["B"][harms]),
                        "expert_margin_when_corrects": quantiles(margin[name][corrected]),
                        "expert_margin_when_harms": quantiles(margin[name][harms]),
                    }
                encoder_union = correction_masks["stage1_raw"] | correction_masks["phasea_raw"]
                all_union = np.logical_or.reduce(tuple(correction_masks.values()))
                panel["truth_oracle_complementarity"] = {
                    "stage1_or_phasea_correctable_B_errors": int(encoder_union.sum()),
                    "stage1_phasea_overlap": int(np.sum(
                        correction_masks["stage1_raw"] & correction_masks["phasea_raw"]
                    )),
                    "stage1_unique_vs_phasea": int(np.sum(
                        correction_masks["stage1_raw"] & ~correction_masks["phasea_raw"]
                    )),
                    "phasea_unique_vs_stage1": int(np.sum(
                        correction_masks["phasea_raw"] & ~correction_masks["stage1_raw"]
                    )),
                    "msg_unique_beyond_both_encoders": int(np.sum(
                        correction_masks["msg_raw"] & ~encoder_union
                    )),
                    "rrf_unique_beyond_encoders_and_msg": int(np.sum(
                        correction_masks["rrf_raw"]
                        & ~(encoder_union | correction_masks["msg_raw"])
                    )),
                    "any_raw_expert_correctable_B_errors": int(all_union.sum()),
                    "unattainable_oracle_recall@1": float(np.mean(b_correct | all_union)),
                    "warning": "uses truth to choose experts; diagnostic ceiling only",
                }
                for name in ARMS:
                    corrected = (~b_correct) & (ranks[name] == 1)
                    introduced = b_correct & (ranks[name] != 1)
                    panel["integrated_arms"][name] = {
                        "recall@1": float(np.mean(ranks[name] == 1)),
                        "corrected": int(corrected.sum()), "introduced": int(introduced.sum()),
                        "net": int(corrected.sum() - introduced.sum()),
                        "risk_net": int(corrected.sum() - 2 * introduced.sum()),
                        "top1_changed_queries": int(np.sum(tops[name] != tops["B"])),
                        "B_margin_corrected": quantiles(margin["B"][corrected]),
                        "B_margin_introduced": quantiles(margin["B"][introduced]),
                    }
                formulas = predictions[f"{panel_name}__formula"].astype("U")
                query_fold = np.asarray([formula_fold(value, folds, seed) for value in formulas])
                query_bound = np.empty(len(query_fold), dtype=np.float64)
                for fold_report in model_report["folds"]:
                    fold = int(fold_report["fold"])
                    query_bound[query_fold == fold] = float(
                        fold_report["arms"]["complete_spectral"]["residual_bound_B_units"]
                    )
                candidate_bound = np.repeat(query_bound, np.diff(ptr))
                residual = scores["complete_spectral"] - scores["B"]
                ratio = np.abs(residual) / candidate_bound
                panel["complete_residual"] = {
                    "absolute_bound_ratio": quantiles(ratio),
                    "candidate_fraction_at_least_95pct_bound": float(np.mean(ratio >= 0.95)),
                    "query_fraction_any_candidate_at_least_95pct_bound": float(np.mean([
                        np.any(ratio[int(left):int(right)] >= 0.95)
                        for left, right in zip(ptr[:-1], ptr[1:])
                    ])),
                    "candidate_fraction_exact_zero": float(np.mean(residual == 0)),
                }
                panel["score_scale"] = {}
                for name in ("B", "p2b_raw", "stage1_raw", "phasea_raw", "msg_raw", "rrf_raw"):
                    panel["score_scale"][name] = {
                        "candidate_score": quantiles(scores[name]),
                        "top1_top2_margin": quantiles(margin[name]),
                        "top1_agreement_with_B": float(np.mean(tops[name] == tops["B"])),
                    }
                result["panels"][panel_name] = panel
    result["input_sha256"] = {
        "predictions": sha256(args.predictions),
        "identity_bundle": sha256(args.identity_bundle),
        "formula_bundle": sha256(args.formula_bundle),
        "model_report": sha256(args.model_report),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
