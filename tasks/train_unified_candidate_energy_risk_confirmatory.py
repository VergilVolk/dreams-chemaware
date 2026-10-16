#!/usr/bin/env python
"""One fixed risk-aligned GNPS confirmation before any Enveda opening.

This is a single successor to the exploratory candidate-energy ablation.  It
uses the already selected, non-duplicated evidence families and a fixed loss
ratio matching the preregistered corrected - 2*introduced utility.  It does
not scan modules, penalties, thresholds, seeds, or Enveda results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_grand_unified_evidence_model import load_bundle
from train_unified_candidate_energy_crossfit import (
    BASE,
    clustered_ci,
    compare,
    crossfit,
    fit_model,
    metric_summary,
    prepare,
    select_modules,
    sha256,
)


MODULES = (
    "noise_v1",
    "weighted_spectral_entropy",
    "p2b_noise_v1_frozen",
    "chemaware_v2_action",
)
BASELINE_CORRECT_PENALTY = 2.0
MAX_MICRO_METRIC_LOSS = 0.001


def near_transition(
    scores: np.ndarray,
    prepared: dict,
    base_scores: np.ndarray,
    seed: int,
    resamples: int,
) -> dict:
    _, candidate_rank = metric_summary(scores, prepared)
    _, baseline_rank = metric_summary(base_scores, prepared)
    near = np.asarray(prepared["near"], dtype=bool)
    if not np.any(near):
        return {"queries": 0, "evaluable": False}
    candidate_rank = candidate_rank[near]
    baseline_rank = baseline_rank[near]
    formulas = np.asarray(prepared["formulas"])[near]
    corrected = (candidate_rank == 1) & (baseline_rank != 1)
    introduced = (candidate_rank != 1) & (baseline_rank == 1)
    delta = (candidate_rank == 1).astype(float) - (baseline_rank == 1).astype(float)
    return {
        "queries": int(np.sum(near)),
        "evaluable": True,
        "delta_recall@1": float(np.mean(delta)),
        "corrected": int(np.sum(corrected)),
        "introduced": int(np.sum(introduced)),
        "risk_net_corrected_minus_2introduced": int(
            np.sum(corrected) - 2 * np.sum(introduced)
        ),
        "formula_cluster_bootstrap_95ci": clustered_ci(
            delta, formulas, seed, resamples,
        ),
    }


def qualification_gates(result: dict, near: dict) -> dict[str, bool]:
    paired = result["paired_vs_wse"]
    candidate = result["candidate"]
    baseline = result["baseline_wse"]
    return {
        "positive_recall1": paired["delta_recall@1"] > 0,
        "positive_formula_cluster_ci": paired["formula_cluster_bootstrap_95ci"][0] > 0,
        "positive_risk_net": paired["risk_net_corrected_minus_2introduced"] > 0,
        "mrr_nonnegative": candidate["mrr"] >= baseline["mrr"],
        "micro_auroc_not_materially_lower": (
            candidate["micro_candidate_auroc"]
            >= baseline["micro_candidate_auroc"] - MAX_MICRO_METRIC_LOSS
        ),
        "micro_auprc_not_materially_lower": (
            candidate["micro_candidate_auprc"]
            >= baseline["micro_candidate_auprc"] - MAX_MICRO_METRIC_LOSS
        ),
        "near_recall1_nonnegative": (
            bool(near.get("evaluable")) and near["delta_recall@1"] >= 0
        ),
        "near_risk_net_nonnegative": (
            bool(near.get("evaluable"))
            and near["risk_net_corrected_minus_2introduced"] >= 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-bundle", type=Path, required=True)
    parser.add_argument("--formula-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-queries", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--anchor-strength", type=float, default=0.01)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    for path in (args.identity_bundle, args.formula_bundle):
        if not path.is_file():
            raise FileNotFoundError(path)

    identity_raw = load_bundle(args.identity_bundle)
    formula_raw = load_bundle(args.formula_bundle)
    identity = prepare(select_modules(identity_raw, MODULES))
    formula = prepare(select_modules(formula_raw, MODULES))
    device = torch.device(args.device)
    predictions, fold_reports = crossfit(
        train=identity,
        targets={"identity": identity, "formula": formula},
        module_names=MODULES,
        folds=args.folds,
        epochs=args.epochs,
        batch_queries=args.batch_queries,
        learning_rate=args.learning_rate,
        anchor_strength=args.anchor_strength,
        seed=args.seed,
        device=device,
        baseline_correct_penalty=BASELINE_CORRECT_PENALTY,
    )

    panels = {}
    gates = {}
    base_index = MODULES.index(BASE)
    for offset, (panel, body) in enumerate((("identity", identity), ("formula", formula))):
        base_scores = body["evidence"][base_index]
        result = compare(
            predictions[panel], body, base_scores,
            args.seed + offset, args.bootstrap_resamples,
        )
        near = near_transition(
            predictions[panel], body, base_scores,
            args.seed + 100 + offset, args.bootstrap_resamples,
        )
        panel_gates = qualification_gates(result, near)
        panels[panel] = {**result, "near_paired_vs_wse": near, "gates": panel_gates}
        gates.update({f"{panel}_{key}": value for key, value in panel_gates.items()})

    eligible = bool(all(gates.values()))
    args.output.mkdir(parents=True)
    score_path = args.output / "crossfit_scores.npz"
    np.savez_compressed(
        score_path,
        scores_identity=predictions["identity"],
        scores_formula=predictions["formula"],
        module_names=np.asarray(MODULES),
        baseline_correct_penalty=np.asarray(BASELINE_CORRECT_PENALTY),
    )

    report = {
        "status": (
            "UNIFIED_RISK_ALIGNED_CONFIRMATORY_PASS"
            if eligible else "UNIFIED_RISK_ALIGNED_CONFIRMATORY_STOP"
        ),
        "dataset": "GNPS Gold/Silver 10-ppm development panels",
        "dataset_role": "development_consumed",
        "external_test_opened": False,
        "selection_history": (
            "single confirmatory successor after the exploratory arm omitted "
            "ChemAware stage-1; no arm, penalty, seed, or threshold scan"
        ),
        "module_names": list(MODULES),
        "objective": "WSE-anchored listwise loss with fixed 2x weight on WSE-correct queries",
        "baseline_correct_penalty": BASELINE_CORRECT_PENALTY,
        "formula_crossfit_folds": args.folds,
        "formula_panel_labels_used_for_fitting": False,
        "optimization": {
            "epochs": args.epochs,
            "batch_queries": args.batch_queries,
            "learning_rate": args.learning_rate,
            "anchor_strength": args.anchor_strength,
            "seed": args.seed,
        },
        "panels_are_independent_confirmations": False,
        "panels": panels,
        "gates": gates,
        "eligible_for_open_set_calibration": eligible,
        "eligible_for_enveda_opening": False,
        "enveda_blocker": (
            "A single GNPS-calibrated abstention threshold and its risk/coverage "
            "contract must still be frozen after ranker qualification."
        ),
        "folds": fold_reports,
        "input_sha256": {
            "identity": sha256(args.identity_bundle),
            "formula": sha256(args.formula_bundle),
            "crossfit_scores": sha256(score_path),
        },
    }

    if eligible:
        queries = np.arange(len(identity["query_ptr"]) - 1, dtype=np.int64)
        model = fit_model(
            identity, MODULES, queries,
            args.epochs, args.batch_queries, args.learning_rate,
            args.anchor_strength, args.seed + 1000, device,
            BASELINE_CORRECT_PENALTY,
        )
        model_path = args.output / "candidate_ranker.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "module_names": MODULES,
            "base_name": BASE,
            "model_class": "AnchoredCandidateEnergy",
            "baseline_correct_penalty": BASELINE_CORRECT_PENALTY,
            "training_seed": args.seed + 1000,
            "external_test_opened": False,
        }, model_path)
        report["candidate_ranker"] = str(model_path)
        report["candidate_ranker_sha256"] = sha256(model_path)
        report["final_mixture_weights"] = dict(zip(
            MODULES,
            map(float, model.mixture_weights().detach().cpu().numpy()),
            strict=True,
        ))

    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
