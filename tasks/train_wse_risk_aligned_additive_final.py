#!/usr/bin/env python
"""One fixed risk-aligned final development experiment for additive model C.

True chemistry and a frozen candidate-mismatch chemistry control use identical
architecture, initialization, train-fold scaler, residual budget, optimizer,
and query weights.  WSE-correct training queries receive weight two and WSE-
incorrect queries weight one, exactly matching corrected-2*introduced utility.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from conditional_null_energy_core import conditional_null_loss, strict_top1
from train_grand_unified_evidence_model import load_bundle
from train_unified_candidate_energy_crossfit import candidate_indices, formula_fold, metric_summary
from train_wse_additive_mechanism_control import capacity_matched_model, parameter_count
from train_wse_conditional_residual import (
    SPECTRAL_FEATURE_NAMES, build_panel, evaluation_scores, fit_scaler, metric_input, paired_comparison,
    sha256, training_residual_bound, transform_features,
)


def variant_features(panel: dict, variant: str) -> tuple[np.ndarray, np.ndarray]:
    if variant == "true_chemistry":
        return panel["features"], panel["null_features"]
    if variant == "candidate_mismatch":
        # Frozen truth-blind mismatch becomes primary.  True chemistry replaces
        # it in the null set, preserving arm count and all non-chemical inputs.
        return panel["null_features"][2], np.stack((
            panel["null_features"][0], panel["null_features"][1], panel["features"],
        ))
    raise ValueError(variant)


def fit_model(panel: dict, train_queries: np.ndarray, variant: str, recipe: dict, seed: int, device: str):
    index, packed_ptr = candidate_indices(panel["query_ptr"], train_queries)
    # Both variants use the scaler fitted from true chemistry on the same
    # training fold, so preprocessing cannot explain their difference.
    mean, scale = fit_scaler(panel["features"], index, panel["chemical_available"])
    actual_raw, null_raw = variant_features(panel, variant)
    actual = transform_features(actual_raw, mean, scale)
    nulls = np.stack([transform_features(block, mean, scale) for block in null_raw])
    bound, capacity = training_residual_bound(panel["base"], panel["query_ptr"], train_queries)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model, chemical_hidden, target_parameters = capacity_matched_model(
        panel["features"].shape[1], int(recipe["hidden"]), bound,
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(recipe["learning_rate"]), weight_decay=1e-4,
    )
    base = torch.as_tensor(panel["base"][index], dtype=torch.float32, device=device)
    features = torch.as_tensor(actual[index], dtype=torch.float32, device=device)
    null_features = torch.as_tensor(nulls[:, index], dtype=torch.float32, device=device)
    labels = torch.as_tensor(panel["labels"][index], dtype=torch.int8, device=device)
    wse_correct = strict_top1(panel["base"][index], panel["labels"][index], packed_ptr).astype(bool)
    weights_np = np.where(wse_correct, 2.0, 1.0).astype(np.float32)
    weights = torch.as_tensor(weights_np, dtype=torch.float32, device=device)
    final = None
    for _ in range(int(recipe["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        loss = conditional_null_loss(
            model=model, base_score=base, actual_features=features,
            null_features=null_features, labels=labels, query_ptr=packed_ptr,
            null_margin=0.08 * bound, null_weight=float(recipe["null_weight"]),
            trust_weight=float(recipe["trust_weight"]), query_weights=weights,
        )
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        final = {
            "total": float(loss.total.detach().cpu()),
            "listwise": float(loss.listwise.detach().cpu()),
            "worst_null": float(loss.worst_null.detach().cpu()),
            "trust": float(loss.trust.detach().cpu()),
        }
    return model, mean, scale, bound, capacity, chemical_hidden, target_parameters, final, {
        "weight_2_wse_correct_queries": int(wse_correct.sum()),
        "weight_1_wse_error_queries": int((~wse_correct).sum()),
    }


@torch.inference_mode()
def predict(model, panel: dict, queries: np.ndarray, variant: str, mean: np.ndarray, scale: np.ndarray, device: str):
    index, _ = candidate_indices(panel["query_ptr"], queries)
    actual, _ = variant_features(panel, variant)
    transformed = transform_features(actual, mean, scale)
    base = torch.as_tensor(panel["base"][index], dtype=torch.float32, device=device)
    features = torch.as_tensor(transformed[index], dtype=torch.float32, device=device)
    return model(base, features).cpu().numpy().astype(np.float32)


def load_parent_scores(panels: dict, ab_predictions: Path, c_predictions: Path) -> dict:
    output = {}
    with np.load(ab_predictions, allow_pickle=False) as ab, np.load(c_predictions, allow_pickle=False) as control:
        for name, panel in panels.items():
            prefix = f"{name}__"
            for key, expected in (("query_ptr", panel["query_ptr"]), ("labels", panel["labels"]),
                                  ("query_ids", panel["query_ids"]), ("candidate_ids", panel["candidate_ids"])):
                if not np.array_equal(np.asarray(ab[prefix + key]).astype(str), np.asarray(expected).astype(str)):
                    raise RuntimeError(f"parent A/B axis mismatch: {name}/{key}")
            output[name] = {
                method: np.asarray(ab[prefix + method], dtype=np.float32)
                for method in ("wse", "old_four_module", "A", "B")
            }
            output[name]["C_original"] = np.asarray(
                control[prefix + "C_additive_no_interaction"], dtype=np.float32,
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-bundle", type=Path, required=True)
    parser.add_argument("--formula-bundle", type=Path, required=True)
    parser.add_argument("--identity-chem", type=Path, required=True)
    parser.add_argument("--formula-chem", type=Path, required=True)
    parser.add_argument("--ab-report", type=Path, required=True)
    parser.add_argument("--ab-predictions", type=Path, required=True)
    parser.add_argument("--c-report", type=Path, required=True)
    parser.add_argument("--c-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    ab_report = json.loads(args.ab_report.read_text(encoding="utf-8"))
    c_report = json.loads(args.c_report.read_text(encoding="utf-8"))
    if ab_report.get("status") != "WSE_CONDITIONAL_RESIDUAL_DEVELOPMENT_COMPLETE":
        raise RuntimeError("A/B parent report incomplete")
    if c_report.get("mechanism_decision") != "CONDITIONAL_INTERACTION_NOT_SUPPORTED":
        raise RuntimeError("additive C was not established as the simpler architecture")
    recipe = ab_report["optimization"]
    folds, seed = int(ab_report["formula_crossfit_folds"]), int(recipe["seed"])
    bundles = {"identity": load_bundle(args.identity_bundle), "formula": load_bundle(args.formula_bundle)}
    panels = {
        "identity": build_panel(bundles["identity"], args.identity_chem),
        "formula": build_panel(bundles["formula"], args.formula_chem),
    }
    parent_scores = load_parent_scores(panels, args.ab_predictions, args.c_predictions)
    identity_fold = np.asarray([formula_fold(value, folds, seed) for value in panels["identity"]["formulas"]])
    target_fold = {
        name: np.asarray([formula_fold(value, folds, seed) for value in panel["formulas"]])
        for name, panel in panels.items()
    }
    variants = ("true_chemistry", "candidate_mismatch")
    outputs = {
        name: {variant: np.full(len(panel["base"]), np.nan, dtype=np.float32) for variant in variants}
        for name, panel in panels.items()
    }
    fold_reports = []
    for fold in range(folds):
        train_queries = np.flatnonzero(identity_fold != fold)
        fold_record = {"fold": fold, "train_queries": int(len(train_queries)), "variants": {}}
        for variant in variants:
            model, mean, scale, bound, capacity, chem_hidden, parameters, final, weight_counts = fit_model(
                panels["identity"], train_queries, variant, recipe, seed + fold, args.device,
            )
            for panel_name, panel in panels.items():
                queries = np.flatnonzero(target_fold[panel_name] == fold)
                index, _ = candidate_indices(panel["query_ptr"], queries)
                outputs[panel_name][variant][index] = predict(
                    model, panel, queries, variant, mean, scale, args.device,
                )
            fold_record["variants"][variant] = {
                "residual_bound_raw_wse_units": bound, "capacity": capacity,
                "parameter_count": parameter_count(model), "expected_parameter_count": parameters,
                "chemical_hidden": chem_hidden, "final_loss": final, **weight_counts,
            }
        fold_reports.append(fold_record)
    if any(np.any(~np.isfinite(value)) for panel in outputs.values() for value in panel.values()):
        raise RuntimeError("risk-aligned OOF prediction ledger incomplete")

    report = {
        "status": "WSE_RISK_ALIGNED_ADDITIVE_FINAL_DEVELOPMENT_COMPLETE",
        "dataset_role": "development_consumed", "enveda_accessed": False,
        "architecture": "capacity-matched additive C; no spectral-chemical hidden interaction",
        "risk_alignment": "training query weight 2 for WSE-correct and 1 for WSE-error",
        "risk_utility": "corrected_minus_2introduced", "hyperparameter_scan": False,
        "chemical_training_control": "frozen truth-blind candidate mismatch with identical preprocessing and initialization",
        "optimization_replayed_from_parent": recipe, "folds": fold_reports, "panels": {},
    }
    risk_pass = True
    chemistry_pass = True
    for panel_index, (panel_name, panel) in enumerate(panels.items()):
        methods = dict(parent_scores[panel_name])
        methods["F_true_risk_aligned"] = outputs[panel_name]["true_chemistry"]
        methods["F_mismatch_risk_aligned"] = outputs[panel_name]["candidate_mismatch"]
        metrics = {
            method: metric_summary(evaluation_scores(score, panel["query_ptr"]), metric_input(panel))[0]
            for method, score in methods.items()
        }
        comparisons = {}
        pairs = (
            ("F_true_risk_aligned", "wse"), ("F_true_risk_aligned", "old_four_module"),
            ("F_true_risk_aligned", "C_original"),
            ("F_true_risk_aligned", "F_mismatch_risk_aligned"),
        )
        for offset, (candidate, baseline) in enumerate(pairs):
            comparisons[f"{candidate}_vs_{baseline}"] = paired_comparison(
                methods[candidate], methods[baseline], panel,
                seed + 80000 + panel_index * 10 + offset, 10000,
            )
        vs_wse = comparisons["F_true_risk_aligned_vs_wse"]
        vs_old = comparisons["F_true_risk_aligned_vs_old_four_module"]
        vs_mismatch = comparisons["F_true_risk_aligned_vs_F_mismatch_risk_aligned"]
        risk_pass &= all(
            block["delta_recall@1"] > 0
            and block["formula_cluster_bootstrap_95ci"][0] > 0
            and block["risk_net_corrected_minus_2introduced"] > 0
            and block.get("near", {}).get("delta_recall@1", 0.0) >= 0
            for block in (vs_wse, vs_old)
        )
        chemistry_pass &= (
            vs_mismatch["delta_recall@1"] > 0
            and vs_mismatch["formula_cluster_bootstrap_95ci"][0] > 0
        )
        report["panels"][panel_name] = {"methods": metrics, "comparisons": comparisons}
    report["final_decision"] = (
        "FREEZE_F_TRUE_FOR_NEW_EXTERNAL_VALIDATION" if risk_pass and chemistry_pass
        else "DO_NOT_FREEZE_RISK_ALIGNED_MODEL"
    )
    report["risk_gate_passed_both_panels"] = bool(risk_pass)
    report["true_chemistry_training_control_passed_both_panels"] = bool(chemistry_pass)
    report["external_boundary"] = "Enveda-180 is consumed; only a genuinely new external dataset is admissible"
    args.output.mkdir(parents=True)
    if risk_pass and chemistry_pass:
        all_queries = np.arange(len(panels["identity"]["query_ids"]), dtype=np.int64)
        model, mean, scale, bound, capacity, chem_hidden, parameters, final, weight_counts = fit_model(
            panels["identity"], all_queries, "true_chemistry", recipe, seed + 1000, args.device,
        )
        model_path = args.output / "frozen_final_model.pt"
        torch.save({
            "state_dict": model.cpu().state_dict(),
            "model_class": "AdditiveBoundedResidualEnergy",
            "architecture": "WSE anchored additive spectral plus chemical residual",
            "spectral_feature_count": len(SPECTRAL_FEATURE_NAMES),
            "spectral_hidden": int(recipe["hidden"]),
            "chemical_hidden": chem_hidden, "parameter_count": parameters,
            "residual_bound_raw_wse_units": bound,
            "scaler_mean": mean, "scaler_scale": scale,
            "risk_query_weights": {"wse_correct": 2.0, "wse_error": 1.0},
            "training_seed": seed + 1000, "training_queries": int(len(all_queries)),
            "enveda_accessed": False,
        }, model_path)
        report["frozen_final_model"] = {
            "path": str(model_path), "sha256": sha256(model_path),
            "fit_scope": "all GNPS identity development queries after OOF selection",
            "capacity": capacity, "final_loss": final, **weight_counts,
        }
    else:
        report["frozen_final_model"] = None
    prediction_path = args.output / "risk_aligned_predictions.npz"
    np.savez_compressed(prediction_path, **{
        f"{panel}__{variant}": value
        for panel, variants_map in outputs.items() for variant, value in variants_map.items()
    })
    report["artifacts"] = {"predictions": str(prediction_path), "predictions_sha256": sha256(prediction_path)}
    report["input_sha256"] = {
        "ab_report": sha256(args.ab_report), "ab_predictions": sha256(args.ab_predictions),
        "c_report": sha256(args.c_report), "c_predictions": sha256(args.c_predictions),
        "identity_bundle": sha256(args.identity_bundle), "formula_bundle": sha256(args.formula_bundle),
        "identity_chem": sha256(args.identity_chem), "formula_chem": sha256(args.formula_chem),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "decision": report["final_decision"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
