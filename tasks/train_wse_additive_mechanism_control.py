#!/usr/bin/env python
"""Single capacity-matched no-interaction control for the finished WSE A/B run.

The control has the same WSE anchor, inputs, formula folds, optimizer, epochs,
null loss, and total residual budget as B.  Its spectral and chemical heads
cannot share hidden units, so any B-over-C gain isolates conditional
spectral-chemical interaction rather than extra inputs or parameter count.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from conditional_null_energy_core import (
    AdditiveBoundedResidualEnergy,
    AnchoredEvidenceEnergy,
    conditional_null_loss,
)
from train_grand_unified_evidence_model import load_bundle
from train_unified_candidate_energy_crossfit import candidate_indices, formula_fold, metric_summary
from train_wse_conditional_residual import (
    SPECTRAL_FEATURE_NAMES,
    build_panel,
    evaluation_scores,
    fit_scaler,
    metric_input,
    paired_comparison,
    sha256,
    training_residual_bound,
    transform_features,
)


def parameter_count(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def capacity_matched_model(feature_count: int, hidden: int, bound: float) -> tuple[AdditiveBoundedResidualEnergy, int, int]:
    # Count B without consuming the initialization stream used by C.  This
    # leaves C's spectral head initialized exactly as B's spectral head under
    # the same fold seed.
    rng_state = torch.random.get_rng_state()
    reference = AnchoredEvidenceEnergy(
        feature_count, mode="joint", spectral_feature_count=len(SPECTRAL_FEATURE_NAMES),
        hidden=hidden, residual_bound=bound,
    )
    target = parameter_count(reference)
    torch.random.set_rng_state(rng_state)
    spectral_count = len(SPECTRAL_FEATURE_NAMES)
    chemical_count = feature_count - spectral_count
    spectral_parameters = (spectral_count + 2) * hidden + 1
    chemical_remainder = target - spectral_parameters - 1
    if chemical_remainder <= 0 or chemical_remainder % (chemical_count + 2):
        raise RuntimeError(f"no exact integer capacity match for B parameter count {target}")
    chemical_hidden = chemical_remainder // (chemical_count + 2)
    model = AdditiveBoundedResidualEnergy(
        feature_count, spectral_feature_count=spectral_count,
        hidden=hidden, chemical_hidden=chemical_hidden, residual_bound=bound,
    )
    if parameter_count(model) != target:
        raise RuntimeError("analytic capacity match drifted")
    return model, chemical_hidden, target


def fit_control(panel: dict, train_queries: np.ndarray, recipe: dict, seed: int, device: str):
    index, packed_ptr = candidate_indices(panel["query_ptr"], train_queries)
    mean, scale = fit_scaler(panel["features"], index, panel["chemical_available"])
    actual = transform_features(panel["features"], mean, scale)
    nulls = np.stack([transform_features(block, mean, scale) for block in panel["null_features"]])
    bound, capacity = training_residual_bound(panel["base"], panel["query_ptr"], train_queries)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model, chemical_hidden, parameters = capacity_matched_model(
        panel["features"].shape[1], int(recipe["hidden"]), bound,
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(recipe["learning_rate"]), weight_decay=1e-4,
    )
    base = torch.as_tensor(panel["base"][index], dtype=torch.float32, device=device)
    feature = torch.as_tensor(actual[index], dtype=torch.float32, device=device)
    null = torch.as_tensor(nulls[:, index], dtype=torch.float32, device=device)
    labels = torch.as_tensor(panel["labels"][index], dtype=torch.int8, device=device)
    final_loss = None
    for _ in range(int(recipe["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        loss = conditional_null_loss(
            model=model, base_score=base, actual_features=feature,
            null_features=null, labels=labels, query_ptr=packed_ptr,
            null_margin=0.08 * bound, null_weight=float(recipe["null_weight"]),
            trust_weight=float(recipe["trust_weight"]),
        )
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        final_loss = float(loss.total.detach().cpu())
    return model, mean, scale, bound, capacity, chemical_hidden, parameters, final_loss


@torch.inference_mode()
def predict(model, panel: dict, queries: np.ndarray, mean: np.ndarray, scale: np.ndarray, device: str) -> np.ndarray:
    index, _ = candidate_indices(panel["query_ptr"], queries)
    transformed = transform_features(panel["features"], mean, scale)
    base = torch.as_tensor(panel["base"][index], dtype=torch.float32, device=device)
    feature = torch.as_tensor(transformed[index], dtype=torch.float32, device=device)
    return model(base, feature).cpu().numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-bundle", type=Path, required=True)
    parser.add_argument("--formula-bundle", type=Path, required=True)
    parser.add_argument("--identity-chem", type=Path, required=True)
    parser.add_argument("--formula-chem", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--parent-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    parent = json.loads(args.parent_report.read_text(encoding="utf-8"))
    if parent.get("status") != "WSE_CONDITIONAL_RESIDUAL_DEVELOPMENT_COMPLETE":
        raise RuntimeError("parent A/B report is not complete")
    recipe = parent["optimization"]
    if bool(recipe.get("hyperparameter_scan")):
        raise RuntimeError("parent recipe unexpectedly used a hyperparameter scan")
    folds, seed = int(parent["formula_crossfit_folds"]), int(recipe["seed"])
    bundles = {
        "identity": load_bundle(args.identity_bundle),
        "formula": load_bundle(args.formula_bundle),
    }
    panels = {
        "identity": build_panel(bundles["identity"], args.identity_chem),
        "formula": build_panel(bundles["formula"], args.formula_chem),
    }
    parent_scores = {}
    with np.load(args.parent_predictions, allow_pickle=False) as source:
        for panel_name, panel in panels.items():
            prefix = f"{panel_name}__"
            for key, expected in (("query_ptr", panel["query_ptr"]), ("labels", panel["labels"]),
                                  ("query_ids", panel["query_ids"]), ("candidate_ids", panel["candidate_ids"])):
                if not np.array_equal(np.asarray(source[prefix + key]).astype(str), np.asarray(expected).astype(str)):
                    raise RuntimeError(f"parent prediction axis mismatch: {panel_name}/{key}")
            parent_scores[panel_name] = {
                method: np.asarray(source[prefix + method], dtype=np.float32)
                for method in ("wse", "old_four_module", "A", "B")
            }

    identity_fold = np.asarray([formula_fold(value, folds, seed) for value in panels["identity"]["formulas"]])
    target_fold = {
        name: np.asarray([formula_fold(value, folds, seed) for value in panel["formulas"]])
        for name, panel in panels.items()
    }
    control_scores = {
        name: np.full(len(panel["base"]), np.nan, dtype=np.float32)
        for name, panel in panels.items()
    }
    fold_reports = []
    for fold in range(folds):
        train_queries = np.flatnonzero(identity_fold != fold)
        model, mean, scale, bound, capacity, chemical_hidden, parameters, final_loss = fit_control(
            panels["identity"], train_queries, recipe, seed + fold, args.device,
        )
        predicted_queries = {}
        for panel_name, panel in panels.items():
            queries = np.flatnonzero(target_fold[panel_name] == fold)
            index, _ = candidate_indices(panel["query_ptr"], queries)
            control_scores[panel_name][index] = predict(model, panel, queries, mean, scale, args.device)
            predicted_queries[panel_name] = int(len(queries))
        fold_reports.append({
            "fold": fold, "train_queries": int(len(train_queries)),
            "residual_bound_raw_wse_units": bound, "capacity": capacity,
            "joint_B_parameter_count": parameters, "C_parameter_count": parameter_count(model),
            "spectral_hidden": int(recipe["hidden"]), "chemical_hidden": chemical_hidden,
            "final_loss": final_loss, "prediction_queries": predicted_queries,
        })
    if any(np.any(~np.isfinite(score)) for score in control_scores.values()):
        raise RuntimeError("control OOF prediction ledger is incomplete")
    if any(record["joint_B_parameter_count"] != record["C_parameter_count"] for record in fold_reports):
        raise RuntimeError("capacity matching failed")

    report = {
        "status": "WSE_ADDITIVE_MECHANISM_CONTROL_COMPLETE",
        "dataset": "GNPS Gold/Silver 10-ppm identity/formula panels",
        "dataset_role": "development_consumed",
        "enveda_accessed": False,
        "control": "C = independent spectral head + independent chemical head; no shared hidden interaction",
        "capacity_match": "exact trainable parameter count and identical total residual bound",
        "parent_decision_preserved": parent["decision"],
        "optimization_replayed_from_parent": recipe,
        "folds": fold_reports,
        "panels": {},
    }
    interaction_supported = True
    comparison_seed = seed + 50000
    for panel_index, (panel_name, panel) in enumerate(panels.items()):
        methods = dict(parent_scores[panel_name])
        methods["C_additive_no_interaction"] = control_scores[panel_name]
        metrics = {}
        for method, score in methods.items():
            metrics[method], _ = metric_summary(evaluation_scores(score, panel["query_ptr"]), metric_input(panel))
        comparisons = {}
        for offset, (candidate, baseline) in enumerate((
            ("B", "C_additive_no_interaction"),
            ("C_additive_no_interaction", "wse"),
            ("C_additive_no_interaction", "old_four_module"),
            ("C_additive_no_interaction", "A"),
        )):
            comparisons[f"{candidate}_vs_{baseline}"] = paired_comparison(
                methods[candidate], methods[baseline], panel,
                comparison_seed + panel_index * 10 + offset, 10000,
            )
        b_vs_c = comparisons["B_vs_C_additive_no_interaction"]
        interaction_supported &= (
            b_vs_c["delta_recall@1"] > 0
            and b_vs_c["formula_cluster_bootstrap_95ci"][0] > 0
            and b_vs_c.get("near", {}).get("delta_recall@1", 0.0) >= 0
        )
        report["panels"][panel_name] = {"methods": metrics, "comparisons": comparisons}
    report["mechanism_decision"] = (
        "CONDITIONAL_INTERACTION_SUPPORTED" if interaction_supported
        else "CONDITIONAL_INTERACTION_NOT_SUPPORTED"
    )
    report["interpretation_boundary"] = (
        "GNPS development mechanism result only; neither outcome is external validation"
    )
    args.output.mkdir(parents=True)
    prediction_path = args.output / "control_predictions.npz"
    np.savez_compressed(prediction_path, **{
        f"{name}__C_additive_no_interaction": score for name, score in control_scores.items()
    })
    report["artifacts"] = {
        "control_predictions": str(prediction_path),
        "control_predictions_sha256": sha256(prediction_path),
    }
    report["input_sha256"] = {
        "parent_report": sha256(args.parent_report),
        "parent_predictions": sha256(args.parent_predictions),
        "identity_bundle": sha256(args.identity_bundle),
        "formula_bundle": sha256(args.formula_bundle),
        "identity_chem": sha256(args.identity_chem),
        "formula_chem": sha256(args.formula_chem),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "decision": report["mechanism_decision"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
