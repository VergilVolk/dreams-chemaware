#!/usr/bin/env python
"""One-shot repair of the B-anchored multi-evidence integration.

The previous incremental head never observed the B anchor state and subtracted
heterogeneous MSG and P2b score summaries.  This fixed experiment changes only
those diagnosed defects:

* B raw score, query gap and Top1--Top2 margin are explicit inputs;
* Stage-1, Phase-A and MSG retain separate score summaries;
* MSG probabilities are converted to logits before fold-local scaling;
* the loss compares the truth candidate with B's strongest false competitor.

The parent B ledger is nested formula-OOF.  GNPS is development/consumed and no
external benchmark is read.  There is no hyperparameter scan.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from train_b_anchored_dual_encoder_multireranker import (
    AnchoredResidual,
    MSG,
    PHASEA,
    STAGE1,
    _axis_equal,
    assert_same_candidate_axis,
    load_parent,
    parent_fit_args,
    predict_parent,
)
from train_grand_unified_evidence_model import load_bundle
from train_unified_candidate_energy_crossfit import candidate_indices, formula_fold, metric_summary
from train_wse_conditional_residual import (
    build_panel as build_parent_panel,
    evaluation_scores,
    fit_arm as fit_parent_arm,
    metric_input,
    paired_comparison,
    query_score_features,
    sha256,
    training_residual_bound,
)


FAMILIES = (STAGE1, PHASEA, MSG)
ARM = "anchor_conditioned_family"
PRIOR = "previous_complete_spectral"


def msg_logit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if np.any((values < 0) | (values > 1)):
        raise ValueError("MSG scores must be probabilities in [0, 1]")
    clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
    return np.log(clipped) - np.log1p(-clipped)


def static_family_features(bundle: dict) -> tuple[np.ndarray, tuple[str, ...]]:
    names = tuple(bundle["module_names"].astype("U"))
    missing = [name for name in FAMILIES if name not in names]
    if missing:
        raise RuntimeError(f"family evidence missing: {missing}")
    ptr = np.asarray(bundle["query_ptr"], dtype=np.int64)
    scores = np.asarray(bundle["scores"], dtype=np.float32)
    availability = np.asarray(bundle["availability"], dtype=np.int8)
    blocks, feature_names = [], []
    for family in FAMILIES:
        index = names.index(family)
        if np.any(availability[index] != 1):
            raise RuntimeError(f"fixed GNPS repair requires complete {family} evidence")
        values = msg_logit(scores[index]) if family == MSG else scores[index]
        blocks.append(query_score_features(values, ptr))
        prefix = "msg_logit" if family == MSG else family
        feature_names.extend((f"{prefix}_raw", f"{prefix}_gap", f"{prefix}_margin"))
    features = np.column_stack(blocks).astype(np.float32)
    if not np.all(np.isfinite(features)):
        raise RuntimeError("non-finite family features")
    return features, tuple(feature_names)


def anchored_features(static: np.ndarray, base_scores: np.ndarray,
                      query_ptr: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    anchor = query_score_features(np.asarray(base_scores, dtype=np.float32), query_ptr)
    output = np.column_stack((anchor, static)).astype(np.float32)
    names = (
        "B_raw", "B_gap", "B_margin",
        f"{STAGE1}_raw", f"{STAGE1}_gap", f"{STAGE1}_margin",
        f"{PHASEA}_raw", f"{PHASEA}_gap", f"{PHASEA}_margin",
        "msg_logit_raw", "msg_logit_gap", "msg_logit_margin",
    )
    if output.shape[1] != len(names) or not np.all(np.isfinite(output)):
        raise RuntimeError("invalid anchor-conditioned feature matrix")
    return output, names


def fit_standardizer(features: np.ndarray, candidate_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(features[candidate_index], dtype=np.float64)
    mean = selected.mean(axis=0).astype(np.float32)
    scale = selected.std(axis=0).astype(np.float32)
    scale = np.maximum(scale, 1e-7)
    return mean, scale


def strongest_false_positions(base: np.ndarray, labels: np.ndarray,
                              packed_ptr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positive, competitor = [], []
    for left, right in zip(packed_ptr[:-1], packed_ptr[1:]):
        left, right = int(left), int(right)
        block_labels = labels[left:right]
        pos = np.flatnonzero(block_labels > 0)
        if len(pos) != 1:
            raise RuntimeError("each query must have exactly one truth candidate")
        pos = left + int(pos[0])
        block = np.asarray(base[left:right], dtype=np.float64).copy()
        block[pos - left] = -np.inf
        neg = left + int(np.argmax(block))
        positive.append(pos); competitor.append(neg)
    return np.asarray(positive, dtype=np.int64), np.asarray(competitor, dtype=np.int64)


def fit_repair(panel: dict, train_queries: np.ndarray, nested_b: np.ndarray,
               recipe: dict, seed: int, device: str):
    candidate_index, packed_ptr = candidate_indices(panel["query_ptr"], train_queries)
    if np.any(~np.isfinite(nested_b[candidate_index])):
        raise RuntimeError("nested B ledger is incomplete")
    dynamic, feature_names = anchored_features(panel["static_features"], nested_b, panel["query_ptr"])
    mean, scale = fit_standardizer(dynamic, candidate_index)
    x_np = (dynamic[candidate_index] - mean) / scale
    base_np = np.asarray(nested_b[candidate_index], dtype=np.float32)
    labels_np = np.asarray(panel["labels"][candidate_index], dtype=np.int8)
    positive, competitor = strongest_false_positions(base_np, labels_np, packed_ptr)
    bound, capacity = training_residual_bound(nested_b, panel["query_ptr"], train_queries)

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = AnchoredResidual(dynamic.shape[1], int(recipe["hidden"]), bound).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(recipe["learning_rate"]), weight_decay=1e-4,
    )
    base = torch.as_tensor(base_np, dtype=torch.float32, device=device)
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    pos = torch.as_tensor(positive, dtype=torch.long, device=device)
    neg = torch.as_tensor(competitor, dtype=torch.long, device=device)
    final = None
    for _ in range(int(recipe["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        residual = model.residual(x)
        score = base + residual
        pairwise = F.softplus(-(score[pos] - score[neg])).mean()
        trust = float(recipe["trust_weight"]) * torch.mean(residual ** 2)
        loss = pairwise + trust
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        final = {"total": float(loss.detach().cpu()),
                 "pairwise": float(pairwise.detach().cpu()), "trust": float(trust.detach().cpu())}
    return model, mean, scale, bound, capacity, final, feature_names


@torch.inference_mode()
def predict_repair(model: AnchoredResidual, panel: dict, queries: np.ndarray,
                   base_scores: np.ndarray, mean: np.ndarray, scale: np.ndarray,
                   device: str) -> tuple[np.ndarray, np.ndarray]:
    candidate_index, _ = candidate_indices(panel["query_ptr"], queries)
    dynamic, _ = anchored_features(panel["static_features"], base_scores, panel["query_ptr"])
    x = torch.as_tensor(
        (dynamic[candidate_index] - mean) / scale, dtype=torch.float32, device=device,
    )
    residual = model.residual(x).cpu().numpy().astype(np.float32)
    return candidate_index, residual


def panel_from_bundle(bundle: dict, parent: dict[str, np.ndarray]) -> dict:
    static, feature_names = static_family_features(bundle)
    return {
        "static_features": static, "static_feature_names": feature_names,
        "query_ptr": np.asarray(bundle["query_ptr"], dtype=np.int64),
        "labels": np.asarray(bundle["labels"], dtype=np.int8),
        "query_ids": np.asarray(bundle["query_ids"]).astype("U"),
        "candidate_ids": np.asarray(bundle["candidate_ids"]).astype("U"),
        "formulas": np.asarray(bundle["group_ids"]).astype("U"),
        "near": np.asarray(bundle["near_query"], dtype=bool),
        "parent": parent,
    }


def load_prior(path: Path, panels: dict[str, dict]) -> dict[str, np.ndarray]:
    output = {}
    with np.load(path, allow_pickle=False) as source:
        for panel_name, panel in panels.items():
            prefix = f"{panel_name}__"
            for key, expected in (
                ("query_ptr", panel["query_ptr"]), ("labels", panel["labels"]),
                ("query_ids", panel["query_ids"]), ("candidate_ids", panel["candidate_ids"]),
                ("formula", panel["formulas"]), ("near", panel["near"]),
            ):
                if prefix + key not in source or not _axis_equal(source[prefix + key], expected):
                    raise RuntimeError(f"prior prediction axis mismatch: {panel_name}/{key}")
            values = np.asarray(source[prefix + "complete_spectral"], dtype=np.float32)
            if values.shape != panel["labels"].shape or not np.all(np.isfinite(values)):
                raise RuntimeError(f"invalid prior complete prediction: {panel_name}")
            output[panel_name] = values
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-bundle", type=Path, required=True)
    parser.add_argument("--formula-bundle", type=Path, required=True)
    parser.add_argument("--core-identity-bundle", type=Path, required=True)
    parser.add_argument("--core-formula-bundle", type=Path, required=True)
    parser.add_argument("--identity-chem", type=Path, required=True)
    parser.add_argument("--formula-chem", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--parent-predictions", type=Path, required=True)
    parser.add_argument("--prior-nested-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    parent_report = json.loads(args.parent_report.read_text(encoding="utf-8"))
    if parent_report.get("status") != "WSE_CONDITIONAL_RESIDUAL_DEVELOPMENT_COMPLETE":
        raise RuntimeError("unsupported parent A/B report")
    recipe = parent_report["optimization"]
    if bool(recipe.get("hyperparameter_scan")):
        raise RuntimeError("parent used a hyperparameter scan")
    folds, seed = int(parent_report["formula_crossfit_folds"]), int(recipe["seed"])

    bundles = {"identity": load_bundle(args.identity_bundle), "formula": load_bundle(args.formula_bundle)}
    historical = load_parent(args.parent_predictions, bundles)
    panels = {name: panel_from_bundle(bundle, historical[name]) for name, bundle in bundles.items()}
    core_bundles = {
        "identity": load_bundle(args.core_identity_bundle),
        "formula": load_bundle(args.core_formula_bundle),
    }
    core_panels = {
        "identity": build_parent_panel(core_bundles["identity"], args.identity_chem),
        "formula": build_parent_panel(core_bundles["formula"], args.formula_chem),
    }
    for name in panels:
        assert_same_candidate_axis(panels[name], core_panels[name], name)
    prior = load_prior(args.prior_nested_predictions, panels)

    identity_fold = np.asarray([formula_fold(v, folds, seed) for v in panels["identity"]["formulas"]])
    target_fold = {
        name: np.asarray([formula_fold(v, folds, seed) for v in panel["formulas"]])
        for name, panel in panels.items()
    }
    nested_by_outer = {
        outer: np.full(len(panels["identity"]["labels"]), np.nan, dtype=np.float32)
        for outer in range(folds)
    }
    nested_records = {outer: [] for outer in range(folds)}
    parent_args = parent_fit_args(recipe, args.device)
    for first in range(folds):
        for second in range(first + 1, folds):
            pair_train = np.flatnonzero((identity_fold != first) & (identity_fold != second))
            fitted_b = fit_parent_arm(
                core_panels["identity"], pair_train, "joint", parent_args,
                # Reproduce the exact nested-B seed used by run_2400157 so
                # the only experimental change is the incremental head.
                seed + 100000 + first * folds + second,
            )
            for outer, inner in ((first, second), (second, first)):
                inner_queries = np.flatnonzero(identity_fold == inner)
                index, score = predict_parent(fitted_b, core_panels["identity"], inner_queries, args.device)
                nested_by_outer[outer][index] = score
                nested_records[outer].append({
                    "inner_fold": int(inner), "shared_omitted_pair": [first, second],
                    "train_queries": int(len(pair_train)), "prediction_queries": int(len(inner_queries)),
                })

    outputs = {
        name: np.full(len(panel["labels"]), np.nan, dtype=np.float32)
        for name, panel in panels.items()
    }
    fold_reports = []
    for fold in range(folds):
        train_queries = np.flatnonzero(identity_fold != fold)
        model, mean, scale, bound, capacity, final, feature_names = fit_repair(
            panels["identity"], train_queries, nested_by_outer[fold], recipe,
            seed + 300000 + fold, args.device,
        )
        prediction_counts = {}
        for panel_name, panel in panels.items():
            queries = np.flatnonzero(target_fold[panel_name] == fold)
            base = historical[panel_name]["B"]
            index, residual = predict_repair(model, panel, queries, base, mean, scale, args.device)
            outputs[panel_name][index] = base[index] + residual
            prediction_counts[panel_name] = int(len(queries))
        fold_reports.append({
            "fold": fold, "train_queries": int(len(train_queries)),
            "nested_parent_predictions": nested_records[fold],
            "residual_bound_B_units": bound, "capacity": capacity,
            "parameters": int(sum(value.numel() for value in model.parameters())),
            "feature_names": list(feature_names), "feature_mean": mean.tolist(),
            "feature_scale": scale.tolist(), "final_loss": final,
            "prediction_queries": prediction_counts,
        })
    if any(np.any(~np.isfinite(values)) for values in outputs.values()):
        raise RuntimeError("repair OOF prediction ledger is incomplete")

    report = {
        "status": "B_ANCHOR_CONDITIONED_FAMILY_REPAIR_NESTED_OOF_COMPLETE",
        "dataset": "GNPS Gold/Silver 10-ppm", "dataset_role": "development_consumed",
        "enveda_accessed": False, "hyperparameter_scan": False,
        "scientific_change": {
            "anchor_conditioning": "B raw/gap/margin are explicit candidate features",
            "family_semantics": "Stage-1 and Phase-A direct summaries; MSG logit summaries; no MSG-P2b subtraction",
            "objective": "truth candidate versus B strongest false competitor plus inherited trust penalty",
            "unchanged": "same parent B recipe, hidden width, epochs, learning rate, trust weight and fold assignment",
        },
        "stacking_isolation": {
            "protocol": "outer formula fold + inner B-parent formula cross-fit",
            "outer_validation_labels_reachable_through_parent_training": False,
            "outer_anchor": "exact frozen historical B formula-OOF bytes",
        },
        "optimization_replayed_from_B_parent": recipe,
        "folds": fold_reports, "panels": {},
    }
    passed = True
    arrays = {}
    for panel_index, (panel_name, panel) in enumerate(panels.items()):
        methods = {
            "wse": historical[panel_name]["wse"], "B": historical[panel_name]["B"],
            PRIOR: prior[panel_name], ARM: outputs[panel_name],
        }
        metrics = {
            name: metric_summary(evaluation_scores(score, panel["query_ptr"]), metric_input(panel))[0]
            for name, score in methods.items()
        }
        paired = {}
        for offset, (candidate, baseline) in enumerate(((ARM, "B"), (ARM, PRIOR), (PRIOR, "B"))):
            paired[f"{candidate}_vs_{baseline}"] = paired_comparison(
                methods[candidate], methods[baseline], panel,
                seed + 400000 + panel_index * 10 + offset, args.bootstrap_resamples,
            )
        primary = paired[f"{ARM}_vs_B"]
        passed &= primary["delta_recall@1"] > 0 and primary["formula_cluster_bootstrap_95ci"][0] > 0
        report["panels"][panel_name] = {"methods": metrics, "comparisons": paired}
        prefix = f"{panel_name}__"
        arrays.update({
            prefix + "query_ptr": panel["query_ptr"], prefix + "labels": panel["labels"],
            prefix + "query_ids": panel["query_ids"], prefix + "candidate_ids": panel["candidate_ids"],
            prefix + "formula": panel["formulas"], prefix + "near": panel["near"],
            prefix + "wse": methods["wse"], prefix + "B": methods["B"],
            prefix + PRIOR: methods[PRIOR], prefix + ARM: methods[ARM],
        })
    report["selection"] = (
        "FREEZE_ANCHOR_CONDITIONED_FAMILY_DEVELOPMENT_CANDIDATE" if passed
        else "KEEP_B_REFERENCE_STOP_SCORE_LEVEL_REPAIR"
    )
    report["claim_boundary"] = (
        "Single pre-specified repair on consumed GNPS development data. "
        "MassSpecGym, GNPS and Enveda are consumed; confirmation requires a new independent dataset."
    )

    args.output.mkdir(parents=True)
    if passed:
        all_identity = np.arange(len(panels["identity"]["query_ids"]), dtype=np.int64)
        model, mean, scale, bound, capacity, final, feature_names = fit_repair(
            panels["identity"], all_identity, historical["identity"]["B"], recipe,
            seed + 399999, args.device,
        )
        model_path = args.output / "anchor_conditioned_family_development_candidate.pt"
        torch.save({
            "state_dict": model.cpu().state_dict(),
            "model_class": "AnchoredResidual", "anchor": "unified_B_development_candidate",
            "arm": ARM, "feature_names": feature_names,
            "feature_mean": mean, "feature_scale": scale,
            "hidden": int(recipe["hidden"]), "residual_bound_B_units": bound,
            "training_seed": seed + 399999, "training_queries": int(len(all_identity)),
            "stacking_fit": "all-identity frozen formula-OOF B candidate scores",
            "enveda_accessed": False,
        }, model_path)
        report["development_candidate_model"] = {
            "path": str(model_path), "sha256": sha256(model_path),
            "residual_bound_B_units": bound, "capacity": capacity, "final_loss": final,
        }
    else:
        report["development_candidate_model"] = None
    prediction_path = args.output / "candidate_predictions.npz"
    np.savez_compressed(prediction_path, **arrays)
    report["artifacts"] = {"candidate_predictions": str(prediction_path), "sha256": sha256(prediction_path)}
    report["input_sha256"] = {
        "identity_bundle": sha256(args.identity_bundle), "formula_bundle": sha256(args.formula_bundle),
        "core_identity_bundle": sha256(args.core_identity_bundle),
        "core_formula_bundle": sha256(args.core_formula_bundle),
        "identity_chem": sha256(args.identity_chem), "formula_chem": sha256(args.formula_chem),
        "parent_report": sha256(args.parent_report),
        "parent_predictions": sha256(args.parent_predictions),
        "prior_nested_predictions": sha256(args.prior_nested_predictions),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "selection": report["selection"],
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
