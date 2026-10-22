#!/usr/bin/env python
"""Cross-fit the fixed-B anchored dual-encoder/multi-reranker integration.

The parent B OOF score is never refit.  New evidence enters only as a bounded
residual relative to the closest already-counted family member:

* Stage-1 and Phase-A are separate contrasts to Noise V1;
* MSG and historical RRF are separate contrasts to frozen P2b-on-Noise.

This makes all assets runnable without treating correlated scores as repeated
independent votes.  Formula-hash OOF fitting is used for every learned arm.
GNPS is development/consumed; this program never reads an external benchmark.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from train_grand_unified_evidence_model import load_bundle
from train_unified_candidate_energy_crossfit import candidate_indices, formula_fold, metric_summary
from train_wse_conditional_residual import (
    evaluation_scores,
    metric_input,
    paired_comparison,
    query_score_features,
    sha256,
    training_residual_bound,
)


WSE = "weighted_spectral_entropy"
NOISE = "noise_v1"
P2B = "p2b_noise_v1_frozen"
STAGE1 = "chemaware_stage1_encoder"
PHASEA = "chemaware_phase_a_encoder"
MSG = "noise_msg_fusion_stack_v1"
RRF = "noise_rrf_1_7"
PARENT_METHODS = ("wse", "old_four_module", "A", "B")
EXPERTS = (
    (STAGE1, NOISE, "stage1_minus_noise"),
    (PHASEA, NOISE, "phasea_minus_noise"),
    (MSG, P2B, "msg_minus_p2b"),
    (RRF, P2B, "rrf_minus_p2b"),
)
ARM_EXPERTS = {
    "stage1_only": (0,),
    "phasea_only": (1,),
    "dual_encoder": (0, 1),
    "complete_spectral": (0, 1, 2),
    "complete_spectral_additive": (0, 1, 2),
    "audit_plus_rrf": (0, 1, 2, 3),
}


class AnchoredResidual(nn.Module):
    """A bounded correction with exact zero-evidence no-op semantics."""

    def __init__(self, width: int, hidden: int, bound: float, linear: bool = False):
        super().__init__()
        self.bound = float(bound)
        self.network = nn.Linear(width, 1) if linear else nn.Sequential(
            nn.Linear(width, hidden), nn.Tanh(), nn.Linear(hidden, 1),
        )
        final = self.network if linear else self.network[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def residual(self, features: torch.Tensor) -> torch.Tensor:
        zero = torch.zeros_like(features)
        raw = self.network(features).squeeze(-1) - self.network(zero).squeeze(-1)
        return self.bound * torch.tanh(raw)

    def forward(self, base: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return base + self.residual(features)


def _axis_equal(observed: np.ndarray, expected: np.ndarray) -> bool:
    if np.asarray(observed).dtype.kind in "OUS" or np.asarray(expected).dtype.kind in "OUS":
        return np.array_equal(np.asarray(observed).astype("U"), np.asarray(expected).astype("U"))
    return np.array_equal(observed, expected)


def load_parent(path: Path, bundles: dict[str, dict]) -> dict[str, dict[str, np.ndarray]]:
    output = {}
    with np.load(path, allow_pickle=False) as source:
        for panel_name, bundle in bundles.items():
            prefix = f"{panel_name}__"
            for key, expected in (
                ("query_ptr", bundle["query_ptr"]), ("labels", bundle["labels"]),
                ("query_ids", bundle["query_ids"]), ("candidate_ids", bundle["candidate_ids"]),
                ("formula", bundle["group_ids"]), ("near", bundle["near_query"]),
            ):
                if prefix + key not in source or not _axis_equal(source[prefix + key], expected):
                    raise RuntimeError(f"parent prediction axis mismatch: {panel_name}/{key}")
            output[panel_name] = {}
            for method in PARENT_METHODS:
                values = np.asarray(source[prefix + method], dtype=np.float32)
                if values.shape != bundle["labels"].shape or not np.all(np.isfinite(values)):
                    raise RuntimeError(f"invalid parent score: {panel_name}/{method}")
                output[panel_name][method] = values
    return output


def build_features(bundle: dict) -> tuple[np.ndarray, tuple[str, ...]]:
    names = tuple(map(str, bundle["module_names"]))
    missing = [name for name in (WSE, NOISE, P2B, STAGE1, PHASEA, MSG, RRF) if name not in names]
    if missing:
        raise RuntimeError(f"evidence bundle misses required modules: {missing}")
    ptr = np.asarray(bundle["query_ptr"], dtype=np.int64)
    scores = np.asarray(bundle["scores"], dtype=np.float32)
    available = np.asarray(bundle["availability"], dtype=np.float32)
    columns, feature_names = [], []
    for child, parent, label in EXPERTS:
        ci, pi = names.index(child), names.index(parent)
        observed = np.minimum(available[ci], available[pi]).astype(np.float32)
        relative = query_score_features(scores[ci], ptr) - query_score_features(scores[pi], ptr)
        relative *= observed[:, None]
        columns.extend((relative[:, 0], relative[:, 1], relative[:, 2], observed))
        feature_names.extend((f"{label}_raw", f"{label}_gap", f"{label}_margin", f"{label}_available"))
    features = np.column_stack(columns).astype(np.float32)
    if not np.all(np.isfinite(features)):
        raise RuntimeError("non-finite integrated evidence")
    return features, tuple(feature_names)


def panel_from_bundle(bundle: dict, parent: dict[str, np.ndarray]) -> dict:
    features, feature_names = build_features(bundle)
    names = tuple(map(str, bundle["module_names"]))
    return {
        "features": features, "feature_names": feature_names,
        "query_ptr": np.asarray(bundle["query_ptr"], dtype=np.int64),
        "labels": np.asarray(bundle["labels"], dtype=np.int8),
        "query_ids": np.asarray(bundle["query_ids"]).astype("U"),
        "candidate_ids": np.asarray(bundle["candidate_ids"]).astype("U"),
        "formulas": np.asarray(bundle["group_ids"]).astype("U"),
        "near": np.asarray(bundle["near_query"], dtype=bool),
        "wse": np.asarray(bundle["scores"], dtype=np.float32)[names.index(WSE)],
        "parent": parent,
    }


def feature_columns(experts: tuple[int, ...]) -> np.ndarray:
    return np.asarray([4 * expert + offset for expert in experts for offset in range(4)], dtype=np.int64)


def fit_scale(features: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Scale contrasts without centering; semantic zero must remain exact."""
    scale = np.ones(features.shape[1], dtype=np.float32)
    for start in range(0, features.shape[1], 4):
        observed = features[index, start + 3] > 0
        if np.any(observed):
            rms = np.sqrt(np.mean(np.square(features[index, start:start + 3][observed]), axis=0))
            scale[start:start + 3] = np.maximum(rms, 1e-7)
    return scale


def listwise_loss(scores: torch.Tensor, labels: torch.Tensor, query_ptr: np.ndarray) -> torch.Tensor:
    lengths = torch.as_tensor(np.diff(query_ptr), dtype=torch.long, device=scores.device)
    qindex = torch.repeat_interleave(torch.arange(len(lengths), device=scores.device), lengths)
    maximum = torch.full((len(lengths),), -torch.inf, dtype=scores.dtype, device=scores.device)
    maximum.scatter_reduce_(0, qindex, scores, reduce="amax", include_self=True)
    exp_sum = torch.zeros_like(maximum)
    exp_sum.scatter_add_(0, qindex, torch.exp(scores - maximum[qindex]))
    positive = labels > 0
    if int(positive.sum()) != len(lengths):
        raise RuntimeError("each packed query must contain one positive")
    return torch.mean(maximum + torch.log(exp_sum) - scores[positive])


def fit_arm(panel: dict, train_queries: np.ndarray, experts: tuple[int, ...], recipe: dict,
            seed: int, device: str, linear: bool = False):
    index, packed_ptr = candidate_indices(panel["query_ptr"], train_queries)
    columns = feature_columns(experts)
    scale_all = fit_scale(panel["features"], index)
    features = panel["features"][:, columns] / scale_all[columns]
    bound, capacity = training_residual_bound(panel["parent"]["B"], panel["query_ptr"], train_queries)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = AnchoredResidual(len(columns), int(recipe["hidden"]), bound, linear=linear).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(recipe["learning_rate"]), weight_decay=1e-4)
    base = torch.as_tensor(panel["parent"]["B"][index], dtype=torch.float32, device=device)
    x = torch.as_tensor(features[index], dtype=torch.float32, device=device)
    y = torch.as_tensor(panel["labels"][index], dtype=torch.int8, device=device)
    final = None
    for _ in range(int(recipe["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        residual = model.residual(x)
        loss = listwise_loss(base + residual, y, packed_ptr) + float(recipe["trust_weight"]) * torch.mean(residual ** 2)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        final = float(loss.detach().cpu())
    return model, columns, scale_all[columns], bound, capacity, final


@torch.inference_mode()
def predict(model: AnchoredResidual, panel: dict, queries: np.ndarray, columns: np.ndarray,
            scale: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    index, _ = candidate_indices(panel["query_ptr"], queries)
    x = torch.as_tensor(panel["features"][index][:, columns] / scale, dtype=torch.float32, device=device)
    residual = model.residual(x).cpu().numpy().astype(np.float32)
    return index, residual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-bundle", type=Path, required=True)
    parser.add_argument("--formula-bundle", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--parent-predictions", type=Path, required=True)
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
    parents = load_parent(args.parent_predictions, bundles)
    panels = {name: panel_from_bundle(bundle, parents[name]) for name, bundle in bundles.items()}
    identity_fold = np.asarray([formula_fold(v, folds, seed) for v in panels["identity"]["formulas"]])
    target_fold = {name: np.asarray([formula_fold(v, folds, seed) for v in body["formulas"]]) for name, body in panels.items()}

    outputs = {name: {arm: np.full(len(body["labels"]), np.nan, dtype=np.float32) for arm in ARM_EXPERTS}
               for name, body in panels.items()}
    folds_report = []
    for fold in range(folds):
        train_queries = np.flatnonzero(identity_fold != fold)
        fold_record = {"fold": fold, "train_queries": int(len(train_queries)), "arms": {}}
        for arm_index, (arm, experts) in enumerate(ARM_EXPERTS.items()):
            model, columns, scale, bound, capacity, final = fit_arm(
                panels["identity"], train_queries, experts, recipe,
                seed + fold + arm_index * 1000, args.device,
                linear=(arm == "complete_spectral_additive"),
            )
            predictions = {}
            for panel_name, panel in panels.items():
                queries = np.flatnonzero(target_fold[panel_name] == fold)
                index, residual = predict(model, panel, queries, columns, scale, args.device)
                outputs[panel_name][arm][index] = panel["parent"]["B"][index] + residual
                predictions[panel_name] = int(len(queries))
            fold_record["arms"][arm] = {
                "experts": [EXPERTS[i][0] for i in experts], "feature_columns": columns.tolist(),
                "residual_bound_B_units": bound, "capacity": capacity,
                "parameters": int(sum(p.numel() for p in model.parameters())),
                "final_loss": final, "prediction_queries": predictions,
            }
        folds_report.append(fold_record)
    if any(np.any(~np.isfinite(v)) for methods in outputs.values() for v in methods.values()):
        raise RuntimeError("OOF prediction ledger is incomplete")

    # Hold the learned full-system residual fixed and replace only B by A.  This
    # is the exact no-V2 replay requested by the fixed comparison matrix.
    for panel_name, panel in panels.items():
        residual = outputs[panel_name]["complete_spectral"] - panel["parent"]["B"]
        outputs[panel_name]["complete_without_v2_replay"] = panel["parent"]["A"] + residual

    comparisons = (
        ("dual_encoder", "B"), ("complete_spectral", "dual_encoder"),
        ("audit_plus_rrf", "complete_spectral"),
        ("complete_spectral", "complete_without_v2_replay"),
        ("complete_spectral", "complete_spectral_additive"),
        ("complete_spectral", "B"), ("complete_spectral", "wse"),
    )
    report = {
        "status": "B_ANCHORED_DUAL_ENCODER_MULTIRERANKER_DEVELOPMENT_COMPLETE",
        "dataset": "GNPS Gold/Silver 10-ppm", "dataset_role": "development_consumed",
        "enveda_accessed": False, "B_fixed_reference": True,
        "evidence_semantics": {
            "encoders": "Stage-1 and Phase-A are distinct score contrasts to Noise V1",
            "rerankers": "MSG and historical RRF are distinct contrasts to P2b-on-Noise",
            "RRF": "retired audit-only input; never silently promoted",
            "V2_ablation": "same learned full residual replayed on parent A instead of parent B",
        },
        "feature_names": list(panels["identity"]["feature_names"]),
        "optimization_replayed_from_B_parent": recipe, "folds": folds_report, "panels": {},
    }
    candidate_pass = True
    for panel_index, (panel_name, panel) in enumerate(panels.items()):
        methods = {**panel["parent"], **outputs[panel_name]}
        metrics = {name: metric_summary(evaluation_scores(score, panel["query_ptr"]), metric_input(panel))[0]
                   for name, score in methods.items()}
        paired = {}
        for offset, (candidate, baseline) in enumerate(comparisons):
            key = f"{candidate}_vs_{baseline}"
            paired[key] = paired_comparison(
                methods[candidate], methods[baseline], panel,
                seed + 70000 + panel_index * 100 + offset, args.bootstrap_resamples,
            )
        gate = paired["complete_spectral_vs_B"]
        candidate_pass &= gate["delta_recall@1"] > 0 and gate["formula_cluster_bootstrap_95ci"][0] > 0
        report["panels"][panel_name] = {"methods": metrics, "comparisons": paired}
    report["selection"] = (
        "FREEZE_COMPLETE_SPECTRAL_DEVELOPMENT_CANDIDATE" if candidate_pass
        else "KEEP_B_REFERENCE_NO_FULL_SYSTEM_PROMOTION"
    )
    report["claim_boundary"] = (
        "Development selection only. A new independent dataset is required after freezing; "
        "MassSpecGym, GNPS, and Enveda are consumed."
    )

    args.output.mkdir(parents=True)
    if candidate_pass:
        all_identity = np.arange(len(panels["identity"]["query_ids"]), dtype=np.int64)
        model, columns, scale, bound, capacity, final = fit_arm(
            panels["identity"], all_identity, ARM_EXPERTS["complete_spectral"], recipe,
            seed + 99999, args.device,
        )
        model_path = args.output / "complete_spectral_development_candidate.pt"
        torch.save({
            "state_dict": model.cpu().state_dict(),
            "model_class": "AnchoredResidual",
            "anchor": "unified_B_development_candidate",
            "arm": "complete_spectral",
            "expert_names": [EXPERTS[i][0] for i in ARM_EXPERTS["complete_spectral"]],
            "feature_names": tuple(np.asarray(panels["identity"]["feature_names"])[columns]),
            "feature_columns": columns,
            "feature_scale": scale,
            "hidden": int(recipe["hidden"]),
            "residual_bound_B_units": bound,
            "training_seed": seed + 99999,
            "training_queries": int(len(all_identity)),
            "enveda_accessed": False,
        }, model_path)
        report["development_candidate_model"] = {
            "path": str(model_path), "sha256": sha256(model_path),
            "fit_scope": "all GNPS identity development queries after formula-OOF selection",
            "residual_bound_B_units": bound, "capacity": capacity, "final_loss": final,
        }
    else:
        report["development_candidate_model"] = None
    arrays = {}
    for panel_name, panel in panels.items():
        prefix = f"{panel_name}__"
        arrays.update({prefix + "query_ptr": panel["query_ptr"], prefix + "labels": panel["labels"],
                       prefix + "query_ids": panel["query_ids"], prefix + "candidate_ids": panel["candidate_ids"],
                       prefix + "formula": panel["formulas"], prefix + "near": panel["near"]})
        for name, values in {**panel["parent"], **outputs[panel_name]}.items():
            arrays[prefix + name] = values
    prediction_path = args.output / "candidate_predictions.npz"
    np.savez_compressed(prediction_path, **arrays)
    report["artifacts"] = {"candidate_predictions": str(prediction_path), "sha256": sha256(prediction_path)}
    report["input_sha256"] = {
        "identity_bundle": sha256(args.identity_bundle), "formula_bundle": sha256(args.formula_bundle),
        "parent_report": sha256(args.parent_report), "parent_predictions": sha256(args.parent_predictions),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "selection": report["selection"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
