#!/usr/bin/env python
"""Refit the centered-residual Full23 Additive from its frozen OOF ledger.

This local replay avoids every upstream encoder/cache.  The frozen ledger
already contains the seven candidate-level expert scores, and the retained V2
archive supplies the six continuous chemistry features.  The script rebuilds
the exact 23-feature Full23 matrix, proves its all-data normalization matches
the released checkpoint, and changes only the residual regularizer.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import tempfile

import numpy as np
import torch

from enveda_unified_candidate_integrator_core import (
    CONTROL_ARM,
    NO_QUERY_CONSTANT_CONTEXT_POLICY,
    formula_fold,
    query_constant_context_names,
)
from gnps_pair_score_cache import sha256_file
from massspecgym_embedding_readout_core import build_independent_scalar_features
from train_enveda_unified_candidate_integrator import (
    ENCODERS,
    MSG,
    PANELS,
    WSE,
    PanelData,
    _comparison,
    _fit_arm,
    _fit_normalization,
    _formula_ci,
    _joint_query_balanced,
    _metric_summary,
    _occurrences,
    _predict_occurrences,
)


EXPERTS = (
    "official_dreams", "noise_v1", WSE, "p2b_noise_v1_frozen", MSG,
    "chemaware_stage1", "chemaware_phasea", "noise_chemaware_fullmsg",
)
PARENT = "full23_v1_frozen"
SUCCESSOR = "full23_centered_residual"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--chem-dir", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-queries", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--anchor-strength", type=float, default=1e-2)
    parser.add_argument("--bootstrap-resamples", type=int, default=5_000)
    parser.add_argument("--device", default="cpu")
    parser.set_defaults(center_residual_penalty=True)
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if (
        payload.get("schema") != "enveda_unified_candidate_integrator_v1"
        or payload.get("control_arm") != CONTROL_ARM
        or payload.get("feature_selection_policy") != NO_QUERY_CONSTANT_CONTEXT_POLICY
    ):
        raise RuntimeError("parent checkpoint is not the frozen 23-feature Full23 model")
    return payload


def _chemical(path: Path, ptr: np.ndarray, formulas: np.ndarray, near: np.ndarray) -> dict:
    with np.load(path, allow_pickle=False) as body:
        if (
            not np.array_equal(body["query_ptr"], ptr)
            or not np.array_equal(body["query_formula"].astype("U"), formulas)
            or not np.array_equal(body["near_query"].astype(bool), near)
        ):
            raise RuntimeError(f"continuous chemistry axis drifted: {path}")
        return {
            "chem_valid": np.asarray(body["availability"], dtype=bool),
            "candidate_utility_correct": np.asarray(body["utility_correct"], dtype=np.float32),
            "candidate_utility_zero": np.asarray(body["utility_zero"], dtype=np.float32),
            "candidate_utility_reversed": np.asarray(body["utility_reversed"], dtype=np.float32),
            "candidate_utility_rotated": np.asarray(
                body["utility_candidate_mismatch"], dtype=np.float32,
            ),
        }


def build_panels(
    ledger_path: Path, chem_dir: Path, parent: dict,
) -> dict[str, PanelData]:
    panels: dict[str, PanelData] = {}
    expected_names = tuple(map(str, parent["scalar_feature_names"]))
    with np.load(ledger_path, allow_pickle=False) as ledger:
        for name in PANELS:
            prefix = f"{name}__"
            required = (
                "query_ptr", "labels", "query_keys", "formula", "near",
                "query_weight", "fold", *EXPERTS, CONTROL_ARM,
            )
            missing = [prefix + field for field in required if prefix + field not in ledger.files]
            if missing:
                raise RuntimeError(f"frozen ledger misses {missing}")
            ptr = np.asarray(ledger[prefix + "query_ptr"], dtype=np.int64)
            labels = np.asarray(ledger[prefix + "labels"], dtype=np.int8)
            keys = np.asarray(ledger[prefix + "query_keys"]).astype("U")
            formulas = np.asarray(ledger[prefix + "formula"]).astype("U")
            near = np.asarray(ledger[prefix + "near"], dtype=bool)
            weights = np.asarray(ledger[prefix + "query_weight"], dtype=np.float32)
            scores = {
                expert: np.asarray(ledger[prefix + expert], dtype=np.float32)
                for expert in EXPERTS
            }
            scores[PARENT] = np.asarray(ledger[prefix + CONTROL_ARM], dtype=np.float32)
            if (
                ptr[0] != 0 or ptr[-1] != len(labels)
                or len(keys) != len(ptr) - 1 or len(formulas) != len(keys)
                or near.shape != weights.shape or weights.shape != keys.shape
            ):
                raise RuntimeError(f"frozen ledger axis is malformed: {name}")
            saved_fold = np.asarray(ledger[prefix + "fold"], dtype=np.int16)
            rebuilt_fold = np.asarray([
                formula_fold(value, 5, 20261026) for value in formulas
            ], dtype=np.int16)
            if not np.array_equal(saved_fold, rebuilt_fold):
                raise RuntimeError(f"frozen formula folds drifted: {name}")

            graph = SimpleNamespace(
                query_ptr=ptr,
                molecule_ptr=np.arange(len(labels) + 1, dtype=np.int64),
                molecule_label=labels,
            )
            chemical = _chemical(
                chem_dir / f"continuous_chem_{name}.npz", ptr, formulas, near,
            )
            scalar, feature_names = build_independent_scalar_features(
                {
                    "v1_cosine": scores["noise_v1"],
                    "weighted_entropy": scores[WSE],
                    "p2b_fused": scores["p2b_noise_v1_frozen"],
                },
                graph,
                {
                    "noise_v1": scores["noise_v1"],
                    "chemaware_stage1": scores["chemaware_stage1"],
                    "chemaware_phasea": scores["chemaware_phasea"],
                    "noise_chemaware_fullmsg": scores["noise_chemaware_fullmsg"],
                },
                "noise_v1",
                chemical,
                {MSG: scores[MSG]},
                include_reliability_context=True,
            )
            excluded = set(query_constant_context_names(feature_names))
            keep = np.asarray([feature not in excluded for feature in feature_names], dtype=bool)
            retained_names = tuple(
                feature for feature in feature_names if feature not in excluded
            )
            if len(excluded) != 31 or retained_names != expected_names:
                raise RuntimeError(f"reconstructed Full23 feature contract drifted: {name}")
            panels[name] = PanelData(
                name=name, graph=graph, scalar=scalar[:, keep],
                feature_names=retained_names, labels=labels, query_ptr=ptr,
                formulas=formulas, query_keys=keys, near=near,
                query_weights=weights, scores=scores,
            )
    return panels


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (
        args.folds != 5 or args.seed != 20261026 or args.epochs != 20
        or args.batch_queries != 256 or args.hidden != 16
        or args.learning_rate != 1e-3 or args.weight_decay != 1e-4
        or args.anchor_strength != 1e-2 or args.bootstrap_resamples < 100
        or args.center_residual_penalty is not True
    ):
        raise ValueError("local replay requires the fixed Full23 v1 training contract")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    parent = load_checkpoint(args.parent_checkpoint)
    panels = build_panels(args.ledger, args.chem_dir, parent)
    occurrences = _occurrences(panels)
    final_mean, final_scale = _fit_normalization(panels, occurrences)
    mean_error = float(np.max(np.abs(final_mean - np.asarray(parent["scalar_mean"]))))
    scale_error = float(np.max(np.abs(final_scale - np.asarray(parent["scalar_scale"]))))
    if mean_error > 1e-6 or scale_error > 1e-6:
        raise RuntimeError(
            f"reconstructed Full23 normalization drifted: mean={mean_error}, scale={scale_error}"
        )

    fold_by_occurrence = np.asarray([
        formula_fold(panels[name].formulas[query], args.folds, args.seed)
        for name, query in occurrences
    ], dtype=np.int16)
    oof = {
        name: np.full(len(panel.labels), np.nan, dtype=np.float32)
        for name, panel in panels.items()
    }
    fold_reports = []
    for fold in range(args.folds):
        train = [occ for index, occ in enumerate(occurrences) if fold_by_occurrence[index] != fold]
        valid = [occ for index, occ in enumerate(occurrences) if fold_by_occurrence[index] == fold]
        mean, scale = _fit_normalization(panels, train)
        model, history = _fit_arm(
            CONTROL_ARM, panels, train, mean, scale, args,
            args.seed + 100 * fold + 1,
        )
        predictions = _predict_occurrences(model, panels, valid, mean, scale, args.device)
        for (name, query), values in predictions.items():
            left, right = panels[name].query_ptr[query:query + 2]
            oof[name][int(left):int(right)] = values
        fold_reports.append({
            "fold": fold, "initial_loss": history[0], "final_loss": history[-1],
            "training_occurrences": len(train), "validation_occurrences": len(valid),
        })
        print(f"[centered local replay] completed formula fold {fold + 1}/{args.folds}", flush=True)
    if any(not np.all(np.isfinite(values)) for values in oof.values()):
        raise RuntimeError("centered OOF coverage is incomplete")

    final_model, final_history = _fit_arm(
        CONTROL_ARM, panels, occurrences, final_mean, final_scale, args,
        args.seed + 50_001,
    )
    method_scores = {}
    panel_reports = {}
    for index, name in enumerate(PANELS):
        panel = panels[name]
        method_scores[name] = {**panel.scores, SUCCESSOR: oof[name]}
        methods = {
            method: _metric_summary(values, panel)[0]
            for method, values in method_scores[name].items()
        }
        panel_reports[name] = {
            "queries": len(panel.query_ptr) - 1,
            "candidate_molecules": len(panel.labels),
            "methods": methods,
            "centered_vs_full23_v1": _comparison(
                oof[name], panel.scores[PARENT], panel,
                args.bootstrap_resamples, args.seed + 7000 + index,
            ),
            "centered_vs_wse": _comparison(
                oof[name], panel.scores[WSE], panel,
                args.bootstrap_resamples, args.seed + 8000 + index,
            ),
        }
    joint = {}
    collapsed = {}
    for method in (WSE, PARENT, SUCCESSOR):
        joint[method], collapsed[method] = _joint_query_balanced(
            panels, method_scores, method,
        )
    keys = sorted(collapsed[SUCCESSOR])
    joint_delta = np.asarray([
        collapsed[SUCCESSOR][key][0] - collapsed[PARENT][key][0] for key in keys
    ])
    joint_formula = np.asarray([collapsed[SUCCESSOR][key][2] for key in keys], dtype="U")

    checkpoint = copy.deepcopy(parent)
    checkpoint["arms"][CONTROL_ARM] = {
        key: value.detach().cpu() for key, value in final_model.state_dict().items()
    }
    checkpoint["scalar_mean"] = final_mean
    checkpoint["scalar_scale"] = final_scale
    checkpoint["arm_selection_from_oof"] = False
    checkpoint["training"] = {
        **checkpoint.get("training", {}),
        "source": "exact Full23 v1 feature reconstruction from frozen OOF ledger",
        "anchor_penalty": "query_centered_residual_mse",
        "query_centered_anchor_penalty": True,
        "hyperparameter_scan": False,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        checkpoint_path = staging / "full23_centered_residual.pt"
        torch.save(checkpoint, checkpoint_path)
        arrays = {}
        for name in PANELS:
            panel = panels[name]
            prefix = f"{name}__"
            arrays.update({
                prefix + "query_ptr": panel.query_ptr,
                prefix + "labels": panel.labels,
                prefix + "query_keys": panel.query_keys,
                prefix + "formula": panel.formulas,
                prefix + "near": panel.near,
                prefix + PARENT: panel.scores[PARENT],
                prefix + SUCCESSOR: oof[name],
                prefix + WSE: panel.scores[WSE],
            })
        prediction_path = staging / "candidate_predictions_oof.npz"
        np.savez_compressed(prediction_path, **arrays)
        report = {
            "status": "FULL23_CENTERED_RESIDUAL_LOCAL_REPLAY_COMPLETE",
            "dataset": "Enveda-180 consumed primary panels",
            "evaluation_role": "fusion_meta_development_consumed",
            "allow_final_external_claim": False,
            "changed_factor": "absolute_residual_mse -> query_centered_residual_mse",
            "unchanged_contract": {
                "features": 23, "arm": CONTROL_ARM, "folds": args.folds,
                "epochs": args.epochs, "batch_queries": args.batch_queries,
                "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
                "anchor_strength": args.anchor_strength, "seed": args.seed,
            },
            "reconstruction": {
                "feature_names_exact": True,
                "scalar_mean_max_abs_error": mean_error,
                "scalar_scale_max_abs_error": scale_error,
            },
            "panels": panel_reports,
            "query_balanced_joint": joint,
            "query_balanced_joint_centered_vs_full23_v1": _formula_ci(
                joint_delta, joint_formula, args.bootstrap_resamples, args.seed + 9000,
            ),
            "folds": fold_reports,
            "final_refit": {
                "initial_loss": final_history[0], "final_loss": final_history[-1],
            },
            "checkpoint": {
                "path": checkpoint_path.name, "sha256": sha256_file(checkpoint_path),
            },
            "prediction_ledger": {
                "path": prediction_path.name, "sha256": sha256_file(prediction_path),
            },
            "inputs": {
                "ledger": str(args.ledger), "ledger_sha256": sha256_file(args.ledger),
                "chem_report_sha256": sha256_file(args.chem_dir / "report.json"),
                "parent_checkpoint": str(args.parent_checkpoint),
                "parent_checkpoint_sha256": sha256_file(args.parent_checkpoint),
            },
        }
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({
        "status": report["status"], "output": str(args.output),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
