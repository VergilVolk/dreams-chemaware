#!/usr/bin/env python
"""Strictly refit the fixed unified Additive contribution matrix on Enveda.

Every arm uses the same global formula folds, optimizer, epoch budget, seed
schedule, WSE anchor penalty, candidate graph and upstream frozen scores.  The
only difference is the declared evidence-family removal.  This is consumed
development evidence and cannot itself support an external/SOTA claim.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from enveda_unified_additive_ablation_core import (
    ABLATION_ORDER,
    CHECKPOINT_SCHEMA,
    FULL,
    VARIANT_ORDER,
    VARIANTS,
    WSE,
    assert_full_reconstruction,
    build_variant_features,
)
from enveda_unified_candidate_integrator_core import CONTROL_ARM, formula_fold
from gnps_pair_score_cache import sha256_file
from train_enveda_unified_candidate_integrator import (
    PANELS,
    _apply_feature_policy,
    _comparison,
    _fit_arm,
    _fit_normalization,
    _formula_ci,
    _joint_query_balanced,
    _make_panels,
    _metric_summary,
    _occurrences,
    _predict_occurrences,
    named_path,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--score-bundle", type=Path, required=True)
    parser.add_argument("--msg-cache", type=Path, required=True)
    parser.add_argument("--stage1-embeddings", type=Path, required=True)
    parser.add_argument("--phasea-embeddings", type=Path, required=True)
    parser.add_argument("--fullmsg-embeddings", type=Path, required=True)
    parser.add_argument("--chem-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-full-run", type=Path, required=True,
        help="The frozen run-2413878 modeling directory used for fail-closed rank reproduction.",
    )
    parser.add_argument("--comparator-embedding", action="append", type=named_path, default=[])
    parser.add_argument("--comparator-score", action="append", type=named_path, default=[])
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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _variant_panels(base: dict, variant: str) -> dict:
    output = {}
    for panel_name, panel in base.items():
        scalar, names = build_variant_features(
            panel.scores, panel.query_ptr, panel.scalar, panel.feature_names, variant,
        )
        if variant == FULL:
            assert_full_reconstruction(scalar, names, panel.scalar, panel.feature_names)
            # Train the reference on the original retained block after proving
            # that the matrix builder reconstructs it. This makes the refit
            # byte-for-byte identical at the feature boundary to run 2413878.
            scalar, names = panel.scalar, panel.feature_names
        output[panel_name] = replace(panel, scalar=scalar, feature_names=names)
    contracts = {panel.feature_names for panel in output.values()}
    if len(contracts) != 1 or WSE not in next(iter(contracts)):
        raise RuntimeError(f"variant feature contract drifted: {variant}")
    return output


def _global_folds(base: dict, occurrences: list[tuple[str, int]], args: argparse.Namespace) -> np.ndarray:
    folds = np.asarray([
        formula_fold(base[name].formulas[query], args.folds, args.seed)
        for name, query in occurrences
    ], dtype=np.int16)
    by_key: dict[str, int] = {}
    for index, (name, query) in enumerate(occurrences):
        key = str(base[name].query_keys[query])
        previous = by_key.setdefault(key, int(folds[index]))
        if previous != int(folds[index]):
            raise RuntimeError(f"repeated query crossed formula folds: {key}")
    return folds


def _joint_comparison(
    full_collapsed: dict, ablated_collapsed: dict, resamples: int, seed: int,
) -> dict:
    keys = sorted(full_collapsed)
    if keys != sorted(ablated_collapsed):
        raise RuntimeError("joint ablation query axes differ")
    delta = np.asarray([
        full_collapsed[key][0] - ablated_collapsed[key][0] for key in keys
    ], dtype=np.float64)
    formula = np.asarray([full_collapsed[key][2] for key in keys], dtype="U")
    result = _formula_ci(delta, formula, resamples, seed)
    result["semantics"] = "positive means full Additive has higher query-balanced Top-1 utility"
    return result


def _reference_contract(
    args: argparse.Namespace, base: dict, policy: str, excluded: tuple[str, ...],
    input_sha256: dict[str, str],
) -> dict:
    report_path = args.reference_full_run / "report.json"
    prediction_path = args.reference_full_run / "candidate_predictions_oof.npz"
    checkpoint_path = args.reference_full_run / "unified_candidate_integrator.pt"
    if not report_path.is_file() or not prediction_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("frozen full Additive reference run is incomplete")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("input_sha256") != input_sha256:
        raise RuntimeError("frozen full Additive reference inputs differ from this matrix")
    features = report.get("features", {})
    if (
        features.get("feature_selection_policy") != policy
        or tuple(features.get("excluded_scalar_feature_names", ())) != tuple(excluded)
        or tuple(features.get("scalar_feature_names", ()))
        != tuple(base[PANELS[0]].feature_names)
    ):
        raise RuntimeError("frozen full Additive reference feature contract drifted")
    expected_optimization = {
        "folds": args.folds, "epochs": args.epochs,
        "batch_queries": args.batch_queries, "hidden": args.hidden,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "anchor_strength": args.anchor_strength, "seed": args.seed,
        "hyperparameter_scan": False,
    }
    observed = report.get("optimization", {})
    failed = [key for key, value in expected_optimization.items() if observed.get(key) != value]
    if failed:
        raise RuntimeError(f"frozen full Additive optimization contract drifted: {failed}")
    declared_checkpoint_sha = report.get("checkpoint", {}).get("sha256")
    if declared_checkpoint_sha != sha256_file(checkpoint_path):
        raise RuntimeError("frozen full Additive checkpoint hash drifted")
    declared_prediction_sha = report.get("prediction_ledger", {}).get("sha256")
    if declared_prediction_sha != sha256_file(prediction_path):
        raise RuntimeError("frozen full Additive prediction-ledger hash drifted")
    return {
        "directory": str(args.reference_full_run),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": declared_checkpoint_sha,
        "prediction_ledger": str(prediction_path),
        "prediction_ledger_sha256": declared_prediction_sha,
    }


def _load_frozen_full_oof(reference_contract: dict, base: dict) -> dict[str, np.ndarray]:
    """Load run 2413878 without retraining and fail closed on every saved axis."""
    output = {}
    with np.load(reference_contract["prediction_ledger"], allow_pickle=False) as reference:
        for panel_name in PANELS:
            panel = base[panel_name]
            prefix = f"{panel_name}__"
            axes = {
                "query_ptr": panel.query_ptr,
                "labels": panel.labels,
                "query_keys": panel.query_keys,
                "formula": panel.formulas,
                "near": panel.near,
                "fold": np.asarray([
                    formula_fold(value, 5, 20261026) for value in panel.formulas
                ], dtype=np.int16),
            }
            for field, expected in axes.items():
                key = prefix + field
                if key not in reference.files:
                    raise RuntimeError(f"frozen full reference lacks {key}")
                observed = np.asarray(reference[key])
                if expected.dtype.kind in "USO":
                    observed = observed.astype("U")
                    expected = np.asarray(expected).astype("U")
                if not np.array_equal(observed, expected):
                    raise RuntimeError(f"frozen full reference axis drifted: {key}")
            score_key = prefix + "additive_linear_control"
            if score_key not in reference.files:
                raise RuntimeError(f"frozen full reference lacks {score_key}")
            frozen = np.asarray(reference[score_key], dtype=np.float32)
            if frozen.shape != panel.labels.shape or not np.all(np.isfinite(frozen)):
                raise RuntimeError(f"frozen full reference score axis drifted: {panel_name}")
            output[panel_name] = frozen
    return output


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (
        args.folds != 5 or args.epochs != 20 or args.batch_queries != 256
        or args.hidden != 16 or args.learning_rate != 1e-3
        or args.weight_decay != 1e-4 or args.anchor_strength != 1e-2
        or args.seed != 20261026
    ):
        raise ValueError("strict matrix requires the frozen run-2413878 training configuration")
    if args.bootstrap_resamples < 100:
        raise ValueError("bootstrap-resamples must be at least 100")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("strict matrix requested CUDA without an allocated GPU")

    base, panel_paths = _make_panels(args)
    policy, excluded = _apply_feature_policy(base, True)
    if len(excluded) != 31 or len(base[PANELS[0]].feature_names) != 23:
        raise RuntimeError("strict matrix did not recover the fixed 23-feature reference")
    input_sha256 = {
        "score_bundle": sha256_file(args.score_bundle),
        "msg_report": sha256_file(args.msg_cache / "report.json"),
        "stage1_embeddings": sha256_file(args.stage1_embeddings),
        "phasea_embeddings": sha256_file(args.phasea_embeddings),
        "fullmsg_embeddings": sha256_file(args.fullmsg_embeddings),
        "chem_report": sha256_file(args.chem_dir / "report.json"),
        **{f"panel_{name}": sha256_file(panel_paths[name]) for name in PANELS},
    }
    reference_contract = _reference_contract(
        args, base, policy, excluded, input_sha256,
    )
    panels = {variant: _variant_panels(base, variant) for variant in VARIANT_ORDER}
    occurrences = _occurrences(base)
    folds = _global_folds(base, occurrences, args)

    oof = {
        variant: {
            panel: np.full(len(base[panel].labels), np.nan, dtype=np.float32)
            for panel in PANELS
        }
        for variant in ABLATION_ORDER
    }
    oof[FULL] = _load_frozen_full_oof(reference_contract, base)
    fold_reports: list[dict] = []
    for fold in range(args.folds):
        train = [occ for index, occ in enumerate(occurrences) if folds[index] != fold]
        valid = [occ for index, occ in enumerate(occurrences) if folds[index] == fold]
        train_formula = {str(base[name].formulas[query]) for name, query in train}
        valid_formula = {str(base[name].formulas[query]) for name, query in valid}
        if not train or not valid or train_formula & valid_formula:
            raise RuntimeError(f"global formula fold {fold} is empty or leaked")
        arms = {}
        for variant in ABLATION_ORDER:
            variant_panels = panels[variant]
            mean, scale = _fit_normalization(variant_panels, train)
            # +1 exactly matches the additive arm seed in run 2413878.
            model, history = _fit_arm(
                CONTROL_ARM, variant_panels, train, mean, scale, args,
                args.seed + 100 * fold + 1,
            )
            predicted = _predict_occurrences(
                model, variant_panels, valid, mean, scale, args.device,
            )
            for (panel_name, query), values in predicted.items():
                ptr = variant_panels[panel_name].query_ptr
                left, right = int(ptr[query]), int(ptr[query + 1])
                if np.any(np.isfinite(oof[variant][panel_name][left:right])):
                    raise RuntimeError("OOF candidate row was predicted twice")
                oof[variant][panel_name][left:right] = values
            arms[variant] = {
                "features": len(variant_panels[PANELS[0]].feature_names),
                "initial_loss": history[0], "final_loss": history[-1],
            }
        fold_reports.append({
            "fold": fold, "training_occurrences": len(train),
            "validation_occurrences": len(valid),
            "training_formulas": len(train_formula),
            "validation_formulas": len(valid_formula), "variants": arms,
        })
        print(f"[additive ablation] completed global formula fold {fold + 1}/{args.folds}", flush=True)

    for variant in ABLATION_ORDER:
        for panel_name in PANELS:
            if not np.all(np.isfinite(oof[variant][panel_name])):
                raise RuntimeError(f"OOF coverage incomplete: {variant}/{panel_name}")

    method_scores: dict[str, dict[str, np.ndarray]] = {}
    panel_reports: dict[str, dict] = {}
    for panel_index, panel_name in enumerate(PANELS):
        method_scores[panel_name] = {
            "official_dreams": base[panel_name].scores["official_dreams"],
            WSE: base[panel_name].scores[WSE],
            **{variant: oof[variant][panel_name] for variant in VARIANT_ORDER},
        }
        metrics = {
            method: _metric_summary(scores, base[panel_name])[0]
            for method, scores in method_scores[panel_name].items()
        }
        full_vs = {
            baseline: _comparison(
                oof[FULL][panel_name], method_scores[panel_name][baseline],
                base[panel_name], args.bootstrap_resamples,
                args.seed + 10_000 * panel_index + index,
            )
            for index, baseline in enumerate((
                "official_dreams", WSE, *VARIANT_ORDER[1:],
            ))
        }
        panel_reports[panel_name] = {
            "queries": int(len(base[panel_name].query_ptr) - 1),
            "candidate_molecules": int(len(base[panel_name].labels)),
            "metrics": metrics,
            "full_additive_vs": full_vs,
        }

    joint, collapsed = {}, {}
    for method in method_scores[PANELS[0]]:
        joint[method], collapsed[method] = _joint_query_balanced(base, method_scores, method)
    joint_full_vs = {
        baseline: _joint_comparison(
            collapsed[FULL], collapsed[baseline], args.bootstrap_resamples,
            args.seed + 90_000 + index,
        )
        for index, baseline in enumerate(("official_dreams", WSE, *VARIANT_ORDER[1:]))
    }

    final_entries: dict[str, dict] = {}
    final_reports: dict[str, dict] = {}
    for variant in ABLATION_ORDER:
        variant_panels = panels[variant]
        mean, scale = _fit_normalization(variant_panels, occurrences)
        model, history = _fit_arm(
            CONTROL_ARM, variant_panels, occurrences, mean, scale, args,
            args.seed + 50_001,
        )
        names = variant_panels[PANELS[0]].feature_names
        final_entries[variant] = {
            "feature_names": names,
            "experts": VARIANTS[variant].experts,
            "include_chemistry": VARIANTS[variant].include_chemistry,
            "scalar_mean": mean,
            "scalar_scale": scale,
            "anchor_index": names.index(WSE),
            "hidden": args.hidden,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        }
        final_reports[variant] = {
            "features": len(names), "initial_loss": history[0], "final_loss": history[-1],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        checkpoint_path = staging / "additive_ablation_matrix.pt"
        checkpoint = {
            "schema": CHECKPOINT_SCHEMA,
            "selection_class": "fixed_additive_contribution_matrix",
            "reference_variant": FULL,
            "frozen_full_reference": reference_contract,
            "selection_from_ablation_outcomes": False,
            "feature_policy": policy,
            "excluded_query_constant_features": excluded,
            "variants": final_entries,
            "training": {
                "dataset": "Enveda-180 consumed primary panels",
                "role": "fusion_contribution_development_consumed",
                "folds": args.folds, "epochs": args.epochs,
                "batch_queries": args.batch_queries, "hidden": args.hidden,
                "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
                "anchor_strength": args.anchor_strength, "seed": args.seed,
                "formula_global_across_panels": True,
                "same_hyperparameters_across_variants": True,
                "hyperparameter_scan": False,
            },
        }
        torch.save(checkpoint, checkpoint_path)

        arrays: dict[str, np.ndarray] = {}
        for panel_name in PANELS:
            panel = base[panel_name]
            prefix = f"{panel_name}__"
            arrays.update({
                prefix + "query_ptr": panel.query_ptr,
                prefix + "labels": panel.labels,
                prefix + "query_keys": panel.query_keys,
                prefix + "formula": panel.formulas,
                prefix + "near": panel.near,
                prefix + "fold": np.asarray([
                    formula_fold(value, args.folds, args.seed) for value in panel.formulas
                ], dtype=np.int16),
                prefix + "official_dreams": panel.scores["official_dreams"],
                prefix + WSE: panel.scores[WSE],
            })
            for variant in VARIANT_ORDER:
                arrays[prefix + variant] = oof[variant][panel_name]
        np.savez_compressed(staging / "candidate_predictions_oof.npz", **arrays)

        report = {
            "status": "ENVEDA_UNIFIED_ADDITIVE_STRICT_REFIT_ABLATION_COMPLETE",
            "dataset": "Enveda-180 cross-condition benchmark primary panels",
            "evaluation_role": "fusion_contribution_development_consumed",
            "truth_status": "consumed",
            "allow_final_external_claim": False,
            "reference_model": FULL,
            "variant_order": list(VARIANT_ORDER),
            "variants": {
                name: {
                    "experts": list(spec.experts),
                    "include_chemistry": spec.include_chemistry,
                    "question": spec.question,
                    "feature_names": list(panels[name][PANELS[0]].feature_names),
                }
                for name, spec in VARIANTS.items()
            },
            "panels": panel_reports,
            "query_balanced_joint": joint,
            "query_balanced_joint_full_vs": joint_full_vs,
            "folds": fold_reports,
            "final_refit": final_reports,
            "frozen_full_reference": {
                **reference_contract,
                "semantics": (
                    "The full 23-feature Additive checkpoint and OOF ledger are reused exactly "
                    "from run 2413878; only the four declared ablations are refit."
                ),
            },
            "checkpoint": {"path": checkpoint_path.name, "sha256": sha256_file(checkpoint_path)},
            "prediction_ledger": {
                "path": "candidate_predictions_oof.npz",
                "sha256": sha256_file(staging / "candidate_predictions_oof.npz"),
            },
            "scientific_contract": {
                "full_feature_reconstruction_max_abs_tolerance": 1e-7,
                "dependent_reliability_summaries_recomputed_after_removal": True,
                "query_constant_context_excluded_for_every_variant": True,
                "same_formula_folds_optimizer_budget_and_seed_schedule": True,
                "full_reference_retrained": False,
                "strict_refits": list(ABLATION_ORDER),
                "no_hyperparameter_or_variant_scan": True,
                "interpretation": (
                    "A full-minus-ablation difference estimates group contribution under this "
                    "fixed low-capacity refit; it does not identify a single module's causal effect."
                ),
            },
            "input_sha256": input_sha256,
        }
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"status": report["status"], "output": str(args.output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
