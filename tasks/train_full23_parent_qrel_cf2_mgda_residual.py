#!/usr/bin/env python
"""Fit the preregistered CF2 cross-family residual on the Full23 v1 parent.

CF2 preserves the parent + query-relative two-source MGDA training and global
formula outer-fold contract.  Its only change is the delta representation:
candidate-varying main effects are augmented by a fixed set of cross-family
products.  The allowed families are representation, spectral, and continuous
ChemAware V2 evidence.  There are no within-family products, query-constant
features, source features, gates, or feature/weight grids.

The four continuous V2 columns are multiplied by their candidate availability
before they enter either a main effect or a product.  Consequently unavailable
chemistry is an exact no-op by construction.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
import copy
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

import train_full23_parent_query_relative_mgda_residual as parent_qrel
from enveda_unified_candidate_integrator_core import CHECKPOINT_SCHEMA, CONTROL_ARM
from gnps_pair_score_cache import sha256_file
from train_enveda_unified_candidate_integrator import PANELS, WSE, PanelData, _comparison, _metric_summary
from train_full23_centered_residual_from_frozen_ledger import PARENT, build_panels, load_checkpoint
from train_full23_multisource_additive import (
    SOURCES,
    combine_panels,
    fit_normalization,
    occurrences,
    rebalance_selected_source_weights,
    set_source_balanced_weights,
)
from train_full23_query_relative_mgda import formula_oof_partition, query_relative_panel
from evaluate_full23_centered_residual_from_frozen_gnps_ledger import _panel as gnps_panel


METHOD = "full23_parent_qrel_cf2_mgda_residual"
SEQUENTIAL_PARENT = parent_qrel.SEQUENTIAL_PARENT
SCHEMA = "full23_parent_qrel_cf2_mgda_residual_v1"
STATUS = "FULL23_PARENT_QREL_CF2_MGDA_RESIDUAL_DEVELOPMENT_COMPLETE"

REPRESENTATION = (
    "noise_v1_query_zscore",
    "chemaware_stage1_cosine_query_zscore",
    "chemaware_phasea_cosine_query_zscore",
    "noise_chemaware_fullmsg_cosine_query_zscore",
)
SPECTRAL = (
    "weighted_spectral_entropy_query_zscore",
    "p2b_noise_v1_frozen_query_zscore",
)
CHEMICAL = (
    "chemaware_v2_correct_minus_zero",
    "chemaware_v2_correct_minus_worst_null",
    "chemaware_v2_null_spread",
    "chemaware_v2_delta_percentile",
)
AVAILABILITY = "chemaware_v2_available"
QUERY_CONSTANT = (
    "reliability_top1_vote_fraction",
    "reliability_mean_percentile",
    "reliability_percentile_std",
    "chemaware_v2_delta_top_gap",
)
EXPECTED_BASE_FEATURES = 19
EXPECTED_INTERACTIONS = 32
EXPECTED_FEATURES = EXPECTED_BASE_FEATURES + EXPECTED_INTERACTIONS


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enveda-ledger", type=Path, required=True)
    parser.add_argument("--enveda-chem-dir", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--gnps-score-dir", type=Path, required=True)
    parser.add_argument("--gnps-axis-ledger", type=Path, required=True)
    parser.add_argument("--gnps-chem-dir", type=Path, required=True)
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
    parser.set_defaults(center_residual_penalty=False)
    return parser.parse_args()


def _product_name(left: str, right: str) -> str:
    # The shared scaler preserves a semantic zero for every feature whose name
    # starts with ``chemaware_v2_``.  Chemistry-bearing interactions therefore
    # deliberately inherit that prefix; otherwise centering would turn an
    # unavailable raw zero into a nonzero normalized pseudo-signal.
    prefix = "chemaware_v2_cf2" if right in CHEMICAL else "cf2"
    return f"{prefix}__{left}__x__{right}"


def cf2_feature_names(base_names: tuple[str, ...]) -> tuple[str, ...]:
    """Return the exact 51-feature CF2 axis after validating Full23 qrel."""
    names = tuple(map(str, base_names))
    required = set(REPRESENTATION + SPECTRAL + CHEMICAL + (AVAILABILITY,) + QUERY_CONSTANT)
    missing = sorted(required - set(names))
    if missing:
        raise RuntimeError(f"CF2 input misses required features: {missing}")
    retained = tuple(name for name in names if name not in QUERY_CONSTANT)
    if len(names) != 23 or len(retained) != EXPECTED_BASE_FEATURES:
        raise RuntimeError("CF2 requires the exact 23-feature query-relative Full23 axis")
    products = tuple(
        [_product_name(left, right) for left in REPRESENTATION for right in SPECTRAL]
        + [_product_name(left, right) for left in REPRESENTATION for right in CHEMICAL]
        + [_product_name(left, right) for left in SPECTRAL for right in CHEMICAL]
    )
    if len(products) != EXPECTED_INTERACTIONS or len(set(products)) != len(products):
        raise AssertionError("CF2 product contract drifted")
    output = retained + products
    if len(output) != EXPECTED_FEATURES or len(set(output)) != len(output):
        raise AssertionError("CF2 feature axis drifted")
    return output


def cf2_panel(panel: PanelData) -> PanelData:
    """Build candidate-varying main effects plus fixed cross-family products."""
    qrel = query_relative_panel(panel)
    names = tuple(map(str, qrel.feature_names))
    output_names = cf2_feature_names(names)
    values = np.asarray(qrel.scalar, dtype=np.float32).copy()
    availability = values[:, names.index(AVAILABILITY)].copy()
    if np.any(~np.isfinite(availability)) or np.any((availability < 0) | (availability > 1)):
        raise RuntimeError("CF2 chemistry availability must be finite in [0,1]")

    # Enforce exact missing-chemistry no-op even if an upstream cache contains
    # stale nonzero placeholders in one of the continuous chemistry columns.
    for name in CHEMICAL:
        values[:, names.index(name)] *= availability

    retained_names = tuple(name for name in names if name not in QUERY_CONSTANT)
    columns = [values[:, names.index(name):names.index(name) + 1] for name in retained_names]
    for left, right in (
        [(left, right) for left in REPRESENTATION for right in SPECTRAL]
        + [(left, right) for left in REPRESENTATION for right in CHEMICAL]
        + [(left, right) for left in SPECTRAL for right in CHEMICAL]
    ):
        columns.append(
            (values[:, names.index(left)] * values[:, names.index(right)])[:, None]
        )
    scalar = np.concatenate(columns, axis=1).astype(np.float32)
    if scalar.shape != (len(values), EXPECTED_FEATURES) or not np.all(np.isfinite(scalar)):
        raise RuntimeError("CF2 scalar feature construction drifted")
    return PanelData(
        name=qrel.name,
        graph=qrel.graph,
        scalar=scalar,
        feature_names=output_names,
        labels=qrel.labels,
        query_ptr=qrel.query_ptr,
        formulas=qrel.formulas,
        query_keys=qrel.query_keys,
        near=qrel.near,
        query_weights=qrel.query_weights.copy(),
        scores=qrel.scores,
    )


def _source_methods(
    original: PanelData, combined_scores: np.ndarray, combined_parent: np.ndarray,
    candidate_slice: slice, method: str = METHOD,
) -> dict[str, np.ndarray]:
    methods = {
        method: combined_scores[candidate_slice],
        SEQUENTIAL_PARENT: combined_parent[candidate_slice],
    }
    for baseline in (PARENT, WSE, "official_dreams", "noise_v1", "p2b_noise_v1_frozen"):
        if baseline in original.scores:
            methods[baseline] = original.scores[baseline]
    return methods


def run(
    args: argparse.Namespace,
    *,
    fit_delta: Callable = parent_qrel.fit_delta_mgda,
    method: str = METHOD,
    schema: str = SCHEMA,
    status: str = STATUS,
    checkpoint_name: str = "full23_parent_qrel_cf2_mgda_residual.pt",
    delta_semantics: str = "released_parent + zero_initialized_linear_CF2_delta",
    task_gradient_combination: str = "closed_form_two_task_MGDA_via_AdamW",
    run_label: str = "Full23 CF2",
) -> dict:
    """Execute the fixed CF2 protocol with one explicitly named optimizer."""
    if args.output.exists():
        raise FileExistsError(args.output)
    if (
        args.folds != 5 or args.seed != 20261026 or args.epochs != 20
        or args.batch_queries != 256 or args.hidden != 16
        or args.learning_rate != 1e-3 or args.weight_decay != 1e-4
        or args.anchor_strength != 1e-2 or args.bootstrap_resamples < 100
        or args.device != "cpu" or args.center_residual_penalty
    ):
        raise ValueError("CF2 has one fixed, non-scanned training contract")

    parent_payload = load_checkpoint(args.parent_checkpoint)
    parent_report = json.loads(args.parent_report.read_text(encoding="utf-8"))
    if (
        parent_report.get("prediction_ledger", {}).get("sha256")
        != sha256_file(args.enveda_ledger)
        or parent_report.get("checkpoint", {}).get("sha256")
        != sha256_file(args.parent_checkpoint)
    ):
        raise RuntimeError("Enveda evidence is not the released Full23 v1 pair")

    enveda_raw = build_panels(args.enveda_ledger, args.enveda_chem_dir, parent_payload)
    gnps_raw: dict[str, PanelData] = {}
    expected_parent_names = tuple(map(str, parent_payload["scalar_feature_names"]))
    with np.load(args.gnps_axis_ledger, allow_pickle=False) as axis:
        for name in PANELS:
            panel, frozen_parent = gnps_panel(
                name, args.gnps_score_dir, axis, args.gnps_chem_dir,
                expected_parent_names,
            )
            panel.scores[PARENT] = frozen_parent
            gnps_raw[name] = panel
    raw_by_source = {"enveda": enveda_raw, "gnps": gnps_raw}
    enveda_cf2 = {name: cf2_panel(panel) for name, panel in enveda_raw.items()}
    gnps_cf2 = {name: cf2_panel(panel) for name, panel in gnps_raw.items()}

    combined: dict[str, PanelData] = {}
    slices: dict[str, dict[str, tuple[slice, slice]]] = {}
    for name in PANELS:
        combined[name], slices[name] = combine_panels(enveda_cf2[name], gnps_cf2[name], name)
    unique_queries = set_source_balanced_weights(combined)
    all_occurrences = occurrences(combined)
    oof = {name: np.full(len(panel.labels), np.nan, np.float32) for name, panel in combined.items()}
    parent_oof = {
        name: np.full(len(panel.labels), np.nan, np.float32) for name, panel in combined.items()
    }
    fold_reports = []
    for held_out in range(args.folds):
        train, valid = formula_oof_partition(
            combined, all_occurrences, held_out, args.folds, args.seed,
        )
        parent_train = parent_qrel.outer_parent_training_occurrences(
            enveda_raw, held_out, args.folds, args.seed,
        )
        parent_train_formulas = {
            str(enveda_raw[name].formulas[query]).strip() for name, query in parent_train
        }
        valid_formulas = {
            str(combined[name].formulas[query]).strip() for name, query in valid
        }
        if parent_train_formulas & valid_formulas:
            raise RuntimeError("outer validation formula leaked into parent fitting")

        parent_mean, parent_scale = parent_qrel._fit_normalization(enveda_raw, parent_train)
        parent_model, parent_history = parent_qrel._fit_arm(
            CONTROL_ARM, enveda_raw, parent_train, parent_mean, parent_scale,
            args, args.seed + 100 * held_out + 1,
        )
        fold_parent_scores = parent_qrel._parent_scores_from_model(
            parent_model, raw_by_source, parent_mean, parent_scale, args.device,
        )

        train_unique, train_source_mass = rebalance_selected_source_weights(combined, train)
        delta_mean, delta_scale = fit_normalization(combined, train)
        delta_model, delta_history = fit_delta(
            combined, train, fold_parent_scores, delta_mean, delta_scale,
            args, args.seed + 100 * held_out + 51,
        )
        predictions = parent_qrel.predict_delta(
            delta_model, combined, valid, fold_parent_scores,
            delta_mean, delta_scale, args.device,
        )
        for (name, query), prediction in predictions.items():
            left, right = map(int, combined[name].query_ptr[query:query + 2])
            if np.any(np.isfinite(oof[name][left:right])):
                raise RuntimeError("outer validation row was predicted more than once")
            oof[name][left:right] = prediction
            parent_oof[name][left:right] = fold_parent_scores[name][left:right]
        fold_reports.append({
            "fold": held_out,
            "training_occurrences": len(train),
            "validation_occurrences": len(valid),
            "training_unique_queries": train_unique,
            "training_source_mass_for_delta_normalization": train_source_mass,
            "parent_training_source": "enveda_only",
            "parent_training_occurrences": len(parent_train),
            "parent_training_formulas": len(parent_train_formulas),
            "validation_formulas": len(valid_formulas),
            "parent_seed": args.seed + 100 * held_out + 1,
            "delta_seed": args.seed + 100 * held_out + 51,
            "parent_initial_loss": parent_history[0],
            "parent_final_loss": parent_history[-1],
            "delta_first_epoch": delta_history[0],
            "delta_last_epoch": delta_history[-1],
            "parent_validation_formula_overlap": 0,
        })
        print(f"[{run_label}] completed outer formula fold {held_out + 1}/{args.folds}", flush=True)
    if any(np.any(~np.isfinite(values)) for values in oof.values()):
        raise RuntimeError("CF2 outer OOF predictions are incomplete")
    if any(np.any(~np.isfinite(values)) for values in parent_oof.values()):
        raise RuntimeError("CF2 outer OOF parent predictions are incomplete")

    reports = {source: {} for source in SOURCES}
    prediction_artifact: dict[str, np.ndarray] = {}
    for panel_index, name in enumerate(PANELS):
        for source_index, source in enumerate(SOURCES):
            original = enveda_raw[name] if source == "enveda" else gnps_raw[name]
            _, candidate_slice = slices[name][source]
            methods = _source_methods(
                original, oof[name], parent_oof[name], candidate_slice, method,
            )
            comparisons = {
                f"{method}_vs_{baseline}": _comparison(
                    methods[method], methods[baseline], original,
                    args.bootstrap_resamples,
                    args.seed + 10_000 + panel_index * 100 + source_index * 10 + offset,
                )
                for offset, baseline in enumerate((SEQUENTIAL_PARENT, PARENT, WSE))
            }
            reports[source][name] = {
                "queries": len(original.query_ptr) - 1,
                "candidate_molecules": len(original.labels),
                "methods": {
                    method: _metric_summary(scores, original)[0]
                    for method, scores in methods.items()
                },
                "comparisons": comparisons,
            }
            prefix = f"{source}__{name}__"
            prediction_artifact.update({
                prefix + "query_ptr": original.query_ptr,
                prefix + "labels": original.labels,
                prefix + "query_keys": original.query_keys,
                prefix + "formula": original.formulas,
                prefix + "near": original.near,
                **{prefix + method: values for method, values in methods.items()},
            })

    final_parent_scores = parent_qrel._released_parent_scores(
        parent_payload, raw_by_source, args.device,
    )
    final_unique, final_source_mass = rebalance_selected_source_weights(
        combined, all_occurrences,
    )
    final_mean, final_scale = fit_normalization(combined, all_occurrences)
    final_delta, final_history = fit_delta(
        combined, all_occurrences, final_parent_scores, final_mean, final_scale,
        args, args.seed + 50_051,
    )
    feature_names = list(next(iter(combined.values())).feature_names)
    checkpoint = {
        "schema": schema,
        "parent_integrator_schema": CHECKPOINT_SCHEMA,
        "parent_arm": CONTROL_ARM,
        "parent_checkpoint_sha256": sha256_file(args.parent_checkpoint),
        "parent_payload": copy.deepcopy(parent_payload),
        "delta_feature_names": feature_names,
        "delta_mean": final_mean,
        "delta_scale": final_scale,
        "delta_state_dict": {
            key: value.detach().cpu() for key, value in final_delta.state_dict().items()
        },
        "delta_semantics": delta_semantics,
        "feature_contract": {
            "candidate_varying_main_effects": EXPECTED_BASE_FEATURES,
            "cross_family_products": EXPECTED_INTERACTIONS,
            "representation_x_spectral": len(REPRESENTATION) * len(SPECTRAL),
            "representation_x_chemical": len(REPRESENTATION) * len(CHEMICAL),
            "spectral_x_chemical": len(SPECTRAL) * len(CHEMICAL),
            "within_family_products": 0,
            "query_constant_features": 0,
            "chemistry_missing_semantics": "availability-times-continuous; exact zero",
        },
        "training": {
            "sources": list(SOURCES),
            "source_indicator": False,
            "source_weight_grid": False,
            "task_gradient_combination": task_gradient_combination,
            "query_centered_delta_penalty": True,
            "final_parent": "released Full23 v1 all-Enveda refit checkpoint",
        },
    }
    report = {
        "status": status,
        "method": method,
        "dataset_roles": {
            "enveda": "consumed development strict global formula outer-fit",
            "gnps": "consumed development strict global formula outer-fit",
        },
        "allow_final_external_claim": False,
        "evaluation_design": "sequential outer-fit; not nested stacking",
        "strict_global_formula_outer_folds": True,
        "source_indicator": False,
        "source_weight_grid": False,
        "features": feature_names,
        "feature_contract": checkpoint["feature_contract"],
        "unique_queries": unique_queries,
        "optimization": {
            "folds": args.folds,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_queries_per_source": args.batch_queries,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "delta_anchor_strength": args.anchor_strength,
            "delta_anchor_penalty": "query-centered delta MSE",
            "hyperparameter_scan": False,
        },
        "folds": fold_reports,
        "panels": reports,
        "final_refit": {
            "parent": "released Full23 v1 checkpoint",
            "parent_checkpoint_sha256": sha256_file(args.parent_checkpoint),
            "unique_queries": final_unique,
            "normalization_source_mass": final_source_mass,
            "delta_first_epoch": final_history[0],
            "delta_last_epoch": final_history[-1],
            "delta_parameters": int(sum(value.numel() for value in final_delta.parameters())),
        },
        "claim_boundary": (
            "Both sources are consumed development. CF2 is a fixed cross-family "
            "candidate requiring a genuinely new source after development selection."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        checkpoint_path = staging / checkpoint_name
        torch.save(checkpoint, checkpoint_path)
        prediction_path = staging / "candidate_predictions_outer_oof.npz"
        np.savez_compressed(prediction_path, **prediction_artifact)
        report["artifacts"] = {
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "predictions": prediction_path.name,
            "predictions_sha256": sha256_file(prediction_path),
            "parent_checkpoint_sha256": sha256_file(args.parent_checkpoint),
            "enveda_ledger_sha256": sha256_file(args.enveda_ledger),
            "gnps_axis_ledger_sha256": sha256_file(args.gnps_axis_ledger),
        }
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"status": report["status"], "output": str(args.output)}, indent=2))
    return report


def main() -> None:
    run(arguments())


if __name__ == "__main__":
    main()
