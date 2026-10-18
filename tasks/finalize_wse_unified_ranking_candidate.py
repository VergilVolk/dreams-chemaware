#!/usr/bin/env python
"""Recover the pre-specified B ranking winner without rerunning GNPS OOF.

The completed A/B experiment was incorrectly stopped because an
application-specific ``corrected - 2 * introduced`` utility was promoted to a
hard ranking-selection gate.  This entrypoint reuses the frozen formula-OOF
predictions, adjudicates the primary retrieval endpoints separately from that
legacy utility, and fits exactly one all-identity development model only when
B passes every primary comparison.  It never reads Enveda.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_grand_unified_evidence_model import load_bundle
from train_unified_candidate_energy_crossfit import metric_summary
from train_wse_conditional_residual import (
    CHEMICAL_FEATURE_NAMES,
    SPECTRAL_FEATURE_NAMES,
    arm_beats,
    build_panel,
    evaluation_scores,
    fit_arm,
    lambda2_risk_beats,
    metric_input,
    metric_profile_noninferior,
    paired_comparison,
    sha256,
)


BASELINES = ("wse", "old_four_module", "A")
PARENT_METHODS = ("wse", "old_four_module", "A", "B")


def _validate_axis(observed: np.ndarray, expected: np.ndarray, label: str) -> None:
    observed = np.asarray(observed)
    expected = np.asarray(expected)
    if observed.dtype.kind in "OUS" or expected.dtype.kind in "OUS":
        equal = np.array_equal(observed.astype("U"), expected.astype("U"))
    else:
        equal = np.array_equal(observed, expected)
    if not equal:
        raise RuntimeError(f"parent prediction axis mismatch: {label}")


def load_parent_scores(path: Path, panels: dict[str, dict]) -> dict[str, dict[str, np.ndarray]]:
    output: dict[str, dict[str, np.ndarray]] = {}
    with np.load(path, allow_pickle=False) as source:
        for panel_name, panel in panels.items():
            prefix = f"{panel_name}__"
            for key, expected in (
                ("query_ptr", panel["query_ptr"]),
                ("labels", panel["labels"]),
                ("query_ids", panel["query_ids"]),
                ("candidate_ids", panel["candidate_ids"]),
                ("formula", panel["formulas"]),
                ("near", panel["near"]),
            ):
                name = prefix + key
                if name not in source:
                    raise RuntimeError(f"parent predictions missing {name}")
                _validate_axis(source[name], expected, f"{panel_name}/{key}")
            output[panel_name] = {}
            for method in PARENT_METHODS:
                name = prefix + method
                if name not in source:
                    raise RuntimeError(f"parent predictions missing {name}")
                values = np.asarray(source[name], dtype=np.float32)
                if values.shape != panel["base"].shape or not np.all(np.isfinite(values)):
                    raise RuntimeError(f"invalid parent scores: {panel_name}/{method}")
                output[panel_name][method] = values
    return output


def adjudicate(
    panels: dict[str, dict],
    scores: dict[str, dict[str, np.ndarray]],
    seed: int,
    resamples: int,
) -> tuple[dict, bool]:
    report_panels: dict[str, dict] = {}
    comparisons_by_baseline: dict[str, dict] = {baseline: {} for baseline in BASELINES}
    for panel_index, (panel_name, panel) in enumerate(panels.items()):
        methods = scores[panel_name]
        metrics = {}
        for method, values in methods.items():
            metrics[method], _ = metric_summary(
                evaluation_scores(values, panel["query_ptr"]), metric_input(panel),
            )
        comparisons = {}
        for baseline_index, baseline in enumerate(BASELINES):
            comparison = paired_comparison(
                methods["B"], methods[baseline], panel,
                seed + panel_index * 100 + baseline_index, resamples,
            )
            comparisons[f"B_vs_{baseline}"] = comparison
            comparisons_by_baseline[baseline][panel_name] = comparison
        report_panels[panel_name] = {"methods": metrics, "comparisons": comparisons}

    primary = {}
    legacy = {}
    panel_names = tuple(panels)
    for baseline in BASELINES:
        primary[baseline] = bool(
            arm_beats(comparisons_by_baseline[baseline], panel_names)
            and metric_profile_noninferior(report_panels, "B", baseline, panel_names)
        )
        legacy[baseline] = bool(
            lambda2_risk_beats(comparisons_by_baseline[baseline], panel_names)
        )
    return {
        "panels": report_panels,
        "primary_ranking_pass": primary,
        "legacy_lambda2_utility_pass": legacy,
    }, bool(all(primary.values()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-bundle", type=Path, required=True)
    parser.add_argument("--formula-bundle", type=Path, required=True)
    parser.add_argument("--identity-chem", type=Path, required=True)
    parser.add_argument("--formula-chem", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--parent-predictions", type=Path, required=True)
    parser.add_argument("--additive-control-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    parent = json.loads(args.parent_report.read_text(encoding="utf-8"))
    if parent.get("status") != "WSE_CONDITIONAL_RESIDUAL_DEVELOPMENT_COMPLETE":
        raise RuntimeError("unsupported parent A/B report")
    additive = json.loads(args.additive_control_report.read_text(encoding="utf-8"))
    if additive.get("status") != "WSE_ADDITIVE_MECHANISM_CONTROL_COMPLETE":
        raise RuntimeError("unsupported additive mechanism-control report")
    recipe = parent["optimization"]
    if bool(recipe.get("hyperparameter_scan")):
        raise RuntimeError("parent recipe unexpectedly used a hyperparameter scan")

    bundles = {
        "identity": load_bundle(args.identity_bundle),
        "formula": load_bundle(args.formula_bundle),
    }
    panels = {
        "identity": build_panel(bundles["identity"], args.identity_chem),
        "formula": build_panel(bundles["formula"], args.formula_chem),
    }
    scores = load_parent_scores(args.parent_predictions, panels)
    adjudication, selected = adjudicate(
        panels, scores, int(recipe["seed"]), args.bootstrap_resamples,
    )

    report = {
        "status": "WSE_UNIFIED_RANKING_CANDIDATE_FINALIZATION_COMPLETE",
        "dataset": "GNPS Gold/Silver 10-ppm identity/formula panels",
        "dataset_role": "development_consumed",
        "enveda_accessed": False,
        "selection_correction": (
            "primary retrieval endpoints are adjudicated separately from the legacy "
            "corrected-2*introduced conservative-use utility"
        ),
        "historical_parent_decision": parent.get("decision"),
        "ranking_decision": (
            "SELECT_B_FULL_COVERAGE_DEVELOPMENT_WINNER"
            if selected else "DO_NOT_SELECT_B"
        ),
        "legacy_lambda2_claim": "reported exactly; no claim that the historical utility passed",
        "mechanism_control": {
            "decision": additive.get("mechanism_decision"),
            "boundary": (
                "B may be selected for ranking performance, but conditional interaction "
                "is not claimed superior to the capacity-matched additive C control"
            ),
        },
        **adjudication,
        "optimization_replayed_from_parent": recipe,
        "external_boundary": (
            "development selection only; Enveda-180 is consumed and only a genuinely "
            "new external dataset can establish generalization"
        ),
    }

    args.output.mkdir(parents=True)
    if selected:
        fit_args = argparse.Namespace(
            epochs=int(recipe["epochs"]), hidden=int(recipe["hidden"]),
            learning_rate=float(recipe["learning_rate"]),
            null_weight=float(recipe["null_weight"]),
            trust_weight=float(recipe["trust_weight"]), device=args.device,
        )
        identity_queries = np.arange(len(panels["identity"]["query_ids"]), dtype=np.int64)
        model, mean, scale, bound, history, capacity = fit_arm(
            panels["identity"], identity_queries, "joint", fit_args,
            int(recipe["seed"]) + 1000,
        )
        model_path = args.output / "unified_B_development_candidate.pt"
        torch.save({
            "state_dict": model.cpu().state_dict(),
            "model_class": "AnchoredEvidenceEnergy",
            "selected_arm": "B_joint_residual",
            "mode": "joint",
            "spectral_feature_names": SPECTRAL_FEATURE_NAMES,
            "chemical_feature_names": CHEMICAL_FEATURE_NAMES,
            "spectral_feature_count": len(SPECTRAL_FEATURE_NAMES),
            "hidden": int(recipe["hidden"]),
            "residual_bound_raw_wse_units": bound,
            "scaler_mean": mean,
            "scaler_scale": scale,
            "training_seed": int(recipe["seed"]) + 1000,
            "training_queries": int(len(identity_queries)),
            "enveda_accessed": False,
        }, model_path)
        report["development_candidate_model"] = {
            "path": str(model_path), "sha256": sha256(model_path),
            "fit_scope": "all GNPS identity development queries after frozen OOF selection",
            "residual_bound_raw_wse_units": bound,
            "capacity": capacity, "final_loss": history[-1],
        }
    else:
        report["development_candidate_model"] = None

    report["input_sha256"] = {
        "identity_bundle": sha256(args.identity_bundle),
        "formula_bundle": sha256(args.formula_bundle),
        "identity_chem": sha256(args.identity_chem),
        "formula_chem": sha256(args.formula_chem),
        "parent_report": sha256(args.parent_report),
        "parent_predictions": sha256(args.parent_predictions),
        "additive_control_report": sha256(args.additive_control_report),
    }
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "ranking_decision": report["ranking_decision"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
