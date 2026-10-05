#!/usr/bin/env python
"""Train the preregistered formula-isolated grand-fusion expert router.

The outer fold estimates routing performance; a distinct inner formula fold
chooses the switch threshold.  The external GNPS panels are never read here.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import joblib
import numpy as np

from grand_fusion_router_core import (
    MODEL_SETTINGS,
    correctness_from_winners,
    feature_names,
    fit_correctness_models,
    predict_correctness,
    query_features,
    risk_ledger,
    route,
    select_threshold,
)
from noise_final_core import sha256_file, stable_fold


ROOT = Path(__file__).resolve().parents[1]
FOLD_COUNT = 5
FOLD_SEED = 20261004
THRESHOLDS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 1.10)
DEFAULT_METHOD_FIELDS = (
    "noise_v1=v1_cosine",
    "weighted_spectral_entropy=weighted_entropy",
    "noise_p2b_v1=p2b_fused",
    "neutral_loss=neutral_loss_sqrt_cosine",
    "sqrt_cosine=sqrt_cosine",
    "spectral_entropy=entropy_similarity",
    "official_dreams=official_cosine",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=ROOT / "tasks/grand_fusion_components_v1.json")
    parser.add_argument("--method", action="append", default=None, metavar="NAME=FIELD")
    parser.add_argument("--default-method", default="weighted_spectral_entropy")
    parser.add_argument("--risk-lambda", type=float, default=2.0)
    parser.add_argument("--allow-audit-only", action="store_true")
    return parser.parse_args()


def parse_methods(values: list[str] | None) -> tuple[tuple[str, ...], dict[str, str]]:
    mapping: dict[str, str] = {}
    for value in values or list(DEFAULT_METHOD_FIELDS):
        if "=" not in value:
            raise ValueError(f"method must be NAME=FIELD: {value}")
        name, field = value.split("=", 1)
        if not name or not field or name in mapping:
            raise ValueError(f"invalid or duplicate method: {value}")
        mapping[name] = field
    if len(mapping) < 2:
        raise ValueError("router requires at least two experts")
    return tuple(mapping), mapping


def registry_gate(registry_path: Path, methods: tuple[str, ...], allow_audit: bool) -> dict:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    entries = {row["name"]: row for row in registry["components"]}
    aliases = {
        "neutral_loss": "neutral_loss_sqrt_cosine",
        "sqrt_cosine": "p2b_sqrt_cosine",
        "spectral_entropy": "p2b_unweighted_entropy",
    }
    forbidden = []
    unknown = []
    for method in methods:
        entry = entries.get(method) or entries.get(aliases.get(method, ""))
        if entry is None:
            unknown.append(method)
            continue
        status = entry["deployment"]
        if status in {"audit_only", "retired_audit_only", "sealed_pending_gate"} and not allow_audit:
            forbidden.append((method, status))
    if unknown:
        raise RuntimeError(f"unregistered expert methods: {unknown}")
    if forbidden:
        raise RuntimeError(f"audit-only experts cannot affect production routing: {forbidden}")
    return registry


def molecule_max(edge_scores: np.ndarray, molecule_ptr: np.ndarray) -> np.ndarray:
    return np.maximum.reduceat(
        np.asarray(edge_scores, dtype=np.float64),
        np.asarray(molecule_ptr, dtype=np.int64)[:-1],
    )


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.risk_lambda < 1.0:
        raise ValueError("risk-lambda must be >=1; introductions may not be underweighted")
    methods, mapping = parse_methods(args.method)
    if args.default_method not in methods:
        raise ValueError("default method must be included in --method")
    registry = registry_gate(args.registry, methods, args.allow_audit_only)

    evidence_file = args.evidence / "evidence.npz"
    evidence_report_file = args.evidence / "report.json"
    evidence_report = json.loads(evidence_report_file.read_text(encoding="utf-8"))
    if evidence_report.get("status") != "NOISE_MSG_PAIR_EVIDENCE_COMPLETE":
        raise RuntimeError("evidence artifact is not frozen and complete")
    with np.load(evidence_file, allow_pickle=False) as body:
        evidence = {name: np.asarray(body[name]) for name in body.files}
    missing = sorted(set(mapping.values()) - set(evidence))
    if missing:
        raise RuntimeError(f"evidence misses expert fields: {missing}")

    query_ptr = np.asarray(evidence["query_ptr"], dtype=np.int64)
    molecule_ptr = np.asarray(evidence["molecule_ptr"], dtype=np.int64)
    labels = np.asarray(evidence["molecule_label"], dtype=np.int8)
    formulas = np.asarray(evidence["query_formula"], dtype=str)
    scores = {name: molecule_max(evidence[field], molecule_ptr) for name, field in mapping.items()}
    features, winners = query_features(scores, query_ptr, methods)
    correctness = correctness_from_winners(winners, query_ptr, labels)
    folds = np.asarray([stable_fold(value, FOLD_COUNT, FOLD_SEED) for value in formulas], dtype=np.int8)

    oof_probabilities = np.full((len(formulas), len(methods)), np.nan, dtype=np.float64)
    oof_chosen = np.full(len(formulas), -1, dtype=np.int64)
    fold_reports: list[dict] = []
    for outer in range(FOLD_COUNT):
        held = folds == outer
        calibration_fold = (outer + 1) % FOLD_COUNT
        calibration = folds == calibration_fold
        fit_inner = ~(held | calibration)
        inner_models = fit_correctness_models(features[fit_inner], correctness[fit_inner], methods)
        calibration_probabilities = predict_correctness(inner_models, features[calibration], methods)
        threshold, threshold_rows = select_threshold(
            calibration_probabilities, winners[calibration], correctness[calibration],
            methods, args.default_method, THRESHOLDS, args.risk_lambda,
        )

        refit = ~held
        outer_models = fit_correctness_models(features[refit], correctness[refit], methods)
        held_probabilities = predict_correctness(outer_models, features[held], methods)
        held_chosen = route(
            held_probabilities, winners[held], methods, args.default_method, threshold,
        )
        oof_probabilities[held] = held_probabilities
        oof_chosen[held] = held_chosen
        fold_report = {
            "outer_fold": outer,
            "held_queries": int(held.sum()),
            "inner_calibration_fold": calibration_fold,
            "threshold": threshold,
            "inner_threshold_ledger": threshold_rows,
            "outer_risk_ledger": risk_ledger(
                correctness[held], held_chosen, methods, args.default_method, args.risk_lambda,
            ),
        }
        fold_reports.append(fold_report)
        print(json.dumps(fold_report["outer_risk_ledger"]), flush=True)

    if np.any(~np.isfinite(oof_probabilities)) or np.any(oof_chosen < 0):
        raise RuntimeError("OOF prediction ledger is incomplete")
    nested_ledger = risk_ledger(
        correctness, oof_chosen, methods, args.default_method, args.risk_lambda,
    )
    deployment_threshold, deployment_rows = select_threshold(
        oof_probabilities, winners, correctness, methods, args.default_method,
        THRESHOLDS, args.risk_lambda,
    )
    final_models = fit_correctness_models(features, correctness, methods)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".grand_router.", dir=args.output.parent))
    try:
        package = {
            "models": final_models,
            "methods": methods,
            "method_fields": mapping,
            "default_method": args.default_method,
            "threshold": deployment_threshold,
            "risk_lambda": args.risk_lambda,
            "feature_names": feature_names(methods),
            "model_settings": MODEL_SETTINGS,
            "fold_count": FOLD_COUNT,
            "fold_seed": FOLD_SEED,
        }
        joblib.dump(package, staging / "router.joblib")
        np.savez_compressed(
            staging / "oof_predictions.npz",
            probabilities=oof_probabilities.astype(np.float32),
            chosen=oof_chosen.astype(np.int16),
            correctness=correctness.astype(np.int8),
            folds=folds,
            method_names=np.asarray(methods),
        )
        report = {
            "status": "GRAND_FUSION_ROUTER_FROZEN",
            "methods": list(methods),
            "method_fields": mapping,
            "default_method": args.default_method,
            "feature_names": list(feature_names(methods)),
            "risk_lambda": args.risk_lambda,
            "nested_formula_oof": {"ledger": nested_ledger, "folds": fold_reports},
            "deployment_threshold": {
                "value": deployment_threshold,
                "selection_ledger": deployment_rows,
                "claim_limit": "Selected from development OOF predictions; external performance must be read only from frozen GNPS panels.",
            },
            "leakage_controls": [
                "formula-grouped outer folds",
                "separate inner threshold fold",
                "no GNPS access",
                "no candidate-count or reference-multiplicity features",
                "strict ties fail and force abstention",
            ],
            "registry_schema": registry["schema"],
            "provenance": {
                "evidence_sha256": sha256_file(evidence_file),
                "evidence_report_sha256": sha256_file(evidence_report_file),
                "registry_sha256": sha256_file(args.registry),
            },
        }
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
