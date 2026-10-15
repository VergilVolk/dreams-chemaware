#!/usr/bin/env python
"""Open one frozen Enveda panel once for the frozen candidate-energy model."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_frozen_grand_unified_external import validate_freeze_chain
from evaluate_gnps_gold_silver_10ppm_embeddings import graph_from_panel
from grand_unified_evidence_core import rank_logit_evidence
from train_grand_unified_evidence_model import load_bundle, sha256_file
from train_unified_candidate_energy_crossfit import (
    AnchoredCandidateEnergy,
    compare,
    metric_summary,
    prepare,
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def selective_curve(scores: np.ndarray, prepared: dict) -> list[dict]:
    ptr = prepared["query_ptr"]
    labels = prepared["labels"]
    correct = np.zeros(len(ptr) - 1, dtype=bool)
    margin = np.zeros(len(ptr) - 1, dtype=np.float64)
    for query, (left, right) in enumerate(zip(ptr[:-1], ptr[1:])):
        left, right = int(left), int(right)
        block = scores[left:right]
        order = np.argsort(-block, kind="stable")
        correct[query] = labels[left + int(order[0])] > 0
        margin[query] = float(block[order[0]] - block[order[1]]) if len(order) > 1 else 0.0
    ranked = np.argsort(-margin, kind="stable")
    output = []
    for coverage in (0.10, 0.25, 0.50, 0.75, 0.90, 1.00):
        count = max(1, int(round(coverage * len(ranked))))
        chosen = ranked[:count]
        output.append({
            "coverage": coverage,
            "queries": int(count),
            "accuracy": float(np.mean(correct[chosen])),
            "error_rate": float(1.0 - np.mean(correct[chosen])),
            "minimum_margin": float(np.min(margin[chosen])),
        })
    return output


@torch.inference_mode()
def predict(model: AnchoredCandidateEnergy, prepared: dict, device: torch.device) -> np.ndarray:
    ptr = prepared["query_ptr"]
    output = np.empty(len(prepared["labels"]), dtype=np.float32)
    for first in range(0, len(ptr) - 1, 2048):
        last = min(first + 2048, len(ptr) - 1)
        left, right = int(ptr[first]), int(ptr[last])
        evidence = torch.as_tensor(prepared["evidence"][:, left:right], device=device)
        availability = torch.as_tensor(prepared["availability"][:, left:right], device=device)
        output[left:right] = model(evidence, availability).cpu().numpy()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--freeze-report", type=Path, required=True)
    parser.add_argument("--test-bundle", type=Path, required=True)
    parser.add_argument("--component-score-bundle", type=Path, required=True)
    parser.add_argument("--reference-score-bundle", type=Path, required=True)
    parser.add_argument("--module-registry", type=Path, required=True)
    parser.add_argument("--benchmark-report", type=Path, required=True)
    parser.add_argument("--benchmark-checksums", type=Path, required=True)
    parser.add_argument("--benchmark-panel", type=Path, required=True)
    parser.add_argument("--opening-ledger", type=Path, required=True)
    parser.add_argument("--global-opening-ledger", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    required = (
        args.model, args.freeze_report, args.test_bundle, args.component_score_bundle,
        args.reference_score_bundle, args.module_registry, args.benchmark_report,
        args.benchmark_checksums, args.benchmark_panel, args.global_opening_ledger,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if args.out_report.exists() or args.opening_ledger.exists():
        raise FileExistsError("one-time Enveda output or opening ledger already exists")

    args.opening_ledger.parent.mkdir(parents=True, exist_ok=True)
    reservation = {
        "schema": "candidate_energy_external_opening_ledger_v1",
        "status": "PANEL_EVALUATION_RESERVED_UNDER_GLOBAL_OPENING",
        "reserved_at_utc": utc_now(),
        "model_sha256": sha256_file(args.model),
        "test_bundle_sha256": sha256_file(args.test_bundle),
    }
    with args.opening_ledger.open("x", encoding="utf-8") as handle:
        json.dump(reservation, handle, indent=2)

    try:
        global_opening = json.loads(args.global_opening_ledger.read_text(encoding="utf-8"))
        if global_opening.get("status") != "OPENING_RESERVED_ALL_SCORES_FROZEN_LABELS_NOT_YET_LOADED":
            raise RuntimeError("global Enveda opening was not reserved before labels")
        if global_opening.get("model_sha256") != reservation["model_sha256"]:
            raise RuntimeError("global Enveda opening model differs")
        freeze = json.loads(args.freeze_report.read_text(encoding="utf-8"))
        benchmark = json.loads(args.benchmark_report.read_text(encoding="utf-8"))
        bundle = load_bundle(args.test_bundle)
        component = load_bundle(args.component_score_bundle)
        for key in (
            "module_names", "scores", "availability", "query_ptr", "labels",
            "group_ids", "query_ids", "candidate_ids",
        ):
            if not np.array_equal(np.asarray(bundle[key]), np.asarray(component[key])):
                raise ValueError(f"sealed/component field differs: {key}")
        modules = validate_freeze_chain(
            bundle=bundle,
            freeze_report=freeze,
            model_sha256=reservation["model_sha256"],
            freeze_report_sha256=sha256_file(args.freeze_report),
            registry_sha256=sha256_file(args.module_registry),
            benchmark_report=benchmark,
            benchmark_report_sha256=sha256_file(args.benchmark_report),
            benchmark_checksums_sha256=sha256_file(args.benchmark_checksums),
            benchmark_panel_sha256=sha256_file(args.benchmark_panel),
            component_score_bundle_sha256=sha256_file(args.component_score_bundle),
        )
        checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
        if checkpoint.get("model_class") != "AnchoredCandidateEnergy":
            raise RuntimeError("frozen checkpoint has the wrong model class")
        if tuple(map(str, checkpoint.get("module_names", ()))) != modules:
            raise RuntimeError("frozen checkpoint module order differs from sealed evidence")
        device = torch.device(args.device)
        model = AnchoredCandidateEnergy(modules).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        prepared = prepare(bundle)
        unified_scores = predict(model, prepared, device)
        base_index = modules.index("weighted_spectral_entropy")
        primary = compare(
            unified_scores, prepared, prepared["evidence"][base_index],
            args.seed, args.bootstrap_resamples,
        )

        graph = graph_from_panel(args.benchmark_panel)
        reference_metrics = {}
        with np.load(args.reference_score_bundle, allow_pickle=False) as source:
            method_names = list(map(str, source["method_names"]))
            panel_name = str(np.asarray(bundle["panel"]).reshape(-1)[0])
            matrix = np.asarray(source[f"scores_{panel_name}"], dtype=np.float32)
        for index, name in enumerate(method_names):
            molecule = np.maximum.reduceat(matrix[index], graph.molecule_ptr[:-1])
            calibrated, _ = rank_logit_evidence(
                molecule[None, :], prepared["query_ptr"],
                np.ones((1, len(molecule)), dtype=np.float32),
            )
            reference_metrics[name], _ = metric_summary(calibrated[0], prepared)

        result = {
            "status": "UNIFIED_CANDIDATE_ENERGY_ENVEDA_OPENED_ONCE_COMPLETE",
            "dataset_id": benchmark["dataset_id"],
            "evaluation_role": "one_time_external_test",
            "panel": str(np.asarray(bundle["panel"]).reshape(-1)[0]),
            "queries": int(len(prepared["query_ptr"]) - 1),
            "opened_at_utc": utc_now(),
            "model_sha256": reservation["model_sha256"],
            "freeze_report_sha256": sha256_file(args.freeze_report),
            "test_bundle_sha256": reservation["test_bundle_sha256"],
            "reference_score_bundle_sha256": sha256_file(args.reference_score_bundle),
            "unified_vs_wse": primary,
            "reference_methods": reference_metrics,
            "selective_annotation_curve": selective_curve(unified_scores, prepared),
            "selection_boundary": "No method, coefficient, threshold or module may change after this result.",
        }
        args.out_report.parent.mkdir(parents=True, exist_ok=True)
        args.out_report.write_text(json.dumps(result, indent=2), encoding="utf-8")
        reservation.update({
            "status": "OPENED_ONCE_COMPLETE_NO_RETRY",
            "completed_at_utc": utc_now(),
            "result_sha256": sha256_file(args.out_report),
        })
        args.opening_ledger.write_text(json.dumps(reservation, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    except Exception as exc:
        reservation.update({
            "status": "OPENING_FAILED_NO_AUTOMATIC_RETRY",
            "failed_at_utc": utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        args.opening_ledger.write_text(json.dumps(reservation, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
