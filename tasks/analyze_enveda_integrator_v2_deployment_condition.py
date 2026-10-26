#!/usr/bin/env python3
"""Replay the frozen Enveda integrator with V2 present versus unavailable.

This is a deployment-condition sensitivity check, not a new performance
estimate.  Both arms use the final refit checkpoint on the already-consumed
Enveda development candidates.  The only changed input is ChemAware V2:
the observed continuous correct/null evidence is replaced by the registered
all-zero, availability-zero no-op used for structure-unavailable application
queries.  No biological response or application outcome is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from enveda_unified_candidate_integrator_core import (
    CHECKPOINT_SCHEMA,
    CONTROL_ARM,
    predict_from_payload,
    unavailable_chemaware,
)
from massspecgym_embedding_readout_core import build_independent_scalar_features


PANELS = ("identity_disjoint", "formula_disjoint")
ENCODERS = (
    "noise_v1", "chemaware_stage1", "chemaware_phasea",
    "noise_chemaware_fullmsg",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before the weights_only keyword.
        payload = torch.load(path, map_location="cpu")
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError("unexpected unified-integrator checkpoint schema")
    if CONTROL_ARM not in payload.get("arms", {}):
        raise RuntimeError("checkpoint lacks the frozen additive arm")
    if tuple(payload.get("optional_families", ())) != (
        "chemaware_v2_continuous_null",
    ):
        raise RuntimeError("checkpoint V2 optional-family contract drifted")
    if payload.get("optional_v2_noop_helper") != (
        "enveda_unified_candidate_integrator_core.unavailable_chemaware"
    ):
        raise RuntimeError("checkpoint V2 no-op contract drifted")
    return payload


def unique_top1_rows(scores: np.ndarray, query_ptr: np.ndarray) -> np.ndarray:
    """Return the unique top row, or -1 when the maximum is tied."""
    output = np.full(len(query_ptr) - 1, -1, dtype=np.int64)
    for query, (left, right) in enumerate(zip(query_ptr[:-1], query_ptr[1:])):
        left, right = int(left), int(right)
        if right <= left:
            raise RuntimeError("development replay encountered an empty query")
        local = np.asarray(scores[left:right], dtype=np.float64)
        winners = np.flatnonzero(local == np.max(local))
        if len(winners) == 1:
            output[query] = left + int(winners[0])
    return output


def load_chemistry(path: Path, query_ptr: np.ndarray) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as body:
        observed_ptr = np.asarray(body["query_ptr"], dtype=np.int64)
        if not np.array_equal(observed_ptr, query_ptr):
            raise RuntimeError(f"continuous chemistry query axis drifted: {path}")
        chemistry = {
            "chem_valid": np.asarray(body["availability"], dtype=bool),
            "candidate_utility_correct": np.asarray(
                body["utility_correct"], dtype=np.float32,
            ),
            "candidate_utility_zero": np.asarray(
                body["utility_zero"], dtype=np.float32,
            ),
            "candidate_utility_reversed": np.asarray(
                body["utility_reversed"], dtype=np.float32,
            ),
            # The integrator's historical ``rotated`` slot is the frozen
            # candidate-mismatch null in this continuous-evidence export.
            "candidate_utility_rotated": np.asarray(
                body["utility_candidate_mismatch"], dtype=np.float32,
            ),
        }
    candidates = int(query_ptr[-1])
    if any(values.shape != (candidates,) for values in chemistry.values()):
        raise RuntimeError(f"continuous chemistry candidate axis drifted: {path}")
    return chemistry


def panel_replay(
    payload: dict,
    prediction_ledger: np.lib.npyio.NpzFile,
    chem_dir: Path,
    panel: str,
    device: torch.device,
) -> dict[str, object]:
    prefix = f"{panel}__"
    query_ptr = np.asarray(prediction_ledger[prefix + "query_ptr"], dtype=np.int64)
    labels = np.asarray(prediction_ledger[prefix + "labels"], dtype=np.int8)
    if (
        query_ptr.ndim != 1 or len(query_ptr) < 2 or query_ptr[0] != 0
        or np.any(np.diff(query_ptr) <= 0) or int(query_ptr[-1]) != len(labels)
    ):
        raise RuntimeError(f"invalid consumed development axis: {panel}")
    if np.any(np.add.reduceat(labels.astype(np.int64), query_ptr[:-1]) != 1):
        raise RuntimeError(f"development queries lack exactly one positive: {panel}")

    graph = SimpleNamespace(
        query_ptr=query_ptr,
        molecule_ptr=np.arange(len(labels) + 1, dtype=np.int64),
        molecule_label=labels,
    )
    encoder_scores = {
        name: np.asarray(prediction_ledger[prefix + name], dtype=np.float64)
        for name in ENCODERS
    }
    evidence = {
        "v1_cosine": encoder_scores["noise_v1"],
        "weighted_entropy": np.asarray(
            prediction_ledger[prefix + "weighted_spectral_entropy"],
            dtype=np.float64,
        ),
        "p2b_fused": np.asarray(
            prediction_ledger[prefix + "p2b_noise_v1_frozen"],
            dtype=np.float64,
        ),
    }
    msg = np.asarray(
        prediction_ledger[prefix + "noise_msg_fusion_stack_v1"],
        dtype=np.float64,
    )
    chemistry_path = chem_dir / f"continuous_chem_{panel}.npz"
    observed_chemistry = load_chemistry(chemistry_path, query_ptr)
    unavailable = unavailable_chemaware(len(labels))

    observed_scalar, observed_names = build_independent_scalar_features(
        evidence, graph, encoder_scores, "noise_v1", observed_chemistry,
        auxiliary_scores={"noise_msg_fusion_stack_v1": msg},
        include_reliability_context=True,
    )
    unavailable_scalar, unavailable_names = build_independent_scalar_features(
        evidence, graph, encoder_scores, "noise_v1", unavailable,
        auxiliary_scores={"noise_msg_fusion_stack_v1": msg},
        include_reliability_context=True,
    )
    if observed_names != unavailable_names:
        raise RuntimeError("V2 deployment replay changed the feature axis")
    observed_scores = predict_from_payload(
        payload, observed_names, observed_scalar, arm=CONTROL_ARM, device=device,
    )
    unavailable_scores = predict_from_payload(
        payload, unavailable_names, unavailable_scalar, arm=CONTROL_ARM, device=device,
    )
    observed_top = unique_top1_rows(observed_scores, query_ptr)
    unavailable_top = unique_top1_rows(unavailable_scores, query_ptr)
    observed_correct = np.zeros(len(observed_top), dtype=bool)
    unavailable_correct = np.zeros(len(unavailable_top), dtype=bool)
    observed_unique = observed_top >= 0
    unavailable_unique = unavailable_top >= 0
    observed_correct[observed_unique] = labels[observed_top[observed_unique]].astype(bool)
    unavailable_correct[unavailable_unique] = labels[
        unavailable_top[unavailable_unique]
    ].astype(bool)
    availability = observed_chemistry["chem_valid"]
    query_has_v2 = np.add.reduceat(availability.astype(np.int64), query_ptr[:-1]) > 0
    changed = observed_top != unavailable_top
    no_v2_queries = ~query_has_v2
    max_abs_no_v2_score_delta = (
        float(np.max(np.abs(observed_scores[np.repeat(no_v2_queries, np.diff(query_ptr))]
                                - unavailable_scores[np.repeat(no_v2_queries, np.diff(query_ptr))])))
        if no_v2_queries.any() else 0.0
    )
    if max_abs_no_v2_score_delta > 1e-6:
        raise RuntimeError("all-unavailable V2 queries changed under the exact no-op replay")

    return {
        "queries": int(len(query_ptr) - 1),
        "candidates": int(len(labels)),
        "v2_available_candidates": int(availability.sum()),
        "queries_with_any_v2": int(query_has_v2.sum()),
        "queries_without_v2": int(no_v2_queries.sum()),
        "top1_changed_when_v2_removed": int(changed.sum()),
        "top1_changed_fraction": float(changed.mean()),
        "top1_agreement": float(np.mean(~changed)),
        "changed_queries_with_any_v2": int(np.sum(changed & query_has_v2)),
        "changed_queries_without_v2": int(np.sum(changed & no_v2_queries)),
        "maximum_absolute_score_delta_on_all_unavailable_queries": max_abs_no_v2_score_delta,
        "final_refit_replay_only_not_oof": {
            "observed_v2_recall1": float(observed_correct.mean()),
            "v2_unavailable_recall1": float(unavailable_correct.mean()),
            "observed_minus_unavailable_pp": float(
                100.0 * (observed_correct.mean() - unavailable_correct.mean())
            ),
            "observed_v2_corrected": int(np.sum(observed_correct & ~unavailable_correct)),
            "observed_v2_introduced": int(np.sum(~observed_correct & unavailable_correct)),
        },
        "chemistry_input": {
            "path": str(chemistry_path), "sha256": sha256(chemistry_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prediction-ledger", type=Path, required=True)
    parser.add_argument("--chem-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    for path in (args.checkpoint, args.prediction_ledger):
        if not path.is_file():
            raise FileNotFoundError(path)
    payload = load_checkpoint(args.checkpoint)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested without an allocated GPU")

    with np.load(args.prediction_ledger, allow_pickle=False) as ledger:
        panels = {
            panel: panel_replay(payload, ledger, args.chem_dir, panel, device)
            for panel in PANELS
        }
    report = {
        "status": "ENVEDA_INTEGRATOR_V2_DEPLOYMENT_CONDITION_REPLAY_COMPLETE",
        "dataset": "Enveda-180 consumed fusion-meta-development panels",
        "evaluation_role": "post-result_deployment_condition_sensitivity_only",
        "truth_status": "consumed",
        "arm": CONTROL_ARM,
        "labels_accessed": True,
        "biological_application_data_accessed": False,
        "candidate_axis_changed": False,
        "model_refit_or_selected": False,
        "interpretation_boundary": (
            "Final-refit paired input replay only, not OOF and not an external performance "
            "estimate. It quantifies sensitivity to the exact V2-unavailable semantics used "
            "for structure-unavailable application candidates; it cannot establish causal "
            "V2 contribution or justify changing the frozen model."
        ),
        "checkpoint": {
            "path": str(args.checkpoint), "sha256": sha256(args.checkpoint),
            "feature_selection_policy": payload.get("feature_selection_policy"),
            "v2_noop_helper": payload.get("optional_v2_noop_helper"),
        },
        "prediction_ledger": {
            "path": str(args.prediction_ledger),
            "sha256": sha256(args.prediction_ledger),
        },
        "panels": panels,
    }
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "report.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
