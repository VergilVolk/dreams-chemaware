#!/usr/bin/env python
"""Truth-free inference for B and the anchor-conditioned family repair.

Queries with zero or one candidate, or incomplete Stage-1/Phase-A/MSG evidence,
remain on B and carry an explicit status.  No missing score is interpreted as
negative evidence.  The output preserves the input query/candidate axis and
also writes a join-ready candidate table for biology workflows.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from apply_b_anchored_dual_encoder_multireranker import (
    build_b_inference_panel,
    load_inference_bundle,
    query_deployment_status,
    same_axis,
)
from conditional_null_energy_core import AnchoredEvidenceEnergy
from train_b_anchor_conditioned_family_repair import FAMILIES, MSG, AnchoredResidual, msg_logit
from train_b_anchored_dual_encoder_multireranker import query_score_features_inference
from train_wse_conditional_residual import sha256, transform_features


def family_inference_features(bundle: dict) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    names = tuple(np.asarray(bundle["module_names"]).astype("U"))
    missing = [name for name in FAMILIES if name not in names]
    if missing:
        raise RuntimeError(f"extended inference bundle misses families: {missing}")
    ptr = np.asarray(bundle["query_ptr"], dtype=np.int64)
    scores = np.asarray(bundle["scores"], dtype=np.float32)
    availability = np.asarray(bundle["availability"], dtype=bool)
    blocks, family_available, feature_names = [], [], []
    for family in FAMILIES:
        position = names.index(family)
        observed = availability[position] & np.isfinite(scores[position])
        values = np.where(observed, scores[position], 0.0).astype(np.float32)
        values = msg_logit(values) if family == MSG else values
        blocks.append(query_score_features_inference(values, ptr))
        family_available.append(observed)
        prefix = "msg_logit" if family == MSG else family
        feature_names.extend((f"{prefix}_raw", f"{prefix}_gap", f"{prefix}_margin"))
    return (
        np.column_stack(blocks).astype(np.float32),
        np.logical_and.reduce(family_available),
        tuple(feature_names),
    )


def query_repair_eligibility(query_ptr: np.ndarray, candidate_available: np.ndarray
                             ) -> tuple[np.ndarray, np.ndarray]:
    ptr = np.asarray(query_ptr, dtype=np.int64)
    eligible = np.zeros(len(ptr) - 1, dtype=bool)
    for query, (left, right) in enumerate(zip(ptr[:-1], ptr[1:])):
        left, right = int(left), int(right)
        eligible[query] = right - left >= 2 and bool(np.all(candidate_available[left:right]))
    return eligible, np.repeat(eligible, np.diff(ptr))


def write_candidate_table(path: Path, bundle: dict, b_score: np.ndarray,
                          integrated: np.ndarray, residual: np.ndarray,
                          candidate_eligible: np.ndarray) -> None:
    query_ids = np.asarray(bundle["query_ids"]).astype("U")
    candidate_ids = np.asarray(bundle["candidate_ids"]).astype("U")
    ptr = np.asarray(bundle["query_ptr"], dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow((
            "query_id", "candidate_id", "B_score", "anchor_conditioned_score",
            "incremental_residual", "repair_evidence_available",
        ))
        for query, (left, right) in enumerate(zip(ptr[:-1], ptr[1:])):
            for row in range(int(left), int(right)):
                writer.writerow((
                    query_ids[query], candidate_ids[row], float(b_score[row]),
                    float(integrated[row]), float(residual[row]), bool(candidate_eligible[row]),
                ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-bundle", type=Path, required=True)
    parser.add_argument("--continuous-chem", type=Path, required=True)
    parser.add_argument("--extended-bundle", type=Path, required=True)
    parser.add_argument("--b-checkpoint", type=Path, required=True)
    parser.add_argument("--repair-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    table_path = args.output.with_suffix(".candidate_scores.tsv.gz")
    report_path = args.output.with_suffix(args.output.suffix + ".json")
    for path in (args.output, table_path, report_path):
        if path.exists():
            raise FileExistsError(path)

    core_bundle = load_inference_bundle(args.core_bundle)
    extended_bundle = load_inference_bundle(args.extended_bundle)
    if not same_axis(core_bundle, extended_bundle):
        raise RuntimeError("core and extended inference candidate axes differ")
    core = build_b_inference_panel(core_bundle, args.continuous_chem)
    query_status, candidate_is_multi = query_deployment_status(core_bundle["query_ptr"])

    b = torch.load(args.b_checkpoint, map_location="cpu", weights_only=False)
    if b.get("model_class") != "AnchoredEvidenceEnergy" or b.get("selected_arm") != "B_joint_residual":
        raise RuntimeError("checkpoint is not the frozen B candidate")
    b_model = AnchoredEvidenceEnergy(
        core["features"].shape[1], mode=str(b["mode"]),
        spectral_feature_count=int(b["spectral_feature_count"]),
        hidden=int(b["hidden"]), residual_bound=float(b["residual_bound_raw_wse_units"]),
    )
    b_model.load_state_dict(b["state_dict"]); b_model.to(args.device).eval()
    b_features = transform_features(
        core["features"], np.asarray(b["scaler_mean"]), np.asarray(b["scaler_scale"]),
    )
    b_score = np.asarray(core["base"], dtype=np.float32).copy()
    with torch.inference_mode():
        selected = np.flatnonzero(candidate_is_multi)
        if len(selected):
            b_score[selected] = b_model(
                torch.as_tensor(core["base"][selected], dtype=torch.float32, device=args.device),
                torch.as_tensor(b_features[selected], dtype=torch.float32, device=args.device),
            ).cpu().numpy().astype(np.float32)

    repair = torch.load(args.repair_checkpoint, map_location="cpu", weights_only=False)
    if (repair.get("model_class") != "AnchoredResidual"
            or repair.get("anchor") != "unified_B_development_candidate"
            or repair.get("arm") != "anchor_conditioned_family"):
        raise RuntimeError("checkpoint is not the anchor-conditioned family repair")
    static, family_available, static_names = family_inference_features(extended_bundle)
    expected_static = tuple(repair["feature_names"])[3:]
    if static_names != expected_static:
        raise RuntimeError("repair family feature contract drifted")
    anchor = query_score_features_inference(b_score, np.asarray(core_bundle["query_ptr"], dtype=np.int64))
    features = np.column_stack((anchor, static)).astype(np.float32)
    if features.shape[1] != len(repair["feature_names"]):
        raise RuntimeError("repair feature width drifted")
    mean = np.asarray(repair["feature_mean"], dtype=np.float32)
    scale = np.asarray(repair["feature_scale"], dtype=np.float32)
    x = (features - mean) / scale
    query_eligible, candidate_eligible = query_repair_eligibility(
        core_bundle["query_ptr"], family_available,
    )
    repair_model = AnchoredResidual(
        features.shape[1], int(repair["hidden"]), float(repair["residual_bound_B_units"]),
    )
    repair_model.load_state_dict(repair["state_dict"]); repair_model.to(args.device).eval()
    residual = np.zeros(len(b_score), dtype=np.float32)
    with torch.inference_mode():
        selected = np.flatnonzero(candidate_eligible)
        if len(selected):
            residual[selected] = repair_model.residual(
                torch.as_tensor(x[selected], dtype=torch.float32, device=args.device),
            ).cpu().numpy().astype(np.float32)
    integrated = b_score + residual
    repair_status = np.where(
        np.diff(np.asarray(core_bundle["query_ptr"], dtype=np.int64)) == 0, "no_candidate",
        np.where(
            np.diff(np.asarray(core_bundle["query_ptr"], dtype=np.int64)) == 1,
            "single_candidate_B_no_repair",
            np.where(query_eligible, "anchor_conditioned_repair", "missing_family_evidence_B_fallback"),
        ),
    ).astype("U")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema=np.asarray("b_anchor_conditioned_family_inference_v1"),
        query_ptr=np.asarray(core_bundle["query_ptr"], dtype=np.int64),
        query_ids=np.asarray(core_bundle["query_ids"]).astype("U"),
        candidate_ids=np.asarray(core_bundle["candidate_ids"]).astype("U"),
        query_status=query_status, repair_status=repair_status,
        repair_evidence_available=candidate_eligible,
        B_score=b_score, integrated_score=integrated, incremental_residual=residual,
    )
    write_candidate_table(
        table_path, core_bundle, b_score, integrated, residual, candidate_eligible,
    )
    report = {
        "status": "B_ANCHOR_CONDITIONED_FAMILY_INFERENCE_COMPLETE",
        "labels_accessed": False, "missing_evidence_semantics": "explicit B fallback",
        "queries": int(len(query_status)), "candidates": int(len(b_score)),
        "repair_status_counts": {
            name: int(np.sum(repair_status == name)) for name in (
                "no_candidate", "single_candidate_B_no_repair", "anchor_conditioned_repair",
                "missing_family_evidence_B_fallback",
            )
        },
        "inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in (
            ("core_bundle", args.core_bundle), ("continuous_chem", args.continuous_chem),
            ("extended_bundle", args.extended_bundle), ("B_checkpoint", args.b_checkpoint),
            ("repair_checkpoint", args.repair_checkpoint),
        )},
        "outputs": {
            "npz": {"path": str(args.output), "sha256": sha256(args.output)},
            "candidate_table": {"path": str(table_path), "sha256": sha256(table_path)},
        },
        "handoff": "join candidate table on query_id and candidate_id without changing biology inclusion criteria",
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
