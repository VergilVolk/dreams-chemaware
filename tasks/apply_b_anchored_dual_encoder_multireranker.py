#!/usr/bin/env python
"""Apply frozen B and the selected dual-encoder/multi-reranker residual.

All inputs are candidate-keyed.  This entrypoint is deliberately independent
of benchmark labels and writes both the B score and the integrated score so a
biology workflow can attribute every changed candidate decision.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from conditional_null_energy_core import AnchoredEvidenceEnergy
from train_b_anchored_dual_encoder_multireranker import (
    AnchoredResidual,
    build_features,
    query_score_features_inference,
)
from train_wse_conditional_residual import (
    SPECTRAL_FEATURE_NAMES,
    build_chemical_features,
    sha256,
    transform_features,
)


WSE = "weighted_spectral_entropy"
NOISE = "noise_v1"
P2B = "p2b_noise_v1_frozen"


def load_inference_bundle(path: Path) -> dict:
    """Load a candidate bundle without labels, groups or truth metadata."""
    required = {"module_names", "scores", "availability", "query_ptr", "query_ids", "candidate_ids"}
    with np.load(path, allow_pickle=False) as source:
        missing = required - set(source.files)
        bundle = {name: np.asarray(source[name]) for name in required if name in source.files}
    if missing:
        raise ValueError(f"{path} missing inference fields: {sorted(missing)}")
    ptr = np.asarray(bundle["query_ptr"], dtype=np.int64)
    n_candidates = len(bundle["candidate_ids"])
    if (ptr.ndim != 1 or len(ptr) != len(bundle["query_ids"]) + 1
            or ptr[0] != 0 or ptr[-1] != n_candidates or np.any(np.diff(ptr) < 0)):
        raise ValueError(f"{path} has an invalid zero/single/multi query pointer")
    scores = np.asarray(bundle["scores"])
    availability = np.asarray(bundle["availability"])
    expected = (len(bundle["module_names"]), n_candidates)
    if scores.shape != expected or availability.shape != expected:
        raise ValueError(f"{path} module/candidate axes differ")
    return bundle


def load_chem_inference(path: Path, bundle: dict) -> dict[str, np.ndarray]:
    required = {
        "schema", "query_ptr", "query_ids", "candidate_ids", "availability",
        "utility_correct", "utility_zero", "utility_reversed", "utility_candidate_mismatch",
    }
    with np.load(path, allow_pickle=False) as source:
        missing = required - set(source.files)
        if missing:
            raise ValueError(f"{path} missing inference chemical fields: {sorted(missing)}")
        # Deliberately do not read query_formula, near_query or any benchmark
        # truth fields even when a training artifact happens to contain them.
        chem = {name: np.asarray(source[name]) for name in required}
    if str(np.asarray(chem["schema"]).item()) != "gnps_chemaware_continuous_null_evidence_v1":
        raise RuntimeError(f"unsupported chemical evidence: {path}")
    checks = {
        "query_ptr": np.array_equal(chem["query_ptr"], bundle["query_ptr"]),
        "query_ids": np.array_equal(chem["query_ids"].astype("U"), bundle["query_ids"].astype("U")),
        "candidate_ids": np.array_equal(
            chem["candidate_ids"].astype("U"), bundle["candidate_ids"].astype("U")
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"chemical evidence inference axis drifted: {failed}")
    return chem


def build_b_inference_panel(bundle: dict, chem_path: Path) -> dict:
    names = tuple(map(str, bundle["module_names"]))
    missing = [name for name in (WSE, NOISE, P2B) if name not in names]
    if missing:
        raise RuntimeError(f"core inference bundle misses modules: {missing}")
    scores = np.asarray(bundle["scores"], dtype=np.float32)
    availability = np.asarray(bundle["availability"], dtype=bool)
    ptr = np.asarray(bundle["query_ptr"], dtype=np.int64)
    positions = {name: names.index(name) for name in (WSE, NOISE, P2B)}
    for name, position in positions.items():
        if not np.all(availability[position]):
            raise RuntimeError(f"{name} must be available on every existing candidate")
    spectral = np.column_stack(tuple(
        query_score_features_inference(scores[positions[name]], ptr)
        for name in (WSE, NOISE, P2B)
    )).astype(np.float32)
    chemical, _nulls, chemical_available = build_chemical_features(
        load_chem_inference(chem_path, bundle)
    )
    return {
        "base": scores[positions[WSE]].copy(),
        "features": np.column_stack((spectral, chemical)).astype(np.float32),
        "chemical_available": chemical_available,
    }


def query_deployment_status(query_ptr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.diff(np.asarray(query_ptr, dtype=np.int64))
    status = np.where(
        lengths == 0, "no_candidate",
        np.where(lengths == 1, "single_candidate_no_reranking", "ranked_multi_candidate"),
    ).astype("U")
    candidate_is_multi = np.repeat(lengths >= 2, lengths)
    return status, candidate_is_multi


def same_axis(left: dict, right: dict) -> bool:
    return (
        np.array_equal(left["query_ptr"], right["query_ptr"])
        and np.array_equal(np.asarray(left["query_ids"]).astype("U"), np.asarray(right["query_ids"]).astype("U"))
        and np.array_equal(np.asarray(left["candidate_ids"]).astype("U"), np.asarray(right["candidate_ids"]).astype("U"))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-bundle", type=Path, required=True)
    parser.add_argument("--continuous-chem", type=Path, required=True)
    parser.add_argument("--extended-bundle", type=Path, required=True)
    parser.add_argument("--b-checkpoint", type=Path, required=True)
    parser.add_argument("--full-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    core_bundle = load_inference_bundle(args.core_bundle)
    extended_bundle = load_inference_bundle(args.extended_bundle)
    if not same_axis(core_bundle, extended_bundle):
        raise RuntimeError("core and extended candidate axes differ")
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
    b_features = transform_features(core["features"], np.asarray(b["scaler_mean"]), np.asarray(b["scaler_scale"]))
    # No learned ranking is applied when there is no competition.  Singleton
    # candidates retain the WSE anchor and are explicitly marked downstream.
    b_score = np.asarray(core["base"], dtype=np.float32).copy()
    with torch.inference_mode():
        selected = np.flatnonzero(candidate_is_multi)
        if len(selected):
            b_score[selected] = b_model(
                torch.as_tensor(core["base"][selected], dtype=torch.float32, device=args.device),
                torch.as_tensor(b_features[selected], dtype=torch.float32, device=args.device),
            ).cpu().numpy().astype(np.float32)

    full = torch.load(args.full_checkpoint, map_location="cpu", weights_only=False)
    if (full.get("model_class") != "AnchoredResidual"
            or full.get("anchor") != "unified_B_development_candidate"
            or full.get("arm") != "complete_spectral"):
        raise RuntimeError("checkpoint is not a B-anchored integrated residual")
    expected_experts = (
        "chemaware_stage1_encoder", "chemaware_phase_a_encoder", "noise_msg_fusion_stack_v1",
    )
    if tuple(full.get("expert_names", ())) != expected_experts:
        raise RuntimeError("deployment checkpoint expert family is unsupported")
    # The retired RRF arm is not a production dependency.  Build only the
    # three feature families named by the selected complete-spectral model.
    all_features, all_names = build_features(
        extended_bundle, allow_noncompetitive=True, experts=(0, 1, 2),
    )
    columns = np.asarray(full["feature_columns"], dtype=np.int64)
    selected_names = tuple(np.asarray(all_names)[columns])
    if selected_names != tuple(full["feature_names"]):
        raise RuntimeError("integrated feature contract drifted")
    x = all_features[:, columns] / np.asarray(full["feature_scale"], dtype=np.float32)
    full_model = AnchoredResidual(
        len(columns), int(full["hidden"]), float(full["residual_bound_B_units"]), linear=False,
    )
    full_model.load_state_dict(full["state_dict"]); full_model.to(args.device).eval()
    residual = np.zeros(len(b_score), dtype=np.float32)
    with torch.inference_mode():
        selected = np.flatnonzero(candidate_is_multi)
        if len(selected):
            residual[selected] = full_model.residual(
                torch.as_tensor(x[selected], dtype=torch.float32, device=args.device)
            ).cpu().numpy().astype(np.float32)
    score = b_score + residual

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema=np.asarray("b_anchored_dual_encoder_multireranker_scores_v1"),
        query_ptr=np.asarray(core_bundle["query_ptr"], dtype=np.int64),
        query_ids=np.asarray(core_bundle["query_ids"]).astype("U"),
        candidate_ids=np.asarray(core_bundle["candidate_ids"]).astype("U"),
        query_status=query_status,
        B_score=b_score, integrated_score=score, incremental_residual=residual,
    )
    report = {
        "status": "B_ANCHORED_DUAL_ENCODER_MULTIRERANKER_INFERENCE_COMPLETE",
        "labels_accessed": False,
        "candidates": int(len(score)), "queries": int(len(core_bundle["query_ptr"]) - 1),
        "query_status_counts": {
            name: int(np.sum(query_status == name)) for name in (
                "no_candidate", "single_candidate_no_reranking", "ranked_multi_candidate"
            )
        },
        "inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in (
            ("core_bundle", args.core_bundle), ("continuous_chem", args.continuous_chem),
            ("extended_bundle", args.extended_bundle), ("B_checkpoint", args.b_checkpoint),
            ("full_checkpoint", args.full_checkpoint),
        )},
        "output_sha256": sha256(args.output),
        "handoff": "candidate-keyed B and integrated scores for identical-condition biology evaluation",
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
