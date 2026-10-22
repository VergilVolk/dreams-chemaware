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
from train_b_anchored_dual_encoder_multireranker import AnchoredResidual, build_features
from train_grand_unified_evidence_model import load_bundle
from train_wse_conditional_residual import build_panel, sha256, transform_features


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

    core_bundle = load_bundle(args.core_bundle)
    extended_bundle = load_bundle(args.extended_bundle)
    if not same_axis(core_bundle, extended_bundle):
        raise RuntimeError("core and extended candidate axes differ")
    core = build_panel(core_bundle, args.continuous_chem)

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
    with torch.inference_mode():
        b_score = b_model(
            torch.as_tensor(core["base"], dtype=torch.float32, device=args.device),
            torch.as_tensor(b_features, dtype=torch.float32, device=args.device),
        ).cpu().numpy().astype(np.float32)

    full = torch.load(args.full_checkpoint, map_location="cpu", weights_only=False)
    if (full.get("model_class") != "AnchoredResidual"
            or full.get("anchor") != "unified_B_development_candidate"
            or full.get("arm") != "complete_spectral"):
        raise RuntimeError("checkpoint is not a B-anchored integrated residual")
    all_features, all_names = build_features(extended_bundle)
    columns = np.asarray(full["feature_columns"], dtype=np.int64)
    selected_names = tuple(np.asarray(all_names)[columns])
    if selected_names != tuple(full["feature_names"]):
        raise RuntimeError("integrated feature contract drifted")
    x = all_features[:, columns] / np.asarray(full["feature_scale"], dtype=np.float32)
    full_model = AnchoredResidual(
        len(columns), int(full["hidden"]), float(full["residual_bound_B_units"]), linear=False,
    )
    full_model.load_state_dict(full["state_dict"]); full_model.to(args.device).eval()
    with torch.inference_mode():
        residual = full_model.residual(torch.as_tensor(x, dtype=torch.float32, device=args.device)).cpu().numpy().astype(np.float32)
    score = b_score + residual

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema=np.asarray("b_anchored_dual_encoder_multireranker_scores_v1"),
        query_ptr=np.asarray(core_bundle["query_ptr"], dtype=np.int64),
        query_ids=np.asarray(core_bundle["query_ids"]).astype("U"),
        candidate_ids=np.asarray(core_bundle["candidate_ids"]).astype("U"),
        B_score=b_score, integrated_score=score, incremental_residual=residual,
    )
    report = {
        "status": "B_ANCHORED_DUAL_ENCODER_MULTIRERANKER_INFERENCE_COMPLETE",
        "labels_accessed": False,
        "candidates": int(len(score)), "queries": int(len(core_bundle["query_ptr"]) - 1),
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
