"""Validate one direct-training arm without making a chemistry claim.

Chemical attribution is impossible from a single arm.  This validator checks
only artifact integrity, deployment-shape contracts, and safety.  It must
never make the arm loadable as a released ChemAware shared encoder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.output_dir / "report.json"
    checkpoint_path = args.output_dir / "final_shared_encoder.pt"
    if not report_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("direct run report/checkpoint is incomplete")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    contracts = report.get("preflight", {}).get("contracts", {})
    required = {
        "formula_disjoint": True,
        "held_formula_candidates_excluded_from_training_gradients": True,
        "same_raw_spectrum_encoder_query_reference": True,
        "official_checkpoint_initialization": True,
        "candidate_input_at_deployment": False,
        "outer_fold_evaluated": False,
    }
    if any(contracts.get(key) is not value for key, value in required.items()):
        raise RuntimeError("direct shared-encoder contract failed")
    gates = report.get("artifact_gates", {})
    if report.get("status") != "ARM_COMPLETE" or not gates or not all(gates.values()):
        raise RuntimeError(f"direct arm failed artifact/safety gates: {gates}")
    if report.get("causal_chemistry_status") != "NOT_EVALUATED_SINGLE_ARM":
        raise RuntimeError("a single arm must not contain a causal chemistry verdict")
    if report.get("selected_step", 0) <= 0:
        raise RuntimeError("direct run selected unchanged official initialization")
    if checkpoint.get("status") != "chemaware_full_candidate_direct_arm_checkpoint":
        raise RuntimeError("unexpected direct checkpoint status")
    if (checkpoint.get("inference_clean_spectrum_only") is not True
            or checkpoint.get("inference_clean_only") is not True
            or checkpoint.get("query_reference_encoder_shared") is not True
            or checkpoint.get("candidate_inputs_at_inference") is not False
            or checkpoint.get("P2b_used") is not False
            or checkpoint.get("formal") is not True
            or checkpoint.get("artifact_validation_pass") is not True
            or checkpoint.get("causal_chemistry_pass") is not False
            or checkpoint.get("release_eligible") is not False
            or checkpoint.get("validation_pass") is not False
            or not isinstance(checkpoint.get("model_state"), dict)):
        raise RuntimeError("direct arm checkpoint quarantine contract failed")
    print(json.dumps({
        "status": "ARM_ARTIFACT_VALIDATED_NOT_CAUSALLY_RELEASED",
        "selected_step": report["selected_step"],
        "delta_recall1": report["final_inner"]["delta_recall1"],
        "delta_mrr": report["final_inner"]["delta_mrr"],
        "preservation": report["preservation"],
        "mean_clip_fraction": report["mean_clip_fraction"],
        "causal_chemistry_pass": False,
        "release_eligible": False,
    }, indent=2))


if __name__ == "__main__":
    main()
