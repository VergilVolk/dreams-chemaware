"""Fail-closed validator for a full-candidate shared embedding pilot."""
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
    checkpoint_path = args.output_dir / "shared_spectrum_adapter.pt"
    if not report_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("pilot report/checkpoint is incomplete")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0 on older cluster environments.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    contracts = report.get("preflight", {}).get("contracts", {})
    gates = report.get("gates", {})
    required_contracts = {
        "formula_disjoint": True,
        "held_formula_candidates_excluded_from_training_gradients": True,
        "same_spectrum_adapter_query_reference": True,
        "outer_fold_evaluated": False,
        "candidate_input_at_deployment": False,
    }
    if any(contracts.get(key) is not value for key, value in required_contracts.items()):
        raise RuntimeError("shared embedding contract failed")
    if report.get("status") != "PASS" or not gates or not all(gates.values()):
        raise RuntimeError(f"pilot did not pass scientific gates: {gates}")
    if report.get("selected_step", 0) <= 0:
        raise RuntimeError("pilot selected the unchanged initialization")
    if report.get("final_inner", {}).get("delta_recall1", 0.0) <= 0:
        raise RuntimeError("pilot did not improve inner Recall@1")
    if checkpoint.get("status") != "chemaware_full_candidate_aligned_embedding":
        raise RuntimeError("unexpected checkpoint status")
    if (checkpoint.get("P2b_used") is not False
            or checkpoint.get("query_reference_encoder_shared") is not True
            or checkpoint.get("candidate_inputs_at_inference") is not False
            or checkpoint.get("molecule_projector_state") is not None
            or checkpoint.get("validation_pass") is not True):
        raise RuntimeError("checkpoint is not deployable shared spectrum embedding")
    print(json.dumps({
        "status": "PASS", "selected_step": report["selected_step"],
        "delta_recall1": report["final_inner"]["delta_recall1"],
        "delta_mrr": report["final_inner"]["delta_mrr"],
        "preservation": report["preservation"]["mean"],
    }, indent=2))


if __name__ == "__main__":
    main()
