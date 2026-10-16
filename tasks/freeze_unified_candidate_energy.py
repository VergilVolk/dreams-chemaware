#!/usr/bin/env python
"""Freeze the preplanned best GNPS candidate-energy ablation before Enveda."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_grand_unified_evidence_model import load_bundle
from train_unified_candidate_energy_crossfit import (
    BASE,
    fit_model,
    prepare,
    select_modules,
    sha256,
)


MODULES = (
    "noise_v1",
    "weighted_spectral_entropy",
    "p2b_noise_v1_frozen",
    "chemaware_v2_action",
)
SELECTED_ARM = "without_chemaware_stage1_encoder"


def validate_selected_arm(crossfit: dict) -> tuple[dict, dict]:
    """Validate the predeclared development arm before any external freeze.

    The cross-fit producer applies a positive risk-net gate to its full arm.
    The final selected ablation must satisfy the same scientific boundary;
    otherwise an ablation search could silently bypass the external-admission
    contract merely by improving mean Recall@1.
    """
    if crossfit.get("status") != "UNIFIED_CANDIDATE_ENERGY_CROSSFIT_COMPLETE":
        raise RuntimeError("GNPS crossfit report is incomplete")
    arm = crossfit.get("arms", {}).get(SELECTED_ARM)
    if arm is None or tuple(arm.get("modules", ())) != MODULES:
        raise RuntimeError("selected preplanned ablation is absent or has drifted")

    development = {}
    for panel in ("identity", "formula"):
        result = arm["panels"][panel]
        selected_r1 = float(result["candidate"]["recall@1"])
        competing = {
            name: float(body["panels"][panel]["candidate"]["recall@1"])
            for name, body in crossfit["arms"].items()
            if name != SELECTED_ARM
        }
        if selected_r1 < max(competing.values()):
            raise RuntimeError(
                f"selected arm is not the highest preplanned Recall@1 arm on {panel}"
            )

        paired = result["paired_vs_wse"]
        if paired["delta_recall@1"] <= 0 or paired["formula_cluster_bootstrap_95ci"][0] <= 0:
            raise RuntimeError(f"selected arm lacks positive GNPS support on {panel}")
        if paired["risk_net_corrected_minus_2introduced"] <= 0:
            raise RuntimeError(
                f"selected arm fails corrected-2*introduced safety on {panel}"
            )
        micro_auc_delta = (
            float(result["candidate"]["micro_candidate_auroc"])
            - float(result["baseline_wse"]["micro_candidate_auroc"])
        )
        if micro_auc_delta < -0.001:
            raise RuntimeError(
                f"selected arm materially lowers micro candidate AUROC on {panel}"
            )
        development[panel] = {
            "queries": result["candidate"]["queries"],
            "recall@1": result["candidate"]["recall@1"],
            "delta_recall@1_vs_wse": paired["delta_recall@1"],
            "formula_cluster_bootstrap_95ci": paired["formula_cluster_bootstrap_95ci"],
            "corrected": paired["corrected"],
            "introduced": paired["introduced"],
            "risk_net_corrected_minus_2introduced": paired[
                "risk_net_corrected_minus_2introduced"
            ],
            "micro_candidate_auroc_delta_vs_wse": micro_auc_delta,
        }
    return arm, development


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-bundle", type=Path, required=True)
    parser.add_argument("--crossfit-report", type=Path, required=True)
    parser.add_argument("--module-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    for path in (args.identity_bundle, args.crossfit_report, args.module_registry):
        if not path.is_file():
            raise FileNotFoundError(path)

    crossfit = json.loads(args.crossfit_report.read_text(encoding="utf-8"))
    arm, development = validate_selected_arm(crossfit)

    raw = load_bundle(args.identity_bundle)
    selected = select_modules(raw, MODULES)
    prepared = prepare(selected)
    queries = np.arange(len(prepared["query_ptr"]) - 1, dtype=np.int64)
    device = torch.device(args.device)
    model = fit_model(
        prepared, MODULES, queries,
        epochs=24, batch_queries=256, learning_rate=2e-3,
        anchor_strength=0.01, seed=20262012, device=device,
    )

    args.output.mkdir(parents=True)
    model_path = args.output / "model.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "module_names": MODULES,
        "base_name": BASE,
        "model_class": "AnchoredCandidateEnergy",
        "evidence_transform": "within-query rank-logit centered and clipped at 4",
        "training_seed": 20262012,
    }, model_path)
    residual = model.coefficients().detach().cpu().numpy()
    mixture = model.mixture_weights().detach().cpu().numpy()
    report = {
        "status": "UNIFIED_CANDIDATE_ENERGY_FINAL_FROZEN",
        "freeze_contract_status": "DEVELOPMENT_MODEL_FROZEN_EXTERNAL_UNOPENED",
        "selection_dataset": "GNPS Gold/Silver development consumed",
        "selection_rule": (
            "highest cross-fitted Top-1 among the preplanned full and "
            "leave-one-channel-out arms on both GNPS development strata"
        ),
        "selected_arm": SELECTED_ARM,
        "module_names": list(MODULES),
        "development_evidence": development,
        "risk_boundary": (
            "Each GNPS development stratum passed positive Recall@1 CI, "
            "corrected-2*introduced > 0, and micro-candidate AUROC delta >= -0.001"
        ),
        "final_residual_coefficients": dict(zip(MODULES, map(float, residual), strict=True)),
        "final_all_available_mixture_weights": dict(zip(MODULES, map(float, mixture), strict=True)),
        "sparse_weight_interpretation": (
            "When ChemAware V2 abstains, its coefficient is absent and the implicit "
            "WSE mass increases by that coefficient."
        ),
        "frozen_model": str(model_path),
        "frozen_model_sha256": sha256(model_path),
        "module_registry": str(args.module_registry),
        "module_registry_sha256": sha256(args.module_registry),
        "identity_bundle_sha256": sha256(args.identity_bundle),
        "crossfit_report_sha256": sha256(args.crossfit_report),
        "external_test_opened": False,
        "massspecgym_used": False,
    }
    report_path = args.output / "freeze_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
