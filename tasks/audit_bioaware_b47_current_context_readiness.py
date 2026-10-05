#!/usr/bin/env python
"""Freeze the evidence that current BioAware assets are not prospective context data.

This is a read-only closure audit.  It fits no model, reads no sealed outcome,
and deliberately fails the readiness decision when the existing ledgers do not
identify sample-local evidence for a genuinely unknown query identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def require(value: bool, message: str) -> None:
    if not bool(value):
        raise RuntimeError(message)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".json", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--b39",
        type=Path,
        default=ROOT / "data/validation/bioaware_b39_m0_localcheck_20260913_v2/report.json",
    )
    parser.add_argument(
        "--b40",
        type=Path,
        default=ROOT / "data/validation/bioaware_b40_m2_localcheck_20260913_v1/report.json",
    )
    parser.add_argument(
        "--b44-audit",
        type=Path,
        default=ROOT / "data/validation/bioaware_b44_external_failure_audit_20260913_v1/report.json",
    )
    parser.add_argument(
        "--b45",
        type=Path,
        default=ROOT / "data/validation/bioaware_b45_coverage_neutral_topology_local_20260913_v1/report.json",
    )
    parser.add_argument(
        "--b46",
        type=Path,
        default=ROOT / "data/validation/bioaware_b46_dual_context_action_local_20260913_v2/report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/validation/bioaware_b47_current_context_readiness_20260913_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")

    paths = {
        "b39": args.b39.resolve(),
        "b40": args.b40.resolve(),
        "b44_audit": args.b44_audit.resolve(),
        "b45": args.b45.resolve(),
        "b46": args.b46.resolve(),
    }
    reports = {name: read_json(path) for name, path in paths.items()}
    require(
        reports["b39"].get("status") == "bioaware_b39_m0_atomic_event_ledger_complete",
        "unexpected B39 report",
    )
    require(
        reports["b40"].get("status") == "bioaware_b40_m2_context_strata_evaluation_complete",
        "unexpected B40 report",
    )
    require(
        reports["b44_audit"].get("status")
        == "bioaware_b44_external_failure_root_cause_complete",
        "unexpected B44 audit report",
    )
    require(
        reports["b45"].get("status") == "bioaware_b45_coverage_neutral_topology_complete",
        "unexpected B45 report",
    )
    require(
        reports["b46"].get("status") == "bioaware_b46_dual_context_action_complete",
        "unexpected B46 report",
    )

    b39 = reports["b39"]
    b40 = reports["b40"]
    b44 = reports["b44_audit"]
    b45 = reports["b45"]
    b46 = reports["b46"]

    semantics = Counter(
        source.get("context_semantics", "missing") for source in b39["sources"].values()
    )
    event_resolution = b39["experimental_evidence_resolution"]
    b45_real = b45["arms"]["degree_real"]
    b46_real = b46["arms"]["real_dual"]
    b44_transport = b44["same_frozen_gate_transport"]["b44"]

    gates = {
        "at_least_one_prospective_unknown_source": int(b39["prospective_unknown_sources"]) >= 1,
        "event_specific_spectral_rows_ge_1000": int(
            event_resolution["event_specific_spectral_rows"]
        )
        >= 1000,
        "event_specific_coabundance_rows_ge_1000": int(
            event_resolution["event_specific_coabundance_rows"]
        )
        >= 1000,
        "sample_local_truth_was_not_preseeded": float(
            b39["st_leave_one_seed_out_audit"]["fraction"]
        )
        == 0.0,
        "sample_local_context_reconstruction_passed": bool(
            b40.get("pass_to_prospective_context_reconstruction", False)
        ),
        "coverage_neutral_topology_passed": bool(b45.get("pass_to_new_external", False)),
        "dual_context_action_passed": bool(
            b46.get("pass_to_external_context_benchmark", False)
        ),
        "independent_catalogue_transport_passed": int(b44_transport["corrected"])
        > 2 * int(b44_transport["introduced"]),
    }
    current_assets_ready = all(gates.values())

    report = {
        "status": "bioaware_b47_current_context_readiness_complete",
        "formal": True,
        "models_fitted": False,
        "sealed_outcomes_read": False,
        "current_assets_ready_for_prospective_context_claim": current_assets_ready,
        "source_context_semantics": dict(sorted(semantics.items())),
        "prospective_unknown_sources": int(b39["prospective_unknown_sources"]),
        "event_specific_evidence": {
            "spectral_rows": int(event_resolution["event_specific_spectral_rows"]),
            "coabundance_rows": int(event_resolution["event_specific_coabundance_rows"]),
        },
        "sample_local_counterfactual_audit": b39["st_leave_one_seed_out_audit"],
        "opened_development_closure": {
            "b40_context_reconstruction_pass": bool(
                b40.get("pass_to_prospective_context_reconstruction", False)
            ),
            "b44_external_catalogue": {
                "queries": int(b44_transport["queries"]),
                "delta_recall1": float(b44_transport["delta_recall1"]),
                "corrected": int(b44_transport["corrected"]),
                "introduced": int(b44_transport["introduced"]),
                "root_cause": b44["root_cause"],
            },
            "b45_coverage_neutral_topology": {
                "queries": int(b45_real["queries"]),
                "eligible_action_queries": int(b45_real["eligible_action_queries"]),
                "recoverable_baseline_errors": int(b45_real["recoverable_baseline_errors"]),
                "delta_recall1": float(b45_real["delta_recall1"]),
                "corrected": int(b45_real["corrected"]),
                "introduced": int(b45_real["introduced"]),
                "formula_ci": b45_real["formula_cluster_bootstrap"],
                "passed": bool(b45.get("pass_to_new_external", False)),
            },
            "b46_coverage_neutral_dual_context": {
                "queries": int(b46_real["queries"]),
                "coverage_neutral_baseline_errors": int(
                    b46_real["coverage_neutral_baseline_errors"]
                ),
                "delta_recall1": float(b46_real["delta_recall1"]),
                "corrected": int(b46_real["corrected"]),
                "introduced": int(b46_real["introduced"]),
                "formula_ci": b46_real["formula_cluster_bootstrap"],
                "passed": bool(b46.get("pass_to_external_context_benchmark", False)),
            },
        },
        "gates": gates,
        "decision": {
            "fit_another_model_on_current_context_ledgers": False,
            "distil_current_context_actions_into_embedding": False,
            "next_stage": "B47 external prospective sample-context benchmark acquisition and truth-sealed readiness preflight",
            "reason": (
                "Current ledgers contain no prospective-unknown source and no event-specific "
                "spectral or coabundance rows. Coverage-neutral topology and the simple dual-context "
                "action do not pass their correction gates."
            ),
        },
        "required_next_evidence": {
            "independent_truth": True,
            "truth_identity_absent_from_every_seed_context": True,
            "sample_by_feature_abundance": True,
            "linked_ms1_ms2_and_retention_time": True,
            "candidate_seed_reaction_event_ids": True,
            "catalogue_coverage_and_degree_controls": True,
            "edge_rewire_seed_permutation_wrong_transform_controls": True,
        },
        "provenance": {
            name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()
        },
        "claim_limit": (
            "This closure audit shows that the current local assets cannot identify prospective "
            "sample-context benefit. It does not show that biological context is useless, and it "
            "does not evaluate a new BioAware model."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
