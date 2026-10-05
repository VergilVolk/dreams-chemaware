#!/usr/bin/env python
"""Fail-closed readiness audit for the three-layer grand-fusion programme."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CORE_BUNDLE_METHODS = {
    "official_dreams", "noise_v1", "weighted_spectral_entropy",
    "p2b_noise_v1_frozen", "p2b_official_frozen",
    "p2b_sqrt_cosine", "p2b_unweighted_entropy", "neutral_loss_sqrt_cosine",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=ROOT / "tasks/grand_fusion_components_v1.json")
    parser.add_argument("--score-bundle", type=Path, default=ROOT / "data/validation/noise_gnps_article_benchmark_run_2349091/method_scores.npz")
    parser.add_argument("--massspecgym-evidence", type=Path)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--noise-checkpoint", type=Path)
    parser.add_argument("--chem-checkpoint", type=Path)
    parser.add_argument("--bio-repair", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def file_state(path: Path | None) -> dict:
    return {"path": str(path) if path else None, "exists": bool(path and path.is_file())}


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    methods: list[str] = []
    bundle_ok = False
    if args.score_bundle.is_file():
        with np.load(args.score_bundle, allow_pickle=False) as body:
            methods = list(map(str, body["method_names"]))
            bundle_ok = all(
                f"scores_{panel}" in body.files
                and body[f"scores_{panel}"].shape[0] == len(methods)
                and np.all(np.isfinite(body[f"scores_{panel}"]))
                for panel in ("identity_disjoint", "formula_disjoint")
            )
    core_missing = sorted(CORE_BUNDLE_METHODS - set(methods))
    chem_bundle = [name for name in methods if name.startswith("chemaware")]

    evidence_ok = False
    evidence_fields: list[str] = []
    if args.massspecgym_evidence:
        evidence_npz = args.massspecgym_evidence / "evidence.npz"
        evidence_report = args.massspecgym_evidence / "report.json"
        if evidence_npz.is_file() and evidence_report.is_file():
            status = json.loads(evidence_report.read_text(encoding="utf-8")).get("status")
            with np.load(evidence_npz, allow_pickle=False) as body:
                evidence_fields = list(body.files)
            evidence_ok = status == "NOISE_MSG_PAIR_EVIDENCE_COMPLETE"

    bio = {"provided": bool(args.bio_repair), "authorized": False, "reason": "not provided"}
    if args.bio_repair:
        report_path = args.bio_repair / "report.json"
        ledger_path = args.bio_repair / "repaired_action_ledger.csv.gz"
        if report_path.is_file() and ledger_path.is_file():
            body = json.loads(report_path.read_text(encoding="utf-8"))
            bio = {
                "provided": True,
                "authorized": bool(body.get("authorization_after_repair")),
                "status": body.get("status"),
                "reason": "authorized" if body.get("authorization_after_repair") else "gate did not authorize truth opening",
            }
        else:
            bio = {"provided": True, "authorized": False, "reason": "report or repaired ledger missing"}

    checkpoints = {
        "base": file_state(args.base_checkpoint),
        "noise": file_state(args.noise_checkpoint),
        "chem": file_state(args.chem_checkpoint),
    }
    task_vector_ready = all(row["exists"] for row in checkpoints.values())
    router_core_ready = bool(bundle_ok and not core_missing and evidence_ok)
    router_with_chem_ready = bool(router_core_ready and chem_bundle and "chemaware_v2_reranker" in evidence_fields)
    report = {
        "status": "GRAND_FUSION_READINESS_AUDIT_COMPLETE",
        "registry_schema": registry.get("schema"),
        "score_bundle": {
            "path": str(args.score_bundle), "valid": bundle_ok,
            "methods": methods, "missing_core_methods": core_missing,
            "chemaware_methods": chem_bundle,
        },
        "massspecgym_evidence": {
            "path": str(args.massspecgym_evidence) if args.massspecgym_evidence else None,
            "valid": evidence_ok, "fields": evidence_fields,
        },
        "checkpoints": checkpoints,
        "bioaware_repair": bio,
        "readiness": {
            "core_noise_spectral_router": router_core_ready,
            "noise_chem_task_vector_scan": task_vector_ready,
            "router_including_chemaware": router_with_chem_ready,
            "bioaware_confirmatory_opening": bio["authorized"],
            "full_three_layer_claim": bool(router_with_chem_ready and task_vector_ready and bio["authorized"]),
        },
        "next_actions": [
            action for condition, action in (
                (not evidence_ok, "Build frozen MassSpecGym pair evidence; do not train a router on GNPS."),
                (not task_vector_ready, "Stage compatible base, Noise and ChemAware slim checkpoints."),
                (not chem_bundle, "Encode/freeze ChemAware GNPS scores and extend the immutable score bundle."),
                ("chemaware_v2_reranker" not in evidence_fields, "Export ChemAware train-side scores with the evidence SHA256 contract."),
                (not bio["authorized"], "Run B47 remediation; keep context layer disabled unless every R1-R5 gate passes."),
            ) if condition
        ],
        "verdict": (
            "FULL_READY" if router_with_chem_ready and task_vector_ready and bio["authorized"]
            else "PARTIAL_ONLY_FAIL_CLOSED"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
