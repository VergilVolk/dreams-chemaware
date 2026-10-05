#!/usr/bin/env python
"""Freeze the outcome-blind BioAware B39-M2 fixed-action matrix.

M2 asks a narrow question: after the already-frozen B37 topology proposal,
does atomically resolved reaction evidence select a safer candidate on the
four internal sources?  This manifest is intentionally created without
reading candidates, embeddings, ranks, labels or action outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def cell(
    cell_id: str,
    name: str,
    *,
    direction: str,
    evidence: str,
    require_hyperedge_complete: bool,
    require_candidate_specific: bool,
    role: str,
    constructible_from_m1: bool,
    interpretation: str,
) -> dict[str, Any]:
    return {
        "cell_id": cell_id,
        "name": name,
        "direction": direction,
        "evidence": evidence,
        "require_rhea": True,
        "exclude_identity_noop": True,
        "require_hyperedge_complete": require_hyperedge_complete,
        "require_candidate_specific": require_candidate_specific,
        "role": role,
        "constructible_from_m1": constructible_from_m1,
        "interpretation": interpretation,
    }


def fixed_cells() -> list[dict[str, Any]]:
    return [
        cell("B39-00", "directed_complete_spectral", direction="supported", evidence="spectral", require_hyperedge_complete=True, require_candidate_specific=True, role="core", constructible_from_m1=True, interpretation="Unidirectional, complete Rhea event with matched spectral edge evidence."),
        cell("B39-01", "directed_complete_coabundance", direction="supported", evidence="coabundance", require_hyperedge_complete=True, require_candidate_specific=True, role="core", constructible_from_m1=True, interpretation="Unidirectional, complete Rhea event with matched co-abundance edge evidence."),
        cell("B39-02", "directed_complete_both", direction="supported", evidence="both", require_hyperedge_complete=True, require_candidate_specific=True, role="core", constructible_from_m1=True, interpretation="Conjunction of directed complete Rhea, spectral and co-abundance evidence."),
        cell("B39-03", "bidirectional_complete_spectral", direction="supported_bidirectional", evidence="spectral", require_hyperedge_complete=True, require_candidate_specific=True, role="core", constructible_from_m1=True, interpretation="Bidirectional complete Rhea event with matched spectral evidence."),
        cell("B39-04", "bidirectional_complete_coabundance", direction="supported_bidirectional", evidence="coabundance", require_hyperedge_complete=True, require_candidate_specific=True, role="core", constructible_from_m1=True, interpretation="Bidirectional complete Rhea event with matched co-abundance evidence."),
        cell("B39-05", "bidirectional_complete_both", direction="supported_bidirectional", evidence="both", require_hyperedge_complete=True, require_candidate_specific=True, role="core", constructible_from_m1=True, interpretation="Conjunction of bidirectional complete Rhea and both experimental layers."),
        cell("B39-06", "direction_agnostic_complete_both", direction="nonconflicted", evidence="both", require_hyperedge_complete=True, require_candidate_specific=True, role="ablation", constructible_from_m1=True, interpretation="Removes the directed-versus-bidirectional distinction but rejects conflicted events."),
        cell("B39-07", "specificity_agnostic_directed_both", direction="supported_or_bidirectional", evidence="both", require_hyperedge_complete=True, require_candidate_specific=False, role="ablation_nonidentifying", constructible_from_m1=True, interpretation="Candidate-specificity ablation; non-identifying if all ledger events are candidate-specific."),
        cell("B39-08", "knowledge_only_directed_complete", direction="supported_or_bidirectional", evidence="knowledge_only", require_hyperedge_complete=True, require_candidate_specific=True, role="ablation", constructible_from_m1=True, interpretation="Reaction topology without experimental edge evidence."),
        cell("B39-09", "conflicted_complete_both", direction="conflicted", evidence="both", require_hyperedge_complete=True, require_candidate_specific=True, role="negative_control", constructible_from_m1=True, interpretation="Same-side or direction-conflicted Rhea events; never eligible for promotion."),
        cell("B39-10", "experimental_matched_non_neighbour", direction="none", evidence="both", require_hyperedge_complete=False, require_candidate_specific=False, role="negative_control", constructible_from_m1=False, interpretation="Requires atomic matched non-neighbour rows not emitted by M1; missing is reported, never zero-filled."),
        cell("B39-11", "structure_matched_reaction_rewire", direction="rewired", evidence="both", require_hyperedge_complete=True, require_candidate_specific=True, role="negative_control", constructible_from_m1=False, interpretation="Requires a preregistered structure/degree-matched reaction-edge rewire not emitted by M1."),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cells = fixed_cells()
    cells_path = args.output_dir / "cells.json"
    atomic_json(cells_path, cells)
    report = {
        "status": "bioaware_b39_m2_fixed_action_manifest_frozen",
        "formal": True,
        "outcomes_read": False,
        "models_fitted": False,
        "cells": len(cells),
        "constructible_cells": sum(item["constructible_from_m1"] for item in cells),
        "roles": {role: sum(item["role"] == role for item in cells) for role in sorted({item["role"] for item in cells})},
        "fixed_evidence_transforms": {
            "spectral": "clip(spectral_excess_dual_view,0,1)",
            "coabundance": "clip(coabundance_excess_positive,0,1)",
            "both": "minimum(spectral,coabundance)",
            "knowledge_only": "one for an eligible atomic Rhea event",
        },
        "fixed_aggregation": {
            "within_dependency_group": "maximum event support",
            "within_candidate_context": "maximum dependency-group support",
            "across_synthetic_rotations": "mean over every visible rotation, including structural zeros",
            "proposal": "unique candidate with the largest strictly-positive support",
        },
        "fixed_application": {
            "starting_ranking": "frozen B37 catalog_topology final candidate",
            "common_risk_layer": "DreaMS baseline_gap <= B37 outer-domain frozen topology gate_margin",
            "risk_layer_inputs": "DreaMS scores only; no BioAware evidence and no outcomes",
            "tie_policy": "non-unique proposal abstains",
        },
        "decision_rule": {
            "core_candidate_for_external_reconstruction": "delta_vs_topology >= 0.03; corrected > 2*introduced; identity and formula cluster CI lower bounds > 0; each internal source nonnegative; at least 100 interventions",
            "negative_controls": "never eligible, regardless of outcome",
            "context_representation": "always false at M2; requires L/H source evidence, prospective unknown evaluation and constructible null controls",
        },
        "forbidden": [
            "post-outcome cell deletion, threshold tuning or relabelling",
            "P2b or another candidate expert as a feature",
            "truth, phenotype or ranking outcome in event support",
            "zero-filling unavailable external evidence or unconstructed nulls",
            "claiming reaction-ID specificity from candidate-seed pair evidence",
        ],
        "provenance": {
            "script_sha256": sha256(Path(__file__)),
            "cells_sha256": sha256(cells_path),
        },
        "claim_limit": "Outcome-blind action preregistration only; contains no BioAware gain and cannot establish shared-embedding improvement.",
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
