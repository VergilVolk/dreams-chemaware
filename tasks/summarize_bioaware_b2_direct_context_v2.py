#!/usr/bin/env python
"""Summarize the preregistered BioAware B2 direct-context experiment matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED = {
    "full_study": ("full", "study"),
    "reaction_smn_study": ("reaction_smn", "study"),
    "reaction_study": ("reaction_only", "study"),
    "smn_study": ("smn_only", "study"),
    "smn_rt_study": ("smn_rt", "study"),
    "rt_study": ("rt_only", "study"),
    "full_formula": ("full", "truth_formula"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path("data/validation/bioaware_b2_direct_context_v2_20260905"),
    )
    args = parser.parse_args()
    destination = args.root / "matrix_report.json"
    if destination.exists():
        raise RuntimeError(f"fail-closed: output exists: {destination}")

    cells: dict[str, dict] = {}
    provenance: dict[str, dict[str, str]] = {}
    dataset_hashes: set[str] = set()
    for name, (evidence_mode, isolation) in EXPECTED.items():
        path = args.root / name / "report.json"
        replay_path = args.root / name / "replay.json"
        flip_path = args.root / name / "flip_mechanism_audit.json"
        if not path.exists():
            raise FileNotFoundError(path)
        if not replay_path.exists():
            raise FileNotFoundError(replay_path)
        if not flip_path.exists():
            raise FileNotFoundError(flip_path)
        body = json.loads(path.read_text(encoding="utf-8"))
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        flip = json.loads(flip_path.read_text(encoding="utf-8"))
        if replay.get("status") != "bioaware_b2_artifact_replay_passed":
            raise RuntimeError(f"{name}: artifact replay did not pass")
        if replay.get("final_rank_mismatches") or replay.get("baseline_rank_mismatches"):
            raise RuntimeError(f"{name}: nonzero artifact replay mismatches")
        if flip.get("status") != "bioaware_b2_flip_mechanism_audit_complete":
            raise RuntimeError(f"{name}: flip audit did not complete")
        if flip.get("rank_mismatches"):
            raise RuntimeError(f"{name}: flip audit rank mismatch")
        if not body.get("formal"):
            raise RuntimeError(f"{name}: non-formal result")
        configuration = body["configuration"]
        if configuration["evidence_mode"] != evidence_mode:
            raise RuntimeError(f"{name}: evidence mode mismatch")
        if configuration["training_isolation"] != isolation:
            raise RuntimeError(f"{name}: isolation mismatch")
        dataset_hashes.add(body["provenance"]["dataset_sha256"])
        provenance[name] = {
            "report_sha256": sha256(path), "replay_sha256": sha256(replay_path),
            "flip_audit_sha256": sha256(flip_path),
        }
        cells[name] = {
            "evidence_mode": evidence_mode,
            "training_isolation": isolation,
            "overall": body["overall"],
            "context_active": body["context_active"],
            "formula_cluster_bootstrap": body["study_formula_cluster_bootstrap"],
            "outer_studies": body["outer_studies"],
            "mean_preservation": body["mean_preservation"],
            "mean_gate": body["mean_gate"],
            "flip_mechanism_audit": flip,
            "pass": body["pass"],
        }
    if len(dataset_hashes) != 1:
        raise RuntimeError(f"matrix cells used different datasets: {sorted(dataset_hashes)}")

    primary = cells["full_study"]
    formula = cells["full_formula"]
    gates = {
        "primary_formula_ci_positive": primary["formula_cluster_bootstrap"]["ci_low"] > 0,
        "primary_corrected_gt_introduced": (
            primary["overall"]["corrected"] > primary["overall"]["introduced"]
        ),
        "primary_every_study_nonnegative": all(
            item["delta_recall1"] >= 0 for item in primary["outer_studies"].values()
        ),
        "formula_and_candidate_disjoint_point_nonnegative": (
            formula["overall"]["delta_recall1"] >= 0
        ),
        "formula_and_candidate_disjoint_corrected_ge_introduced": (
            formula["overall"]["corrected"] >= formula["overall"]["introduced"]
        ),
        "all_cells_preserve_universal_embedding": all(
            cell["mean_preservation"] >= .995 for cell in cells.values()
        ),
    }
    report = {
        "status": "bioaware_b2_direct_context_v2_matrix_complete",
        "formal": True,
        "selection_policy": (
            "No cell is selected post hoc. Full/study is the fixed primary; evidence-family "
            "cells are attribution controls; identity/formula cells are generalisation sensitivities."
        ),
        "cells": cells,
        "gates_to_last_block_direct_finetuning": gates,
        "pass_to_last_block_direct_finetuning": bool(all(gates.values())),
        "next_stage": (
            "If and only if the gate passes, compare frozen-backbone adapter against direct "
            "fine-tuning of the final DreaMS block under the identical LOSO protocol."
        ),
        "contracts": {
            "P2b_used": False,
            "phenotype_used": False,
            "outer_study_labels_used_for_training": False,
            "distillation_used": False,
            "full_study_is_primary": True,
        },
        "provenance": {
            "dataset_sha256": next(iter(dataset_hashes)),
            "cell_report_sha256": provenance,
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "This is cross-study OOF contextual candidate-embedding evidence. It is not an "
            "untouched blind benchmark, universal single-spectrum embedding gain, or SOTA claim."
        ),
    }
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
