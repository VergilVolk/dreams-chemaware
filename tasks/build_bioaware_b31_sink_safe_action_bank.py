#!/usr/bin/env python
"""Build a sink-safe direct shared-embedding action bank from B20 and B30.

Corrective supervision contains only the 54 B30-retained row actions.  Safety
supervision retains all seven original B17 introduced-error pairs, including
the four candidate-sink actions vetoed by B30.  The three B17 corrections that
conflict with the cross-source sink veto are marked uncertain and receive no
corrective gradient.  Exact B20 query/reference tensors are reused unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from audit_bioaware_b11_catalog_interaction_action import atomic_json, sha256  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b20-dir", type=Path, required=True)
    parser.add_argument("--b30-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    b20_report_path = args.b20_dir / "report.json"
    b20_table_path = args.b20_dir / "direct_actions.csv.gz"
    b20_manifest_path = args.b20_dir / "direct_action_manifest.npz"
    b30_report_path = args.b30_dir / "report.json"
    b30_table_path = args.b30_dir / "nested_sink_veto_transitions.csv.gz"
    for path in (b20_report_path, b20_table_path, b20_manifest_path, b30_report_path, b30_table_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    b20_report = json.loads(b20_report_path.read_text(encoding="utf-8"))
    b30_report = json.loads(b30_report_path.read_text(encoding="utf-8"))
    if (b20_report.get("pass_to_direct_gradient_canary") is not True
            or b30_report.get("strictly_better_action_than_B17") is not True):
        raise RuntimeError("B31 requires passing B20 and B30 artifacts")
    if b20_report["provenance"].get("manifest_sha256") != sha256(b20_manifest_path):
        raise RuntimeError("B31 B20 manifest hash mismatch")
    if b30_report["provenance"].get("transitions_sha256") != sha256(b30_table_path):
        raise RuntimeError("B31 B30 transition hash mismatch")

    b20 = pd.read_csv(b20_table_path)
    b30 = pd.read_csv(b30_table_path)
    with np.load(b20_manifest_path, allow_pickle=False) as handle:
        body = {name: handle[name] for name in handle.files}
    if len(b20) != 860 or len(b30) != 860 or len(body["query_id"]) != 860:
        raise RuntimeError("B31 input coverage changed")
    if not b20["query_id"].astype(str).equals(pd.Series(body["query_id"].astype(str))):
        raise RuntimeError("B31 B20 table/tensor order mismatch")
    joined = b20.merge(
        b30[["query_id", "final_candidate_id", "corrected", "introduced", "b30_veto"]],
        on="query_id", how="left", suffixes=("_b20", "_b30"), validate="one_to_one",
    )
    if len(joined) != 860 or joined[["corrected_b30", "introduced_b30", "b30_veto"]].isna().any().any():
        raise RuntimeError("B31 failed to align B30 decisions")

    direct_corrective = joined["corrected_b30"].astype(bool).to_numpy()
    # Keep every empirically harmful B17 pair as a safety constraint, even if
    # B30 correctly vetoed that action at inference.
    direct_safety = joined["introduced_b20"].astype(bool).to_numpy()
    uncertain_sink_conflict = (
        joined["corrected_b20"].astype(bool)
        & joined["b30_veto"].astype(bool)
    ).to_numpy()
    if np.any(direct_corrective & direct_safety):
        raise RuntimeError("B31 corrective/safety roles overlap")
    if np.any(uncertain_sink_conflict & (direct_corrective | direct_safety)):
        raise RuntimeError("B31 uncertain sink-conflict role overlaps supervision")
    if (int(direct_corrective.sum()), int(direct_safety.sum()), int(uncertain_sink_conflict.sum())) != (54, 7, 3):
        raise RuntimeError("B31 action role counts changed")

    joined["direct_corrective"] = direct_corrective
    joined["direct_safety"] = direct_safety
    joined["uncertain_sink_conflict"] = uncertain_sink_conflict
    joined["action_candidate_id"] = joined["final_candidate_id_b20"].astype(str)
    joined["deployed_candidate_id"] = joined["final_candidate_id_b30"].astype(str)
    joined["physical_key"] = joined["source"].astype(str) + "::" + joined["physical_query_id"].astype(str)
    physical = joined.drop_duplicates("physical_key")
    physical_corrective = int(physical["direct_corrective"].sum())
    physical_safety = int(physical["direct_safety"].sum())
    physical_uncertain = int(physical["uncertain_sink_conflict"].sum())
    if (physical_corrective, physical_safety, physical_uncertain) != (50, 7, 3):
        raise RuntimeError("B31 physical action role counts changed")

    output_body = dict(body)
    output_body["direct_corrective"] = direct_corrective.astype(bool)
    output_body["direct_safety"] = direct_safety.astype(bool)
    output_body["uncertain_sink_conflict"] = uncertain_sink_conflict.astype(bool)
    output_body["deployed_final_candidate_id"] = joined["deployed_candidate_id"].to_numpy(str)
    # final_candidate_id/final_reference_position deliberately remain the B17
    # action candidate: safety pairs need the vetoed harmful reference as n-.
    if not np.array_equal(output_body["final_candidate_id"].astype(str), joined["action_candidate_id"].to_numpy(str)):
        raise RuntimeError("B31 action-candidate tensor alignment changed")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    table_path = args.output_dir / "direct_actions.csv.gz"
    manifest_path = args.output_dir / "direct_action_manifest.npz"
    joined.to_csv(table_path, index=False, compression="gzip")
    np.savez_compressed(manifest_path, **output_body)
    gates = {
        "all_860_rows_materialised": len(joined) == 860,
        "corrective_rows_54": int(direct_corrective.sum()) == 54,
        "safety_rows_7": int(direct_safety.sum()) == 7,
        "uncertain_sink_conflict_rows_3": int(uncertain_sink_conflict.sum()) == 3,
        "physical_corrective_50": physical_corrective == 50,
        "physical_safety_7": physical_safety == 7,
        "physical_uncertain_3": physical_uncertain == 3,
        "corrective_and_safety_disjoint": not np.any(direct_corrective & direct_safety),
        "exact_B20_tensors_reused": bool(
            np.array_equal(body["query_tensor"], output_body["query_tensor"])
            and np.array_equal(body["reference_tensor"], output_body["reference_tensor"])
        ),
    }
    report = {
        "status": "bioaware_b31_sink_safe_action_bank_complete",
        "formal": True,
        "action_bank": {
            "evaluation_rows": 860,
            "physical_query_spectra": 753,
            "corrective_rows": 54,
            "corrective_physical_queries": physical_corrective,
            "corrective_identities": int(joined.loc[direct_corrective, "truth_candidate_id"].nunique()),
            "corrective_formulas": int(joined.loc[direct_corrective, "truth_formula"].nunique()),
            "safety_rows": 7,
            "safety_physical_queries": physical_safety,
            "uncertain_sink_conflict_rows": 3,
            "uncertain_sink_conflict_physical_queries": physical_uncertain,
        },
        "gates": gates,
        "pass_to_direct_gradient_canary": bool(all(gates.values())),
        "contracts": {
            "corrective_supervision_is_B30_retained_only": True,
            "all_B17_introduced_pairs_are_safety_constraints": True,
            "sink_conflicting_B17_corrections_have_zero_corrective_weight": True,
            "action_candidate_reference_retained_for_safety": True,
            "B20_exact_spectrum_tensors_reused": True,
            "P2b_used": False, "phenotype_used": False,
            "shared_embedding_changed": False,
        },
        "provenance": {
            "B20_report_sha256": sha256(b20_report_path),
            "B20_actions_sha256": sha256(b20_table_path),
            "B20_manifest_sha256": sha256(b20_manifest_path),
            "B30_report_sha256": sha256(b30_report_path),
            "B30_transitions_sha256": sha256(b30_table_path),
            "actions_sha256": sha256(table_path),
            "manifest_sha256": sha256(manifest_path),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": "A direct-training action bank derived from opened outcomes. It is not shared-embedding performance, blind validation, SOTA, or a deployable candidate router.",
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
