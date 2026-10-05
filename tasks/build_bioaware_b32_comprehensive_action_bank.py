#!/usr/bin/env python
"""Freeze the complete BioAware direct-training bank used by the B32 bridge.

B32 unifies two independently developed opened action sources without treating
either source as clean shared-embedding performance:

* B30 contributes every cross-source sink-veto-retained correction and B17
  contributes its observed harms through the frozen B31 bank.
* the older same-formula ``current_v4`` action contributes only corrections
  not already retained or contradicted by B30, while the union of *all* its
  observed harms becomes one-sided safety supervision.
* the three B30 sink-conflicting current-v4 corrections remain uncertain and
  receive zero corrective weight.

Every supervised row is backed by an exact query and candidate spectrum.  MoNA
reference tensors absent from B31 are materialised once into the frozen B32
artifact.  P2b, phenotype labels, action probabilities and catalogue scores are
not exported to the training manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from audit_bioaware_b15_action_spectrum_support import load_mona_reference_tensors  # noqa: E402


def sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(block_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b31-dir", type=Path, required=True)
    parser.add_argument("--current-v4-actions", type=Path, required=True)
    parser.add_argument("--mona-mgf", type=Path, required=True)
    parser.add_argument("--mona-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def physical_counts(frame: pd.DataFrame) -> dict[str, int]:
    physical = frame.drop_duplicates(["source", "physical_query_id"])
    return {
        "corrective": int(physical["direct_corrective"].astype(bool).sum()),
        "safety": int(physical["direct_safety"].astype(bool).sum()),
        "uncertain": int(physical["uncertain_sink_conflict"].astype(bool).sum()),
    }


def main() -> None:
    args = arguments()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    paths = {
        "b31_report": args.b31_dir / "report.json",
        "b31_actions": args.b31_dir / "direct_actions.csv.gz",
        "b31_manifest": args.b31_dir / "direct_action_manifest.npz",
        "current_v4_actions": args.current_v4_actions,
        "mona_mgf": args.mona_mgf,
        "mona_manifest": args.mona_manifest,
    }
    for path in paths.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    b31_report = json.loads(paths["b31_report"].read_text(encoding="utf-8"))
    if (
        b31_report.get("status") != "bioaware_b31_sink_safe_action_bank_complete"
        or b31_report.get("pass_to_direct_gradient_canary") is not True
    ):
        raise RuntimeError("B32 requires a passing B31 action bank")
    if b31_report["provenance"].get("manifest_sha256") != sha256(paths["b31_manifest"]):
        raise RuntimeError("B32 B31 manifest hash mismatch")
    if b31_report["provenance"].get("actions_sha256") != sha256(paths["b31_actions"]):
        raise RuntimeError("B32 B31 action-table hash mismatch")

    b31 = pd.read_csv(paths["b31_actions"])
    old = pd.read_csv(paths["current_v4_actions"])
    with np.load(paths["b31_manifest"], allow_pickle=False) as handle:
        body = {name: handle[name] for name in handle.files}
    if len(b31) != 860 or len(body["query_id"]) != 860:
        raise RuntimeError("B32 B31 coverage changed")
    if not b31["query_id"].astype(str).equals(pd.Series(body["query_id"].astype(str))):
        raise RuntimeError("B32 B31 table/tensor order mismatch")
    if len(old) != 50 or old["query_id"].duplicated().any():
        raise RuntimeError("current-v4 action coverage changed")
    if set(old["action_outcome"].astype(str)) != {"corrected", "introduced"}:
        raise RuntimeError("unexpected current-v4 action outcome")
    if (int(old["corrected"].astype(bool).sum()), int(old["introduced"].astype(bool).sum())) != (39, 11):
        raise RuntimeError("current-v4 corrected/introduced counts changed")
    if not set(old["query_id"].astype(str)).issubset(set(b31["query_id"].astype(str))):
        raise RuntimeError("current-v4 contains a query outside the frozen B31 universe")
    if not (old["recipe"].astype(str) == "same_formula_uncertainty").all():
        raise RuntimeError("current-v4 recipe changed")

    old_columns = [
        "query_id", "action_outcome", "corrected", "introduced",
        "proposed_candidate_id", "proposal_best_library_row",
    ]
    joined = b31.merge(
        old[old_columns].rename(columns={
            "action_outcome": "current_v4_outcome",
            "corrected": "current_v4_corrected",
            "introduced": "current_v4_introduced",
            "proposed_candidate_id": "current_v4_candidate_id",
            "proposal_best_library_row": "current_v4_reference_row",
        }),
        on="query_id", how="left", validate="one_to_one",
    )
    has_old = joined["current_v4_outcome"].notna().to_numpy()
    old_corrected = joined["current_v4_corrected"].fillna(False).astype(bool).to_numpy()
    old_introduced = joined["current_v4_introduced"].fillna(False).astype(bool).to_numpy()
    b31_corrective = joined["direct_corrective"].astype(bool).to_numpy()
    b31_safety = joined["direct_safety"].astype(bool).to_numpy()
    b31_uncertain = joined["uncertain_sink_conflict"].astype(bool).to_numpy()

    current_v4_unique_corrective = old_corrected & ~b31_corrective & ~b31_uncertain
    current_v4_unique_safety = old_introduced & ~b31_safety
    direct_corrective = b31_corrective | current_v4_unique_corrective
    direct_safety = b31_safety | old_introduced
    uncertain = b31_uncertain.copy()
    if np.any(direct_corrective & direct_safety):
        raise RuntimeError("B32 corrective/safety roles overlap")
    if np.any(uncertain & (direct_corrective | direct_safety)):
        raise RuntimeError("B32 uncertain role overlaps supervised roles")
    observed_counts = (
        int(b31_corrective.sum()), int(b31_safety.sum()), int(b31_uncertain.sum()),
        int(current_v4_unique_corrective.sum()), int(current_v4_unique_safety.sum()),
        int(direct_corrective.sum()), int(direct_safety.sum()), int(uncertain.sum()),
    )
    expected_counts = (54, 7, 3, 9, 7, 63, 14, 3)
    if observed_counts != expected_counts:
        raise RuntimeError(f"B32 action-role counts changed: {observed_counts} != {expected_counts}")

    reference_rows = np.asarray(body["reference_tensor_rows"], dtype=np.int64)
    reference_tensor = np.asarray(body["reference_tensor"], dtype=np.float32)
    if len(reference_rows) != len(reference_tensor) or len(np.unique(reference_rows)) != len(reference_rows):
        raise RuntimeError("B31 reference row/tensor alignment is invalid")
    existing_position = {int(row): position for position, row in enumerate(reference_rows)}
    harm_rows = np.sort(
        joined.loc[old_introduced, "current_v4_reference_row"].astype(np.int64).unique()
    )
    missing_rows = np.asarray(
        [int(row) for row in harm_rows if int(row) not in existing_position],
        dtype=np.int64,
    )
    if len(missing_rows):
        appended = load_mona_reference_tensors(
            paths["mona_mgf"], paths["mona_manifest"], missing_rows,
        )
        reference_rows = np.concatenate([reference_rows, missing_rows])
        reference_tensor = np.concatenate([reference_tensor, appended], axis=0)
    reference_position = {int(row): position for position, row in enumerate(reference_rows)}

    final_candidate = np.asarray(body["final_candidate_id"].astype(str)).copy()
    final_position = np.asarray(body["final_reference_position"], dtype=np.int64).copy()
    # For current-v4 harms, the harmful proposal is n- in the safety floor.
    # For current-v4 corrections, truth versus the frozen official Top-1 is the
    # corrective pair; final is set to truth only to keep action semantics exact.
    for index in np.flatnonzero(old_introduced):
        candidate = str(joined.iloc[index]["current_v4_candidate_id"])
        row = int(joined.iloc[index]["current_v4_reference_row"])
        final_candidate[index] = candidate
        final_position[index] = int(reference_position[row])
    for index in np.flatnonzero(old_corrected):
        final_candidate[index] = str(joined.iloc[index]["truth_candidate_id"])
        final_position[index] = int(body["truth_reference_position"][index])

    origin = np.full(len(joined), "none", dtype=object)
    origin[b31_corrective] = "b30_corrective"
    origin[b31_safety] = "b17_safety"
    origin[current_v4_unique_corrective] = "current_v4_unique_corrective"
    origin[current_v4_unique_safety] = "current_v4_unique_safety"
    origin[b31_corrective & old_corrected] = "b30_and_current_v4_corrective"
    origin[b31_safety & old_introduced] = "b17_and_current_v4_safety"
    origin[uncertain] = "sink_conflict_zero_weight"
    joined["direct_corrective"] = direct_corrective
    joined["direct_safety"] = direct_safety
    joined["uncertain_sink_conflict"] = uncertain
    joined["supervision_origin"] = origin.astype(str)
    joined["training_candidate_id"] = final_candidate.astype(str)
    joined["training_reference_position"] = final_position

    output_body = dict(body)
    output_body["reference_tensor_rows"] = reference_rows.astype(np.int64)
    output_body["reference_tensor"] = reference_tensor.astype(np.float32)
    output_body["final_candidate_id"] = final_candidate.astype(str)
    output_body["final_reference_position"] = final_position.astype(np.int64)
    output_body["direct_corrective"] = direct_corrective.astype(bool)
    output_body["direct_safety"] = direct_safety.astype(bool)
    output_body["uncertain_sink_conflict"] = uncertain.astype(bool)
    output_body["supervision_origin"] = origin.astype(str)

    physical = physical_counts(joined)
    if physical != {"corrective": 59, "safety": 14, "uncertain": 3}:
        raise RuntimeError(f"B32 physical action counts changed: {physical}")
    if np.any(final_position < 0) or np.any(final_position >= len(reference_tensor)):
        raise RuntimeError("B32 final reference position is invalid")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    table_path = args.output_dir / "direct_actions.csv.gz"
    manifest_path = args.output_dir / "direct_action_manifest.npz"
    joined.to_csv(table_path, index=False, compression="gzip")
    np.savez_compressed(manifest_path, **output_body)
    gates = {
        "all_860_rows_materialised": len(joined) == 860,
        "B30_corrective_rows_54_retained": int((direct_corrective & b31_corrective).sum()) == 54,
        "current_v4_unique_corrective_rows_9_added": int(current_v4_unique_corrective.sum()) == 9,
        "current_v4_sink_conflicts_3_zero_weight": int(uncertain.sum()) == 3,
        "union_corrective_rows_63": int(direct_corrective.sum()) == 63,
        "union_safety_rows_14": int(direct_safety.sum()) == 14,
        "current_v4_harms_11_all_protected": int((direct_safety & old_introduced).sum()) == 11,
        "roles_disjoint": bool(
            not np.any(direct_corrective & direct_safety)
            and not np.any(uncertain & (direct_corrective | direct_safety))
        ),
        "physical_corrective_rows_59": physical["corrective"] == 59,
        "physical_safety_rows_14": physical["safety"] == 14,
        "every_supervised_action_has_spectrum": bool(
            np.all(final_position[direct_corrective | direct_safety] >= 0)
        ),
    }
    report = {
        "status": "bioaware_b32_comprehensive_action_bank_complete",
        "formal": True,
        "action_bank": {
            "evaluation_rows": 860,
            "physical_query_spectra": int(
                joined.drop_duplicates(["source", "physical_query_id"]).shape[0]
            ),
            "corrective_rows": int(direct_corrective.sum()),
            "corrective_physical_queries": physical["corrective"],
            "corrective_identities": int(joined.loc[direct_corrective, "truth_candidate_id"].nunique()),
            "corrective_formulas": int(joined.loc[direct_corrective, "truth_formula"].nunique()),
            "safety_rows": int(direct_safety.sum()),
            "safety_physical_queries": physical["safety"],
            "uncertain_sink_conflict_rows": int(uncertain.sum()),
            "uncertain_sink_conflict_physical_queries": physical["uncertain"],
            "B30_corrective_rows": int(b31_corrective.sum()),
            "current_v4_corrective_rows": int(old_corrected.sum()),
            "current_v4_unique_corrective_rows_added": int(current_v4_unique_corrective.sum()),
            "current_v4_harm_rows": int(old_introduced.sum()),
            "current_v4_unique_safety_rows_added": int(current_v4_unique_safety.sum()),
            "new_reference_tensors_appended": int(len(missing_rows)),
        },
        "gates": gates,
        "pass_to_direct_gradient_canary": bool(all(gates.values())),
        "contracts": {
            "B30_sink_safe_corrections_retained": True,
            "current_v4_unique_nonconflicting_corrections_added": True,
            "all_B17_and_current_v4_harms_are_safety_constraints": True,
            "sink_conflicting_corrections_have_zero_weight": True,
            "opened_action_outcomes_are_not_embedding_performance": True,
            "query_reference_encoder_must_be_shared": True,
            "P2b_used": False,
            "phenotype_used": False,
            "shared_embedding_changed": False,
        },
        "provenance": {
            "B31_report_sha256": sha256(paths["b31_report"]),
            "B31_actions_sha256": sha256(paths["b31_actions"]),
            "B31_manifest_sha256": sha256(paths["b31_manifest"]),
            "current_v4_actions_sha256": sha256(paths["current_v4_actions"]),
            "mona_manifest_sha256": sha256(paths["mona_manifest"]),
            "actions_sha256": sha256(table_path),
            "manifest_sha256": sha256(manifest_path),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "An opened-outcome, exact-spectrum direct-training bank. It does not "
            "establish shared-embedding gain, blind generalisation, SOTA, or a "
            "deployable BioAware result."
        ),
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
