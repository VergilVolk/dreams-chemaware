#!/usr/bin/env python
"""Materialise the frozen B17 BioAware router as direct spectrum actions.

B20 joins the strictly replayed B17 decisions to the exact query tensors and
MoNA reference spectra established by B15.  Any B17-only final candidate is
resolved with the same per-query best-reference protocol used by B15.  No
model is fitted and no catalogue score is distilled.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from audit_bioaware_b15_action_spectrum_support import (  # noqa: E402
    atomic_json,
    candidate_reference_map,
    load_mona_reference_tensors,
    sha256,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--b17-dir", type=Path,
        default=ROOT / "data/validation/bioaware_b17_nested_union_localcheck_20260907_v2",
    )
    parser.add_argument(
        "--b15-dir", type=Path,
        default=ROOT / "data/validation/bioaware_b15_action_spectrum_support_localcheck_20260907_v4",
    )
    parser.add_argument(
        "--internal-candidates", type=Path,
        default=ROOT / "data/validation/bioaware_b3_reaction_coabundance_local_20260906/candidate_features.csv.gz",
    )
    parser.add_argument(
        "--st-manifest-dir", type=Path,
        default=ROOT / "data/validation/bioaware_st001154_hilic_extension_manifest_v1",
    )
    parser.add_argument(
        "--st-candidate-scores", type=Path,
        default=ROOT / "data/validation/bioaware_st001154_hilic_extension_v3_evaluation_v1/candidate_features.csv.gz",
    )
    parser.add_argument(
        "--st-evaluation-dir", type=Path,
        default=ROOT / "data/validation/bioaware_st001154_hilic_extension_v3_evaluation_v1",
    )
    parser.add_argument(
        "--kgmn-manifest-dir", type=Path,
        default=ROOT / "data/validation/bioaware_kgmn200std_confirmation_manifest_v2",
    )
    parser.add_argument(
        "--mona-mgf", type=Path, default=ROOT / "data/models/mona_neg_full.mgf",
    )
    parser.add_argument(
        "--mona-manifest", type=Path,
        default=ROOT / "data/models/mona_neg_dreams_emb/manifest.csv",
    )
    parser.add_argument(
        "--mona-embeddings", type=Path,
        default=ROOT / "data/models/mona_neg_dreams_emb/embeddings.npy",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    required = (
        args.b17_dir / "report.json",
        args.b17_dir / "nested_domain_loso_transitions.csv.gz",
        args.b15_dir / "report.json",
        args.b15_dir / "spectrum_supported_actions.csv.gz",
        args.b15_dir / "spectrum_support_manifest.npz",
        args.internal_candidates,
        args.st_manifest_dir / "candidate_references.csv.gz",
        args.st_manifest_dir / "queries.csv.gz",
        args.st_candidate_scores,
        args.st_evaluation_dir / "query_embeddings.npy",
        args.kgmn_manifest_dir / "candidate_scores.csv.gz",
        args.mona_mgf,
        args.mona_manifest,
        args.mona_embeddings,
    )
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    b17_report = json.loads((args.b17_dir / "report.json").read_text(encoding="utf-8"))
    if b17_report.get("strictly_better_action_than_B12") is not True:
        raise RuntimeError("B17 did not pass its frozen action gate")
    b15_report = json.loads((args.b15_dir / "report.json").read_text(encoding="utf-8"))
    if b15_report.get("pass_to_direct_shared_embedding_design") is not True:
        raise RuntimeError("B15 spectrum support did not pass")

    b17 = pd.read_csv(args.b17_dir / "nested_domain_loso_transitions.csv.gz")
    b15 = pd.read_csv(args.b15_dir / "spectrum_supported_actions.csv.gz")
    if (
        len(b17) != 860 or b17["query_id"].nunique() != 860
        or int(b17["corrected"].sum()) != 57
        or int(b17["introduced"].sum()) != 7
    ):
        raise RuntimeError("B17 action counts changed")
    support_columns = [
        "query_id", "source_query_id", "physical_query_id",
        "truth_candidate_id", "truth_formula", "baseline_candidate_id",
        "truth_reference_row", "baseline_reference_row",
    ]
    joined = b17.merge(
        b15[support_columns], on="query_id", suffixes=("", "_b15"),
        validate="one_to_one",
    )
    if len(joined) != 860:
        raise RuntimeError("B20 B17/B15 query join lost rows")
    for column in ("truth_candidate_id", "truth_formula", "baseline_candidate_id"):
        if not joined[column].astype(str).equals(joined[f"{column}_b15"].astype(str)):
            raise RuntimeError(f"B20 B17/B15 mismatch: {column}")

    b15_manifest = np.load(
        args.b15_dir / "spectrum_support_manifest.npz", allow_pickle=False
    )
    b15_query_ids = b15_manifest["query_id"].astype(str)
    if len(b15_query_ids) != 860 or len(set(b15_query_ids)) != 860:
        raise RuntimeError("B15 tensor manifest query coverage changed")
    tensor_position = {qid: index for index, qid in enumerate(b15_query_ids)}
    query_positions = np.asarray(
        [tensor_position[str(qid)] for qid in joined["query_id"]], dtype=np.int64
    )
    query_tensors = np.asarray(
        b15_manifest["query_tensor"][query_positions], dtype=np.float32
    )
    if query_tensors.shape != (860, 101, 2) or not np.isfinite(query_tensors).all():
        raise RuntimeError("B20 invalid query tensors")

    reference_map = candidate_reference_map(args)
    final_reference_rows: list[int] = []
    missing: list[tuple[str, str, str]] = []
    for row in joined.itertuples(index=False):
        key = (str(row.source), str(row.source_query_id), str(row.final_candidate_id))
        reference = reference_map.get(key)
        if reference is None:
            missing.append(key)
        else:
            final_reference_rows.append(int(reference))
    if missing:
        raise RuntimeError(f"B20 missing {len(missing)} final references: {missing[:5]}")
    joined["final_reference_row"] = final_reference_rows

    selected_rows = np.asarray(sorted(set(
        joined["truth_reference_row"].astype(int)
        .tolist()
        + joined["baseline_reference_row"].astype(int).tolist()
        + joined["final_reference_row"].astype(int).tolist()
    )), dtype=np.int64)
    reference_tensors = load_mona_reference_tensors(
        args.mona_mgf, args.mona_manifest, selected_rows
    )
    row_position = {int(row): position for position, row in enumerate(selected_rows)}
    for role in ("truth", "baseline", "final"):
        joined[f"{role}_reference_position"] = [
            row_position[int(row)] for row in joined[f"{role}_reference_row"]
        ]

    final_replay = joined["final_candidate_id"].astype(str).eq(
        joined["truth_candidate_id"].astype(str)
    )
    if not final_replay.equals(joined["final_correct"].astype(bool)):
        raise RuntimeError("B20 final-candidate correctness replay mismatch")
    joined["physical_duplicate_weight"] = 1.0 / joined.groupby(
        ["source", "physical_query_id"]
    )["query_id"].transform("size").astype(float)
    joined["identity_equal_weight"] = 1.0 / joined.groupby(
        "truth_candidate_id"
    )["query_id"].transform("size").astype(float)
    joined["direct_corrective"] = joined["corrected"].astype(bool)
    joined["direct_safety"] = joined["introduced"].astype(bool)

    corrected = joined.loc[joined["corrected"]]
    introduced = joined.loc[joined["introduced"]]
    physical_corrected = corrected.drop_duplicates(["source", "physical_query_id"])
    physical_introduced = introduced.drop_duplicates(["source", "physical_query_id"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    action_path = args.output_dir / "direct_actions.csv.gz"
    joined.to_csv(action_path, index=False, compression="gzip")
    manifest_path = args.output_dir / "direct_action_manifest.npz"
    np.savez_compressed(
        manifest_path,
        query_id=joined["query_id"].to_numpy(dtype=str),
        source=joined["source"].to_numpy(dtype=str),
        physical_query_id=joined["physical_query_id"].to_numpy(dtype=str),
        truth_candidate_id=joined["truth_candidate_id"].to_numpy(dtype=str),
        truth_formula=joined["truth_formula"].to_numpy(dtype=str),
        baseline_candidate_id=joined["baseline_candidate_id"].to_numpy(dtype=str),
        final_candidate_id=joined["final_candidate_id"].to_numpy(dtype=str),
        query_tensor=query_tensors,
        reference_tensor_rows=selected_rows,
        reference_tensor=reference_tensors,
        truth_reference_position=joined["truth_reference_position"].to_numpy(np.int64),
        baseline_reference_position=joined["baseline_reference_position"].to_numpy(np.int64),
        final_reference_position=joined["final_reference_position"].to_numpy(np.int64),
        baseline_correct=joined["baseline_correct"].to_numpy(bool),
        final_correct=joined["final_correct"].to_numpy(bool),
        direct_corrective=joined["direct_corrective"].to_numpy(bool),
        direct_safety=joined["direct_safety"].to_numpy(bool),
        physical_duplicate_weight=joined["physical_duplicate_weight"].to_numpy(np.float32),
        identity_equal_weight=joined["identity_equal_weight"].to_numpy(np.float32),
    )

    gates = {
        "all_860_queries_materialised": len(joined) == 860,
        "all_final_candidates_have_exact_reference_spectra": not missing,
        "corrective_physical_queries_ge_50": len(physical_corrected) >= 50,
        "corrective_identities_ge_25": corrected["truth_candidate_id"].nunique() >= 25,
        "corrective_formulas_ge_25": corrected["truth_formula"].nunique() >= 25,
        "safety_physical_queries_le_7": len(physical_introduced) <= 7,
        "B17_final_correctness_replay_exact": True,
    }
    report = {
        "status": "bioaware_b20_direct_action_manifest_complete",
        "formal": True,
        "B17_replay": {
            "evaluation_rows": int(len(joined)),
            "physical_query_spectra": int(joined["physical_query_id"].nunique()),
            "corrected_rows": int(len(corrected)),
            "corrected_physical_queries": int(len(physical_corrected)),
            "corrected_identities": int(corrected["truth_candidate_id"].nunique()),
            "corrected_formulas": int(corrected["truth_formula"].nunique()),
            "introduced_rows": int(len(introduced)),
            "introduced_physical_queries": int(len(physical_introduced)),
            "introduced_identities": int(introduced["truth_candidate_id"].nunique()),
            "introduced_formulas": int(introduced["truth_formula"].nunique()),
        },
        "materialisation": {
            "query_tensors": int(len(query_tensors)),
            "unique_candidate_reference_spectra": int(len(selected_rows)),
            "query_tensor_shape": list(query_tensors.shape),
            "reference_tensor_shape": list(reference_tensors.shape),
        },
        "gates": gates,
        "pass_to_direct_gradient_canary": bool(all(gates.values())),
        "contracts": {
            "model_fitted": False,
            "catalogue_score_distilled": False,
            "reaction_neighbour_used_as_identity_positive": False,
            "query_reference_shared_encoder_required_next": True,
            "physical_duplicates_downweighted": True,
            "identity_equal_weights_exported": True,
            "P2b_used": False,
            "phenotype_used": False,
        },
        "provenance": {
            "B17_report_sha256": sha256(args.b17_dir / "report.json"),
            "B17_transitions_sha256": sha256(
                args.b17_dir / "nested_domain_loso_transitions.csv.gz"
            ),
            "B15_report_sha256": sha256(args.b15_dir / "report.json"),
            "B15_manifest_sha256": sha256(
                args.b15_dir / "spectrum_support_manifest.npz"
            ),
            "actions_sha256": sha256(action_path),
            "manifest_sha256": sha256(manifest_path),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "B20 proves exact spectrum-level support for the opened B17 action. "
            "It is not a trained shared embedding, blind validation, reaction "
            "mechanism, or expected +5.81 pp embedding gain."
        ),
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["pass_to_direct_gradient_canary"]:
        raise RuntimeError(f"B20 direct-action gate failed: {gates}")


if __name__ == "__main__":
    main()
