#!/usr/bin/env python3
"""Package every v10 complete-axis method for biology-side comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from score_contrast_atlas_v10_complete_axis_full23 import METHODS, MSG_FAMILY, sha256
from validate_contrast_atlas_method_pair_scores import expected_reciprocal_edges, validate_package


METHOD_VERSION = {
    "weighted_spectral_entropy": "full23_frozen_top100_ms_entropy_1.5.2",
    "official_dreams": "official_finetuned_dreams_frozen",
    "noise_v1": "noise_v1_frozen",
    "p2b_noise_v1_frozen": "p2b_noise_v1_frozen",
    MSG_FAMILY: "noise_msg_fusion_stack_v1",
    "chemaware_stage1": "chemaware_stage1_frozen",
    "chemaware_phasea": "chemaware_phasea_frozen",
    "noise_chemaware_fullmsg": "noise_chemaware_fullmsg_frozen",
    "full23_v1": "unified_additive_full23_v1",
}

RAW_SCORE_SEMANTICS = {
    "weighted_spectral_entropy": "similarity_score",
    "official_dreams": "similarity_score",
    "noise_v1": "similarity_score",
    "p2b_noise_v1_frozen": "ranking_score",
    MSG_FAMILY: "ranking_score",
    "chemaware_stage1": "similarity_score",
    "chemaware_phasea": "similarity_score",
    "noise_chemaware_fullmsg": "similarity_score",
    "full23_v1": "ranking_score",
}

RAW_SCORE_DETAIL = {
    "weighted_spectral_entropy": "weighted_spectral_entropy_similarity",
    "official_dreams": "embedding_cosine_similarity",
    "noise_v1": "embedding_cosine_similarity",
    "p2b_noise_v1_frozen": "frozen_candidate_ranking_score",
    MSG_FAMILY: "frozen_classifier_positive_class_score",
    "chemaware_stage1": "embedding_cosine_similarity",
    "chemaware_phasea": "embedding_cosine_similarity",
    "noise_chemaware_fullmsg": "embedding_cosine_similarity",
    "full23_v1": "unified_additive_candidate_ranking_score",
}


def validate_bundle_axis(
    axis_dir: Path,
    directed_pair_ids: np.ndarray,
    query_ptr: np.ndarray,
    query_ids: np.ndarray,
) -> None:
    """Fail closed if a score bundle is not in the frozen axis order."""
    pairs = pd.read_csv(
        axis_dir / "directed_complete_candidate_pairs.tsv.gz",
        sep="\t", usecols=["directed_pair_id"], dtype=str,
    )
    blocks = pd.read_csv(
        axis_dir / "query_target_source_blocks.tsv.gz",
        sep="\t", usecols=[
            "query_block_id", "candidate_pair_offset_start",
            "candidate_pair_offset_end",
        ], dtype={"query_block_id": str},
    )
    expected_pair_ids = pairs["directed_pair_id"].astype(str).to_numpy(dtype="U")
    expected_query_ids = blocks["query_block_id"].astype(str).to_numpy(dtype="U")
    starts = blocks["candidate_pair_offset_start"].to_numpy(np.int64)
    stops = blocks["candidate_pair_offset_end"].to_numpy(np.int64)
    expected_ptr = np.concatenate((starts[:1], stops)).astype(np.int64)
    if not np.array_equal(np.asarray(directed_pair_ids).astype("U"), expected_pair_ids):
        raise RuntimeError("score bundle directed-pair order differs from frozen axis")
    if not np.array_equal(np.asarray(query_ids).astype("U"), expected_query_ids):
        raise RuntimeError("score bundle query-block order differs from frozen axis")
    if not np.array_equal(np.asarray(query_ptr, dtype=np.int64), expected_ptr):
        raise RuntimeError("score bundle query pointer differs from frozen axis")


def package_method(
    axis_dir: Path,
    method: str,
    scores: np.ndarray,
    artifact_sha256: str,
    output: Path,
    contract_path: Path,
    *,
    wse_matched_fragments: np.ndarray | None = None,
    tie_tolerance: float = 1e-12,
) -> dict[str, object]:
    pairs_path = axis_dir / "directed_complete_candidate_pairs.tsv.gz"
    blocks_path = axis_dir / "query_target_source_blocks.tsv.gz"
    report_path = axis_dir / "report.json"
    pairs = pd.read_csv(
        pairs_path, sep="\t", low_memory=False,
        dtype={"directed_pair_id": str, "query_block_id": str, "query_study_id": str,
               "query_entity_id": str, "candidate_study_id": str, "candidate_entity_id": str},
    )
    blocks = pd.read_csv(blocks_path, sep="\t", dtype={"query_block_id": str})
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(pairs),):
        raise RuntimeError(f"{method} score axis differs from frozen candidate pairs")
    if method == "weighted_spectral_entropy":
        if wse_matched_fragments is None or np.asarray(wse_matched_fragments).shape != scores.shape:
            raise RuntimeError("WSE packaging requires aligned matched-fragment counts")
        matched = np.asarray(wse_matched_fragments, dtype=np.int64)
    else:
        matched = np.zeros(len(scores), dtype=np.int64)

    available = np.isfinite(scores)
    pair_scores = pairs[[
        "directed_pair_id", "query_block_id", "query_block_index", "candidate_index_within_block",
    ]].copy()
    pair_scores["raw_score"] = scores
    pair_scores["score_available"] = available
    pair_scores["unavailable_reason"] = np.where(available, "", "missing_required_method_evidence")

    decisions: list[dict[str, object]] = []
    starts = blocks["candidate_pair_offset_start"].to_numpy(np.int64)
    stops = blocks["candidate_pair_offset_end"].to_numpy(np.int64)
    pair_ids = pairs["directed_pair_id"].astype(str).to_numpy()
    candidate_ids = pairs["candidate_entity_id"].astype(str).to_numpy()
    for block in blocks.itertuples(index=False):
        start = int(block.candidate_pair_offset_start)
        stop = int(block.candidate_pair_offset_end)
        if stop == start:
            continue
        values = scores[start:stop]
        complete = bool(np.isfinite(values).all())
        decision_status = "abstain_incomplete_block_evidence"
        answered = False
        selected_pair = ""
        selected_candidate = ""
        selected_score = math.nan
        top1_tie_count = 0
        margin = math.nan
        if complete:
            best = float(np.max(values))
            winners = np.flatnonzero(values >= best - tie_tolerance)
            top1_tie_count = len(winners)
            if len(values) > 1:
                ordered = np.sort(values)
                margin = float(ordered[-1] - ordered[-2])
            if len(winners) != 1:
                decision_status = "abstain_tied_top1"
            else:
                local = int(winners[0])
                global_index = start + local
                method_accepts = not (
                    method == "weighted_spectral_entropy"
                    and (best <= 0 or matched[global_index] <= 0)
                )
                if method_accepts:
                    decision_status = "selected_unique_top1"
                    answered = True
                    selected_pair = pair_ids[global_index]
                    selected_candidate = candidate_ids[global_index]
                    selected_score = best
                else:
                    decision_status = "abstain_method_rule"
        decisions.append({
            "query_block_id": str(block.query_block_id),
            "query_block_index": int(block.query_block_index),
            "candidate_count": stop - start,
            "scored_candidate_count": int(np.isfinite(values).sum()),
            "decision_status": decision_status,
            "answered": answered,
            "selected_directed_pair_id": selected_pair,
            "selected_candidate_entity_id": selected_candidate,
            "selected_raw_score": selected_score,
            "top1_tie_count": top1_tie_count,
            "top1_minus_top2_raw_margin": margin,
        })
    decision_frame = pd.DataFrame(decisions)
    score_by_id = dict(zip(pair_ids, scores, strict=True))
    expected_edges = expected_reciprocal_edges(pairs, decision_frame, score_by_id)
    edge_columns = [
        "reciprocal_edge_id", "left_study_id", "left_entity_id", "right_study_id",
        "right_entity_id", "left_to_right_directed_pair_id", "right_to_left_directed_pair_id",
        "left_to_right_raw_score", "right_to_left_raw_score",
    ]
    edge_frame = pd.DataFrame(list(expected_edges.values()), columns=edge_columns)

    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    pair_output = output / "method_pair_scores.tsv.gz"
    block_output = output / "method_block_decisions.tsv.gz"
    edge_output = output / "reciprocal_edges.tsv.gz"
    compression = {"method": "gzip", "mtime": 0}
    pair_scores.to_csv(pair_output, sep="\t", index=False, compression=compression)
    decision_frame.to_csv(block_output, sep="\t", index=False, compression=compression)
    edge_frame.to_csv(edge_output, sep="\t", index=False, compression=compression)

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    report = {
        "schema": "contrast_atlas_method_pair_score_package_v1",
        "status": "DEVELOPMENT_CONSUMED_COMPLETE_AXIS_METHOD_PACKAGE",
        "data_role": "development_consumed",
        "method": {
            "method_id": method,
            "method_version": METHOD_VERSION[method],
            "artifact_sha256": artifact_sha256,
            "raw_score_semantics": RAW_SCORE_SEMANTICS[method],
            "raw_score_detail": RAW_SCORE_DETAIL[method],
            "higher_is_better": True,
            "probability_interpretation": False,
            "raw_score_clipped": False,
            "raw_score_transformation": "none",
        },
        "axis": {
            "axis_report_sha256": sha256(report_path),
            "directed_pair_axis_sha256": sha256(pairs_path),
            "query_block_axis_sha256": sha256(blocks_path),
        },
        "truth_blinding": {
            "response_beta_read": False, "response_se_read": False,
            "phenotype_read": False, "registered_relations_read": False,
            "identity_truth_read": False,
        },
        "missingness": {
            "unavailable_pair_policy": "blank_score",
            "incomplete_block_policy": "abstain_fail_closed",
            "external_score_fallback_used": False,
            "wse_fallback_used": False,
        },
        "decision": {
            "selection_scope": "complete_nonempty_candidate_block",
            "block_decisions_are_method_specific": True,
            "reciprocal_rule": "both_directions_selected_unique_top1",
            "reciprocal_edges_preserve_both_directions": True,
            "tie_tolerance": tie_tolerance,
            "method_rule": (
                "positive_score_and_at_least_one_matched_fragment"
                if method == "weighted_spectral_entropy"
                else "unique_top1_ranking_support_only_no_open_set_acceptance"
            ),
        },
        "denominators": {
            "directed_candidate_pairs": len(pairs),
            "available_pair_scores": int(available.sum()),
            "nonempty_blocks": len(decision_frame),
            "answered_blocks": int(decision_frame["answered"].sum()),
            "reciprocal_edges": len(edge_frame),
        },
        "outputs": {
            pair_output.name: sha256(pair_output),
            block_output.name: sha256(block_output),
            edge_output.name: sha256(edge_output),
        },
        "claim_boundary": contract["claim_boundary"] + [
            "Selected singleton and multi-candidate Top-1 rows are ranking hypotheses, not calibrated identity acceptances.",
            "This package is development-consumed and establishes no biological validity by itself.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    validation = validate_package(axis_dir, output, contract_path)
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis-dir", type=Path, required=True)
    parser.add_argument("--scoring-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    report_path = args.scoring_dir / "report.json"
    bundle_path = args.scoring_dir / "candidate_score_bundle.npz"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "DEVELOPMENT_CONSUMED_CONTRAST_ATLAS_V10_COMPLETE_AXIS_FULL23_SCORED":
        raise RuntimeError("input is not the frozen v10 complete-axis scoring result")
    if report.get("outputs", {}).get(bundle_path.name) != sha256(bundle_path):
        raise RuntimeError("candidate score bundle hash drifted")
    with np.load(bundle_path, allow_pickle=False) as body:
        if str(np.asarray(body["schema"]).item()) != "contrast_atlas_v10_complete_axis_full23_score_bundle_v1":
            raise RuntimeError("score bundle schema drifted")
        scores = {method: np.asarray(body[method], dtype=np.float64) for method in METHODS}
        matched = np.asarray(body["wse_matched_fragments"], dtype=np.int64)
        directed_pair_ids = np.asarray(body["directed_pair_ids"]).astype("U")
        query_ptr = np.asarray(body["query_ptr"], dtype=np.int64)
        query_ids = np.asarray(body["query_ids"]).astype("U")
    validate_bundle_axis(args.axis_dir, directed_pair_ids, query_ptr, query_ids)
    if (
        report.get("inputs", {}).get("pair_axis", {}).get("sha256")
        != sha256(args.axis_dir / "directed_complete_candidate_pairs.tsv.gz")
        or report.get("inputs", {}).get("block_axis", {}).get("sha256")
        != sha256(args.axis_dir / "query_target_source_blocks.tsv.gz")
    ):
        raise RuntimeError("scoring report axis provenance differs from packaging axis")
    args.output_dir.mkdir(parents=True)
    results = {}
    for method in METHODS:
        results[method] = package_method(
            args.axis_dir,
            method,
            scores[method],
            str(report["method_artifacts"][method]["sha256"]),
            args.output_dir / method,
            args.contract,
            wse_matched_fragments=matched if method == "weighted_spectral_entropy" else None,
        )
    summary = {
        "schema": "contrast_atlas_v10_complete_axis_method_packages_v1",
        "status": "PASS_ALL_METHOD_PACKAGES_VALIDATED",
        "methods": results,
        "source_scoring_report": {"path": str(report_path), "sha256": sha256(report_path)},
    }
    (args.output_dir / "report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
