#!/usr/bin/env python3
"""Evaluate frozen rankers on phenotype-blind IBD cross-adduct ion families.

An ion family qualifies when an explicit [M+H]+ member has a single strict
library-supported InChIKey and one or more non-protonated members share its
FBMN correlation group, annotation network and neutral mass.  The anchored
InChIKey is a frozen policy anchor for the non-protonated query candidate
ranking.  It is not treated as exact chemical truth.

The benchmark is small and development-consumed.  It is a chemically grounded
algorithm-to-application positive control, not external biological validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_ibd200_candidate_context_evidence import candidate_axis
from evaluate_ibd200_algorithm_public_response_links import parse_score_spec


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rank_metrics(scores: np.ndarray, candidates: np.ndarray, truth: str) -> dict[str, object]:
    scores = np.asarray(scores, dtype=np.float64)
    candidates = np.asarray(candidates).astype("U")
    finite = np.isfinite(scores)
    matches = np.flatnonzero(candidates == truth)
    if len(matches) != 1:
        raise RuntimeError("truth candidate is absent or duplicated")
    truth_index = int(matches[0])
    truth_score = float(scores[truth_index])
    if not np.isfinite(truth_score):
        return {
            "truth_score_available": False, "truth_score": None,
            "optimistic_rank": None, "conservative_rank": None,
            "truth_in_top_tie": False, "top_tie_size": None,
            "unique_correct_top1": False, "deterministic_top1_correct": False,
            "deterministic_top1_candidate": "",
        }
    available_scores = scores[finite]
    maximum = float(np.max(available_scores))
    top_mask = finite & np.isclose(scores, maximum, rtol=0.0, atol=1e-8)
    top_candidates = np.sort(candidates[top_mask])
    deterministic_top = str(top_candidates[0])
    return {
        "truth_score_available": True,
        "truth_score": truth_score,
        "optimistic_rank": int(1 + np.sum(available_scores > truth_score + 1e-8)),
        "conservative_rank": int(np.sum(available_scores >= truth_score - 1e-8)),
        "truth_in_top_tie": bool(top_mask[truth_index]),
        "top_tie_size": int(top_mask.sum()),
        "unique_correct_top1": bool(top_mask[truth_index] and top_mask.sum() == 1),
        "deterministic_top1_correct": deterministic_top == truth,
        "deterministic_top1_candidate": deterministic_top,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice-dir", required=True, type=Path)
    parser.add_argument("--identity-partition", required=True, type=Path)
    parser.add_argument("--author-annotations", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--score", action="append", required=True, help="METHOD=PATH:FIELD")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    queries_path = args.slice_dir / "query_entities.tsv.gz"
    reference_score = args.slice_dir / "wse_candidate_scores.npz"
    candidates_path = args.slice_dir / "candidate_structures.tsv.gz"
    for path in (
        queries_path, reference_score, candidates_path, args.identity_partition,
        args.author_annotations,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    query_ptr, query_ids, candidate_ids = candidate_axis(reference_score)
    query_index = {str(entity): index for index, entity in enumerate(query_ids)}

    queries = pd.read_csv(queries_path, sep="\t")
    identity = pd.read_csv(args.identity_partition, sep="\t")
    supported = identity["library_supported"].astype(str).str.lower().eq("true")
    identity = identity.loc[supported, ["entity_id", "library_inchikey", "wse", "matched_peaks"]]
    queries = queries.merge(identity, on="entity_id", how="left", validate="one_to_one")
    queries["ion_family_id"] = (
        queries["correlation_group_id"].astype(str) + "|"
        + queries["annotation_network_number"].astype(str) + "|"
        + queries["neutral_mass"].round(5).astype(str)
    )

    truth_rows: list[dict[str, object]] = []
    for family_id, family in queries.loc[queries["neutral_mass"].notna()].groupby("ion_family_id"):
        anchors = family.loc[
            family["best_ion"].eq("[M+H]+") & family["library_inchikey"].notna()
        ]
        truth_candidates = sorted(set(anchors["library_inchikey"].astype(str)))
        if len(truth_candidates) != 1:
            continue
        truth = truth_candidates[0]
        anchor = anchors.sort_values(["wse", "matched_peaks", "entity_id"], ascending=[False, False, True]).iloc[0]
        for query in family.loc[
            ~family["best_ion"].eq("[M+H]+") & family["algorithm_answerable"].astype(bool)
        ].itertuples(index=False):
            index = query_index[str(query.entity_id)]
            start, stop = int(query_ptr[index]), int(query_ptr[index + 1])
            if truth not in set(candidate_ids[start:stop]):
                continue
            truth_rows.append({
                "ion_family_id": str(family_id),
                "query_entity_id": str(query.entity_id),
                "query_best_ion": str(query.best_ion),
                "query_neutral_mass": float(query.neutral_mass),
                "anchor_entity_id": str(anchor["entity_id"]),
                "anchor_raw_entity_id": str(anchor["raw_entity_id"]),
                "truth_candidate_id": truth,
                "anchor_wse": float(anchor["wse"]),
                "anchor_matched_peaks": int(anchor["matched_peaks"]),
                "candidate_count": int(stop - start),
            })
    truth = pd.DataFrame(truth_rows)
    if truth.empty:
        raise RuntimeError("no qualified cross-adduct hidden-truth queries")

    annotations = pd.read_csv(args.author_annotations, sep="\t")
    annotations["anchor_raw_entity_id"] = annotations["#Scan#"].astype(str)
    annotations = annotations.loc[
        annotations["InChIKey"].notna()
        & (pd.to_numeric(annotations["MQScore"], errors="coerce") >= 0.70)
        & (pd.to_numeric(annotations["SharedPeaks"], errors="coerce") >= 6)
        & (pd.to_numeric(annotations["MZErrorPPM"], errors="coerce").abs() <= 10.0)
    ].sort_values(
        ["anchor_raw_entity_id", "MQScore", "SharedPeaks"],
        ascending=[True, False, False], kind="mergesort",
    ).drop_duplicates("anchor_raw_entity_id")
    annotation_map = dict(zip(
        annotations["anchor_raw_entity_id"].astype(str),
        annotations["InChIKey"].astype(str),
    ))
    truth["author_annotation_inchikey"] = truth["anchor_raw_entity_id"].map(annotation_map).fillna("")
    truth["author_annotation_available"] = truth["author_annotation_inchikey"].ne("")
    truth["author_annotation_exact_agreement"] = (
        truth["author_annotation_available"]
        & truth["author_annotation_inchikey"].eq(truth["truth_candidate_id"])
    )

    score_specs = [parse_score_spec(spec) for spec in args.score]
    result_rows: list[dict[str, object]] = []
    method_summaries: dict[str, object] = {}
    score_provenance: dict[str, object] = {}
    for method, score_path, field in score_specs:
        axis = candidate_axis(score_path)
        if not all(np.array_equal(left, right) for left, right in zip(axis, (query_ptr, query_ids, candidate_ids))):
            raise RuntimeError(f"candidate axis differs for {method}")
        bundle = np.load(score_path, allow_pickle=False)
        if field not in bundle.files:
            raise RuntimeError(f"missing score field for {method}: {field}")
        values = np.asarray(bundle[field], dtype=np.float64)
        for row in truth.itertuples(index=False):
            index = query_index[str(row.query_entity_id)]
            start, stop = int(query_ptr[index]), int(query_ptr[index + 1])
            metrics = rank_metrics(values[start:stop], candidate_ids[start:stop], str(row.truth_candidate_id))
            result_rows.append({"method": method, **row._asdict(), **metrics})
        method_frame = pd.DataFrame([row for row in result_rows if row["method"] == method])
        available = method_frame["truth_score_available"].astype(bool)
        method_summaries[method] = {
            "queries": int(len(method_frame)),
            "ion_families": int(method_frame["ion_family_id"].nunique()),
            "truth_score_available": int(available.sum()),
            "unique_correct_top1": int(method_frame["unique_correct_top1"].sum()),
            "truth_in_top_tie": int(method_frame["truth_in_top_tie"].sum()),
            "deterministic_top1_correct": int(method_frame["deterministic_top1_correct"].sum()),
            "mean_optimistic_reciprocal_rank": float(
                np.mean([1.0 / rank for rank in method_frame.loc[available, "optimistic_rank"]])
            ) if available.any() else None,
            "mean_conservative_reciprocal_rank": float(
                np.mean([1.0 / rank for rank in method_frame.loc[available, "conservative_rank"]])
            ) if available.any() else None,
        }
        score_provenance[method] = {
            "path": str(score_path), "field": field, "sha256": sha256(score_path),
        }

    results = pd.DataFrame(result_rows)
    args.output_dir.mkdir(parents=True)
    truth_path = args.output_dir / "cross_adduct_hidden_truth.tsv"
    results_path = args.output_dir / "method_cross_adduct_results.tsv"
    truth.to_csv(truth_path, sep="\t", index=False)
    results.to_csv(results_path, sep="\t", index=False)
    report = {
        "schema": "ibd200_cross_adduct_hidden_truth_benchmark_v1",
        "status": "DEVELOPMENT_CONSUMED_POLICY_ANCHOR_CONSISTENCY_ONLY",
        "data_role": "development_consumed",
        "denominators": {
            "ion_families": int(truth["ion_family_id"].nunique()),
            "nonprotonated_queries": int(len(truth)),
            "adduct_classes": sorted(truth["query_best_ion"].unique().tolist()),
            "policy_anchor_entities": int(truth["anchor_entity_id"].nunique()),
            "policy_anchors_with_qualified_author_annotation": int(
                truth.loc[truth["author_annotation_available"], "anchor_entity_id"].nunique()
            ),
            "policy_anchors_exactly_agreeing_with_author_annotation": int(
                truth.loc[truth["author_annotation_exact_agreement"], "anchor_entity_id"].nunique()
            ),
        },
        "method_results": method_summaries,
        "metric_contract": {
            "primary": "unique frozen-policy InChIKey Top-1; arbitrary candidate-ID tie breaking is not credited",
            "secondary": "policy-anchor membership in the top tie set and optimistic/conservative reciprocal rank",
            "policy_anchor": "one strict [M+H]+ library-supported member within the same FBMN ion family",
        },
        "leakage_status": "ion annotations, neutral masses, spectra and strict library anchors only; no phenotype or response effect was read",
        "provenance": {
            "query_entities_sha256": sha256(queries_path),
            "candidate_structures_sha256": sha256(candidates_path),
            "identity_partition_sha256": sha256(args.identity_partition),
            "author_annotations_sha256": sha256(args.author_annotations),
            "score_bundles": score_provenance,
            "truth_ledger_sha256": sha256(truth_path),
            "results_sha256": sha256(results_path),
            "implementation_sha256": sha256(Path(__file__)),
        },
        "claim_boundary": [
            "The benchmark has only two ion families and cannot establish general cross-adduct superiority.",
            "The frozen policy anchors conflict with qualified author-library InChIKeys and are not exact structure truth.",
            "This evaluates reproduction of a frozen candidate policy across adducts, not chemical identity correctness.",
            "This does not establish biological-response retrieval or unknown-information increment.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
