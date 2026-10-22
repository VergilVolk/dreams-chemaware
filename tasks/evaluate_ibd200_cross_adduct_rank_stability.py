#!/usr/bin/env python3
"""Evaluate within-ion-family candidate-rank stability across adducts.

For each phenotype-blind FBMN ion family with one scoreable [M+H]+ feature and
one or more scoreable non-protonated features, compare a method's candidate
ranking across the shared neutral-mass candidate set.  This does not assume
that any candidate structure is correct; it measures condition robustness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_ibd200_candidate_context_evidence import candidate_axis, rank_average
from evaluate_ibd200_algorithm_public_response_links import parse_score_spec


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ranking_stability(
    anchor_scores: np.ndarray,
    query_scores: np.ndarray,
    candidate_ids: np.ndarray,
) -> dict[str, object]:
    anchor_scores = np.asarray(anchor_scores, dtype=np.float64)
    query_scores = np.asarray(query_scores, dtype=np.float64)
    candidate_ids = np.asarray(candidate_ids).astype("U")
    paired = np.isfinite(anchor_scores) & np.isfinite(query_scores)
    if not paired.any():
        return {
            "paired_scores": 0, "top_tie_overlap": False, "top_tie_jaccard": None,
            "anchor_top_tie_size": None, "query_top_tie_size": None,
            "unique_top1_comparable": False, "unique_top1_concordant": False,
            "spearman_rank_correlation": None,
        }
    a = anchor_scores[paired]
    q = query_scores[paired]
    c = candidate_ids[paired]
    a_top = np.isclose(a, np.max(a), rtol=0.0, atol=1e-8)
    q_top = np.isclose(q, np.max(q), rtol=0.0, atol=1e-8)
    a_set = set(c[a_top])
    q_set = set(c[q_top])
    union = a_set | q_set
    unique_comparable = len(a_set) == 1 and len(q_set) == 1
    rho = None
    if len(a) >= 3:
        ar = rank_average(a)
        qr = rank_average(q)
        if np.std(ar) > 0 and np.std(qr) > 0:
            rho = float(np.corrcoef(ar, qr)[0, 1])
    return {
        "paired_scores": int(len(a)),
        "top_tie_overlap": bool(a_set & q_set),
        "top_tie_jaccard": float(len(a_set & q_set) / len(union)),
        "anchor_top_tie_size": int(len(a_set)),
        "query_top_tie_size": int(len(q_set)),
        "unique_top1_comparable": bool(unique_comparable),
        "unique_top1_concordant": bool(unique_comparable and a_set == q_set),
        "spearman_rank_correlation": rho,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--score", action="append", required=True, help="METHOD=PATH:FIELD")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    queries_path = args.slice_dir / "query_entities.tsv.gz"
    reference_score = args.slice_dir / "wse_candidate_scores.npz"
    candidates_path = args.slice_dir / "candidate_structures.tsv.gz"
    for path in (queries_path, reference_score, candidates_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    query_ptr, query_ids, candidate_ids = candidate_axis(reference_score)
    index_by_query = {str(query): index for index, query in enumerate(query_ids)}
    queries = pd.read_csv(queries_path, sep="\t")
    queries["ion_family_id"] = (
        queries["correlation_group_id"].astype(str) + "|"
        + queries["annotation_network_number"].astype(str) + "|"
        + queries["neutral_mass"].round(5).astype(str)
    )

    pairs: list[dict[str, object]] = []
    for family_id, family in queries.loc[queries["neutral_mass"].notna()].groupby("ion_family_id"):
        anchors = family.loc[family["best_ion"].eq("[M+H]+") & family["algorithm_answerable"].astype(bool)]
        targets = family.loc[~family["best_ion"].eq("[M+H]+") & family["algorithm_answerable"].astype(bool)]
        if len(anchors) != 1 or targets.empty:
            continue
        anchor_id = str(anchors.iloc[0]["entity_id"])
        anchor_index = index_by_query[anchor_id]
        a_start, a_stop = int(query_ptr[anchor_index]), int(query_ptr[anchor_index + 1])
        anchor_candidates = candidate_ids[a_start:a_stop]
        for target in targets.itertuples(index=False):
            query_id = str(target.entity_id)
            query_index = index_by_query[query_id]
            q_start, q_stop = int(query_ptr[query_index]), int(query_ptr[query_index + 1])
            if not np.array_equal(anchor_candidates, candidate_ids[q_start:q_stop]):
                raise RuntimeError(f"ion-family candidate axes differ: {family_id}")
            pairs.append({
                "ion_family_id": str(family_id),
                "anchor_entity_id": anchor_id,
                "query_entity_id": query_id,
                "query_best_ion": str(target.best_ion),
                "neutral_mass": float(target.neutral_mass),
                "candidate_count": int(q_stop - q_start),
            })
    pairs_frame = pd.DataFrame(pairs)
    if pairs_frame.empty:
        raise RuntimeError("no qualified cross-adduct ion-family pairs")

    result_rows: list[dict[str, object]] = []
    method_results: dict[str, object] = {}
    provenance: dict[str, object] = {}
    for method, path, field in [parse_score_spec(value) for value in args.score]:
        axis = candidate_axis(path)
        if not all(np.array_equal(left, right) for left, right in zip(axis, (query_ptr, query_ids, candidate_ids))):
            raise RuntimeError(f"candidate axis differs for {method}")
        bundle = np.load(path, allow_pickle=False)
        if field not in bundle.files:
            raise RuntimeError(f"missing score field for {method}: {field}")
        values = np.asarray(bundle[field], dtype=np.float64)
        for pair in pairs_frame.itertuples(index=False):
            ai = index_by_query[str(pair.anchor_entity_id)]
            qi = index_by_query[str(pair.query_entity_id)]
            a_start, a_stop = int(query_ptr[ai]), int(query_ptr[ai + 1])
            q_start, q_stop = int(query_ptr[qi]), int(query_ptr[qi + 1])
            metrics = ranking_stability(
                values[a_start:a_stop], values[q_start:q_stop], candidate_ids[a_start:a_stop]
            )
            result_rows.append({"method": method, **pair._asdict(), **metrics})
        frame = pd.DataFrame([row for row in result_rows if row["method"] == method])
        correlations = pd.to_numeric(frame["spearman_rank_correlation"], errors="coerce").dropna()
        comparable = frame["unique_top1_comparable"].astype(bool)
        method_results[method] = {
            "ion_families": int(frame["ion_family_id"].nunique()),
            "cross_adduct_queries": int(len(frame)),
            "score_pairs_available": int(frame["paired_scores"].gt(0).sum()),
            "top_tie_overlap": int(frame["top_tie_overlap"].sum()),
            "mean_top_tie_jaccard": float(frame["top_tie_jaccard"].dropna().mean()),
            "unique_top1_comparable": int(comparable.sum()),
            "unique_top1_concordant": int(frame.loc[comparable, "unique_top1_concordant"].sum()),
            "rank_correlation_available": int(len(correlations)),
            "median_spearman_rank_correlation": float(correlations.median()) if len(correlations) else None,
        }
        provenance[method] = {"path": str(path), "field": field, "sha256": sha256(path)}

    results = pd.DataFrame(result_rows)
    args.output_dir.mkdir(parents=True)
    pairs_path = args.output_dir / "ion_family_pairs.tsv"
    results_path = args.output_dir / "method_rank_stability.tsv"
    pairs_frame.to_csv(pairs_path, sep="\t", index=False)
    results.to_csv(results_path, sep="\t", index=False)
    report = {
        "schema": "ibd200_cross_adduct_rank_stability_v1",
        "status": "DEVELOPMENT_CONSUMED_CROSS_ADDUCT_STABILITY_COMPLETE",
        "data_role": "development_consumed",
        "denominators": {
            "ion_families": int(pairs_frame["ion_family_id"].nunique()),
            "cross_adduct_queries": int(len(pairs_frame)),
            "adduct_classes": sorted(pairs_frame["query_best_ion"].unique().tolist()),
        },
        "method_results": method_results,
        "scientific_contract": {
            "object": "within-ion-family candidate-rank stability relative to the method's own [M+H]+ member",
            "primary": "top-tie overlap and unique-Top1 concordance",
            "secondary": "candidate-rank Spearman correlation",
            "truth_requirement": "none; this is condition robustness, not structure correctness",
        },
        "leakage_status": "phenotype, beta, SE, response relation and candidate identity truth were not loaded",
        "provenance": {
            "query_entities_sha256": sha256(queries_path),
            "candidate_structures_sha256": sha256(candidates_path),
            "score_bundles": provenance,
            "pair_ledger_sha256": sha256(pairs_path),
            "results_sha256": sha256(results_path),
            "implementation_sha256": sha256(Path(__file__)),
        },
        "claim_boundary": [
            "Rank stability can improve even when the common candidate is chemically wrong.",
            "The panel has six ion families and is a development application stress test, not external validation.",
            "No biological-response or unknown-information increment follows from stability alone.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
