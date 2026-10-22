#!/usr/bin/env python3
"""Compare real and degree-preserving wrong context on cross-adduct policy anchors."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_ibd200_candidate_context_specificity import exact_cluster_signflip_p


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-truth", required=True, type=Path)
    parser.add_argument("--context-evidence", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    truth = pd.read_csv(args.hidden_truth, sep="\t")
    evidence = np.load(args.context_evidence, allow_pickle=False)
    query_ptr = np.asarray(evidence["query_ptr"], dtype=np.int64)
    query_ids = np.asarray(evidence["query_ids"]).astype("U")
    candidate_ids = np.asarray(evidence["candidate_ids"]).astype("U")
    folds = np.asarray(evidence["outer_folds"], dtype=np.int64)
    real = np.asarray(evidence["real_context_support"], dtype=np.float64)
    wrong = np.asarray(evidence["wrong_context_support"], dtype=np.float64)
    paired = np.asarray(evidence["paired_context_available"], dtype=bool)
    query_index = {str(query_id): index for index, query_id in enumerate(query_ids)}

    rows: list[dict[str, object]] = []
    ranking_rows: list[dict[str, object]] = []
    for item in truth.itertuples(index=False):
        index = query_index[str(item.query_entity_id)]
        start, stop = int(query_ptr[index]), int(query_ptr[index + 1])
        local_candidates = candidate_ids[start:stop]
        truth_local = np.flatnonzero(local_candidates == str(item.truth_candidate_id))
        if len(truth_local) != 1:
            raise RuntimeError("cross-adduct truth is absent or duplicated on context axis")
        truth_edge = start + int(truth_local[0])
        for fold_index, fold in enumerate(folds):
            if paired[fold_index, truth_edge]:
                rows.append({
                    "ion_family_id": str(item.ion_family_id),
                    "query_entity_id": str(item.query_entity_id),
                    "query_best_ion": str(item.query_best_ion),
                    "truth_candidate_id": str(item.truth_candidate_id),
                    "outer_fold": int(fold),
                    "real_context_support": float(real[fold_index, truth_edge]),
                    "wrong_context_support": float(wrong[fold_index, truth_edge]),
                    "real_minus_wrong": float(real[fold_index, truth_edge] - wrong[fold_index, truth_edge]),
                })
            local_paired = paired[fold_index, start:stop]
            if local_paired.sum() < 2 or not local_paired[truth_local[0]]:
                continue
            available_candidates = local_candidates[local_paired]
            real_values = real[fold_index, start:stop][local_paired]
            wrong_values = wrong[fold_index, start:stop][local_paired]
            real_top = available_candidates[np.lexsort((available_candidates, -real_values))[0]]
            wrong_top = available_candidates[np.lexsort((available_candidates, -wrong_values))[0]]
            ranking_rows.append({
                "ion_family_id": str(item.ion_family_id),
                "query_entity_id": str(item.query_entity_id),
                "outer_fold": int(fold),
                "coverage_matched_candidates": int(local_paired.sum()),
                "real_context_truth_top1": str(real_top) == str(item.truth_candidate_id),
                "wrong_context_truth_top1": str(wrong_top) == str(item.truth_candidate_id),
            })
    ledger = pd.DataFrame(rows)
    ranking = pd.DataFrame(ranking_rows)
    if ledger.empty:
        raise RuntimeError("no paired real/wrong context on cross-adduct truth")
    family_delta = ledger.groupby("ion_family_id")["real_minus_wrong"].mean()
    p_value = exact_cluster_signflip_p(family_delta.to_numpy(float))

    args.output_dir.mkdir(parents=True)
    ledger_path = args.output_dir / "cross_adduct_truth_context.tsv"
    ranking_path = args.output_dir / "coverage_matched_context_ranks.tsv"
    ledger.to_csv(ledger_path, sep="\t", index=False)
    ranking.to_csv(ranking_path, sep="\t", index=False)
    report = {
        "schema": "ibd200_cross_adduct_context_audit_v1",
        "status": "REAL_CONTEXT_POLICY_ANCHOR_SIGNAL_NOT_QUALIFIED",
        "data_role": "development_consumed_posthoc_diagnostic",
        "denominators": {
            "ion_families": int(ledger["ion_family_id"].nunique()),
            "queries_with_paired_truth_context": int(ledger["query_entity_id"].nunique()),
            "paired_query_fold_truth_edges": int(len(ledger)),
            "nontrivial_coverage_matched_query_folds": int(len(ranking)),
        },
        "results": {
            "truth_edges_real_greater_than_wrong": int((ledger["real_minus_wrong"] > 0).sum()),
            "truth_edges_real_less_than_wrong": int((ledger["real_minus_wrong"] < 0).sum()),
            "median_real_minus_wrong": float(ledger["real_minus_wrong"].median()),
            "positive_ion_family_clusters": int((family_delta > 0).sum()),
            "one_sided_exact_family_signflip_p": float(p_value),
            "real_context_coverage_matched_truth_top1": int(ranking["real_context_truth_top1"].sum()) if len(ranking) else 0,
            "wrong_context_coverage_matched_truth_top1": int(ranking["wrong_context_truth_top1"].sum()) if len(ranking) else 0,
        },
        "interpretation": (
            "Real sample context consistently favors the frozen cross-adduct policy anchor over the degree-preserving "
            "wrong context, including a nontrivial sodium-adduct comparison, but only two ion families "
            "exist and BioAware is not statistically qualified."
        ),
        "leakage_status": "phenotype and response effects were not loaded",
        "provenance": {
            "hidden_truth_sha256": sha256(args.hidden_truth),
            "context_evidence_sha256": sha256(args.context_evidence),
            "truth_context_ledger_sha256": sha256(ledger_path),
            "coverage_matched_ranks_sha256": sha256(ranking_path),
            "implementation_sha256": sha256(Path(__file__)),
        },
        "claim_boundary": [
            "The ion-family label is a conflicting library-policy anchor, not authentic-standard confirmation.",
            "Candidate context is sparse and missing context is not negative evidence.",
            "Two ion families cannot support a general BioAware performance claim.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
