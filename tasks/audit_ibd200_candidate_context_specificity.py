#!/usr/bin/env python3
"""Audit real versus degree-preserving wrong IBD candidate context.

This is a development diagnostic on strict library-supported query edges.  It
tests whether the candidate-specific sample evidence is stronger for the real
query profile than for its frozen detection-degree-matched wrong profile.  The
strict anchor policy is shared by construction, so this audit cannot serve as
independent identity truth or as a biological-benefit claim.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_cluster_signflip_p(cluster_values: np.ndarray) -> float:
    values = np.asarray(cluster_values, dtype=np.float64)
    if not 1 <= len(values) <= 20 or not np.isfinite(values).all():
        raise ValueError("exact sign flip requires 1..20 finite clusters")
    observed = float(values.mean())
    exceed = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        total += 1
        if float(np.mean(values * np.asarray(signs))) >= observed - 1e-15:
            exceed += 1
    return exceed / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-evidence", required=True, type=Path)
    parser.add_argument("--identity-partition", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    evidence = np.load(args.context_evidence, allow_pickle=False)
    query_ptr = np.asarray(evidence["query_ptr"], dtype=np.int64)
    query_ids = np.asarray(evidence["query_ids"]).astype("U")
    candidate_ids = np.asarray(evidence["candidate_ids"]).astype("U")
    edge_queries = np.repeat(query_ids, np.diff(query_ptr))
    folds = np.asarray(evidence["outer_folds"], dtype=np.int64)
    paired = np.asarray(evidence["paired_context_available"], dtype=bool)
    real = np.asarray(evidence["real_context_support"], dtype=np.float64)
    wrong = np.asarray(evidence["wrong_context_support"], dtype=np.float64)

    identity = pd.read_csv(args.identity_partition, sep="\t", dtype=str)
    supported = identity["library_supported"].str.lower().eq("true")
    truth = dict(zip(
        identity.loc[supported, "entity_id"].astype(str),
        identity.loc[supported, "library_inchikey"].astype(str),
    ))
    is_policy_match = np.asarray([
        truth.get(str(query_id)) == str(candidate_id)
        for query_id, candidate_id in zip(edge_queries, candidate_ids)
    ])

    rows: list[dict[str, object]] = []
    for fold_index, fold in enumerate(folds):
        selected = paired[fold_index] & is_policy_match
        for edge_index in np.flatnonzero(selected):
            rows.append({
                "outer_fold": int(fold),
                "entity_id": str(edge_queries[edge_index]),
                "candidate_id": str(candidate_ids[edge_index]),
                "real_context_support": float(real[fold_index, edge_index]),
                "wrong_context_support": float(wrong[fold_index, edge_index]),
                "real_minus_wrong": float(real[fold_index, edge_index] - wrong[fold_index, edge_index]),
            })
    ledger = pd.DataFrame(rows)
    if ledger.empty:
        raise RuntimeError("no paired strict-policy match edges are available")
    cluster = (
        ledger.groupby("candidate_id", as_index=False)
        .agg(edges=("real_minus_wrong", "size"), mean_delta=("real_minus_wrong", "mean"),
             median_delta=("real_minus_wrong", "median"))
    )
    p_value = exact_cluster_signflip_p(cluster["mean_delta"].to_numpy())
    query_delta = ledger.groupby("entity_id")["real_minus_wrong"].mean()

    args.output_dir.mkdir(parents=True)
    ledger_path = args.output_dir / "paired_policy_match_edges.tsv"
    cluster_path = args.output_dir / "structure_cluster_summary.tsv"
    ledger.to_csv(ledger_path, sep="\t", index=False)
    cluster.to_csv(cluster_path, sep="\t", index=False)
    report = {
        "schema": "ibd200_plate3_candidate_context_specificity_audit_v1",
        "status": "REAL_CONTEXT_TREND_NOT_QUALIFIED",
        "data_role": "development_consumed_posthoc_diagnostic",
        "denominators": {
            "paired_policy_match_fold_edges": int(len(ledger)),
            "unique_query_entities": int(ledger["entity_id"].nunique()),
            "independent_candidate_structure_clusters": int(len(cluster)),
            "outer_folds": int(ledger["outer_fold"].nunique()),
        },
        "results": {
            "median_fold_edge_real_minus_wrong": float(ledger["real_minus_wrong"].median()),
            "mean_fold_edge_real_minus_wrong": float(ledger["real_minus_wrong"].mean()),
            "positive_fold_edges": int((ledger["real_minus_wrong"] > 0).sum()),
            "negative_fold_edges": int((ledger["real_minus_wrong"] < 0).sum()),
            "median_query_mean_delta": float(query_delta.median()),
            "positive_structure_clusters": int((cluster["mean_delta"] > 0).sum()),
            "one_sided_exact_structure_cluster_signflip_p": float(p_value),
        },
        "interpretation": (
            "Real context is directionally stronger than the degree-preserving wrong context, "
            "but only a handful of candidate structures contribute and the exact structure-cluster "
            "test is not significant. BioAware is therefore not qualified by this slice."
        ),
        "leakage_status": "no phenotype, beta, SE, held-out effect or retrieval truth was loaded",
        "provenance": {
            "context_evidence_sha256": sha256(args.context_evidence),
            "identity_partition_sha256": sha256(args.identity_partition),
            "edge_ledger_sha256": sha256(ledger_path),
            "cluster_summary_sha256": sha256(cluster_path),
            "implementation_sha256": sha256(Path(__file__)),
        },
        "claim_boundary": [
            "The strict library policy supplies both anchors and policy-match labels, so this is not independent identity validation.",
            "The result does not establish BioAware ranking gain or biological-response gain.",
            "Context-unavailable candidate edges remain abstentions and stay in the full 662-query denominator.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
