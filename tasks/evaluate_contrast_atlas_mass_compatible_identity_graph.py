#!/usr/bin/env python3
"""Open strict known identities after a mass-compatible graph is frozen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluate_contrast_atlas_frozen_graph_exact_identity import (
    evaluate,
    orient_edges,
    sha256,
    strict_truth_groups,
)


GRAPH_STATUS = "DEVELOPMENT_CONSUMED_MASS_COMPATIBLE_IDENTITY_GRAPH_FROZEN"
TRUTH_STATUS = "DEVELOPMENT_CONSUMED_POSTTRUTH_EXACT_IDENTITY_CALIBRATION_COMPLETE"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", required=True, type=Path)
    parser.add_argument("--truth-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    graph_report_path = args.graph_dir / "report.json"
    graph_edges_path = args.graph_dir / "source_pair_frozen_edges.tsv.gz"
    truth_report_path = args.truth_dir / "report.json"
    truth_pairs_path = args.truth_dir / "directed_cross_source_entity_truth_pairs.tsv.gz"
    graph_report = json.loads(graph_report_path.read_text(encoding="utf-8"))
    truth_report = json.loads(truth_report_path.read_text(encoding="utf-8"))
    if graph_report.get("status") != GRAPH_STATUS:
        raise ValueError("input is not a frozen mass-compatible graph")
    if truth_report.get("status") != TRUTH_STATUS:
        raise ValueError("input is not a strict posttruth identity ledger")
    if "strict_identity" in json.dumps(graph_report).lower() or str(truth_pairs_path) in json.dumps(graph_report):
        raise ValueError("graph report appears to declare opened identity truth")

    edges = pd.read_csv(graph_edges_path, sep="\t", dtype=str)
    edges["chemical_similarity"] = pd.to_numeric(edges.chemical_similarity, errors="raise")
    truth_pairs = pd.read_csv(truth_pairs_path, sep="\t", dtype=str)
    oriented = orient_edges(edges)
    allowed = set(oriented.query_study_id) | set(oriented.candidate_study_id)
    groups = strict_truth_groups(truth_pairs, allowed)
    summary, detail = evaluate(oriented, groups)

    args.output_dir.mkdir(parents=True)
    summary_path = args.output_dir / "world_summary.tsv"
    detail_path = args.output_dir / "strict_truth_edge_audit.tsv.gz"
    summary.to_csv(summary_path, sep="\t", index=False, na_rep="NA")
    detail.to_csv(
        detail_path, sep="\t", index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    all_row = summary.loc[summary.evidence_world.eq("all")].iloc[0]
    json_summary = summary.astype(object).where(pd.notna(summary), None)
    report = {
        "schema": "contrast_atlas_mass_compatible_identity_calibration_v1",
        "status": "DEVELOPMENT_CONSUMED_POSTTRUTH_MASS_COMPATIBLE_IDENTITY_CALIBRATION_COMPLETE",
        "evidence_role": {
            "dataset_status": "development_consumed",
            "leakage_state": "strict_identity_opened_only_after_graph_was_written",
            "external_confirmation_claim": False,
        },
        "primary_all_world_result": {
            "truth_query_target_source_groups": int(all_row.strict_truth_query_target_source_groups),
            "answered": int(all_row.frozen_top1_edge_available),
            "coverage": float(all_row.frozen_top1_edge_coverage),
            "correct": int(all_row.top1_exact_identity_correct),
            "precision_among_answered": float(all_row.top1_exact_identity_accuracy_conditional_on_edge),
        },
        "world_summary": json_summary.to_dict("records"),
        "claim_boundary": [
            "This calibrates known exact planar identities, not unresolved entities.",
            "It validates neither a biological response nor an unknown-information increment.",
            "The graph uses official DreaMS only; frozen Noise/P2b/unified comparisons remain required.",
            "All studies are development/consumed."
        ],
        "provenance": {
            "graph_report_sha256": sha256(graph_report_path),
            "graph_edges_sha256": sha256(graph_edges_path),
            "truth_report_sha256": sha256(truth_report_path),
            "truth_pairs_sha256": sha256(truth_pairs_path),
            "world_summary_sha256": sha256(summary_path),
            "detail_sha256": sha256(detail_path),
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
