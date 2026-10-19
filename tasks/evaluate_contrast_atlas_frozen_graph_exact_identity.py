#!/usr/bin/env python3
"""Calibrate a frozen Contrast Atlas spectral graph against strict identities.

This is deliberately a *post-freeze* entity-retrieval audit.  It never reads
response effects, contrast labels, biological relations, names, precursor
mass, or retention time.  A correct edge means only that the frozen k=1
cross-source edge lands on any target entity with the same strict planar
InChIKey.  It is not evidence for a biological-response relation or for an
unknown-entity increment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH = ROOT / "data/validation/contrast_atlas_official_local_alignment_v1_20261019"
DEFAULT_TRUTH = ROOT / "data/validation/contrast_atlas_exact_identity_calibration_v3_20261019"
DEFAULT_OUTPUT = ROOT / "data/validation/contrast_atlas_official_local_exact_identity_calibration_v1_20261019"
GRAPH_STATUS = "DEVELOPMENT_CONSUMED_LOCAL_ALIGNMENT_BASELINE_COMPLETE"
TRUTH_STATUS = "DEVELOPMENT_CONSUMED_POSTTRUTH_EXACT_IDENTITY_CALIBRATION_COMPLETE"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def orient_edges(edges: pd.DataFrame) -> pd.DataFrame:
    """Convert stored ordered-pair edges into actual query-to-candidate rows."""
    required = {
        "left_study_id", "right_study_id", "evidence_world", "direction",
        "left_entity_id", "right_entity_id", "chemical_similarity",
    }
    if missing := required.difference(edges.columns):
        raise ValueError(f"edge table missing columns: {sorted(missing)}")
    forward = edges.loc[edges["direction"].eq("left_to_right")].copy()
    reverse = edges.loc[edges["direction"].eq("right_to_left")].copy()
    if len(forward) + len(reverse) != len(edges):
        raise ValueError("unrecognised frozen edge direction")
    forward = forward.rename(columns={
        "left_study_id": "query_study_id",
        "left_entity_id": "query_entity_id",
        "right_study_id": "candidate_study_id",
        "right_entity_id": "candidate_entity_id",
    })
    reverse = reverse.rename(columns={
        "right_study_id": "query_study_id",
        "right_entity_id": "query_entity_id",
        "left_study_id": "candidate_study_id",
        "left_entity_id": "candidate_entity_id",
    })
    columns = [
        "evidence_world", "query_study_id", "query_entity_id",
        "candidate_study_id", "candidate_entity_id", "chemical_similarity",
    ]
    oriented = pd.concat([forward[columns], reverse[columns]], ignore_index=True)
    key = ["evidence_world", "query_study_id", "query_entity_id", "candidate_study_id"]
    if oriented.duplicated(key).any():
        raise ValueError("frozen k=1 graph contains duplicate query/target-source edges")
    return oriented.sort_values(key, kind="stable").reset_index(drop=True)


def strict_truth_groups(truth_pairs: pd.DataFrame, allowed_studies: set[str]) -> pd.DataFrame:
    required = {
        "planar_inchikey", "left_study_id", "left_entity_id",
        "right_study_id", "right_entity_id", "truth_class", "truth_scope",
    }
    if missing := required.difference(truth_pairs.columns):
        raise ValueError(f"truth table missing columns: {sorted(missing)}")
    truth = truth_pairs.loc[
        truth_pairs["truth_class"].eq("exact_planar_inchikey_identity")
        & truth_pairs["truth_scope"].eq("posttruth_spectral_entity_retrieval_calibration_only")
        & truth_pairs["left_study_id"].isin(allowed_studies)
        & truth_pairs["right_study_id"].isin(allowed_studies)
    ].copy()
    keys = ["left_study_id", "left_entity_id", "right_study_id"]
    grouped = (
        truth.groupby(keys, sort=True)
        .agg(
            strict_planar_inchikey=("planar_inchikey", "first"),
            correct_candidate_entity_ids=("right_entity_id", lambda values: ";".join(sorted(set(map(str, values))))),
            correct_candidate_count=("right_entity_id", "nunique"),
        )
        .reset_index()
        .rename(columns={
            "left_study_id": "query_study_id",
            "left_entity_id": "query_entity_id",
            "right_study_id": "candidate_study_id",
        })
    )
    if grouped.empty:
        raise ValueError("no strict cross-source truth groups overlap the frozen graph studies")
    return grouped


def evaluate(oriented_edges: pd.DataFrame, truth_groups: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    join = ["query_study_id", "query_entity_id", "candidate_study_id"]
    worlds = sorted(oriented_edges["evidence_world"].unique())
    summary_rows: list[dict[str, object]] = []
    detail_rows: list[pd.DataFrame] = []
    for world in worlds:
        world_edges = oriented_edges.loc[oriented_edges["evidence_world"].eq(world)].copy()
        detail = truth_groups.merge(world_edges, on=join, how="left", validate="one_to_one")
        detail["edge_available"] = detail["candidate_entity_id"].notna()
        correct_sets = detail["correct_candidate_entity_ids"].str.split(";")
        detail["top1_exact_identity_correct"] = [
            bool(available and str(candidate) in set(targets))
            for available, candidate, targets in zip(
                detail["edge_available"], detail["candidate_entity_id"], correct_sets
            )
        ]
        # The merge carries the world from the frozen edges; assigning rather
        # than inserting also keeps the pure-function test input ergonomic.
        detail["evidence_world"] = world
        covered = int(detail["edge_available"].sum())
        correct = int(detail["top1_exact_identity_correct"].sum())
        summary_rows.append({
            "evidence_world": world,
            "strict_truth_query_target_source_groups": int(len(detail)),
            "frozen_top1_edge_available": covered,
            "frozen_top1_edge_coverage": covered / len(detail),
            "top1_exact_identity_correct": correct,
            "top1_exact_identity_accuracy_conditional_on_edge": (
                correct / covered if covered else None
            ),
        })
        detail_rows.append(detail)
    return pd.DataFrame(summary_rows), pd.concat(detail_rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--truth-dir", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    graph_dir, truth_dir, output_dir = (
        args.graph_dir.resolve(), args.truth_dir.resolve(), args.output_dir.resolve()
    )
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    graph_report_path = graph_dir / "report.json"
    truth_report_path = truth_dir / "report.json"
    edge_path = graph_dir / "source_pair_frozen_edges.tsv.gz"
    truth_path = truth_dir / "directed_cross_source_entity_truth_pairs.tsv.gz"
    for path in (graph_report_path, truth_report_path, edge_path, truth_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    graph_report = json.loads(graph_report_path.read_text(encoding="utf-8"))
    truth_report = json.loads(truth_report_path.read_text(encoding="utf-8"))
    if graph_report.get("status") != GRAPH_STATUS:
        raise ValueError("graph is not a completed frozen local-alignment baseline")
    if truth_report.get("status") != TRUTH_STATUS:
        raise ValueError("truth asset is not the strict post-truth calibration ledger")
    # Guard against a graph that directly declares the opened truth artifact as input.
    if str(truth_path) in set(graph_report.get("input_sha256", {})):
        raise ValueError("graph report declares the post-truth ledger as an input")

    edges = pd.read_csv(edge_path, sep="\t", compression="gzip", dtype=str)
    edges["chemical_similarity"] = edges["chemical_similarity"].astype(float)
    truth_pairs = pd.read_csv(truth_path, sep="\t", compression="gzip", dtype=str)
    oriented = orient_edges(edges)
    allowed = set(oriented["query_study_id"]) | set(oriented["candidate_study_id"])
    groups = strict_truth_groups(truth_pairs, allowed)
    summary, detail = evaluate(oriented, groups)

    output_dir.mkdir(parents=True)
    # Keep the JSON report standards-compliant: an unavailable unresolved
    # identity accuracy is null, never a non-standard NaN literal.
    summary = summary.astype(object).where(pd.notna(summary), None)
    summary_path = output_dir / "world_summary.tsv"
    detail_path = output_dir / "strict_truth_edge_audit.tsv.gz"
    summary.to_csv(summary_path, sep="\t", index=False)
    detail.to_csv(detail_path, sep="\t", index=False, compression={"method": "gzip", "mtime": 0})
    report = {
        "schema": "contrast_atlas_frozen_local_graph_exact_identity_calibration_v1",
        "status": "DEVELOPMENT_CONSUMED_POSTTRUTH_FROZEN_GRAPH_ENTITY_CALIBRATION_COMPLETE",
        "data_role": "development_consumed",
        "posttruth_boundary": {
            "graph_input_declares_truth_ledger": False,
            "identity_opening": "strict planar InChIKey only after frozen graph construction",
            "forbidden_identity_fallbacks": ["compound name", "precursor m/z", "retention time", "threshold retuning"],
        },
        "denominators": {
            "frozen_directional_edges": int(len(oriented)),
            "frozen_graph_studies": int(len(allowed)),
            "strict_directed_truth_pairs_within_graph": int(
                ((truth_pairs["left_study_id"].isin(allowed)) & (truth_pairs["right_study_id"].isin(allowed))).sum()
            ),
            "strict_truth_query_target_source_groups": int(len(groups)),
        },
        "claim_boundary": [
            "This calibrates only frozen spectral entity retrieval against strict exact planar identity.",
            "It does not establish a biological-response relation, direction, mechanism, or unknown-entity increment.",
            "Every truth item is library-supported; unresolved entities have no identity truth in this audit.",
            "This result cannot be used to tune the frozen graph or its cosine threshold.",
        ],
        "input_sha256": {
            str(graph_report_path.relative_to(ROOT).as_posix()): sha256(graph_report_path),
            str(edge_path.relative_to(ROOT).as_posix()): sha256(edge_path),
            str(truth_report_path.relative_to(ROOT).as_posix()): sha256(truth_report_path),
            str(truth_path.relative_to(ROOT).as_posix()): sha256(truth_path),
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output_dir / "checksums.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in (summary_path, detail_path, report_path)),
        encoding="utf-8",
    )
    print(json.dumps({**report, "world_summary": summary.to_dict("records")}, indent=2))


if __name__ == "__main__":
    main()
