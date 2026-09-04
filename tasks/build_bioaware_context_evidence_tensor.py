"""Build leakage-safe, dependency-collapsed BioAware context tensors.

This stage converts typed Rhea paths into model inputs.  Outcome columns are
written to a separate labels table and are never present in the candidate or
edge tensors.  Multiple Rhea records supported by the same observed seed, or
multiple incomplete paths depending on the same missing co-substrate set, are
collapsed before learning/inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


FORBIDDEN_INPUT_TOKENS = (
    "truth", "correct", "introduced", "phenotype", "tumor", "normal",
    "case", "control", "qvalue", "pvalue", "outcome",
)

RELATION_TYPES = {
    "complete_direction_supported": 0,
    "complete_direction_unknown": 1,
    "incomplete_direction_supported": 2,
    "incomplete_direction_unknown": 3,
    "direction_conflicted": 4,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def dependency_key(row: pd.Series) -> str:
    if bool(row.source_side_complete):
        # Complete paths sharing one observed seed are one biological vote.
        return f"complete_seed:{text(row.seed_compound_id)}"
    missing = text(row.missing_source_signature)
    return f"missing:{missing or '__UNSPECIFIED__'}"


def relation_name(row: pd.Series) -> str:
    if bool(row.curated_direction_conflicted):
        return "direction_conflicted"
    complete = bool(row.source_side_complete)
    supported = bool(row.curated_direction_supported)
    if complete and supported:
        return "complete_direction_supported"
    if complete:
        return "complete_direction_unknown"
    if supported:
        return "incomplete_direction_supported"
    return "incomplete_direction_unknown"


def validate_input_columns(frame: pd.DataFrame, label: str) -> None:
    suspicious = [
        column for column in frame.columns
        if any(token in column.lower() for token in FORBIDDEN_INPUT_TOKENS)
    ]
    if suspicious:
        raise RuntimeError(f"{label} leaks forbidden columns: {suspicious}")


def build_one(candidate_path: Path, path_path: Path, output_dir: Path, cohort: str) -> dict:
    candidates_raw = pd.read_csv(candidate_path)
    paths = pd.read_csv(path_path)
    required_candidate = {"query_id", "candidate_id", "spectral_score", "truth_candidate_id", "truth_formula"}
    required_path = {
        "query_id", "candidate_id", "seed_compound_id", "seed_query_id", "reaction_id",
        "seed_score", "contribution", "source_side_complete", "source_side_completeness",
        "target_side_completeness", "missing_source_signature", "curated_direction_supported",
        "curated_direction_conflicted", "candidate_specificity",
    }
    if not required_candidate.issubset(candidates_raw.columns):
        raise RuntimeError(f"{cohort}: candidate schema missing {required_candidate - set(candidates_raw.columns)}")
    if not required_path.issubset(paths.columns):
        raise RuntimeError(f"{cohort}: path schema missing {required_path - set(paths.columns)}")
    if candidates_raw.duplicated(["query_id", "candidate_id"]).any():
        raise RuntimeError(f"{cohort}: duplicate query/candidate rows")

    label_columns = ["query_id", "candidate_id", "truth_candidate_id", "truth_formula"]
    for optional in ("is_pseudo_or_published_truth",):
        if optional in candidates_raw.columns:
            label_columns.append(optional)
    labels = candidates_raw[label_columns].copy()
    labels["is_positive"] = labels.candidate_id.astype(str) == labels.truth_candidate_id.astype(str)

    candidates = candidates_raw[["query_id", "candidate_id", "spectral_score"]].copy()
    candidates["query_id"] = candidates.query_id.astype(str)
    candidates["candidate_id"] = candidates.candidate_id.astype(str)

    paths = paths.copy()
    for column in ("query_id", "candidate_id", "seed_compound_id", "seed_query_id"):
        paths[column] = paths[column].astype(str)
    valid_pairs = set(zip(candidates.query_id, candidates.candidate_id))
    path_pairs = set(zip(paths.query_id, paths.candidate_id))
    if not path_pairs.issubset(valid_pairs):
        raise RuntimeError(f"{cohort}: typed path references a non-candidate pair")
    paths["dependency_key"] = paths.apply(dependency_key, axis=1)
    paths["relation_name"] = paths.apply(relation_name, axis=1)
    paths["relation_type"] = paths.relation_name.map(RELATION_TYPES).astype(np.int16)
    paths["path_confidence"] = np.clip(
        paths.get("specificity_weighted_contribution", paths.contribution).astype(float), 0.0, 1.0
    )
    paths["experimental_support"] = np.clip(paths.seed_score.astype(float), 0.0, 1.0)
    paths["reaction_completeness"] = np.sqrt(
        np.clip(paths.source_side_completeness.astype(float), 0.0, 1.0)
        * np.clip(paths.target_side_completeness.astype(float), 0.0, 1.0)
    )
    competition_conflict = 1.0 - np.clip(paths.candidate_specificity.astype(float), 0.0, 1.0)
    direction_conflict = paths.curated_direction_conflicted.astype(bool).astype(float)
    paths["conflict"] = np.maximum(competition_conflict, direction_conflict)

    # One independent message per query/candidate/seed/dependency group.  The
    # strongest calibrated member represents that dependent group.
    paths = paths.sort_values(
        ["query_id", "candidate_id", "seed_compound_id", "dependency_key", "path_confidence"],
        ascending=[True, True, True, True, False], kind="stable",
    )
    collapsed = paths.drop_duplicates(
        ["query_id", "candidate_id", "seed_compound_id", "dependency_key"], keep="first"
    ).copy()
    edge_columns = [
        "query_id", "candidate_id", "seed_query_id", "seed_compound_id", "reaction_id",
        "dependency_key", "relation_name", "relation_type", "path_confidence",
        "experimental_support", "reaction_completeness", "conflict",
    ]
    edges = collapsed[edge_columns].copy()

    edge_summary = edges.groupby(["query_id", "candidate_id"], sort=False).agg(
        context_edges=("dependency_key", "size"),
        context_seed_compounds=("seed_compound_id", "nunique"),
        context_confidence_max=("path_confidence", "max"),
        context_conflict_max=("conflict", "max"),
        context_complete_edges=("reaction_completeness", lambda x: int(np.sum(np.asarray(x) >= 0.999))),
    ).reset_index()
    candidates = candidates.merge(edge_summary, on=["query_id", "candidate_id"], how="left", validate="one_to_one")
    for column in ("context_edges", "context_seed_compounds", "context_complete_edges"):
        candidates[column] = candidates[column].fillna(0).astype(int)
    for column in ("context_confidence_max", "context_conflict_max"):
        candidates[column] = candidates[column].fillna(0.0).astype(float)
    candidates["context_state"] = "no_evidence"
    has = candidates.context_edges > 0
    conflict = candidates.context_conflict_max > 0
    candidates.loc[has & ~conflict, "context_state"] = "supported"
    candidates.loc[has & conflict, "context_state"] = "mixed_or_conflicted"

    validate_input_columns(candidates, f"{cohort} candidates")
    validate_input_columns(edges, f"{cohort} edges")
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_out = output_dir / f"{cohort}__candidates.csv.gz"
    edge_out = output_dir / f"{cohort}__edges.csv.gz"
    label_out = output_dir / f"{cohort}__labels.csv.gz"
    candidates.to_csv(candidate_out, index=False)
    edges.to_csv(edge_out, index=False)
    labels.to_csv(label_out, index=False)
    return {
        "queries": int(candidates.query_id.nunique()),
        "candidate_rows": int(len(candidates)),
        "candidate_identities": int(candidates.candidate_id.nunique()),
        "raw_paths": int(len(paths)),
        "dependency_collapsed_edges": int(len(edges)),
        "edge_reduction_fraction": float(1.0 - len(edges) / max(1, len(paths))),
        "queries_with_context": int(candidates.loc[candidates.context_edges > 0, "query_id"].nunique()),
        "candidates_with_context": int((candidates.context_edges > 0).sum()),
        "conflicted_candidates": int((candidates.context_conflict_max > 0).sum()),
        "candidate_sha256": sha256(candidate_out),
        "edge_sha256": sha256(edge_out),
        "label_sha256": sha256(label_out),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir", type=Path,
        default=Path("data/validation/bioaware_reaction_context_broad_noop_filtered_20260830"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_context_evidence_tensor_20260830"),
    )
    parser.add_argument("--cohorts", nargs="+", default=["mtbls13729_expanded", "mtbls1905_auto"])
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    results = {}
    provenance = {}
    for cohort in args.cohorts:
        candidate_path = args.source_dir / f"{cohort}__candidate_context.csv.gz"
        path_path = args.source_dir / f"{cohort}__path_context.csv.gz"
        if not candidate_path.exists() or not path_path.exists():
            raise FileNotFoundError(f"missing context inputs for {cohort}")
        results[cohort] = build_one(candidate_path, path_path, args.output_dir, cohort)
        provenance[cohort] = {
            "candidate_context_sha256": sha256(candidate_path),
            "path_context_sha256": sha256(path_path),
        }
    report = {
        "status": "bioaware_context_evidence_tensor_complete",
        "formal": False,
        "source": str(args.source_dir),
        "cohorts": results,
        "relation_types": RELATION_TYPES,
        "contracts": {
            "phenotype_blind": True,
            "outcomes_separated_from_inputs": True,
            "identity_noop_source_filtered": "required by source-directory contract",
            "dependency_collapse": "complete paths collapse by observed seed; incomplete paths collapse by missing-source signature",
            "missing_relation_semantics": "unknown, never negative",
            "consumed_development_only": True,
        },
        "provenance": provenance,
        "claim_limit": "This is a leakage-safe tensorization of consumed development evidence; it is not external performance.",
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
