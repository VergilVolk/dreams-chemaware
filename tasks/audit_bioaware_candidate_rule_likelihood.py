#!/usr/bin/env python
"""Candidate-specific rule-likelihood headroom for unresolved BioAware errors.

Unlike the old rule-overlap label, this audit compares each query's observed
mass motifs with the empirical motif prevalence of every candidate's reference
spectra.  Candidate identities are not inferred from rule presence alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from annotation.rule_evidence import load_main_rules, spectrum_rule_vector


EPS = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def weighted_overlap(query: np.ndarray, candidate: np.ndarray, weight: np.ndarray) -> float:
    denominator = float(np.sum(weight * query))
    if denominator <= EPS:
        return np.nan
    return float(np.sum(weight * query * candidate) / denominator)


def weighted_jaccard(query: np.ndarray, candidate: np.ndarray, weight: np.ndarray) -> float:
    numerator = float(np.sum(weight * np.minimum(query, candidate)))
    denominator = float(np.sum(weight * np.maximum(query, candidate)))
    return numerator / denominator if denominator > EPS else np.nan


def top1(group: pd.DataFrame, column: str) -> bool:
    values = pd.to_numeric(group[column], errors="coerce")
    if not np.isfinite(values).any():
        return False
    maximum = float(np.nanmax(values))
    tied = group[np.isclose(values, maximum, rtol=0, atol=1e-12)]
    truth = str(group["truth_candidate_id"].iloc[0])
    return bool(len(tied) == 1 and str(tied.iloc[0].candidate_id) == truth)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=Path(
        "data/validation/bioaware_metdna3_dreams_official_v1/candidate_scores.csv.gz"))
    parser.add_argument("--queries", type=Path, default=Path(
        "data/validation/bioaware_metdna3_dreams_cache_v1/queries.csv.gz"))
    parser.add_argument("--query-tensors", type=Path, default=Path(
        "data/validation/bioaware_metdna3_dreams_cache_v1/query_tensors.npz"))
    parser.add_argument("--references", type=Path, default=Path(
        "data/validation/bioaware_metdna3_dreams_cache_v1/candidate_references.csv.gz"))
    parser.add_argument("--hdf5", type=Path, default=Path(
        "data/models/MassSpecGym_MurckoHist_split.hdf5"))
    parser.add_argument("--unresolved", type=Path, default=Path(
        "data/validation/bioaware_10pp_headroom_v1/unresolved_error_queries.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "data/validation/bioaware_candidate_rule_likelihood_v1"))
    parser.add_argument(
        "--all-queries-unresolved", action="store_true",
        help="evaluation-only mode: do not load an outcome-derived unresolved-query subset",
    )
    args = parser.parse_args()
    required_paths = [args.scores, args.queries, args.query_tensors, args.references, args.hdf5]
    if not args.all_queries_unresolved:
        required_paths.append(args.unresolved)
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"fail-closed: non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(args.scores)
    queries = pd.read_csv(args.queries)
    references = pd.read_csv(args.references)
    unresolved = None if args.all_queries_unresolved else pd.read_csv(args.unresolved)
    tensors = np.load(args.query_tensors)["query_tensor"]
    if len(queries) != len(tensors):
        raise RuntimeError("query tensor/table row mismatch")
    query_index = {str(value): index for index, value in enumerate(queries["query_id"])}
    rules = load_main_rules()
    categories = np.asarray([str(rule["category"]) for rule in rules])
    useful = np.isin(categories, ["NL", "CF", "ISO"])
    sparse = np.isin(categories, ["CF", "ISO"])

    query_hits = np.zeros((len(queries), len(rules)), dtype=np.uint8)
    for index, tensor in enumerate(tensors):
        precursor = float(tensor[0, 0])
        query_hits[index] = spectrum_rule_vector(tensor[1:, 0], precursor, rules)

    unique_rows = np.sort(references["reference_row"].unique().astype(np.int64))
    with h5py.File(args.hdf5, "r") as handle:
        spectra = np.asarray(handle["spectrum"][unique_rows], dtype=np.float32)
        precursor = np.asarray(handle["precursor_mz"][unique_rows], dtype=np.float32)
    reference_hits = np.zeros((len(unique_rows), len(rules)), dtype=np.uint8)
    for index in range(len(unique_rows)):
        reference_hits[index] = spectrum_rule_vector(spectra[index, 0], float(precursor[index]), rules)
        if (index + 1) % 1000 == 0:
            print(f"[rule-likelihood] {index + 1:,}/{len(unique_rows):,} references", flush=True)
    row_position = {int(row): index for index, row in enumerate(unique_rows)}
    prevalence = reference_hits.mean(axis=0)
    idf = np.log((len(reference_hits) + 1.0) / (reference_hits.sum(axis=0) + 1.0))

    candidate_rows: list[dict] = []
    for base in scores.itertuples(index=False):
        query_id = str(base.query_id)
        subset = references[
            (references["query_id"].astype(str) == query_id)
            & (references["candidate_id"].astype(str) == str(base.candidate_id))
        ]
        positions = [row_position[int(row)] for row in subset["reference_row"]]
        candidate_prevalence = reference_hits[positions].mean(axis=0)
        query_vector = query_hits[query_index[query_id]].astype(float)
        candidate_rows.append({
            "query_id": query_id,
            "candidate_id": str(base.candidate_id),
            "truth_candidate_id": str(base.truth_candidate_id),
            "truth_formula": str(base.truth_formula),
            "spectral_score": float(base.spectral_score),
            "reference_spectra": int(len(positions)),
            "rule_overlap_idf": weighted_overlap(
                query_vector[useful], candidate_prevalence[useful], idf[useful]
            ),
            "rule_jaccard_idf": weighted_jaccard(
                query_vector[useful], candidate_prevalence[useful], idf[useful]
            ),
            "sparse_rule_overlap": weighted_overlap(
                query_vector[sparse], candidate_prevalence[sparse], np.ones(int(sparse.sum()))
            ),
        })
    candidate_table = pd.DataFrame(candidate_rows)
    unresolved_ids = set(scores.query_id.astype(str)) if unresolved is None else set(unresolved["query_id"].astype(str))
    query_rows: list[dict] = []
    score_columns = ["rule_overlap_idf", "rule_jaccard_idf", "sparse_rule_overlap"]
    for query_id, group in candidate_table.groupby("query_id", sort=False):
        row = {
            "query_id": str(query_id),
            "truth_candidate_id": str(group["truth_candidate_id"].iloc[0]),
            "truth_formula": str(group["truth_formula"].iloc[0]),
            "baseline_correct": top1(group, "spectral_score"),
            "unresolved_before_g2": str(query_id) in unresolved_ids,
        }
        for column in score_columns:
            row[f"{column}_correct"] = top1(group, column)
        query_rows.append(row)
    query_table = pd.DataFrame(query_rows)
    arm_report = {}
    for column in score_columns:
        correct = query_table[f"{column}_correct"]
        corrected = (~query_table["baseline_correct"]) & correct
        introduced = query_table["baseline_correct"] & ~correct
        independent = corrected & query_table["unresolved_before_g2"]
        arm_report[column] = {
            "recall1": float(correct.mean()),
            "corrected": int(corrected.sum()),
            "introduced": int(introduced.sum()),
            "new_unresolved_headroom": int(independent.sum()),
            "new_unresolved_identities": int(query_table.loc[independent, "truth_candidate_id"].nunique()),
            "new_unresolved_query_ids": sorted(query_table.loc[independent, "query_id"].astype(str)),
        }
    candidate_path = output / "candidate_rule_scores.csv.gz"
    query_path = output / "query_headroom.csv.gz"
    candidate_table.to_csv(candidate_path, index=False)
    query_table.to_csv(query_path, index=False)
    payload = {
        "status": "bioaware_candidate_rule_likelihood_headroom_complete",
        "formal": True,
        "queries": int(len(query_table)),
        "baseline_recall1": float(query_table["baseline_correct"].mean()),
        "rules": int(len(rules)),
        "useful_rules": int(useful.sum()),
        "sparse_rules": int(sparse.sum()),
        "reference_spectra": int(len(unique_rows)),
        "arms": arm_report,
        "decision": (
            "No arm is a deployable override. Proceed only if independent unresolved "
            "headroom replicates across at least two fixed rule-score definitions."
        ),
        "provenance": {
            "scores_sha256": sha256(args.scores),
            "query_tensors_sha256": sha256(args.query_tensors),
            "references_sha256": sha256(args.references),
            "unresolved_sha256": None if unresolved is None else sha256(args.unresolved),
            "candidate_scores_sha256": sha256(candidate_path),
            "query_headroom_sha256": sha256(query_path),
        },
        "claim_limit": (
            "Observed mass-motif compatibility is not unique structural assignment. "
            "This consumed-development audit measures only candidate-specific headroom."
        ),
    }
    atomic_json(output / "report.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
