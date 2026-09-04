#!/usr/bin/env python
"""Unify frozen BioAware candidate evidence without outcome-based filtering."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["query_id", "candidate_id", "truth_candidate_id", "truth_formula"]


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


def aggregate_paths(path: Path, prefix: str, maximum_depth: int = 3) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "maximum_depth" in frame:
        frame = frame[frame["maximum_depth"] == maximum_depth].copy()
    if frame.duplicated(["fold", "query_id", "candidate_id"]).any():
        raise RuntimeError(f"duplicate fold/query/candidate rows in {path}")
    grouped = frame.groupby(KEY, as_index=False).agg(
        **{
            f"{prefix}_path_fraction": ("complete_ms2_paths", lambda x: float(np.mean(np.asarray(x) > 0))),
            f"{prefix}_identity_path_fraction": ("identity_paths", lambda x: float(np.mean(np.asarray(x) > 0))),
            f"{prefix}_best_bottleneck": ("best_bottleneck", lambda x: float(np.nanmedian(x)) if np.isfinite(x).any() else 0.0),
            f"{prefix}_median_bottleneck": ("median_bottleneck", lambda x: float(np.nanmedian(x)) if np.isfinite(x).any() else 0.0),
            f"{prefix}_rotations": ("fold", "nunique"),
        }
    )
    return grouped


def aggregate_smn(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.duplicated(["fold", "query_id", "candidate_id"]).any():
        raise RuntimeError(f"duplicate SMN fold/query/candidate rows in {path}")
    return frame.groupby(KEY, as_index=False).agg(
        smn_path_fraction=("path_available", "mean"),
        smn_best_bottleneck=("best_bottleneck", lambda x: float(np.nanmedian(x)) if np.isfinite(x).any() else 0.0),
        smn_supporting_paths=("supporting_paths", "median"),
        smn_rotations=("fold", "nunique"),
    )


def aggregate_rt(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.duplicated(["fold", "query_id", "candidate_id"]).any():
        raise RuntimeError(f"duplicate RT fold/query/candidate rows in {path}")
    return frame.groupby(KEY, as_index=False).agg(
        rt_score=("rt_score", "median"),
        rt_pass_fraction=("rt_pass", "mean"),
        rt_rotations=("fold", "nunique"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dreams", type=Path, default=Path(
        "data/validation/bioaware_metdna3_dreams_official_v1/candidate_scores.csv.gz"))
    parser.add_argument("--known-edge", type=Path, default=Path(
        "data/validation/bioaware_metdna3_candidate_edge_ms2_v1/candidate_edge_evidence.csv.gz"))
    parser.add_argument("--predicted-edge", type=Path, default=Path(
        "data/validation/bioaware_metdna3_candidate_edge_ms2_step1_v1/candidate_edge_evidence.csv.gz"))
    parser.add_argument("--smn", type=Path, default=Path(
        "data/validation/bioaware_metdna3_smn_headroom_v1/candidate_structural_evidence.csv.gz"))
    parser.add_argument("--rt", type=Path, default=Path(
        "data/validation/bioaware_metdna3_rt_headroom_v1/candidate_rt_evidence.csv.gz"))
    parser.add_argument("--decoder", type=Path, default=Path(
        "data/validation/bioaware_candidate_fragment_decoder_v1/candidate_scores.csv.gz"))
    parser.add_argument("--rules", type=Path, default=Path(
        "data/validation/bioaware_candidate_rule_likelihood_v1/candidate_rule_scores.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "data/validation/bioaware_candidate_evidence_ledger_v1"))
    args = parser.parse_args()
    sources = {
        "dreams": args.dreams, "known_edge": args.known_edge,
        "predicted_edge": args.predicted_edge, "smn": args.smn,
        "rt": args.rt, "decoder": args.decoder, "rules": args.rules,
    }
    for path in sources.values():
        if not path.exists():
            raise FileNotFoundError(path)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"fail-closed: non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    dreams = pd.read_csv(args.dreams)
    if dreams.duplicated(["query_id", "candidate_id"]).any():
        raise RuntimeError("DreaMS candidate table contains duplicates")
    ledger = dreams[KEY + ["spectral_score", "reference_spectra"]].copy()
    decoder = pd.read_csv(args.decoder)[KEY + ["decoder_score", "heldout_rotations"]]
    rules = pd.read_csv(args.rules)[KEY + [
        "rule_overlap_idf", "rule_jaccard_idf", "sparse_rule_overlap"
    ]]
    components = [
        decoder,
        rules,
        aggregate_paths(args.known_edge, "known_edge"),
        aggregate_paths(args.predicted_edge, "predicted_edge"),
        aggregate_smn(args.smn),
        aggregate_rt(args.rt),
    ]
    for component in components:
        before = len(ledger)
        ledger = ledger.merge(component, on=KEY, how="left", validate="one_to_one")
        if len(ledger) != before:
            raise RuntimeError("candidate evidence merge changed candidate count")
    evidence_columns = [
        "decoder_score", "rule_overlap_idf", "rule_jaccard_idf",
        "sparse_rule_overlap", "known_edge_path_fraction",
        "known_edge_best_bottleneck", "predicted_edge_path_fraction",
        "predicted_edge_best_bottleneck", "smn_path_fraction",
        "smn_best_bottleneck", "rt_score", "rt_pass_fraction",
    ]
    missing = {column: int(ledger[column].isna().sum()) for column in evidence_columns}
    # Missing network paths are genuine absence; missing decoder/rule/RT values
    # indicate an implementation failure and are not silently imputed.
    strict = ["decoder_score", "rule_overlap_idf", "rule_jaccard_idf", "rt_score"]
    bad = {column: missing[column] for column in strict if missing[column]}
    if bad:
        raise RuntimeError(f"strict evidence columns contain missing values: {bad}")
    network = [column for column in evidence_columns if column not in strict and column != "sparse_rule_overlap"]
    ledger[network] = ledger[network].fillna(0.0)
    ledger["sparse_rule_overlap"] = ledger["sparse_rule_overlap"].fillna(0.0)
    if ledger.groupby("query_id")["truth_candidate_id"].nunique().max() != 1:
        raise RuntimeError("query has inconsistent truth identity")
    if not np.all(ledger.groupby("query_id").apply(
        lambda group: int((group["candidate_id"] == group["truth_candidate_id"]).sum()) == 1,
        include_groups=False,
    )):
        raise RuntimeError("every query must contain exactly one truth candidate")
    ledger_path = output / "candidate_evidence.csv.gz"
    ledger.to_csv(ledger_path, index=False)
    payload = {
        "status": "bioaware_candidate_evidence_ledger_complete",
        "formal": True,
        "queries": int(ledger["query_id"].nunique()),
        "candidates": int(len(ledger)),
        "identities": int(ledger["truth_candidate_id"].nunique()),
        "formulas": int(ledger["truth_formula"].nunique()),
        "evidence_columns": evidence_columns,
        "missing_before_semantic_fill": missing,
        "rotation_count_range": {
            "minimum": int(min(
                ledger[column].min() for column in
                ["known_edge_rotations", "predicted_edge_rotations", "smn_rotations", "rt_rotations"]
            )),
            "maximum": int(max(
                ledger[column].max() for column in
                ["known_edge_rotations", "predicted_edge_rotations", "smn_rotations", "rt_rotations"]
            )),
        },
        "contracts": {
            "P2b": "forbidden",
            "phenotype": "forbidden",
            "truth_columns": "retained only for OOF training/evaluation, never as model features",
            "rotation_aggregation": "median/fraction over preregistered truth-identity-heldout rotations",
        },
        "provenance": {name: sha256(path) for name, path in sources.items()},
        "ledger_sha256": sha256(ledger_path),
    }
    atomic_json(output / "report.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
