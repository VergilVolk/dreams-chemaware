#!/usr/bin/env python
"""Build a formula-OOF-ready typed BioAware candidate ledger.

The same Level-1 query is evaluated against several frozen seed rotations.
Reaction-context features are computed independently in each rotation and only
then aggregated.  Truth identities are labels, never context inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from annotation.bioaware import BioAwareConfig, build_one_hop_evidence
from annotation.bioaware_context import extract_reaction_context_features


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores", type=Path,
        default=Path("data/validation/bioaware_metdna3_dreams_official_v1/candidate_scores.csv.gz"),
    )
    parser.add_argument(
        "--base-ledger", type=Path,
        default=Path("data/validation/bioaware_candidate_evidence_ledger_v1/candidate_evidence.csv.gz"),
    )
    parser.add_argument(
        "--splits", type=Path,
        default=Path("data/validation/bioaware_metdna3_development_v1/identity_splits.csv.gz"),
    )
    parser.add_argument(
        "--participants", type=Path,
        default=Path("data/reference/bioaware_rhea_offline_20260827/rhea_participants.csv.gz"),
    )
    parser.add_argument(
        "--reaction-directions", type=Path,
        default=Path("data/reference/bioaware_rhea_offline_20260827/rhea2reactome.tsv"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_typed_candidate_ledger_v1"),
    )
    args = parser.parse_args()
    for path in (
        args.scores, args.base_ledger, args.splits, args.participants,
        args.reaction_directions,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(args.scores)
    base = pd.read_csv(args.base_ledger)
    split = pd.read_csv(args.splits)
    participants = pd.read_csv(args.participants)
    directions = pd.read_csv(args.reaction_directions, sep="\t")
    graph_identities = set(participants["compound_id"].astype(str))
    config = BioAwareConfig()
    rotation_features: list[pd.DataFrame] = []

    for fold in sorted(split["fold"].unique()):
        seed_identities = set(
            split[(split["fold"] == fold) & (split["role"] == "seed")]["ik14"].astype(str)
        )
        heldout_identities = set(
            split[(split["fold"] == fold) & (split["role"] == "heldout")]["ik14"].astype(str)
        )
        fold_candidates = scores[
            scores["truth_candidate_id"].astype(str).isin(heldout_identities)
        ].copy()
        if not len(fold_candidates):
            raise RuntimeError(f"fold {fold} has no held-out candidates")
        eligible_seeds = sorted(seed_identities & graph_identities)
        seeds = pd.DataFrame(
            {
                "seed_query_id": [f"fold{fold}:seed:{identity}" for identity in eligible_seeds],
                "seed_compound_id": eligible_seeds,
                "seed_score": 1.0,
            }
        )
        paths = build_one_hop_evidence(participants, seeds, config)
        features, _ = extract_reaction_context_features(
            fold_candidates,
            paths,
            participants,
            seeds,
            reaction_directions=directions,
        )
        features["rotation_fold"] = int(fold)
        rotation_features.append(features)
        print(
            f"[typed-ledger] fold={fold} queries={fold_candidates['query_id'].nunique()} "
            f"seeds={len(eligible_seeds)}",
            flush=True,
        )

    rotations = pd.concat(rotation_features, ignore_index=True)
    if rotations.duplicated(["rotation_fold", "query_id", "candidate_id"]).any():
        raise RuntimeError("duplicate rotation/query/candidate feature rows")
    metadata = {"query_id", "candidate_id", "rotation_fold"}
    numeric = [
        column for column in rotations.columns
        if column not in metadata and pd.api.types.is_numeric_dtype(rotations[column])
    ]
    aggregations: dict[str, list[str]] = {column: ["mean", "max"] for column in numeric}
    aggregated = rotations.groupby(["query_id", "candidate_id"], sort=False).agg(aggregations)
    aggregated.columns = [f"typed_{column}_{stat}" for column, stat in aggregated.columns]
    aggregated = aggregated.reset_index()
    positive_columns = [
        "raw_network_support",
        "dependency_corrected_network_support",
        "candidate_specific_network_support",
        "complete_network_support",
        "direction_supported_network_support",
    ]
    for column in positive_columns:
        fraction = (
            rotations.assign(_positive=pd.to_numeric(rotations[column], errors="raise") > 0)
            .groupby(["query_id", "candidate_id"], sort=False)["_positive"]
            .mean()
            .rename(f"typed_{column}_positive_fraction")
            .reset_index()
        )
        aggregated = aggregated.merge(
            fraction, on=["query_id", "candidate_id"], how="left", validate="one_to_one"
        )
    rotation_count = (
        rotations.groupby(["query_id", "candidate_id"], sort=False)
        .size()
        .rename("typed_rotation_count")
        .reset_index()
    )
    aggregated = aggregated.merge(
        rotation_count, on=["query_id", "candidate_id"], validate="one_to_one"
    )

    key = ["query_id", "candidate_id"]
    if base.duplicated(key).any() or aggregated.duplicated(key).any():
        raise RuntimeError("candidate key is not unique")
    ledger = base.merge(aggregated, on=key, how="left", validate="one_to_one")
    typed_columns = [column for column in ledger if column.startswith("typed_")]
    if ledger[typed_columns].isna().any().any():
        missing = int(ledger[typed_columns].isna().any(axis=1).sum())
        raise RuntimeError(f"typed aggregation missing for {missing} candidate rows")
    if len(ledger) != len(base) or ledger["query_id"].nunique() != 117:
        raise RuntimeError("ledger coverage changed")
    ledger_path = args.output_dir / "candidate_evidence_typed.csv.gz"
    rotation_path = args.output_dir / "rotation_typed_features.csv.gz"
    ledger.to_csv(ledger_path, index=False, compression="gzip")
    rotations.to_csv(rotation_path, index=False, compression="gzip")
    report = {
        "status": "bioaware_typed_candidate_ledger_complete",
        "formal": True,
        "queries": int(ledger["query_id"].nunique()),
        "candidate_rows": int(len(ledger)),
        "identities": int(ledger["truth_candidate_id"].nunique()),
        "formulas": int(ledger["truth_formula"].nunique()),
        "rotation_rows": int(len(rotations)),
        "typed_features": int(len(typed_columns)),
        "excluded_identity_noop_paths": int(
            rotations["excluded_identity_noop_path_count"].sum()
        ),
        "contracts": {
            "P2b": "forbidden",
            "phenotype": "forbidden",
            "truth_identity": "label only; never a seed or context feature",
            "reaction_noops": "excluded before candidate competition and support aggregation",
            "rotation_aggregation": "outcome-blind mean/max/nonzero fraction",
        },
        "provenance": {
            "scores": sha256(args.scores),
            "base_ledger": sha256(args.base_ledger),
            "splits": sha256(args.splits),
            "participants": sha256(args.participants),
            "directions": sha256(args.reaction_directions),
            "ledger": sha256(ledger_path),
            "rotations": sha256(rotation_path),
        },
        "claim_limit": (
            "Consumed Level-1 development ledger. This artifact supports nested "
            "formula-OOF model development, not external performance claims."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
