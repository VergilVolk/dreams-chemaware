#!/usr/bin/env python3
"""Fail-closed validation for the outcome-free KGMN/DreaMS edge manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values],
        dtype=object,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--edges", type=Path, default=Path("data/reference/metdna2_emrn_network_20260828/metdna2_emrn_edges.csv.gz"))
    parser.add_argument("--hdf5", type=Path, default=Path("data/models/MassSpecGym_MurckoHist_split.hdf5"))
    args = parser.parse_args()

    report_path = args.manifest_dir / "report.json"
    table_path = args.manifest_dir / "paired_reaction_decoy_triples.csv.gz"
    if not report_path.is_file() or not table_path.is_file():
        raise FileNotFoundError("edge calibration manifest is incomplete")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    table = pd.read_csv(table_path)
    if report.get("status") != "kgmn_dreams_edge_calibration_manifest_frozen":
        raise RuntimeError("unexpected manifest status")
    if report.get("outcome_columns_present") is not False:
        raise RuntimeError("manifest must be frozen before any score outcome")
    forbidden_fragments = ("dreams_score", "author_score", "probability", "corrected", "introduced", "label")
    forbidden = [column for column in table.columns if any(fragment in column.lower() for fragment in forbidden_fragments)]
    if forbidden:
        raise RuntimeError(f"outcome-like columns are forbidden in manifest: {forbidden}")
    if len(table) != report["counts"]["paired_triples"]:
        raise RuntimeError("paired-triple count mismatch")
    if sha256(table_path) != report["provenance"]["triples_sha256"]:
        raise RuntimeError("paired-triple hash mismatch")
    if sha256(args.edges) != report["provenance"]["edges_sha256"]:
        raise RuntimeError("MetDNA2 edge hash mismatch")
    if sha256(args.hdf5) != report["provenance"]["hdf5_sha256"]:
        raise RuntimeError("HDF5 hash mismatch")
    if not (table["positive_formula"] == table["decoy_formula"]).all():
        raise RuntimeError("positive and decoy formulas differ")
    if (table["source_ik14"] == table["positive_ik14"]).any() or (table["source_ik14"] == table["decoy_ik14"]).any():
        raise RuntimeError("source identity is reused as target")
    if (table["positive_ik14"] == table["decoy_ik14"]).any():
        raise RuntimeError("positive and decoy identities overlap")

    edges = pd.read_csv(args.edges, usecols=["ik14_a", "ik14_b"])
    network_pairs = {
        tuple(sorted((str(left), str(right))))
        for left, right in edges.dropna().itertuples(index=False, name=None)
        if str(left) != str(right)
    }
    decoy_pairs = {
        tuple(sorted(pair))
        for pair in table[["source_ik14", "decoy_ik14"]].itertuples(index=False, name=None)
    }
    if network_pairs & decoy_pairs:
        raise RuntimeError("one or more formula-matched decoys are MetDNA2 network neighbors")

    weight_sums = table.groupby("edge_id")["edge_equal_weight"].sum().to_numpy(dtype=float)
    if not np.allclose(weight_sums, 1.0, atol=1e-10, rtol=0):
        raise RuntimeError("edge-equal weights do not sum to one")
    fold_identities: dict[int, set[str]] = {}
    for fold, frame in table.groupby("component_fold"):
        fold_identities[int(fold)] = set(frame[["source_ik14", "positive_ik14", "decoy_ik14"]].to_numpy().ravel())
    folds = sorted(fold_identities)
    for left_index, left_fold in enumerate(folds):
        for right_fold in folds[left_index + 1 :]:
            if fold_identities[left_fold] & fold_identities[right_fold]:
                raise RuntimeError(f"identity leakage between folds {left_fold} and {right_fold}")

    with h5py.File(args.hdf5, "r") as handle:
        identity = decode(handle["INCHIKEY"][:])
        formula = decode(handle["FORMULA"][:])
        fold = decode(handle["fold"][:])
        instrument = decode(handle["INSTRUMENT_TYPE"][:])
        adduct = decode(handle["adduct"][:])
    for role in ("source", "positive", "decoy"):
        rows = table[f"{role}_row"].to_numpy(dtype=int)
        if (rows < 0).any() or (rows >= len(identity)).any():
            raise RuntimeError(f"invalid {role} rows")
        if not (fold[rows] == "train").all():
            raise RuntimeError(f"non-train {role} rows")
        if not (identity[rows] == table[f"{role}_ik14"].astype(str).to_numpy()).all():
            raise RuntimeError(f"{role} row identity mismatch")
        if role in {"positive", "decoy"} and not (formula[rows] == table[f"{role}_formula"].astype(str).to_numpy()).all():
            raise RuntimeError(f"{role} row formula mismatch")
        if not (instrument[rows] == table["instrument"].astype(str).to_numpy()).all():
            raise RuntimeError(f"{role} row instrument mismatch")
    exact = table["match_level"] == "instrument_adduct"
    if exact.any():
        for role in ("source", "positive", "decoy"):
            rows = table.loc[exact, f"{role}_row"].to_numpy(dtype=int)
            expected = table.loc[exact, f"{role}_adduct"].astype(str).to_numpy()
            if not (adduct[rows] == expected).all():
                raise RuntimeError(f"exact-match {role} adduct mismatch")

    print(
        json.dumps(
            {
                "status": "kgmn_dreams_edge_calibration_manifest_validation_passed",
                "paired_triples": int(len(table)),
                "edge_ids": int(table["edge_id"].nunique()),
                "component_folds": len(folds),
                "identity_overlap_across_folds": 0,
                "outcome_columns": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
