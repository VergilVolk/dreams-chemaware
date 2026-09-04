#!/usr/bin/env python
"""Seal a metadata-only MoNA identity-disjoint MS/MS retrieval panel.

This is a transfer audit, not a claim that the spectra were absent from DreaMS
pretraining.  Every query and candidate identity is excluded from the local
MassSpecGym HDF5 used to train/develop the adapter.  No embedding or model score
is consulted while constructing the candidate graph.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_hdf_strings(handle: h5py.File, key: str) -> set[str]:
    output = set()
    for value in handle[key][:]:
        text = value.decode("utf-8", "ignore") if isinstance(value, bytes) else str(value)
        if text:
            output.add(text[:14])
    return output


def formula_from_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    return rdMolDescriptors.CalcMolFormula(mol) if mol is not None else ""


def ppm_close(values: np.ndarray, center: float, ppm: float) -> np.ndarray:
    return np.abs(values - center) <= max(abs(center), 1.0) * ppm * 1e-6


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/models/mona_neg_dreams_emb/manifest.csv")
    parser.add_argument("--embeddings", type=Path, default=ROOT / "data/models/mona_neg_dreams_emb/embeddings.npy")
    parser.add_argument("--mgf", type=Path, default=ROOT / "data/models/mona_neg_full.mgf")
    parser.add_argument("--hdf5", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/mona_identity_disjoint_transfer_panel")
    parser.add_argument("--ppm", type=float, default=20.0)
    parser.add_argument("--queries-per-identity", type=int, default=2)
    parser.add_argument("--references-per-candidate", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--overwrite", action="store_true", help="development only; never use after formal sealing")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    required = (args.manifest, args.embeddings, args.mgf, args.hdf5)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"refusing to overwrite sealed panel: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.manifest).fillna("")
    embeddings = np.load(args.embeddings, mmap_mode="r")
    if len(frame) != len(embeddings):
        raise RuntimeError("MoNA manifest/embedding row mismatch")
    if not {"smiles", "inchikey", "precursor_mz"}.issubset(frame.columns):
        raise RuntimeError("MoNA manifest lacks required metadata")
    with h5py.File(args.hdf5, "r") as handle:
        development_identities = read_hdf_strings(handle, "INCHIKEY")

    RDLogger.DisableLog("rdApp.*")
    frame["row"] = np.arange(len(frame), dtype=np.int64)
    frame["ik14"] = frame.inchikey.astype(str).str.slice(0, 14)
    frame["formula"] = [formula_from_smiles(value) for value in frame.smiles]
    frame["precursor_mz"] = pd.to_numeric(frame.precursor_mz, errors="coerce")
    frame = frame[
        frame.ik14.str.len().eq(14)
        & frame.formula.ne("")
        & np.isfinite(frame.precursor_mz)
        & ~frame.ik14.isin(development_identities)
    ].copy()

    # Require repeated spectra so query and positive reference are distinct.
    identity_count = frame.groupby("ik14").size()
    repeated = set(identity_count[identity_count >= 2].index.astype(str))
    frame = frame[frame.ik14.isin(repeated)].copy()
    identity_formula_count = frame[["formula", "ik14"]].drop_duplicates().groupby("formula").size()
    candidate_formulas = set(identity_formula_count[identity_formula_count >= 2].index.astype(str))
    frame = frame[frame.formula.isin(candidate_formulas)].copy()

    rows_by_formula_identity: dict[str, dict[str, np.ndarray]] = {}
    for formula, formula_frame in frame.groupby("formula", sort=True):
        rows_by_formula_identity[str(formula)] = {
            str(identity): group.sort_values(["precursor_mz", "row"]).row.to_numpy(np.int64)
            for identity, group in formula_frame.groupby("ik14", sort=True)
        }
    mz = pd.to_numeric(pd.read_csv(args.manifest).precursor_mz, errors="coerce").to_numpy(float)

    query_row: list[int] = []
    query_ik14: list[str] = []
    query_formula: list[str] = []
    query_ptr = [0]
    molecule_ik14: list[str] = []
    molecule_label: list[bool] = []
    molecule_ptr = [0]
    candidate_row: list[int] = []

    for formula in sorted(rows_by_formula_identity):
        identities = rows_by_formula_identity[formula]
        for identity in sorted(identities):
            source_rows = identities[identity]
            if len(source_rows) <= args.queries_per_identity:
                selected_queries = source_rows[: args.queries_per_identity]
            else:
                positions = np.linspace(0, len(source_rows) - 1, args.queries_per_identity, dtype=int)
                selected_queries = source_rows[np.unique(positions)]
            for query in selected_queries:
                center = float(mz[query])
                candidate_groups: list[tuple[str, np.ndarray]] = []
                for candidate_identity, available in identities.items():
                    allowed = available[ppm_close(mz[available], center, args.ppm)]
                    if candidate_identity == identity:
                        allowed = allowed[allowed != query]
                    if len(allowed):
                        candidate_groups.append((candidate_identity, allowed[: args.references_per_candidate]))
                positive = [item for item in candidate_groups if item[0] == identity]
                negative = sorted((item for item in candidate_groups if item[0] != identity), key=lambda item: item[0])
                if len(positive) != 1 or not negative:
                    continue
                query_row.append(int(query))
                query_ik14.append(identity)
                query_formula.append(formula)
                for candidate_identity, references in positive + negative:
                    molecule_ik14.append(candidate_identity)
                    molecule_label.append(candidate_identity == identity)
                    candidate_row.extend(map(int, references))
                    molecule_ptr.append(len(candidate_row))
                query_ptr.append(len(molecule_ik14))

    arrays = {
        "query_row": np.asarray(query_row, dtype=np.int64),
        "query_ik14": np.asarray(query_ik14, dtype="U14"),
        "query_formula": np.asarray(query_formula, dtype="U64"),
        "query_ptr": np.asarray(query_ptr, dtype=np.int64),
        "molecule_ik14": np.asarray(molecule_ik14, dtype="U14"),
        "molecule_label": np.asarray(molecule_label, dtype=bool),
        "molecule_ptr": np.asarray(molecule_ptr, dtype=np.int64),
        "candidate_row": np.asarray(candidate_row, dtype=np.int64),
    }
    if len(query_row) < 500 or len(set(query_ik14)) < 300 or len(set(query_formula)) < 100:
        raise RuntimeError(
            f"transfer panel is underpowered: queries={len(query_row)} identities={len(set(query_ik14))} "
            f"formulas={len(set(query_formula))}"
        )
    if set(query_ik14) & development_identities or set(molecule_ik14) & development_identities:
        raise RuntimeError("identity leakage into transfer panel")
    for query in range(len(query_row)):
        left, right = query_ptr[query:query + 2]
        if np.sum(arrays["molecule_label"][left:right]) != 1 or not arrays["molecule_label"][left]:
            raise RuntimeError(f"query {query} does not have unique-first positive")

    np.savez_compressed(args.output_dir / "panel.npz", **arrays)
    candidate_counts = np.diff(arrays["query_ptr"])
    reference_counts = np.diff(arrays["molecule_ptr"])
    report = {
        "status": "mona_identity_disjoint_transfer_panel_sealed",
        "formal": True,
        "n_queries": len(query_row),
        "n_query_identities": len(set(query_ik14)),
        "n_candidate_identities": len(set(molecule_ik14)),
        "n_formulas": len(set(query_formula)),
        "n_candidate_molecules": len(molecule_ik14),
        "n_candidate_spectra": len(candidate_row),
        "candidates_per_query": {
            "median": float(np.median(candidate_counts)), "p90": float(np.quantile(candidate_counts, 0.9)),
            "maximum": int(candidate_counts.max()),
        },
        "references_per_candidate": {
            "median": float(np.median(reference_counts)), "p90": float(np.quantile(reference_counts, 0.9)),
            "maximum": int(reference_counts.max()),
        },
        "development_identity_overlap": 0,
        "construction_uses_model_scores": False,
        "candidate_protocol": f"same molecular formula and {args.ppm:g} ppm precursor window; unique-first positive; ties count against positive",
        "claim_limit": (
            "Identity-disjoint transfer relative to the local MassSpecGym development HDF5. "
            "Because MoNA may have contributed to DreaMS pretraining, this is not proof of pretraining-corpus novelty."
        ),
        "parameters": vars(args) | {"manifest": str(args.manifest), "embeddings": str(args.embeddings), "mgf": str(args.mgf), "hdf5": str(args.hdf5), "output_dir": str(args.output_dir)},
        "provenance": {
            "manifest_sha256": sha256_file(args.manifest), "embeddings_sha256": sha256_file(args.embeddings),
            "mgf_sha256": sha256_file(args.mgf), "hdf5_sha256": sha256_file(args.hdf5),
            "panel_sha256": sha256_file(args.output_dir / "panel.npz"),
            "script_sha256": sha256_file(Path(__file__)),
        },
    }
    # Convert Path values introduced by vars(args).
    report["parameters"] = {key: str(value) if isinstance(value, Path) else value for key, value in report["parameters"].items()}
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
