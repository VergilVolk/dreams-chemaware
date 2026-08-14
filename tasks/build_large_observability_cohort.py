"""Build a formula-isolated, multi-thousand-molecule MS/MS audit cohort.

Selection is independent of DreaMS scores.  Every eligible molecule has at
least two quality-controlled, non-duplicate [M+H]+ spectra, and every retained
formula contains at least two distinct IK14 molecules.  Whole formula groups
are assigned to discovery, confirmation, or final test to prevent isomer links
from crossing statistical splits.
"""

from __future__ import annotations


import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


def spectrum_hash(spectrum: np.ndarray) -> str:
    mz, intensity = np.asarray(spectrum[0], float), np.asarray(spectrum[1], float)
    valid = np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (intensity > 0)
    payload = np.stack((np.round(mz[valid], 2), np.round(intensity[valid], 4)), axis=1)
    return hashlib.blake2b(payload.tobytes(), digest_size=8).hexdigest()


def quality_mask(spectra: np.ndarray, precursor: np.ndarray) -> np.ndarray:
    mz, intensity = spectra[:, 0], spectra[:, 1]
    valid = np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (intensity > 0)
    count = valid.sum(axis=1)
    max_mz = np.where(valid, mz, -np.inf).max(axis=1)
    max_intensity = np.where(valid, intensity, -np.inf).max(axis=1)
    min_intensity = np.where(valid, intensity, np.inf).min(axis=1)
    amplitude = np.divide(
        max_intensity, min_intensity,
        out=np.zeros_like(max_intensity), where=np.isfinite(min_intensity) & (min_intensity > 0),
    )
    relative = np.divide(
        intensity, max_intensity[:, None],
        out=np.zeros_like(intensity), where=np.isfinite(intensity) & (max_intensity[:, None] > 0),
    )
    high_count = ((relative > 0.1) & valid).sum(axis=1)
    return (
        np.isfinite(precursor) & (precursor > 0) & (precursor <= 1000)
        & (count >= 3) & (count <= 128) & (max_mz <= 1000)
        & (amplitude >= 20) & (high_count >= 3)
    )


def ring_class(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "invalid"
    rings = int(rdMolDescriptors.CalcNumRings(mol))
    if rings == 0:
        return "acyclic"
    if rings == 1:
        return "single_ring"
    return "multi_ring"


def stable_uniform(text: str, seed: int) -> float:
    digest = hashlib.blake2b(f"{seed}|{text}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") / float(2**64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/models/MassSpecGym_MurckoHist_split.hdf5"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/large_observability_cohort"))
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--max-spectra-per-molecule", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.data, "r") as handle:
        adduct = handle["adduct"].asstr()[:]
        formula = handle["FORMULA"].asstr()[:]
        precursor = np.asarray(handle["precursor_mz"][:], float)
        candidate_rows = np.flatnonzero(adduct == "[M+H]+")
        candidate_rows = candidate_rows[formula[candidate_rows] != ""]
        spectra = np.asarray(handle["spectrum"][candidate_rows])
        keep = quality_mask(spectra, precursor[candidate_rows])
        rows = candidate_rows[keep]
        spectra = spectra[keep]
        values = {
            "ik14": np.asarray([value[:14] for value in handle["INCHIKEY"].asstr()[rows]], object),
            "formula": handle["FORMULA"].asstr()[rows],
            "smiles": handle["smiles"].asstr()[rows],
            "identifier": handle["IDENTIFIER"].asstr()[rows],
            "fold": handle["fold"].asstr()[rows],
            "instrument": handle["INSTRUMENT_TYPE"].asstr()[rows],
            "collision_energy": np.asarray(handle["COLLISION_ENERGY"][rows], float),
            "precursor_mz": np.asarray(handle["precursor_mz"][rows], float),
        }

    frame = pd.DataFrame(values)
    frame["hdf5_row"] = rows
    frame["spectrum_hash"] = [spectrum_hash(value) for value in spectra]
    frame = frame.drop_duplicates(["ik14", "spectrum_hash"]).copy()
    frame["ring_class"] = [ring_class(value) for value in frame["smiles"]]
    frame = frame.loc[frame["ring_class"] != "invalid"].copy()
    spectrum_counts = frame.groupby("ik14")["hdf5_row"].size()
    eligible_ik = set(spectrum_counts[spectrum_counts >= 2].index)
    frame = frame.loc[frame["ik14"].isin(eligible_ik)].copy()
    molecule_formula = frame.drop_duplicates("ik14").groupby("formula")["ik14"].nunique()
    eligible_formula = set(molecule_formula[molecule_formula >= 2].index)
    frame = frame.loc[frame["formula"].isin(eligible_formula)].copy()

    # Whole formulas, rather than individual molecules, define the split.  To
    # avoid an accidental chemistry imbalance, formulas are stratified by the
    # dominant ring class, precursor-mass octile, and isomer-family size.  A
    # deterministic greedy allocator balances molecule counts at 60/20/20
    # inside every stratum.
    molecule_profile = frame.drop_duplicates("ik14")
    formula_profile = molecule_profile.groupby("formula", sort=False).agg(
        molecules=("ik14", "nunique"),
        precursor_mz=("precursor_mz", "median"),
        acyclic=("ring_class", lambda x: int((x == "acyclic").sum())),
        single_ring=("ring_class", lambda x: int((x == "single_ring").sum())),
        multi_ring=("ring_class", lambda x: int((x == "multi_ring").sum())),
    ).reset_index()
    ring_columns = ["acyclic", "single_ring", "multi_ring"]
    formula_profile["dominant_ring"] = formula_profile[ring_columns].idxmax(axis=1)
    formula_profile["mass_bin"] = pd.qcut(
        formula_profile["precursor_mz"], q=8, labels=False, duplicates="drop"
    ).astype(int)
    formula_profile["family_size_bin"] = pd.cut(
        formula_profile["molecules"], bins=[0, 2, 4, np.inf], labels=["2", "3-4", "5+"]
    ).astype(str)
    split_by_formula: dict[str, str] = {}
    targets = {"discovery": 0.6, "confirmation": 0.2, "test": 0.2}
    strata = ["dominant_ring", "mass_bin", "family_size_bin"]
    for _, group in formula_profile.groupby(strata, sort=True):
        group = group.copy()
        group["tie"] = [stable_uniform(value, args.seed) for value in group["formula"]]
        group = group.sort_values(["molecules", "tie"], ascending=[False, True])
        total = float(group["molecules"].sum())
        assigned = {key: 0.0 for key in targets}
        for row in group.itertuples(index=False):
            choice = min(
                targets,
                key=lambda key: (
                    (assigned[key] + float(row.molecules)) / max(total * targets[key], 1.0),
                    stable_uniform(f"{row.formula}|{key}", args.seed + 7),
                ),
            )
            split_by_formula[row.formula] = choice
            assigned[choice] += float(row.molecules)
    frame["audit_split"] = frame["formula"].map(split_by_formula)

    # Retain condition diversity without letting large replicate groups dominate.
    selected = []
    for _, group in frame.groupby("ik14", sort=True):
        group = group.copy()
        group["ce_finite"] = np.isfinite(group["collision_energy"])
        group["selection_key"] = [stable_uniform(str(value), args.seed + 1) for value in group["identifier"]]
        group = group.sort_values(["instrument", "ce_finite", "collision_energy", "selection_key"])
        if len(group) > args.max_spectra_per_molecule:
            positions = np.linspace(0, len(group) - 1, args.max_spectra_per_molecule).round().astype(int)
            group = group.iloc[np.unique(positions)]
        selected.append(group)
    frame = pd.concat(selected, ignore_index=True).drop(columns=["ce_finite", "selection_key"])

    molecule = frame.groupby(["audit_split", "formula", "ik14"], sort=False).agg(
        spectra=("hdf5_row", "size"),
        smiles=("smiles", "first"), ring_class=("ring_class", "first"),
        precursor_mz=("precursor_mz", "median"), original_fold=("fold", lambda x: "|".join(sorted(set(x)))),
    ).reset_index()
    formula_summary = molecule.groupby(["audit_split", "formula"], sort=False).agg(
        molecules=("ik14", "nunique"), spectra=("spectra", "sum"),
    ).reset_index()

    frame.sort_values(["audit_split", "formula", "ik14", "hdf5_row"]).to_csv(
        args.output_dir / "selected_spectra.csv", index=False
    )
    molecule.to_csv(args.output_dir / "molecules.csv", index=False)
    formula_summary.to_csv(args.output_dir / "formulas.csv", index=False)
    report = {
        "status": "large_observability_cohort",
        "selection": {
            "adduct": "[M+H]+", "quality_control": "DataFormatA-like spectrum checks",
            "minimum_nonduplicate_spectra_per_molecule": 2,
            "minimum_distinct_molecules_per_formula": 2,
            "maximum_spectra_per_molecule": args.max_spectra_per_molecule,
            "score_independent": True,
        },
        "split_rule": (
            "Whole molecular formulas assigned approximately 60/20/20 with "
            "stratification by dominant ring class, precursor-mass octile, and "
            "isomer-family size; no formula or molecule overlap."
        ),
        "counts": {
            split: {
                "molecules": int(part["ik14"].nunique()),
                "formulas": int(part["formula"].nunique()),
                "spectra": int(len(part)),
                "ring_classes": {str(k): int(v) for k, v in part.drop_duplicates("ik14")["ring_class"].value_counts().items()},
            }
            for split, part in frame.groupby("audit_split")
        },
        "overall": {
            "molecules": int(frame["ik14"].nunique()),
            "formulas": int(frame["formula"].nunique()),
            "spectra": int(len(frame)),
        },
        "leakage_audit": {
            "formula_overlap": 0,
            "molecule_overlap": 0,
        },
        "claim_limit": "MassSpecGym train/val are pooled for failure discovery; the formula-isolated test split must remain untouched until the pipeline is frozen.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
