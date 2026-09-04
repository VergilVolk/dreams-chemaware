#!/usr/bin/env python
"""Validate MONA-negative rows before treating them as [M-H]- references."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors


PROTON_MASS = 1.007276466621


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def structure_record(smiles: str) -> dict:
    molecule = Chem.MolFromSmiles(smiles) if smiles else None
    if molecule is None:
        return {
            "structure_parseable": False, "calculated_ik14": "",
            "calculated_formula": "", "expected_m_h_mz": np.nan,
        }
    try:
        ik14 = Chem.MolToInchiKey(molecule)[:14]
    except Exception:  # pragma: no cover - RDKit build dependent
        ik14 = ""
    return {
        "structure_parseable": True,
        "calculated_ik14": ik14,
        "calculated_formula": rdMolDescriptors.CalcMolFormula(molecule),
        "expected_m_h_mz": float(Descriptors.ExactMolWt(molecule) - PROTON_MASS),
    }


def audit_manifest(manifest: pd.DataFrame, ppm: float) -> pd.DataFrame:
    required = {"smiles", "inchikey", "precursor_mz"}
    if missing := required - set(manifest.columns):
        raise RuntimeError(f"manifest missing columns: {sorted(missing)}")
    RDLogger.DisableLog("rdApp.*")
    structures = {
        smiles: structure_record(smiles)
        for smiles in manifest["smiles"].fillna("").astype(str).unique()
    }
    result = manifest.copy()
    result.insert(0, "library_row", np.arange(len(result), dtype=np.int64))
    structure_frame = pd.DataFrame([
        structures[smiles]
        for smiles in result["smiles"].fillna("").astype(str)
    ])
    result = pd.concat([result.reset_index(drop=True), structure_frame], axis=1)
    result["manifest_ik14"] = result["inchikey"].fillna("").astype(str).str[:14].str.upper()
    observed = pd.to_numeric(result["precursor_mz"], errors="coerce").to_numpy(float)
    expected = pd.to_numeric(result["expected_m_h_mz"], errors="coerce").to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result["m_h_ppm_error"] = np.abs(observed - expected) / expected * 1e6
    result["structure_identity_consistent"] = (
        result["structure_parseable"].astype(bool)
        & result["calculated_ik14"].astype(str).eq(result["manifest_ik14"].astype(str))
    )
    result["m_h_mass_consistent"] = (
        np.isfinite(result["m_h_ppm_error"].to_numpy(float))
        & (result["m_h_ppm_error"].to_numpy(float) <= ppm)
    )
    result["approved_m_h_reference"] = (
        result["structure_identity_consistent"] & result["m_h_mass_consistent"]
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("data/models/mona_neg_dreams_emb/manifest.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/mona_negative_library_chemical_integrity_v1"),
    )
    parser.add_argument("--ppm", type=float, default=10.0)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {args.output_dir}")
    audited = audit_manifest(pd.read_csv(args.manifest), args.ppm)
    approved = audited.loc[audited["approved_m_h_reference"], "library_row"].to_numpy(np.int64)
    args.output_dir.mkdir(parents=True)
    table_path = args.output_dir / "library_row_integrity.csv.gz"
    rows_path = args.output_dir / "approved_m_h_library_rows.npy"
    audited.to_csv(table_path, index=False, compression="gzip")
    np.save(rows_path, approved)
    report = {
        "status": "mona_negative_library_chemical_integrity_complete",
        "formal": True,
        "declared_adduct_scope": "[M-H]-",
        "mass_tolerance_ppm": float(args.ppm),
        "library_rows": int(len(audited)),
        "parseable_structure_rows": int(audited["structure_parseable"].sum()),
        "structure_identity_consistent_rows": int(audited["structure_identity_consistent"].sum()),
        "mass_consistent_m_h_rows": int(audited["m_h_mass_consistent"].sum()),
        "approved_m_h_rows": int(len(approved)),
        "rejected_rows": int(len(audited) - len(approved)),
        "approved_fraction": float(len(approved) / len(audited)),
        "contracts": {
            "uses_spectrum_outcome": False,
            "uses_query_truth": False,
            "approval_requires_smiles_ik14_match": True,
            "approval_requires_theoretical_m_h_mass_match": True,
        },
        "provenance": {
            "manifest_sha256": sha256(args.manifest),
            "integrity_table_sha256": sha256(table_path),
            "approved_rows_sha256": sha256(rows_path),
        },
        "claim_limit": "Approval establishes metadata consistency with a singly deprotonated structure, not spectrum identity or library quality.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
