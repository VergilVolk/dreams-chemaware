#!/usr/bin/env python
"""Audit the Utrecht endogenous-standard library as an external DreaMS panel.

This stage is deliberately chemistry-only.  It checks that negative spectra
really encode singly deprotonated structures and asks whether the frozen MONA
candidate library creates a non-trivial strict-10-ppm retrieval problem.  It
does not score spectra, fit BioAware, or manufacture biological context.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors


PROTON_MASS = 1.007276466621


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_mgf(path: Path) -> Iterator[dict]:
    record: dict[str, object] | None = None
    peaks: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8-sig", errors="strict") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            if line == "BEGIN IONS":
                if record is not None:
                    raise RuntimeError(f"nested BEGIN IONS at line {line_number}")
                record, peaks = {}, []
                continue
            if line == "END IONS":
                if record is None:
                    raise RuntimeError(f"orphan END IONS at line {line_number}")
                record["peaks"] = peaks
                yield record
                record, peaks = None, []
                continue
            if record is None:
                raise RuntimeError(f"content outside MGF block at line {line_number}")
            if "=" in line:
                key, value = line.split("=", 1)
                if key in record:
                    raise RuntimeError(f"duplicate metadata key {key!r} at line {line_number}")
                record[key] = value
            else:
                fields = line.split()
                if len(fields) != 2:
                    raise RuntimeError(f"invalid peak line {line_number}: {line!r}")
                mz, intensity = map(float, fields)
                if not np.isfinite(mz) or not np.isfinite(intensity) or mz <= 0 or intensity <= 0:
                    raise RuntimeError(f"invalid peak values at line {line_number}")
                peaks.append((mz, intensity))
    if record is not None:
        raise RuntimeError("unterminated MGF block")


def chemical_record(record: dict, ppm: float) -> dict:
    smiles = str(record.get("SMILES", ""))
    declared_ik = str(record.get("INCHIKEY", "")).upper()
    molecule = Chem.MolFromSmiles(smiles) if smiles else None
    parseable = molecule is not None
    calculated_ik = Chem.MolToInchiKey(molecule).upper() if molecule is not None else ""
    formula = rdMolDescriptors.CalcMolFormula(molecule) if molecule is not None else ""
    expected = float(Descriptors.ExactMolWt(molecule) - PROTON_MASS) if molecule is not None else np.nan
    observed = float(str(record.get("PEPMASS", "nan")).split()[0])
    ppm_error = abs(observed - expected) / expected * 1e6 if np.isfinite(expected) else np.nan
    exact_m_h = (
        str(record.get("IONMODE", "")).casefold() == "negative"
        and str(record.get("ADDUCT", "")) == "[M-H]-"
        and str(record.get("CHARGE", "")) == "1-"
    )
    return {
        "spectrum_id": str(record.get("SPECTRUMID", "")),
        "name": str(record.get("NAME", "")),
        "declared_inchikey": declared_ik,
        "truth_ik14": declared_ik[:14],
        "declared_formula": str(record.get("FORMULA", "")),
        "calculated_formula": formula,
        "precursor_mz": observed,
        "n_fragment_peaks": len(record.get("peaks", [])),
        "ion_mode": str(record.get("IONMODE", "")),
        "adduct": str(record.get("ADDUCT", "")),
        "collision_energy": str(record.get("COLLISION_ENERGY", "")),
        "structure_parseable": parseable,
        "structure_identity_consistent": bool(parseable and calculated_ik == declared_ik),
        "formula_consistent": bool(parseable and formula == str(record.get("FORMULA", ""))),
        "m_h_ppm_error": ppm_error,
        "approved_exact_m_h": bool(
            exact_m_h and parseable and calculated_ik == declared_ik
            and formula == str(record.get("FORMULA", ""))
            and np.isfinite(ppm_error) and ppm_error <= ppm
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mgf", type=Path,
        default=Path("data/reference/bioaware_public_cohort_probe_20260901/UU_QE_Metabolomics_Standards_Lib_noRT_public.mgf"),
    )
    parser.add_argument(
        "--compound-table", type=Path,
        default=Path("data/reference/bioaware_public_cohort_probe_20260901/UU_QE_Metabolomics_Standards_Lib_noRT_unique_compounds.csv"),
    )
    parser.add_argument(
        "--mona-manifest", type=Path,
        default=Path("data/models/mona_neg_dreams_emb/manifest.csv"),
    )
    parser.add_argument(
        "--approved-mona-rows", type=Path,
        default=Path("data/validation/mona_negative_library_chemical_integrity_v1/approved_m_h_library_rows.npy"),
    )
    parser.add_argument(
        "--development-features", type=Path,
        default=Path("data/validation/bioaware_metdna3_external_negative_loso_ranker_v3_identity_purged/candidate_features.csv.gz"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_uu_negative_benchmark_readiness_v1"),
    )
    parser.add_argument("--ppm", type=float, default=10.0)
    args = parser.parse_args()
    required = [
        args.mgf, args.compound_table, args.mona_manifest,
        args.approved_mona_rows, args.development_features,
    ]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {args.output_dir}")

    RDLogger.DisableLog("rdApp.*")
    compounds = pd.read_csv(args.compound_table)
    if compounds["inchikey"].duplicated().any():
        raise RuntimeError("compound table contains duplicate InChIKeys")
    records = list(parse_mgf(args.mgf))
    spectrum_table = pd.DataFrame([chemical_record(record, args.ppm) for record in records])
    if spectrum_table["spectrum_id"].eq("").any() or spectrum_table["spectrum_id"].duplicated().any():
        raise RuntimeError("spectrum identifiers must be non-empty and unique")

    mona = pd.read_csv(args.mona_manifest)
    approved_rows = np.load(args.approved_mona_rows, allow_pickle=False).astype(np.int64)
    if approved_rows.ndim != 1 or len(np.unique(approved_rows)) != len(approved_rows):
        raise RuntimeError("approved MONA rows are not a unique vector")
    if approved_rows.min() < 0 or approved_rows.max() >= len(mona):
        raise RuntimeError("approved MONA row is out of range")
    mona = mona.iloc[approved_rows].copy()
    mona["candidate_id"] = mona["inchikey"].astype(str).str[:14].str.upper()
    mona_mz = pd.to_numeric(mona["precursor_mz"], errors="coerce").to_numpy(float)

    approved_spectra = spectrum_table[spectrum_table["approved_exact_m_h"]].copy()
    candidate_counts: list[int] = []
    truth_present: list[bool] = []
    for row in approved_spectra.itertuples(index=False):
        window = np.flatnonzero(np.abs(mona_mz - row.precursor_mz) / row.precursor_mz * 1e6 <= args.ppm)
        identities = set(mona.iloc[window]["candidate_id"].astype(str))
        candidate_counts.append(len(identities))
        truth_present.append(str(row.truth_ik14) in identities)
    approved_spectra["candidate_molecules"] = candidate_counts
    approved_spectra["truth_in_mona_window"] = truth_present
    approved_spectra["evaluable"] = (
        approved_spectra["truth_in_mona_window"]
        & approved_spectra["candidate_molecules"].ge(2)
        & approved_spectra["n_fragment_peaks"].ge(3)
    )

    development = pd.read_csv(args.development_features, usecols=["truth_candidate_id", "truth_formula"])
    development_ids = set(development["truth_candidate_id"].astype(str))
    development_formulas = set(development["truth_formula"].astype(str))
    approved_spectra["identity_seen_in_bioaware_development"] = approved_spectra["truth_ik14"].isin(development_ids)
    approved_spectra["formula_seen_in_bioaware_development"] = approved_spectra["declared_formula"].isin(development_formulas)

    evaluable = approved_spectra[approved_spectra["evaluable"]]
    args.output_dir.mkdir(parents=True)
    table_path = args.output_dir / "spectrum_readiness.csv.gz"
    approved_spectra.to_csv(table_path, index=False, compression="gzip")
    report = {
        "status": "bioaware_uu_negative_benchmark_readiness_complete",
        "formal": True,
        "source": {
            "title": "Endogenous Metabolite Spectral Library",
            "zenodo_record": "21394918",
            "resolved_record": "21394919",
        },
        "spectra": {
            "all": int(len(spectrum_table)),
            "declared_negative": int(spectrum_table["ion_mode"].str.casefold().eq("negative").sum()),
            "approved_exact_m_h": int(len(approved_spectra)),
            "approved_exact_m_h_identities": int(approved_spectra["truth_ik14"].nunique()),
            "evaluable": int(len(evaluable)),
            "evaluable_identities": int(evaluable["truth_ik14"].nunique()),
            "evaluable_formulas": int(evaluable["declared_formula"].nunique()),
            "evaluable_unseen_development_identities": int(evaluable.loc[~evaluable["identity_seen_in_bioaware_development"], "truth_ik14"].nunique()),
            "evaluable_unseen_development_formulas": int(evaluable.loc[~evaluable["formula_seen_in_bioaware_development"], "declared_formula"].nunique()),
        },
        "candidate_graph": {
            "median_candidate_molecules": float(evaluable["candidate_molecules"].median()) if len(evaluable) else None,
            "maximum_candidate_molecules": int(evaluable["candidate_molecules"].max()) if len(evaluable) else 0,
            "truth_missing_from_mona_window": int((~approved_spectra["truth_in_mona_window"]).sum()),
            "single_candidate_only": int((approved_spectra["candidate_molecules"] < 2).sum()),
        },
        "gates": {
            "approved_exact_m_h_spectra_ge_200": len(approved_spectra) >= 200,
            "approved_exact_m_h_identities_ge_50": approved_spectra["truth_ik14"].nunique() >= 50,
            "evaluable_spectra_ge_100": len(evaluable) >= 100,
            "evaluable_identities_ge_30": evaluable["truth_ik14"].nunique() >= 30,
            "unseen_development_identities_ge_10": evaluable.loc[~evaluable["identity_seen_in_bioaware_development"], "truth_ik14"].nunique() >= 10,
        },
        "provenance": {
            "mgf_sha256": sha256(args.mgf),
            "compound_table_sha256": sha256(args.compound_table),
            "mona_manifest_sha256": sha256(args.mona_manifest),
            "approved_mona_rows_sha256": sha256(args.approved_mona_rows),
            "development_features_sha256": sha256(args.development_features),
            "readiness_table_sha256": sha256(table_path),
            "script_sha256": sha256(Path(__file__).resolve()),
        },
        "contracts": {
            "chemically_verified_exact_m_h_only": True,
            "strict_10ppm_candidate_window": True,
            "candidate_identity_aggregation": "per IK14 maximum after DreaMS encoding",
            "outcomes_used": False,
            "BioAware_used": False,
            "P2b_used": False,
        },
        "claim_limit": "Readiness for an independent standard-spectrum DreaMS benchmark only. The library has no biological sample context and cannot confirm BioAware network benefit.",
    }
    report["pass_to_official_dreams_encoding"] = all(report["gates"].values())
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
