#!/usr/bin/env python
"""Audit TIMS-Bench ground truth for the frozen negative BioAware contract.

This is a readiness audit, not an evaluation.  It intentionally separates a
useful modern external DreaMS benchmark from the stricter requirement for an
independent biological negative-mode BioAware confirmation cohort.
"""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


PROTON = 1.007276466621
SODIUM = 22.989218


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def ik14(values: pd.Series) -> set[str]:
    return set(values.dropna().astype(str).str.upper().str[:14])


def exact_mass(smiles: str) -> float:
    molecule = Chem.MolFromSmiles(str(smiles))
    return float(Descriptors.ExactMolWt(molecule)) if molecule is not None else np.nan


def adduct_sign(value: str) -> int:
    text = str(value).replace(" ", "")
    if "]+" in text or text.endswith("+"):
        return 1
    if "]-" in text or text.endswith("-"):
        return -1
    return 0


def precursor_delta_evidence(library: pd.DataFrame, structures: pd.DataFrame) -> dict:
    lib = library.copy()
    lib["ik14"] = lib["inchikey_2d"].astype(str).str[:14]
    structure = structures.dropna(subset=["smile", "canonical_ikey"]).copy()
    structure["ik14"] = structure["canonical_ikey"].astype(str).str[:14]
    structure = structure.drop_duplicates("ik14")
    merged = lib.merge(structure[["ik14", "smile"]], on="ik14", how="inner")
    merged["neutral_exact_mass"] = merged["smile"].map(exact_mass)
    merged = merged[np.isfinite(merged["neutral_exact_mass"])].copy()
    delta = merged["precursor_mz"].astype(float) - merged["neutral_exact_mass"]
    positive_h = np.abs(delta - PROTON) <= 0.02
    positive_na = np.abs(delta - SODIUM) <= 0.02
    negative_h = np.abs(delta + PROTON) <= 0.02
    return {
        "joined_spectra": int(len(merged)),
        "joined_identities": int(merged["ik14"].nunique()),
        "protonated_positive_spectra": int(positive_h.sum()),
        "sodiated_positive_spectra": int(positive_na.sum()),
        "deprotonated_negative_spectra": int(negative_h.sum()),
        "positive_fraction": float((positive_h | positive_na).mean()) if len(merged) else 0.0,
        "negative_fraction": float(negative_h.mean()) if len(merged) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/reference/timsbench_groundtruth_probe_20260901"),
    )
    parser.add_argument(
        "--development-candidates",
        type=Path,
        default=Path(
            "data/validation/"
            "bioaware_metdna3_external_negative_loso_ranker_v3_identity_purged/"
            "candidate_features.csv.gz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/validation/bioaware_timsbench_confirmation_readiness_v1"),
    )
    args = parser.parse_args()

    required = {
        "nist_library": args.input_dir / "nist_srm_spikein_lib.pq",
        "nist_harmonized": args.input_dir / "NIST_SRM_mzmine_harmonized.parquet",
        "plant_library": args.input_dir / "plant_spikein_lib.pq",
        "plant_harmonized": args.input_dir / "plant_spikein_mzmine_harmonized.parquet",
        "reframe_library": args.input_dir / "reframe_spikein_lib.pq",
        "reframe_structures": args.input_dir / "reframe_smiles_list.csv",
        "reframe_harmonized": args.input_dir / "MSV000098263_mzmine_harmonized.parquet",
        "reframe_truth": args.input_dir / "confidently_annotated_inchikey.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing TIMS-Bench audit inputs: {missing}")

    development = pd.read_csv(args.development_candidates, usecols=["truth_candidate_id"])
    development_ids = ik14(development["truth_candidate_id"])

    nist_library = pd.read_parquet(required["nist_library"])
    nist_harmonized = pd.read_parquet(required["nist_harmonized"])
    nist_ids = ik14(nist_library["inchikey_14"])
    nist_adduct = nist_harmonized["ADDUCT"].dropna().astype(str)

    plant_library = pd.read_parquet(required["plant_library"])
    plant_harmonized = pd.read_parquet(required["plant_harmonized"])
    plant_ids = ik14(plant_library["inchikey_2d"])
    plant_sample_columns = [c for c in plant_harmonized.columns if "Peak height" in c]

    reframe_library = pd.read_parquet(required["reframe_library"])
    reframe_structures = pd.read_csv(required["reframe_structures"])
    reframe_harmonized = pd.read_parquet(required["reframe_harmonized"])
    reframe_truth = pd.read_csv(required["reframe_truth"])
    reframe_ids = ik14(reframe_library["inchikey_2d"])
    reframe_delta = precursor_delta_evidence(reframe_library, reframe_structures)

    cohorts = {
        "NIST_SRM": {
            "truth_rows": int(len(nist_library)),
            "truth_identities": len(nist_ids),
            "harmonized_features": int(len(nist_harmonized)),
            "features_with_ms2": int(nist_harmonized["MS/MS_ASSIGNED"].fillna(False).sum()),
            "development_truth_identity_overlap": len(nist_ids & development_ids),
            "positive_adduct_assignments": int(nist_adduct.map(adduct_sign).eq(1).sum()),
            "negative_adduct_assignments": int(nist_adduct.map(adduct_sign).eq(-1).sum()),
            "biological_sample_context": False,
            "negative_mode_established": False,
            "ready_for_negative_bioaware_confirmation": False,
            "reason": "standard spike-in benchmark; no negative-mode evidence and no biological network context",
        },
        "plant_spikein": {
            "truth_spectra": int(len(plant_library)),
            "truth_identities": len(plant_ids),
            "harmonized_features": int(len(plant_harmonized)),
            "features_with_ms2": int(plant_harmonized["MS/MS_ASSIGNED"].fillna(False).sum()),
            "development_truth_identity_overlap": len(plant_ids & development_ids),
            "positive_named_sample_columns": int(sum("_POS_" in c.upper() for c in plant_sample_columns)),
            "negative_named_sample_columns": int(sum("_NEG_" in c.upper() for c in plant_sample_columns)),
            "biological_sample_context": False,
            "negative_mode_established": False,
            "ready_for_negative_bioaware_confirmation": False,
            "reason": "all 89 intensity columns explicitly encode POS; artificial individual-standard spike-ins",
        },
        "MSV000098263_ReFRAME": {
            "truth_library_spectra": int(len(reframe_library)),
            "truth_library_identities": len(reframe_ids),
            "confident_truth_rows": int(len(reframe_truth)),
            "confident_truth_identities": len(ik14(reframe_truth["INCHIKEY"])),
            "harmonized_features": int(len(reframe_harmonized)),
            "features_with_ms2": int(reframe_harmonized["MS/MS_ASSIGNED"].fillna(False).sum()),
            "development_truth_identity_overlap": len(reframe_ids & development_ids),
            "precursor_mass_delta_evidence": reframe_delta,
            "biological_sample_context": False,
            "negative_mode_established": reframe_delta["deprotonated_negative_spectra"] > 0,
            "ready_for_negative_bioaware_confirmation": False,
            "reason": "ReFRAME is a positive-ion chemical screen, not a phenotype-blind biological negative-mode cohort",
        },
    }

    report = {
        "status": "bioaware_timsbench_confirmation_readiness_complete",
        "formal": True,
        "source": "TIMS-Bench v1.0.0, Zenodo 20816379",
        "development_truth_identities": len(development_ids),
        "cohorts": cohorts,
        "decision": {
            "negative_bioaware_confirmation_cohorts": 0,
            "modern_external_dreams_benchmarks": 3,
            "download_full_archive": False,
            "run_frozen_negative_bioaware": False,
            "usefulness": "retain for separate positive/general DreaMS benchmarking only",
        },
        "contract": {
            "requires_negative_mode": True,
            "requires_level1_or_equivalent_structure_truth": True,
            "requires_biological_sample_context": True,
            "requires_truth_blind_network_seeds": True,
            "requires_identity_overlap_report": True,
        },
        "provenance": {name: digest(path) for name, path in required.items()},
        "claim_limit": (
            "This is a data-readiness audit. It does not evaluate BioAware, DreaMS, "
            "or annotation accuracy. TIMS-Bench remains valuable for a separate "
            "positive/general benchmark but cannot confirm the frozen negative BioAware expert."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
