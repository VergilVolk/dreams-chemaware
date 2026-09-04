"""Audit whether frozen-gate test corrections rely on trivial replicate leakage."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode(value) -> str:
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-dir", type=Path,
        default=ROOT / "data/validation/chemaware_frozen_spectral_gate_test_20260902",
    )
    parser.add_argument(
        "--input-dir", type=Path,
        default=ROOT / "data/validation/chemaware_frozen_gate_test_inputs_20260902",
    )
    parser.add_argument(
        "--hdf5", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/validation/chemaware_frozen_gate_trigger_case_audit_20260902",
    )
    args = parser.parse_args()
    ledger_path = args.test_dir / "test_gate_ledger.csv.gz"
    manifest_path = args.input_dir / "test_manifest.csv"
    pair_path = args.input_dir / "test_pair_features.csv.gz"
    required = [ledger_path, manifest_path, pair_path, args.hdf5, args.test_dir / "report.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    ledger = pd.read_csv(ledger_path)
    manifest = pd.read_csv(manifest_path)
    pairs = pd.read_csv(pair_path)
    active = ledger.loc[ledger["route_activated"]].copy()
    if len(active) != 9 or not ((~active["dreams_correct"]) & active["consensus_correct"]).all():
        raise RuntimeError("frozen test trigger count/outcomes changed")
    smiles_by_identity = manifest.drop_duplicates("ik14").set_index("ik14")["smiles"].astype(str).to_dict()
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    cases = []
    with h5py.File(args.hdf5, "r") as handle:
        for gate_row in active.itertuples(index=False):
            query_index = int(gate_row.query_index)
            truth = str(gate_row.ik14)
            query_hdf5 = int(manifest.at[query_index, "hdf5_row"])
            edges = pairs.loc[(pairs["left"] == query_index) | (pairs["right"] == query_index)]
            candidates = np.where(
                edges["left"].to_numpy(np.int64) == query_index,
                edges["right"].to_numpy(np.int64),
                edges["left"].to_numpy(np.int64),
            )
            positive = candidates[manifest.iloc[candidates]["ik14"].astype(str).to_numpy() == truth]
            query_instrument = decode(handle["INSTRUMENT_TYPE"][query_hdf5])
            query_ce = float(handle["COLLISION_ENERGY"][query_hdf5])
            refs = []
            for candidate_index in positive:
                candidate_hdf5 = int(manifest.at[int(candidate_index), "hdf5_row"])
                candidate_instrument = decode(handle["INSTRUMENT_TYPE"][candidate_hdf5])
                candidate_ce = float(handle["COLLISION_ENERGY"][candidate_hdf5])
                refs.append({
                    "manifest_index": int(candidate_index),
                    "hdf5_row": candidate_hdf5,
                    "spectrum_hash": str(manifest.at[int(candidate_index), "spectrum_hash"]),
                    "instrument": candidate_instrument,
                    "collision_energy": candidate_ce if np.isfinite(candidate_ce) else None,
                    "fold": decode(handle["fold"][candidate_hdf5]),
                    "simulation_challenge_membership": decode(handle["SIMULATION_CHALLENGE"][candidate_hdf5]),
                    "known_cross_instrument": bool(
                        query_instrument != "nan" and candidate_instrument != "nan"
                        and query_instrument != candidate_instrument
                    ),
                    "distinct_observed_collision_energy": bool(
                        np.isfinite(query_ce) and np.isfinite(candidate_ce)
                        and abs(query_ce - candidate_ce) > 1e-9
                    ),
                })
            truth_smiles = smiles_by_identity[truth]
            wrong_smiles = smiles_by_identity[str(gate_row.dreams_prediction)]
            truth_mol, wrong_mol = Chem.MolFromSmiles(truth_smiles), Chem.MolFromSmiles(wrong_smiles)
            if truth_mol is None or wrong_mol is None:
                raise RuntimeError("trigger case contains invalid SMILES")
            similarity = float(DataStructs.TanimotoSimilarity(
                fpgen.GetFingerprint(truth_mol), fpgen.GetFingerprint(wrong_mol)
            ))
            cases.append({
                "query_index": query_index,
                "query_hdf5_row": query_hdf5,
                "formula": str(gate_row.formula),
                "truth_ik14": truth,
                "dreams_wrong_ik14": str(gate_row.dreams_prediction),
                "truth_smiles": truth_smiles,
                "dreams_wrong_smiles": wrong_smiles,
                "morgan_tanimoto_truth_vs_dreams_wrong": similarity,
                "gate_probability": float(gate_row.gate_probability),
                "dreams_top2_margin": float(gate_row.dreams_top2_margin),
                "consensus_votes": int(gate_row.consensus_votes),
                "candidate_molecules": int(gate_row.candidate_molecules),
                "candidate_reference_spectra": int(gate_row.candidate_reference_spectra),
                "query_spectrum_hash": str(manifest.at[query_index, "spectrum_hash"]),
                "query_instrument": query_instrument,
                "query_collision_energy": query_ce if np.isfinite(query_ce) else None,
                "query_fold": decode(handle["fold"][query_hdf5]),
                "query_simulation_challenge_membership": decode(
                    handle["SIMULATION_CHALLENGE"][query_hdf5]
                ),
                "positive_references": refs,
                "positive_hash_all_distinct_from_query": all(
                    ref["spectrum_hash"] != str(manifest.at[query_index, "spectrum_hash"])
                    for ref in refs
                ),
                "has_known_cross_instrument_positive": any(
                    ref["known_cross_instrument"] for ref in refs
                ),
                "has_distinct_observed_collision_energy_positive": any(
                    ref["distinct_observed_collision_energy"] for ref in refs
                ),
            })
    args.output_dir.mkdir(parents=True)
    cases_path = args.output_dir / "trigger_cases.json"
    cases_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    similarities = np.asarray([row["morgan_tanimoto_truth_vs_dreams_wrong"] for row in cases])
    report = {
        "status": "chemaware_frozen_gate_trigger_cases_audited",
        "test_result_was_modified": False,
        "triggered_cases": len(cases),
        "all_corrected": all(row["truth_ik14"] != row["dreams_wrong_ik14"] for row in cases),
        "all_positive_hashes_distinct_from_query": all(
            row["positive_hash_all_distinct_from_query"] for row in cases
        ),
        "cases_with_known_cross_instrument_positive": sum(
            row["has_known_cross_instrument_positive"] for row in cases
        ),
        "cases_with_distinct_observed_collision_energy_positive": sum(
            row["has_distinct_observed_collision_energy_positive"] for row in cases
        ),
        "query_fold_counts": dict(sorted(Counter(row["query_fold"] for row in cases).items())),
        "query_simulation_challenge_membership_counts": dict(sorted(Counter(
            row["query_simulation_challenge_membership"] for row in cases
        ).items())),
        "membership_is_not_provenance": True,
        "morgan_tanimoto_truth_vs_dreams_wrong": {
            "min": float(similarities.min()),
            "median": float(np.median(similarities)),
            "max": float(similarities.max()),
        },
        "provenance": {
            "test_report_sha256": sha256(args.test_dir / "report.json"),
            "test_ledger_sha256": sha256(ledger_path),
            "test_manifest_sha256": sha256(manifest_path),
            "test_pair_features_sha256": sha256(pair_path),
            "hdf5_sha256": sha256(args.hdf5),
            "cases_sha256": sha256(cases_path),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "Nine internal held-out corrections are a sparse mechanism signal. Distinct rounded "
            "spectrum hashes exclude exact duplicates under the cohort hash, but do not prove "
            "cross-laboratory generalization or structural causality."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
