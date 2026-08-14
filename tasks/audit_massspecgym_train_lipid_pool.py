"""Leakage-safe phospholipid hard-negative gate inside MassSpecGym train fold."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors


RDLogger.DisableLog("rdApp.error")


def decode(values) -> np.ndarray:
    return np.asarray([value.decode("utf-8", errors="ignore") if isinstance(value, bytes) else str(value) for value in values])


def spectrum_hash(spectrum: np.ndarray) -> str:
    mz, intensity = np.asarray(spectrum[0]), np.asarray(spectrum[1])
    keep = (mz > 0) & (intensity > 0)
    packed = np.stack((np.rint(mz[keep] / 0.01), np.rint(intensity[keep] / 0.01)), axis=1).astype(np.int32)
    return hashlib.blake2b(packed.tobytes(), digest_size=8).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=Path, default=Path("data/models/MassSpecGym_MurckoHist_split.hdf5"))
    parser.add_argument("--external-manifest", type=Path, default=Path("data/validation/external_ring_stratified_cohort/manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/massspecgym_train_lipid_pool_gate"))
    parser.add_argument("--adduct", default="[M+H]+")
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    held_out = {unit["ik14"] for unit in json.loads(args.external_manifest.read_text(encoding="utf-8"))["units"]}

    with h5py.File(args.hdf5, "r") as handle:
        folds, adducts = decode(handle["fold"][:]), decode(handle["adduct"][:])
        iks, smiles, formulas = decode(handle["INCHIKEY"][:]), decode(handle["smiles"][:]), decode(handle["FORMULA"][:])
        selected = np.flatnonzero((folds == "train") & (adducts == args.adduct))
        chemistry_cache = {}; candidate_rows = []; audit = Counter()
        for row in selected:
            ik14 = iks[row][:14]
            if ik14 in held_out:
                audit["external_holdout_overlap_rejected"] += 1; continue
            smi = smiles[row]
            if smi not in chemistry_cache:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    chemistry_cache[smi] = None
                else:
                    counts = Counter(atom.GetSymbol() for atom in mol.GetAtoms())
                    chemistry_cache[smi] = {
                        "eligible": rdMolDescriptors.CalcNumRings(mol) == 0 and counts["C"] >= 20 and counts["P"] >= 1,
                        "canonical_smiles": Chem.MolToSmiles(mol, isomericSmiles=True),
                        "formula_from_smiles": rdMolDescriptors.CalcMolFormula(mol),
                        "carbon_count": counts["C"], "nitrogen_count": counts["N"],
                    }
            chem = chemistry_cache[smi]
            if not chem or not chem["eligible"]:
                continue
            candidate_rows.append((int(row), ik14, chem, formulas[row]))
        candidate_indices = [value[0] for value in candidate_rows]
        spectra = handle["spectrum"][candidate_indices] if candidate_indices else np.zeros((0, 2, 128))

    by_ik = defaultdict(list)
    metadata = {}
    for (row, ik14, chem, formula_hdf5), spectrum in zip(candidate_rows, spectra):
        token = spectrum_hash(spectrum)
        by_ik[ik14].append((row, token))
        metadata[ik14] = {
            "ik14": ik14, "smiles": chem["canonical_smiles"],
            "formula": formula_hdf5 or chem["formula_from_smiles"],
            "carbon_count": chem["carbon_count"], "nitrogen_count": chem["nitrogen_count"],
        }
    positive_ready = {
        ik14: values for ik14, values in by_ik.items()
        if len({token for _, token in values}) >= 2
    }
    by_formula = defaultdict(list)
    for ik14 in positive_ready:
        by_formula[metadata[ik14]["formula"]].append(ik14)
    dense = {formula: values for formula, values in by_formula.items() if len(values) >= 2}
    eligible_iks = {ik14 for values in dense.values() for ik14 in values}
    units = []
    for ik14 in sorted(eligible_iks):
        rows = sorted({row for row, _ in positive_ready[ik14]})
        alternatives = sorted(value for value in dense[metadata[ik14]["formula"]] if value != ik14)
        units.append({**metadata[ik14], "hdf5_rows": rows, "nonduplicate_spectra": len({t for _, t in positive_ready[ik14]}),
                      "same_formula_negative_ik14": alternatives})
    (args.output_dir / "manifest.json").write_text(json.dumps({"units": units}, indent=2, ensure_ascii=False), encoding="utf-8")
    directed = sum(len(values) * (len(values) - 1) for values in dense.values())
    largest = max((len(values) for values in dense.values()), default=0)
    gates = {
        "at_least_100_molecules": len(units) >= 100,
        "at_least_30_formula_groups": len(dense) >= 30,
        "largest_formula_at_most_15_percent": largest / len(units) <= 0.15 if units else False,
        "at_least_500_directed_identity_negative_choices": directed >= 500,
        "external_holdout_overlap_zero": audit["external_holdout_overlap_rejected"] == 0,
    }
    report = {
        "status": "massspecgym_train_lipid_pool_gate", "fold": "train", "adduct": args.adduct,
        "domain_definition": "acyclic; C>=20; P>=1; >=2 nonduplicate spectra; same-formula alternate molecule",
        "selected_train_spectra": len(selected), "candidate_spectra": len(candidate_rows),
        "unique_candidate_molecules": len(by_ik), "positive_ready_molecules": len(positive_ready),
        "eligible_same_formula_molecules": len(units), "independent_formula_groups": len(dense),
        "largest_formula_group": largest, "directed_identity_negative_choices": directed,
        "formula_group_size_distribution": dict(Counter(len(values) for values in dense.values())),
        "audit": dict(audit), "gates": gates, "gate_pass": all(gates.values()),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
