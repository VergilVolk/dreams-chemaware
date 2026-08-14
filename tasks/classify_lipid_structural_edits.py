"""Candidate structural-edit taxonomy for phospholipid hard-negative pairs.

Labels are deliberately named ``*_candidate``: graph signatures narrow the
chemical edit, but sn-position and double-bond localization need expert review.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors


RDLogger.DisableLog("rdApp.error")


def carbon_chain_signature(mol: Chem.Mol) -> tuple[tuple[int, int, int], ...]:
    """Carbon-only components: (carbon count, C=C count, branch excess)."""
    carbon = {atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6}
    adjacency = {idx: [] for idx in carbon}
    double_edges = set()
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in carbon and b in carbon:
            adjacency[a].append(b); adjacency[b].append(a)
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                double_edges.add(tuple(sorted((a, b))))
    seen = set(); components = []
    for root in carbon:
        if root in seen: continue
        stack = [root]; component = set()
        while stack:
            node = stack.pop()
            if node in component: continue
            component.add(node); seen.add(node); stack.extend(adjacency[node])
        if len(component) < 4: continue
        edges = sum(len(adjacency[node]) for node in component) // 2
        doubles = sum(a in component and b in component for a, b in double_edges)
        branch_excess = sum(max(0, len(adjacency[node]) - 2) for node in component)
        components.append((len(component), int(doubles), int(branch_excess + max(0, edges - len(component) + 1))))
    return tuple(sorted(components, reverse=True))


def bond_orderless_smiles(mol: Chem.Mol) -> str:
    clone = Chem.RWMol(mol)
    for bond in clone.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    for atom in clone.GetAtoms():
        atom.SetIsAromatic(False); atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    value = clone.GetMol()
    Chem.SanitizeMol(value)
    return Chem.MolToSmiles(value, isomericSmiles=False)


def phosphorus_environment(mol: Chem.Mol, radius: int = 3) -> tuple[str, ...]:
    outputs = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 15: continue
        bonds = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom.GetIdx())
        atom_ids = {atom.GetIdx()}
        for bond_id in bonds:
            bond = mol.GetBondWithIdx(bond_id)
            atom_ids.update((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        outputs.append(Chem.MolFragmentToSmiles(
            mol, atomsToUse=sorted(atom_ids), bondsToUse=list(bonds),
            isomericSmiles=False, canonical=True,
        ))
    return tuple(sorted(outputs))


def describe(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: raise ValueError(f"Invalid SMILES: {smiles}")
    stereo = Chem.MolToSmiles(mol, isomericSmiles=True)
    nonstereo = Chem.MolToSmiles(mol, isomericSmiles=False)
    c_double = sum(
        bond.GetBondType() == Chem.BondType.DOUBLE
        and bond.GetBeginAtom().GetAtomicNum() == 6
        and bond.GetEndAtom().GetAtomicNum() == 6
        for bond in mol.GetBonds()
    )
    return {
        "stereo": stereo, "nonstereo": nonstereo,
        "bond_orderless": bond_orderless_smiles(mol),
        "p_environment": phosphorus_environment(mol),
        "chain_signature": carbon_chain_signature(mol),
        "c_double_bonds": int(c_double),
        "stereocenters": len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
    }


def classify(a: dict, b: dict) -> tuple[str, str]:
    if a["nonstereo"] == b["nonstereo"]:
        if a["stereo"] != b["stereo"]:
            return "stereochemical_or_sn_configuration_candidate", "same connectivity; stereochemical SMILES differ"
        return "structure_duplicate_candidate", "canonical structures are identical"
    if a["bond_orderless"] == b["bond_orderless"]:
        return "double_bond_position_or_geometry_candidate", "connectivity matches after ignoring bond order/stereo"
    same_p = a["p_environment"] == b["p_environment"]
    same_chains = a["chain_signature"] == b["chain_signature"]
    if same_p and same_chains:
        return "sn_or_attachment_position_candidate", "same P environment and carbon-chain multiset; connectivity differs"
    if same_p and not same_chains:
        return "fatty_chain_allocation_candidate", "same P environment; carbon-chain length/unsaturation allocation differs"
    if not same_p and same_chains:
        return "headgroup_or_backbone_attachment_candidate", "carbon-chain multiset retained; P environment differs"
    return "compound_lipid_connectivity_candidate", "both P environment and carbon-chain allocation differ"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-pilot", type=Path, default=Path("data/validation/external_ring_balanced_pilot"))
    parser.add_argument("--training-manifest", type=Path, default=Path("data/validation/massspecgym_train_lipid_pool_gate/manifest.json"))
    parser.add_argument("--adapter-results", type=Path, default=Path("data/validation/lipid_projection_adapter_external/query_results.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/lipid_structural_edit_taxonomy"))
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = {}
    def get(smiles):
        if smiles not in cache: cache[smiles] = describe(smiles)
        return cache[smiles]

    rows = []
    # Enumerate unique training pairs once per formula.
    train_units = json.loads(args.training_manifest.read_text(encoding="utf-8"))["units"]
    train_by_ik = {unit["ik14"]: unit for unit in train_units}
    seen = set()
    for unit in train_units:
        for negative_ik in unit["same_formula_negative_ik14"]:
            key = tuple(sorted((unit["ik14"], negative_ik)))
            if key in seen: continue
            seen.add(key); negative = train_by_ik[negative_ik]
            label, reason = classify(get(unit["smiles"]), get(negative["smiles"]))
            rows.append({
                "source": "massspecgym_train", "split": "train_pool", "query_ik14": unit["ik14"],
                "negative_ik14": negative_ik, "formula": unit["formula"], "edit_candidate": label,
                "reason": reason, "query_chain_signature": str(get(unit["smiles"])["chain_signature"]),
                "negative_chain_signature": str(get(negative["smiles"])["chain_signature"]),
                "query_smiles": unit["smiles"], "negative_smiles": negative["smiles"],
            })
    # Enumerate the best negative selected by official embedding for each external query view.
    effects = pd.read_csv(args.adapter_results)
    official = effects.loc[effects["seed"] == -1].copy()
    for split in ("discovery", "confirmation"):
        units = json.loads((args.external_pilot / f"{split}_manifest.json").read_text(encoding="utf-8"))["units"]
        by_id = {int(unit["pair_id"]): unit for unit in units}
        for record in official[(official["split"] == split) & official["phospholipid_like"]].to_dict(orient="records"):
            query = next(unit for unit in units if unit["ik14"] == record["ik14"])
            negative = by_id[int(record["best_negative_pair_id"])]
            label, reason = classify(get(query["smiles"]), get(negative["smiles"]))
            rows.append({
                "source": "external_official_best_negative", "split": split,
                "query_ik14": query["ik14"], "negative_ik14": negative["ik14"],
                "formula": query["formula"], "edit_candidate": label, "reason": reason,
                "query_chain_signature": str(get(query["smiles"])["chain_signature"]),
                "negative_chain_signature": str(get(negative["smiles"])["chain_signature"]),
                "query_smiles": query["smiles"], "negative_smiles": negative["smiles"],
                "query_view": int(record["view"]), "official_margin": float(record["margin"]),
                "official_top1": bool(record["top1"]),
            })
    frame = pd.DataFrame(rows); frame.to_csv(args.output_dir / "pairs.csv", index=False)
    summary = frame.groupby(["source", "split", "edit_candidate"]).agg(
        pairs=("query_ik14", "size"), query_molecules=("query_ik14", "nunique"),
        formulas=("formula", "nunique"),
    ).reset_index()
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    report = {
        "status": "lipid_structural_edit_candidate_taxonomy",
        "training_unique_pairs": int((frame["source"] == "massspecgym_train").sum()),
        "external_query_views": int((frame["source"] == "external_official_best_negative").sum()),
        "counts": summary.to_dict(orient="records"),
        "claim_limit": "Graph-signature candidate labels. sn-position and double-bond localization require expert/manual or lipid-nomenclature validation.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(summary.to_string(index=False)); print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
