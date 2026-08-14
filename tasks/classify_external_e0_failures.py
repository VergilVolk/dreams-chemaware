"""Candidate chemical taxonomy for unique external E0 failure pairs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from myopic_mces import MCES
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Fragments, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


def info(smiles: str, fpgen) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    canonical = Chem.MolToSmiles(mol, isomericSmiles=False)
    stereo = Chem.MolToSmiles(mol, isomericSmiles=True)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    groups = {}
    for name in dir(Fragments):
        if name.startswith("fr_") and callable(getattr(Fragments, name)):
            value = int(getattr(Fragments, name)(mol))
            if value: groups[name] = value
    carbons = sum(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())
    phosphorus = sum(atom.GetAtomicNum() == 15 for atom in mol.GetAtoms())
    nitrogens = sum(atom.GetAtomicNum() == 7 for atom in mol.GetAtoms())
    rings = rdMolDescriptors.CalcNumRings(mol)
    if rings == 0 and carbons >= 20 and phosphorus >= 1:
        domain = "acyclic_phospholipid_like"
    elif rings == 0 and carbons >= 20:
        domain = "acyclic_long_chain_lipid_like"
    elif rings == 0 and nitrogens >= 2 and groups.get("fr_amide", 0) >= 2:
        domain = "acyclic_polyamide_or_peptide_like"
    elif rings == 0:
        domain = "other_acyclic"
    elif rings == 1:
        domain = "single_ring"
    else:
        domain = "multi_ring"
    return {
        "mol": mol, "canonical": canonical, "stereo": stereo,
        "scaffold": scaffold, "groups": groups, "rings": rings,
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "heavy_atoms": mol.GetNumHeavyAtoms(), "mw": Descriptors.ExactMolWt(mol),
        "domain": domain, "fingerprint": fpgen.GetFingerprint(mol),
    }


def classify(a: dict, b: dict, mces: float) -> tuple[str, list[str]]:
    group_changes = sorted(
        name for name in set(a["groups"]) | set(b["groups"])
        if a["groups"].get(name, 0) != b["groups"].get(name, 0)
    )
    if a["canonical"] == b["canonical"]:
        if a["stereo"] != b["stereo"]:
            return "stereoisomer_candidate", group_changes
        return "connectivity_equivalent_duplicate_candidate", group_changes
    same_scaffold = bool(a["scaffold"] and a["scaffold"] == b["scaffold"])
    if not a["scaffold"] and not b["scaffold"]:
        same_scaffold = True
    if mces <= 2 and same_scaffold and not group_changes:
        return "positional_or_local_connectivity_candidate", group_changes
    if mces <= 2 and same_scaffold:
        return "same_scaffold_functional_group_candidate", group_changes
    if mces <= 2 and (a["rings"] != b["rings"] or a["aromatic_rings"] != b["aromatic_rings"]):
        return "local_ring_topology_candidate", group_changes
    if mces <= 2:
        return "close_connectivity_different_scaffold_candidate", group_changes
    if mces <= 5:
        return "moderate_constitutional_isomer_candidate", group_changes
    return "distant_constitutional_isomer_candidate", group_changes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", type=Path, default=Path("data/validation/external_ring_balanced_pilot"))
    parser.add_argument("--failures", type=Path, default=Path("data/validation/external_ring_balanced_e0/failures.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/external_e0_failure_taxonomy"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    units = json.loads((args.pilot_dir / "confirmation_manifest.json").read_text(encoding="utf-8"))["units"]
    by_pair = {int(unit["pair_id"]): unit for unit in units}
    failures = pd.read_csv(args.failures)
    failures = failures.loc[
        (failures["split"] == "confirmation")
        & (failures["candidate_protocol"] == "same_formula_negative_pair_ids")
    ].copy()
    grouped = failures.groupby(["ik14", "best_negative_ik14"], sort=False)
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    cache = {}
    rows = []
    for (query_ik, negative_ik), group in grouped:
        query = by_pair[int(group.iloc[0]["pair_id"])]
        negative = by_pair[int(group.iloc[0]["best_negative_pair_id"])]
        for item in (query, negative):
            cache.setdefault(item["smiles"], info(item["smiles"], fpgen))
        a, b = cache[query["smiles"]], cache[negative["smiles"]]
        _, distance, _, mode = MCES(query["smiles"], negative["smiles"], threshold=10, catch_errors=True)
        label, group_changes = classify(a, b, float(distance))
        rows.append({
            "query_ik14": query_ik, "negative_ik14": negative_ik,
            "query_ring_class": query["ring_class"], "query_domain": a["domain"],
            "negative_domain": b["domain"], "formula": query["formula"],
            "query_smiles": query["smiles"], "negative_smiles": negative["smiles"],
            "query_views_failed": len(group), "margin_mean": float(group["margin"].mean()),
            "mces": float(distance), "mces_mode": int(mode),
            "morgan_tanimoto": float(DataStructs.TanimotoSimilarity(a["fingerprint"], b["fingerprint"])),
            "same_scaffold": bool(a["scaffold"] == b["scaffold"]),
            "edit_class_candidate": label,
            "functional_group_changes": "|".join(group_changes),
            "manual_review_required": True,
        })
    frame = pd.DataFrame(rows).sort_values(["query_ring_class", "margin_mean"])
    frame.to_csv(args.output_dir / "failure_pairs.csv", index=False)
    edit = frame.groupby(["query_ring_class", "edit_class_candidate"]).agg(
        unique_pairs=("query_ik14", "size"), query_molecules=("query_ik14", "nunique"),
        median_mces=("mces", "median"), median_tanimoto=("morgan_tanimoto", "median"),
        mean_margin=("margin_mean", "mean"),
    ).reset_index()
    edit.to_csv(args.output_dir / "edit_class_summary.csv", index=False)
    domain = frame.groupby(["query_domain", "edit_class_candidate"]).size().reset_index(name="unique_pairs")
    domain.to_csv(args.output_dir / "domain_summary.csv", index=False)
    report = {
        "status": "external_e0_failure_taxonomy",
        "unique_failure_pairs": len(frame),
        "unique_query_molecules": int(frame["query_ik14"].nunique()),
        "query_domain_counts": dict(Counter(frame["query_domain"])),
        "edit_class_counts": dict(Counter(frame["edit_class_candidate"])),
        "claim_limit": "Automated candidate taxonomy; stereochemistry and chemical edit labels require expert review.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(edit.to_string(index=False))


if __name__ == "__main__":
    main()
