"""Classify high-consensus DreaMS residual pairs into auditable edit families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from myopic_mces import MCES
from rdkit import Chem, DataStructs
from rdkit.Chem import Fragments, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


def molecule_info(smiles: str, fpgen) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    groups = {}
    for name in dir(Fragments):
        fn = getattr(Fragments, name)
        if name.startswith("fr_") and callable(fn):
            value = int(fn(mol))
            if value:
                groups[name] = value
    return {
        "mol": mol,
        "canonical": Chem.MolToSmiles(mol, isomericSmiles=False),
        "isomeric": Chem.MolToSmiles(mol, isomericSmiles=True),
        "scaffold": MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False),
        "rings": int(rdMolDescriptors.CalcNumRings(mol)),
        "aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "groups": groups,
        "fp": fpgen.GetFingerprint(mol),
    }


def label(a: dict, b: dict, distance: float, group_changes: list[str]) -> str:
    if a["canonical"] == b["canonical"]:
        return "stereochemistry_or_label_equivalence" if a["isomeric"] != b["isomeric"] else "connectivity_equivalent"
    same_scaffold = bool(a["scaffold"] and a["scaffold"] == b["scaffold"])
    if not a["scaffold"] and not b["scaffold"]:
        same_scaffold = True
    if distance <= 2 and same_scaffold and not group_changes:
        return "positional_or_local_connectivity"
    if distance <= 2 and same_scaffold:
        return "same_scaffold_functional_group"
    if distance <= 2 and (a["rings"] != b["rings"] or a["aromatic_rings"] != b["aromatic_rings"]):
        return "local_ring_topology"
    if distance <= 2:
        return "close_connectivity_different_scaffold"
    if distance <= 5:
        return "moderate_constitutional_isomer"
    return "distant_constitutional_isomer"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, default=Path("data/validation/large_observability_residual_audit"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/large_residual_edit_taxonomy"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    cache = {}
    rows = []
    for split in ("discovery", "confirmation"):
        frame = pd.read_csv(args.audit_dir / f"{split}_query_audit.csv")
        frame = frame.loc[frame["robust_model_residual_candidate"]].copy()
        # Collapse multiple query spectra, but preserve direction because the
        # query and highest-scoring wrong candidate play different roles.
        for (query_ik, negative_ik), group in frame.groupby(["ik14", "dreams_best_negative_ik14"], sort=False):
            query_smiles = group["smiles"].iloc[0]
            negative_smiles = group["dreams_best_negative_smiles"].iloc[0]
            for smiles in (query_smiles, negative_smiles):
                if smiles not in cache:
                    cache[smiles] = molecule_info(smiles, fpgen)
            a, b = cache[query_smiles], cache[negative_smiles]
            _, distance, _, mode = MCES(query_smiles, negative_smiles, threshold=15, catch_errors=True)
            changes = sorted(
                name for name in set(a["groups"]) | set(b["groups"])
                if a["groups"].get(name, 0) != b["groups"].get(name, 0)
            )
            rows.append({
                "split": split, "query_ik14": query_ik, "negative_ik14": negative_ik,
                "formula": group["formula"].iloc[0], "ring_class": group["ring_class"].iloc[0],
                "query_smiles": query_smiles, "negative_smiles": negative_smiles,
                "query_spectra": len(group), "dreams_margin_mean": float(group["dreams_margin"].mean()),
                "raw_margin_mean": float(group["raw_margin"].mean()),
                "mces": float(distance), "mces_mode": int(mode),
                "morgan_tanimoto": float(DataStructs.TanimotoSimilarity(a["fp"], b["fp"])),
                "same_scaffold": bool(a["scaffold"] == b["scaffold"]),
                "ring_delta": b["rings"] - a["rings"],
                "aromatic_ring_delta": b["aromatic_rings"] - a["aromatic_rings"],
                "edit_class_candidate": label(a, b, float(distance), changes),
                "functional_group_changes": "|".join(changes),
            })
    result = pd.DataFrame(rows)
    result.to_csv(args.output_dir / "residual_pairs.csv", index=False)
    summary = result.groupby(["split", "edit_class_candidate"], sort=False).agg(
        directed_pairs=("query_ik14", "size"), query_molecules=("query_ik14", "nunique"),
        formulas=("formula", "nunique"), median_mces=("mces", "median"),
        median_tanimoto=("morgan_tanimoto", "median"), mean_dreams_margin=("dreams_margin_mean", "mean"),
    ).reset_index()
    summary.to_csv(args.output_dir / "edit_summary.csv", index=False)
    replicated = []
    for edit, group in result.groupby("edit_class_candidate"):
        counts = group.groupby("split").agg(molecules=("query_ik14", "nunique"), formulas=("formula", "nunique"))
        replicated.append({
            "edit_class_candidate": edit,
            "discovery_molecules": int(counts.loc["discovery", "molecules"]) if "discovery" in counts.index else 0,
            "confirmation_molecules": int(counts.loc["confirmation", "molecules"]) if "confirmation" in counts.index else 0,
            "discovery_formulas": int(counts.loc["discovery", "formulas"]) if "discovery" in counts.index else 0,
            "confirmation_formulas": int(counts.loc["confirmation", "formulas"]) if "confirmation" in counts.index else 0,
        })
    replicated = pd.DataFrame(replicated)
    replicated["passes_peak_localization_scale_gate"] = (
        (replicated["discovery_molecules"] >= 30) & (replicated["confirmation_molecules"] >= 15)
        & (replicated["discovery_formulas"] >= 20) & (replicated["confirmation_formulas"] >= 10)
    )
    replicated.to_csv(args.output_dir / "replicated_edit_families.csv", index=False)
    report = {
        "status": "large_residual_edit_taxonomy",
        "pairs": len(result), "query_molecules": int(result["query_ik14"].nunique()),
        "formulas": int(result["formula"].nunique()),
        "replicated_edit_families_passing_scale_gate": replicated.loc[
            replicated["passes_peak_localization_scale_gate"], "edit_class_candidate"
        ].tolist(),
        "gate": "Discovery >=30 molecules/20 formulas and confirmation >=15 molecules/10 formulas.",
        "claim_limit": "Automated edit families are screening strata; specific reaction or fragmentation mechanisms require peak-level and expert validation.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(replicated.to_string(index=False))


if __name__ == "__main__":
    main()
