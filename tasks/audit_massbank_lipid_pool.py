"""Structure- and metadata-level audit of MassBank for lipid hard negatives."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


def norm(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def load_sets(msg_path: Path, external_path: Path) -> tuple[set[str], set[str]]:
    with msg_path.open(encoding="utf-8") as stream:
        msg = {row["inchikey14"][:14] for row in csv.DictReader(stream)}
    external = json.loads(external_path.read_text(encoding="utf-8"))
    held_out = {unit["ik14"] for unit in external["units"]}
    return msg, held_out


def is_lipid_like(smiles: str) -> tuple[bool, dict]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, {}
    counts = Counter(atom.GetSymbol() for atom in mol.GetAtoms())
    rings = rdMolDescriptors.CalcNumRings(mol)
    eligible = rings == 0 and counts["C"] >= 20 and counts["P"] >= 1
    return eligible, {
        "carbon_count": counts["C"], "nitrogen_count": counts["N"],
        "phosphorus_count": counts["P"], "oxygen_count": counts["O"],
        "ring_count": rings,
    }


def summarize(records: list[dict], label: str) -> tuple[dict, list[dict]]:
    by_ik: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        by_ik[row["ik14"]].append(row)
    molecule_rows = []
    for ik14, spectra in by_ik.items():
        first = spectra[0]
        conditions = {
            (row["instrument_type"], row["collision_energy"], row["adduct"])
            for row in spectra
        }
        molecule_rows.append({
            **{key: first[key] for key in (
                "ik14", "full_inchikey", "smiles", "formula", "carbon_count",
                "nitrogen_count", "phosphorus_count", "oxygen_count", "ring_count",
            )},
            "records": len(spectra), "unique_splashes": len({row["splash"] for row in spectra if row["splash"]}),
            "unique_conditions": len(conditions),
        })
    positive_ready = [row for row in molecule_rows if row["unique_splashes"] >= 2]
    by_formula: dict[str, list[dict]] = defaultdict(list)
    for row in positive_ready:
        by_formula[row["formula"]].append(row)
    dense = {formula: values for formula, values in by_formula.items() if len(values) >= 2}
    eligible = [row for values in dense.values() for row in values]
    directed = sum(len(values) * (len(values) - 1) for values in dense.values())
    largest = max((len(values) for values in dense.values()), default=0)
    report = {
        "cohort": label,
        "records": len(records),
        "unique_lipid_like_molecules": len(molecule_rows),
        "molecules_with_two_splashes": len(positive_ready),
        "eligible_same_formula_molecules": len(eligible),
        "independent_formula_groups": len(dense),
        "largest_formula_group": largest,
        "directed_identity_negative_choices": directed,
        "formula_group_size_distribution": dict(Counter(len(values) for values in dense.values())),
    }
    eligible_iks = {row["ik14"] for row in eligible}
    eligible_records = [row for row in records if row["ik14"] in eligible_iks]
    return report, eligible_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("data/massbank/massbank_202406_msms.csv"))
    parser.add_argument("--massspecgym-metadata", type=Path, default=Path("data/massspecgym/metadata.csv"))
    parser.add_argument("--external-manifest", type=Path, default=Path("data/validation/external_ring_stratified_cohort/manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/massbank_lipid_pool_gate"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    msg, held_out = load_sets(args.massspecgym_metadata, args.external_manifest)
    frame = pd.read_csv(args.csv, low_memory=False)
    records = []
    chemistry_cache: dict[str, tuple[bool, dict]] = {}
    audit = Counter()
    for row in frame.to_dict(orient="records"):
        audit["all_records"] += 1
        ik = norm(row.get("InChIKey")); smiles = norm(row.get("SMILES"))
        if not ik:
            audit["missing_inchikey"] += 1; continue
        if not smiles:
            audit["missing_smiles"] += 1; continue
        ik14 = ik[:14]
        if ik14 in held_out:
            audit["external_holdout_excluded"] += 1; continue
        ion_mode = norm(row.get("MS_ION_MODE")).upper()
        adduct = norm(row.get("PRECURSOR_TYPE_ADDUCT"))
        ms_type = norm(row.get("MS_TYPE")).upper()
        if ion_mode != "POSITIVE" or ms_type not in {"MS2", "MS/MS", "MSMS"}:
            audit["wrong_mode_or_level"] += 1; continue
        if adduct not in {"[M+H]+", "[M+Na]+"}:
            audit["unsupported_adduct"] += 1; continue
        if smiles not in chemistry_cache:
            chemistry_cache[smiles] = is_lipid_like(smiles)
        eligible, chemistry = chemistry_cache[smiles]
        if not eligible:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            audit["invalid_smiles"] += 1; continue
        formula = rdMolDescriptors.CalcMolFormula(mol)
        records.append({
            "accession": norm(row.get("ACCESSION")), "ik14": ik14,
            "full_inchikey": ik, "smiles": smiles, "formula": formula,
            "adduct": adduct, "precursor_mz": norm(row.get("PRECURSOR_MZ")),
            "instrument": norm(row.get("INSTRUMENT")),
            "instrument_type": norm(row.get("INSTRUMENT_TYPE")),
            "collision_energy": norm(row.get("MS_COLLISION_ENERGY")),
            "splash": norm(row.get("SPLASH")),
            "in_massspecgym": ik14 in msg, **chemistry,
        })

    all_report, all_records = summarize(records, "external_holdout_excluded")
    novel_report, novel_records = summarize([row for row in records if not row["in_massspecgym"]], "also_massspecgym_excluded")
    for name, output in (("all_eligible_records.csv", all_records), ("novel_eligible_records.csv", novel_records)):
        with (args.output_dir / name).open("w", newline="", encoding="utf-8") as stream:
            fields = list(output[0]) if output else ["accession"]
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(output)
    report = {
        "status": "massbank_lipid_pool_metadata_gate",
        "domain_definition": "acyclic; C>=20; P>=1; positive MS2; [M+H]+ or [M+Na]+; at least two distinct SPLASH records per identity",
        "audit": dict(audit), "external_holdout_molecules": len(held_out),
        "massspecgym_molecules": len(msg), "cohorts": [all_report, novel_report],
        "next_gate": "Parse full MSP peak lists for the eligible accessions; verify spectrum quality and nonduplicate positive pairs.",
    }
    (args.output_dir / "metadata_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
