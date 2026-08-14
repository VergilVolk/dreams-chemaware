"""Metadata gate for a MoNA phospholipid-like hard-negative training pool."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


SMILES_RE = re.compile(r'"SMILES=([^"\r\n]+)"', re.IGNORECASE)


def load_sets(msg_path: Path, external_path: Path) -> tuple[set[str], set[str]]:
    with msg_path.open(encoding="utf-8") as stream:
        msg = {row["inchikey14"][:14] for row in csv.DictReader(stream)}
    external = json.loads(external_path.read_text(encoding="utf-8"))
    held_out = {unit["ik14"] for unit in external["units"]}
    return msg, held_out


def chemistry(smiles: str) -> dict | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    counts = Counter(atom.GetSymbol() for atom in mol.GetAtoms())
    rings = rdMolDescriptors.CalcNumRings(mol)
    if rings != 0 or counts["C"] < 20 or counts["P"] < 1:
        return None
    return {
        "canonical_smiles": Chem.MolToSmiles(mol, isomericSmiles=True),
        "formula_from_smiles": rdMolDescriptors.CalcMolFormula(mol),
        "carbon_count": counts["C"], "nitrogen_count": counts["N"],
        "phosphorus_count": counts["P"], "oxygen_count": counts["O"],
    }


def iter_headers(path: Path):
    current = {}
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for raw in stream:
            line = raw.rstrip("\r\n")
            if not line:
                if current:
                    yield current
                    current = {}
                continue
            if ": " in line and not line[0].isdigit():
                key, value = line.split(": ", 1)
                current.setdefault(key, value)
    if current:
        yield current


def summarize(records: list[dict], label: str) -> tuple[dict, list[dict]]:
    by_ik = defaultdict(list)
    for record in records:
        by_ik[record["ik14"]].append(record)
    positive_ready = {}
    for ik14, values in by_ik.items():
        record_ids = {value["record_id"] for value in values}
        conditions = {
            (value["instrument_type"], value["collision_energy"], value["adduct"])
            for value in values
        }
        if len(record_ids) >= 2 and len(conditions) >= 2:
            positive_ready[ik14] = values
    by_formula = defaultdict(list)
    for ik14, values in positive_ready.items():
        by_formula[values[0]["formula"]].append(ik14)
    dense = {formula: values for formula, values in by_formula.items() if len(values) >= 2}
    eligible_iks = {ik14 for values in dense.values() for ik14 in values}
    eligible_records = [record for record in records if record["ik14"] in eligible_iks]
    directed = sum(len(values) * (len(values) - 1) for values in dense.values())
    largest = max((len(values) for values in dense.values()), default=0)
    return {
        "cohort": label, "records": len(records), "unique_lipid_like_molecules": len(by_ik),
        "molecules_with_two_conditions": len(positive_ready),
        "eligible_same_formula_molecules": len(eligible_iks),
        "independent_formula_groups": len(dense), "largest_formula_group": largest,
        "directed_identity_negative_choices": directed,
        "formula_group_size_distribution": dict(Counter(len(values) for values in dense.values())),
    }, eligible_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--msp", type=Path, default=Path("data/mona/MoNA-export-LC-MS-MS_Spectra.msp"))
    parser.add_argument("--massspecgym-metadata", type=Path, default=Path("data/massspecgym/metadata.csv"))
    parser.add_argument("--external-manifest", type=Path, default=Path("data/validation/external_ring_stratified_cohort/manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/mona_lipid_pool_gate"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    msg, held_out = load_sets(args.massspecgym_metadata, args.external_manifest)

    audit = Counter(); records = []; chemistry_cache = {}
    for header in iter_headers(args.msp):
        audit["all_records"] += 1
        ik = header.get("InChIKey", "").strip()
        if not ik:
            audit["missing_inchikey"] += 1; continue
        ik14 = ik[:14]
        if ik14 in held_out:
            audit["external_holdout_excluded"] += 1; continue
        if header.get("Spectrum_type", "").upper() != "MS2" or header.get("Ion_mode", "").upper() not in {"P", "POSITIVE"}:
            audit["wrong_mode_or_level"] += 1; continue
        adduct = header.get("Precursor_type", "").strip()
        if adduct not in {"[M+H]+", "[M+Na]+"}:
            audit["unsupported_adduct"] += 1; continue
        match = SMILES_RE.search(header.get("Comments", ""))
        if not match:
            audit["missing_smiles"] += 1; continue
        smiles = match.group(1).strip()
        if smiles not in chemistry_cache:
            chemistry_cache[smiles] = chemistry(smiles)
        chem = chemistry_cache[smiles]
        if chem is None:
            continue
        formula = header.get("Formula", "").strip() or chem["formula_from_smiles"]
        records.append({
            "record_id": header.get("DB#", "").strip(), "ik14": ik14,
            "full_inchikey": ik, "smiles": chem["canonical_smiles"], "formula": formula,
            "adduct": adduct, "precursor_mz": header.get("PrecursorMZ", "").strip(),
            "instrument": header.get("Instrument", "").strip(),
            "instrument_type": header.get("Instrument_type", "").strip(),
            "collision_energy": header.get("Collision_energy", "").strip(),
            "num_peaks": header.get("Num Peaks", "").strip(), "in_massspecgym": ik14 in msg,
            **{key: chem[key] for key in ("carbon_count", "nitrogen_count", "phosphorus_count", "oxygen_count")},
        })

    all_report, all_records = summarize(records, "external_holdout_excluded")
    novel_report, novel_records = summarize([record for record in records if not record["in_massspecgym"]], "also_massspecgym_excluded")
    for filename, values in (("all_eligible_records.csv", all_records), ("novel_eligible_records.csv", novel_records)):
        with (args.output_dir / filename).open("w", newline="", encoding="utf-8") as stream:
            fields = list(values[0]) if values else ["record_id"]
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(values)
    report = {
        "status": "mona_lipid_pool_metadata_gate",
        "domain_definition": "acyclic; C>=20; P>=1; positive MS2; [M+H]+ or [M+Na]+; >=2 record IDs and acquisition conditions per identity",
        "audit": dict(audit), "cohorts": [all_report, novel_report],
        "next_gate": "If sufficient, parse only eligible peak lists, remove duplicate spectra, and compare overlap with MassBank by spectrum/accession.",
    }
    (args.output_dir / "metadata_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
