"""Metadata gate for a GNPS phospholipid-like hard-negative pool."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, inchi, rdMolDescriptors


PROTON = 1.007276466621
SODIUM = 22.989218
RDLogger.DisableLog("rdApp.error")


def load_sets(msg_path: Path, external_path: Path) -> tuple[set[str], set[str]]:
    with msg_path.open(encoding="utf-8") as stream:
        msg = {row["inchikey14"][:14] for row in csv.DictReader(stream)}
    external = json.loads(external_path.read_text(encoding="utf-8"))
    held_out = {unit["ik14"] for unit in external["units"]}
    return msg, held_out


def iter_headers(path: Path):
    current = {}
    in_record = False
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for raw in stream:
            line = raw.strip()
            if line == "BEGIN IONS":
                current = {}; in_record = True
            elif line == "END IONS":
                if current:
                    yield current
                current = {}; in_record = False
            elif in_record and "=" in line and not (line[0].isdigit() or line[0] in ".-"):
                key, value = line.split("=", 1)
                current.setdefault(key.upper(), value.strip())


def structure_record(smiles: str, inchi_text: str) -> dict | None:
    mol = None
    if smiles and smiles.upper() not in {"N/A", "NA", "NO_SMILES"}:
        mol = Chem.MolFromSmiles(smiles)
    if mol is None and inchi_text.startswith("InChI="):
        mol = Chem.MolFromInchi(inchi_text)
    if mol is None:
        return None
    counts = Counter(atom.GetSymbol() for atom in mol.GetAtoms())
    if rdMolDescriptors.CalcNumRings(mol) != 0 or counts["C"] < 20 or counts["P"] < 1:
        return None
    try:
        full_ik = inchi.MolToInchiKey(mol)
    except Exception:
        return None
    if not full_ik:
        return None
    neutral = Descriptors.ExactMolWt(mol)
    return {
        "ik14": full_ik[:14], "full_inchikey": full_ik,
        "smiles": Chem.MolToSmiles(mol, isomericSmiles=True),
        "formula": rdMolDescriptors.CalcMolFormula(mol), "neutral_mass": neutral,
        "carbon_count": counts["C"], "nitrogen_count": counts["N"],
        "phosphorus_count": counts["P"], "oxygen_count": counts["O"],
    }


def summarize(records: list[dict], label: str) -> tuple[dict, list[dict]]:
    by_ik = defaultdict(list)
    for record in records:
        by_ik[record["ik14"]].append(record)
    positive_ready = {
        ik14: values for ik14, values in by_ik.items()
        if len({value["spectrum_id"] for value in values}) >= 2
    }
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
        "molecules_with_two_records": len(positive_ready),
        "eligible_same_formula_molecules": len(eligible_iks),
        "independent_formula_groups": len(dense), "largest_formula_group": largest,
        "directed_identity_negative_choices": directed,
        "formula_group_size_distribution": dict(Counter(len(values) for values in dense.values())),
    }, eligible_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mgf", type=Path, default=Path("data/gnps/GNPS_ALL_GNPS.mgf"))
    parser.add_argument("--massspecgym-metadata", type=Path, default=Path("data/massspecgym/metadata.csv"))
    parser.add_argument("--external-manifest", type=Path, default=Path("data/validation/external_ring_stratified_cohort/manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/gnps_lipid_pool_gate"))
    parser.add_argument("--adduct-ppm", type=float, default=20.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    msg, held_out = load_sets(args.massspecgym_metadata, args.external_manifest)

    audit = Counter(); records = []; cache = {}
    for header in iter_headers(args.mgf):
        audit["all_records"] += 1
        if header.get("MSLEVEL", "") != "2" or header.get("IONMODE", "").upper() != "POSITIVE":
            audit["wrong_mode_or_level"] += 1; continue
        smiles = header.get("SMILES", "").strip(); inchi_text = header.get("INCHI", "").strip()
        cache_key = (smiles, inchi_text)
        if cache_key not in cache:
            cache[cache_key] = structure_record(smiles, inchi_text)
        chemistry = cache[cache_key]
        if chemistry is None:
            continue
        if chemistry["ik14"] in held_out:
            audit["external_holdout_excluded"] += 1; continue
        try:
            precursor = float(header.get("PEPMASS", "").split()[0])
        except (ValueError, IndexError):
            audit["missing_precursor"] += 1; continue
        candidates = {"[M+H]+": chemistry["neutral_mass"] + PROTON,
                      "[M+Na]+": chemistry["neutral_mass"] + SODIUM}
        adduct, expected = min(candidates.items(), key=lambda item: abs(item[1] - precursor))
        ppm_error = abs(expected - precursor) / expected * 1e6
        if ppm_error > args.adduct_ppm:
            audit["adduct_mass_rejected"] += 1; continue
        spectrum_id = header.get("SPECTRUMID", "") or header.get("USI", "") or header.get("SCANS", "")
        if not spectrum_id:
            audit["missing_spectrum_id"] += 1; continue
        records.append({
            **chemistry, "spectrum_id": spectrum_id, "usi": header.get("USI", ""),
            "precursor_mz": precursor, "adduct": adduct, "adduct_ppm_error": ppm_error,
            "instrument": header.get("SOURCE_INSTRUMENT", ""),
            "library_quality": header.get("LIBRARYQUALITY", ""),
            "name": header.get("NAME", ""), "in_massspecgym": chemistry["ik14"] in msg,
        })

    all_report, all_records = summarize(records, "external_holdout_excluded")
    novel_report, novel_records = summarize([record for record in records if not record["in_massspecgym"]], "also_massspecgym_excluded")
    for filename, values in (("all_eligible_records.csv", all_records), ("novel_eligible_records.csv", novel_records)):
        with (args.output_dir / filename).open("w", newline="", encoding="utf-8") as stream:
            fields = list(values[0]) if values else ["spectrum_id"]
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(values)
    report = {
        "status": "gnps_lipid_pool_metadata_gate",
        "domain_definition": "acyclic; C>=20; P>=1; positive MS2; structure-derived identity; inferred [M+H]+/[M+Na]+ within 20 ppm; >=2 spectrum IDs",
        "audit": dict(audit), "unique_structure_cache_entries": len(cache),
        "cohorts": [all_report, novel_report],
        "next_gate": "If sufficient, rescan only eligible spectrum IDs, apply peak-quality/nonduplicate checks, then split by complete formula groups.",
    }
    (args.output_dir / "metadata_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
