"""Pre-training gate for same-formula phospholipid-like hard negatives.

The audit is molecule-first and excludes every molecule used by MassSpecGym or
the external discovery/confirmation cohort.  It does not create training pairs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


PROTON = 1.007276466621
SODIUM = 22.989218


def load_exclusions(metadata: Path, external_manifest: Path) -> tuple[set[str], dict]:
    with metadata.open(encoding="utf-8") as stream:
        msg = {row["inchikey14"][:14] for row in csv.DictReader(stream)}
    external = json.loads(external_manifest.read_text(encoding="utf-8"))
    held_out = {unit["ik14"] for unit in external["units"]}
    return msg | held_out, {"massspecgym": len(msg), "external_cohort": len(held_out)}


def candidate_record(key: str, index: dict, adduct_ppm: float) -> dict | None:
    smiles = index["ik_to_smi"][key]
    formula = index["ik_to_fm"].get(key)
    precursor = index["ik_to_pm"].get(key)
    if formula is None or precursor is None or int(index["ik_counts"].get(key, 0)) < 2:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or rdMolDescriptors.CalcNumRings(mol) != 0:
        return None
    counts = Counter(atom.GetSymbol() for atom in mol.GetAtoms())
    # Broad, explicit gate definition.  It includes glycerophospholipids and
    # related long-chain phosphorus lipids without prescribing a headgroup.
    if counts["C"] < 20 or counts["P"] < 1:
        return None
    neutral = Descriptors.ExactMolWt(mol)
    candidates = {"[M+H]+": neutral + PROTON, "[M+Na]+": neutral + SODIUM}
    adduct, expected = min(candidates.items(), key=lambda item: abs(item[1] - float(precursor)))
    ppm_error = abs(expected - float(precursor)) / expected * 1e6
    if ppm_error > adduct_ppm:
        return None
    headgroup = "other_P_lipid"
    if counts["N"]:
        headgroup = "P_lipid_with_N"
    if any(atom.GetAtomicNum() == 7 and atom.GetFormalCharge() > 0 for atom in mol.GetAtoms()):
        headgroup = "quaternary_ammonium_P_lipid"
    elif counts["N"]:
        headgroup = "neutral_N_P_lipid"
    return {
        "ik14": key[:14], "full_inchikey": key, "smiles": smiles,
        "formula": formula, "index_spectrum_count": int(index["ik_counts"][key]),
        "adduct": adduct, "expected_precursor_mz": expected,
        "first_precursor_mz": float(precursor), "adduct_ppm_error": ppm_error,
        "carbon_count": counts["C"], "nitrogen_count": counts["N"],
        "phosphorus_count": counts["P"], "oxygen_count": counts["O"],
        "headgroup_proxy": headgroup,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", type=Path, default=Path("tasks/_cache/indices.json"))
    parser.add_argument("--massspecgym-metadata", type=Path, default=Path("data/massspecgym/metadata.csv"))
    parser.add_argument("--external-manifest", type=Path, default=Path("data/validation/external_ring_stratified_cohort/manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/lipid_hard_negative_pool_gate"))
    parser.add_argument("--adduct-ppm", type=float, default=20.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    index = json.loads(args.indices.read_text(encoding="utf-8"))
    excluded, exclusion_counts = load_exclusions(args.massspecgym_metadata, args.external_manifest)
    records_by_ik: dict[str, dict] = {}
    audit = Counter()
    for key in index["ik_to_smi"]:
        audit["indexed_full_inchikeys"] += 1
        if key[:14] in excluded:
            audit["excluded_molecules"] += 1
            continue
        record = candidate_record(key, index, args.adduct_ppm)
        if record is None:
            continue
        audit["broad_phospholipid_like_candidates"] += 1
        existing = records_by_ik.get(record["ik14"])
        if existing is None or record["index_spectrum_count"] > existing["index_spectrum_count"]:
            records_by_ik[record["ik14"]] = record

    by_formula: dict[str, list[dict]] = defaultdict(list)
    for record in records_by_ik.values():
        by_formula[record["formula"]].append(record)
    dense = {formula: values for formula, values in by_formula.items() if len(values) >= 2}
    eligible = [record for values in dense.values() for record in values]
    eligible.sort(key=lambda row: (row["formula"], row["ik14"]))
    with (args.output_dir / "index_level_candidates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(eligible[0]) if eligible else ["ik14"])
        writer.writeheader(); writer.writerows(eligible)

    formula_rows = []
    for formula, values in sorted(dense.items(), key=lambda item: (-len(item[1]), item[0])):
        formula_rows.append({
            "formula": formula, "molecules": len(values),
            "indexed_spectra": sum(value["index_spectrum_count"] for value in values),
            "precursor_mz_median": sorted(value["expected_precursor_mz"] for value in values)[len(values)//2],
            "headgroup_proxies": "|".join(sorted(Counter(value["headgroup_proxy"] for value in values))),
        })
    with (args.output_dir / "formula_groups.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = list(formula_rows[0]) if formula_rows else ["formula"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(formula_rows)

    total = len(eligible)
    largest = max((len(values) for values in dense.values()), default=0)
    gates = {
        "at_least_100_molecules": total >= 100,
        "at_least_30_formula_groups": len(dense) >= 30,
        "largest_formula_at_most_15_percent": (largest / total <= 0.15) if total else False,
        "at_least_500_directed_identity_negative_choices": sum(len(v) * (len(v) - 1) for v in dense.values()) >= 500,
        "raw_spectrum_quality_scan_required": True,
    }
    report = {
        "status": "index_level_lipid_hard_negative_pool_gate",
        "domain_definition": "acyclic; C>=20; P>=1; inferred [M+H]+ or [M+Na]+ within 20 ppm; >=2 indexed spectra",
        "exclusion_counts": exclusion_counts,
        "audit": dict(audit),
        "unique_candidate_molecules": len(records_by_ik),
        "eligible_same_formula_molecules": total,
        "independent_formula_groups": len(dense),
        "largest_formula_group": largest,
        "directed_identity_negative_choices": sum(len(v) * (len(v) - 1) for v in dense.values()),
        "formula_group_size_distribution": dict(Counter(len(v) for v in dense.values())),
        "headgroup_proxy_distribution": dict(Counter(v["headgroup_proxy"] for v in eligible)),
        "gates": gates,
        "gate_pass_before_raw_scan": all(value for key, value in gates.items() if key != "raw_spectrum_quality_scan_required"),
        "claim_limit": "Index-level audit only; positive-ion, peak-quality and nonduplicate-spectrum requirements must be checked by streaming the raw MGF.",
    }
    (args.output_dir / "index_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
