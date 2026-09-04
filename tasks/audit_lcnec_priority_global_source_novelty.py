"""Audit four frozen LCNEC hypotheses against every source-atlas identity row.

The audit deliberately uses an exact normalized-name resolver and reports its
unresolved fraction. It can establish absence under that conservative resolver,
not synonym-complete chemical novelty.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi
from rdkit.Chem.rdMolDescriptors import CalcMolFormula


ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data/validation/lcnec_zenodo19005638_preflight/article_mmc7.xlsx"
HMDB = ROOT / "data/external/netid_v1/source/LiChenPU-NetID-9f63202/FDR_example/hmdb_library.csv"
PRIORITIES = ROOT / "data/validation/lcnec_hsst3n_identity_claim_defense_v1/priority_identity_claim_ledger.csv"
OUT = ROOT / "data/validation/lcnec_hsst3n_priority_global_source_novelty_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_name(value: object) -> str:
    text = str(value).casefold().replace("α", "alpha").replace("γ", "gamma")
    return re.sub(r"[^a-z0-9]", "", text)


ELEMENT_MASS = {
    "C": 12.0,
    "H": 1.00782503223,
    "N": 14.00307400443,
    "O": 15.99491461957,
    "P": 30.97376199842,
    "S": 31.9720711744,
}
PROTON = 1.007276466621
ADDUCT_SHIFT = {
    "[M+H]+": PROTON,
    "[M-H]-": -PROTON,
    "[M+NH4]+": 18.033825553,
    "[M+CH3COO]-": 59.013851,
    "[M+CHO2]-": 44.998201,
    "[M]+": 0.0,
    "[M+Na]+": 22.989218,
    "[M-H2O+H]+": PROTON - 18.010564684,
}


def formula_exact_mass(formula: str) -> float:
    pieces = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    if not pieces or "".join(f"{element}{count}" for element, count in pieces) != formula:
        raise RuntimeError(f"cannot parse formula {formula!r}")
    total = 0.0
    for element, count in pieces:
        if element not in ELEMENT_MASS:
            raise RuntimeError(f"unsupported element {element} in {formula}")
        total += ELEMENT_MASS[element] * (int(count) if count else 1)
    return total


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = pd.read_excel(WORKBOOK, sheet_name="Table S1", header=11)
    source = source.loc[
        source["Metabolite"].notna() & source["Platform"].notna() & source["m/z"].notna()
    ].copy()
    if len(source) != 1054:
        raise RuntimeError(f"source atlas row count changed: {len(source)}")
    source["normalized_name"] = source["Metabolite"].map(normalize_name)
    source["neutral_mass_estimate"] = source.apply(
        lambda row: float(row["m/z"]) - ADDUCT_SHIFT[str(row["Ion form"])]
        if str(row["Ion form"]) in ADDUCT_SHIFT else float("nan"),
        axis=1,
    )

    priorities = pd.read_csv(PRIORITIES)
    if len(priorities) != 4:
        raise RuntimeError("frozen priority ledger no longer has four rows")

    hmdb = pd.read_csv(HMDB, low_memory=False)
    hmdb = hmdb.loc[hmdb["name"].notna() & hmdb["SMILES"].notna()].copy()
    hmdb["normalized_name"] = hmdb["name"].map(normalize_name)
    needed_names = set(source["normalized_name"])
    RDLogger.DisableLog("rdApp.*")

    name_to_structures: dict[str, dict[str, dict[str, str]]] = {}
    priority_aliases: dict[str, set[str]] = {
        str(row.ik14): {normalize_name(row.spectral_hypothesis)} for row in priorities.itertuples(index=False)
    }
    for row in hmdb.itertuples(index=False):
        normalized = str(row.normalized_name)
        # Exact-name resolution only needs HMDB entries whose normalized names
        # actually occur in the source atlas. This keeps the deterministic audit
        # bounded and does not change its declared resolver.
        if normalized not in needed_names:
            continue
        molecule = Chem.MolFromSmiles(str(row.SMILES))
        if molecule is None:
            continue
        key = inchi.MolToInchiKey(molecule)
        if not key:
            continue
        ik14 = key[:14]
        name_to_structures.setdefault(normalized, {})[ik14] = {
            "formula": CalcMolFormula(molecule),
            "hmdb_name": str(row.name),
        }

    resolved_rows: list[dict[str, object]] = []
    for row in source.itertuples(index=False):
        normalized = str(row.normalized_name)
        structures = name_to_structures.get(normalized, {})
        keys = sorted(structures)
        status = "unique" if len(keys) == 1 else ("ambiguous" if len(keys) > 1 else "unresolved")
        ik14 = keys[0] if len(keys) == 1 else ""
        formula = structures[ik14]["formula"] if ik14 else ""
        resolved_rows.append({
            "source_metabolite": row.Metabolite,
            "normalized_name": normalized,
            "platform": row.Platform,
            "msi_level": row._10 if hasattr(row, "_10") else None,
            "resolution_status": status,
            "resolved_ik14": ik14,
            "resolved_formula": formula,
            "candidate_ik14s": ";".join(keys),
        })
    resolved = pd.DataFrame(resolved_rows)
    resolved.to_csv(OUT / "source_identity_resolution_ledger.csv", index=False)

    audit_rows: list[dict[str, object]] = []
    mass_match_rows: list[dict[str, object]] = []
    for priority in priorities.itertuples(index=False):
        ik14 = str(priority.ik14)
        formula = str(priority.formula)
        aliases = priority_aliases.get(ik14, set()) | {normalize_name(priority.spectral_hypothesis)}
        exact_alias_rows = source.loc[source["normalized_name"].isin(aliases)]
        structure_rows = resolved.loc[resolved["resolved_ik14"].eq(ik14)]
        formula_rows = resolved.loc[resolved["resolved_formula"].eq(formula)]
        other_formula_rows = formula_rows.loc[~formula_rows["resolved_ik14"].eq(ik14)]
        neutral_mass = formula_exact_mass(formula)
        mass_ppm = (source["neutral_mass_estimate"] - neutral_mass).abs() / neutral_mass * 1e6
        exact_mass_rows = source.loc[mass_ppm <= 5].copy()
        exact_mass_rows["neutral_mass_error_ppm"] = mass_ppm.loc[exact_mass_rows.index]
        for _, match in exact_mass_rows.iterrows():
            mass_match_rows.append({
                "priority_name": priority.priority_name,
                "priority_formula": formula,
                "priority_neutral_exact_mass": neutral_mass,
                "source_metabolite": match["Metabolite"],
                "source_platform": match["Platform"],
                "source_msi_level": match["MSI Level"],
                "source_mz": match["m/z"],
                "source_ion_form": match["Ion form"],
                "source_neutral_mass_estimate": match["neutral_mass_estimate"],
                "neutral_mass_error_ppm": match["neutral_mass_error_ppm"],
            })
        audit_rows.append({
            "priority_name": priority.priority_name,
            "spectral_hypothesis": priority.spectral_hypothesis,
            "ik14": ik14,
            "formula": formula,
            "priority_hmdb_aliases": ";".join(sorted(aliases)),
            "source_exact_alias_rows": len(exact_alias_rows),
            "source_unique_structure_rows": len(structure_rows),
            "source_same_formula_rows": len(formula_rows),
            "source_same_formula_other_structure_rows": len(other_formula_rows),
            "source_neutral_mass_matches_5ppm": len(exact_mass_rows),
            "source_exact_alias_names": ";".join(sorted(set(exact_alias_rows["Metabolite"].astype(str)))),
            "source_structure_match_names": ";".join(sorted(set(structure_rows["source_metabolite"].astype(str)))),
            "source_same_formula_names": ";".join(sorted(set(formula_rows["source_metabolite"].astype(str)))),
            "absent_from_global_source_under_exact_and_mass_resolver": bool(
                len(exact_alias_rows) == 0 and len(structure_rows) == 0 and len(exact_mass_rows) == 0
            ),
        })
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(OUT / "priority_global_source_novelty_audit.csv", index=False)
    mass_matches = pd.DataFrame(mass_match_rows)
    mass_matches.to_csv(OUT / "priority_global_source_neutral_mass_matches.csv", index=False)

    report = {
        "status": "lcnec_priority_global_source_novelty_audit_complete",
        "formal": True,
        "source_rows_all_platforms": len(source),
        "source_unique_names": int(source["normalized_name"].nunique()),
        "source_rows_unique_structure_resolved": int((resolved["resolution_status"] == "unique").sum()),
        "source_rows_ambiguous": int((resolved["resolution_status"] == "ambiguous").sum()),
        "source_rows_unresolved": int((resolved["resolution_status"] == "unresolved").sum()),
        "priorities": len(audit),
        "source_rows_with_supported_adduct_for_neutral_mass": int(source["neutral_mass_estimate"].notna().sum()),
        "source_rows_with_unsupported_adduct_for_neutral_mass": int(source["neutral_mass_estimate"].isna().sum()),
        "priorities_absent_under_exact_and_mass_global_resolver": int(
            audit["absent_from_global_source_under_exact_and_mass_resolver"].sum()
        ),
        "priorities_with_source_same_formula_rows": int((audit["source_same_formula_rows"] > 0).sum()),
        "per_priority": audit.to_dict("records"),
        "provenance": {
            "source_workbook_sha256": sha256(WORKBOOK),
            "hmdb_library_sha256": sha256(HMDB),
            "priority_ledger_sha256": sha256(PRIORITIES),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "Absence means no exact normalized alias, no unique exact-name-to-HMDB IK14 match and no supported-adduct neutral-mass "
            "match within 5 ppm among all source-atlas rows. It is not proof of chemical novelty. Unresolved source names, unsupported "
            "adducts and same-formula alternatives remain "
            "explicit limitations; every priority remains Level 2 or a connectivity-family hypothesis."
        ),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
