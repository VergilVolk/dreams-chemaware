"""Benchmark LCNEC identity concordance on source-matched positive controls.

The source table has names but no structures.  To avoid manual synonym tuning,
this script resolves only exact normalized source names that map to one unique
IK14 in a frozen local HMDB library.  Unresolved or structurally ambiguous names
are reported and excluded from the identity-concordance denominator.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi
from rdkit.Chem.rdMolDescriptors import CalcMolFormula


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/validation/lcnec_hsst3n_same_universe_comparison_v1/per_family_comparison.csv"
HMDB = ROOT / "data/external/netid_v1/source/LiChenPU-NetID-9f63202/FDR_example/hmdb_library.csv"
OUT = ROOT / "data/validation/lcnec_hsst3n_source_positive_control_identity_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_name(value: object) -> str:
    text = str(value).lower().replace("α", "alpha").replace("γ", "gamma")
    return re.sub(r"[^a-z0-9]", "", text)


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [math.nan, math.nan]
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - radius, center + radius]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE)
    source = source.loc[source["author_matched"].astype(bool)].copy()
    target_names = set(source["author_metabolite"].map(normalize_name))
    hmdb = pd.read_csv(HMDB, low_memory=False)
    hmdb["normalized_name"] = hmdb["name"].map(normalize_name)
    hmdb = hmdb.loc[hmdb["normalized_name"].isin(target_names)].copy()
    RDLogger.DisableLog("rdApp.*")

    lookup: dict[str, dict[str, str]] = {}
    for _, row in hmdb.iterrows():
        name = row["normalized_name"]
        smiles = row.get("SMILES")
        if not name or pd.isna(smiles):
            continue
        molecule = Chem.MolFromSmiles(str(smiles))
        if molecule is None:
            continue
        key = inchi.MolToInchiKey(molecule)
        if key:
            lookup.setdefault(name, {})[key[:14]] = CalcMolFormula(molecule)

    rows = []
    for _, row in source.iterrows():
        normalized = normalize_name(row["author_metabolite"])
        structures = lookup.get(normalized, {})
        keys = sorted(structures)
        status = "unique" if len(keys) == 1 else ("ambiguous" if len(keys) > 1 else "unresolved")
        truth = keys[0] if len(keys) == 1 else ""
        source_formula = structures.get(truth, "")
        dreams_candidate = str(row.get("dreams_top_ik14", ""))
        p2b_candidate = str(row.get("p2b_top_ik14", ""))
        candidate_molecule = Chem.MolFromSmiles(str(row.get("p2b_top_smiles", "")))
        candidate_formula = CalcMolFormula(candidate_molecule) if candidate_molecule is not None else ""
        dreams_correct = bool(truth and dreams_candidate == truth)
        p2b_correct = bool(truth and p2b_candidate == truth)
        retained = bool(row["multi_evidence_retained"])
        rows.append({
            "family_id": int(row["family_id"]),
            "author_metabolite": row["author_metabolite"],
            "author_msi_level": row["author_msi_level"],
            "resolution_status": status,
            "source_ik14": truth,
            "source_formula": source_formula,
            "source_ik14_candidates": ";".join(keys),
            "dreams_top_name": row.get("dreams_top_name", ""),
            "dreams_top_ik14": dreams_candidate,
            "dreams_top1_concordant": dreams_correct,
            "p2b_top_name": row.get("p2b_top_name", ""),
            "p2b_top_ik14": p2b_candidate,
            "p2b_top_formula": candidate_formula,
            "p2b_top1_concordant": p2b_correct,
            "multi_evidence_retained": retained,
            "full_tool_concordant_retained": bool(retained and p2b_correct),
            "full_tool_discordant_retained": bool(retained and not p2b_correct),
            "same_formula_isomer_error": bool(
                retained and truth and not p2b_correct and source_formula == candidate_formula
            ),
            "annotation_confidence": row.get("annotation_confidence", ""),
        })
    ledger = pd.DataFrame(rows)
    ledger.to_csv(OUT / "positive_control_identity_ledger.csv", index=False)
    evaluable = ledger.loc[ledger["resolution_status"] == "unique"].copy()
    if len(evaluable) < 15:
        raise RuntimeError(f"too few structure-resolvable source positive controls: {len(evaluable)}")

    def metric(column: str) -> dict[str, object]:
        success = int(evaluable[column].astype(bool).sum())
        total = int(len(evaluable))
        return {"success": success, "total": total, "fraction": success / total, "wilson_95ci": wilson(success, total)}

    retained = evaluable.loc[evaluable["multi_evidence_retained"].astype(bool)]
    retained_correct = int(retained["p2b_top1_concordant"].astype(bool).sum())
    retained_total = int(len(retained))
    report = {
        "status": "lcnec_source_positive_control_identity_benchmark_complete",
        "formal": True,
        "protocol": "exact normalized source name -> unique local-HMDB IK14; no manual synonym rescue",
        "source_matched_families": int(len(ledger)),
        "structure_resolvable_unique": int(len(evaluable)),
        "structure_ambiguous": int((ledger["resolution_status"] == "ambiguous").sum()),
        "structure_unresolved": int((ledger["resolution_status"] == "unresolved").sum()),
        "metrics": {
            "official_dreams_top1_concordance": metric("dreams_top1_concordant"),
            "p2b_top1_concordance": metric("p2b_top1_concordant"),
            "full_tool_correct_evidence_yield": metric("full_tool_concordant_retained"),
            "full_tool_precision_among_retained": {
                "success": retained_correct,
                "total": retained_total,
                "fraction": retained_correct / retained_total if retained_total else math.nan,
                "wilson_95ci": wilson(retained_correct, retained_total),
            },
            "full_tool_incorrect_retained": int(evaluable["full_tool_discordant_retained"].astype(bool).sum()),
            "full_tool_incorrect_retained_same_formula_isomers": int(
                evaluable["same_formula_isomer_error"].astype(bool).sum()
            ),
        },
        "provenance": {"source_sha256": sha256(SOURCE), "hmdb_sha256": sha256(HMDB)},
        "claim_limit": "This is source-name/structure concordance on a conservative positive-control subset. MSI Level-2 source annotations are not authentic-standard truth, unresolved names are excluded, and the result is not a global annotation-accuracy estimate.",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
