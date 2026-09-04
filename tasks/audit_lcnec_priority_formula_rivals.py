"""Audit same-formula spectral-library rivals for the four LCNEC priorities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


ROOT = Path(__file__).resolve().parents[1]
TOP = ROOT / "data/validation/lcnec_hsst3n_all_robust_annotation/priority_annotation_top5.csv"
LEDGER = ROOT / "data/validation/lcnec_hsst3n_manuscript_supplement/table_s5_4_priority_evidence_ledger.csv"
HMDB = ROOT / "data/external/netid_v1/source/LiChenPU-NetID-9f63202/FDR_example/hmdb_library.csv"
OUT = ROOT / "data/validation/lcnec_hsst3n_priority_formula_rivals_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def formula(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    return rdMolDescriptors.CalcMolFormula(mol) if mol is not None else None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    priority = pd.read_csv(LEDGER)[["family_id", "priority_name", "spectral_hypothesis", "formula"]]
    top = pd.read_csv(TOP).drop_duplicates(["family_id", "rank", "ik14"]).copy()
    top = top[top["family_id"].isin(priority["family_id"])].copy()
    top["computed_formula"] = [formula(value) for value in top["smiles"]]
    top = top.merge(priority, on="family_id", how="left", validate="many_to_one")
    top["same_formula_as_priority"] = top["computed_formula"] == top["formula"]
    top.to_csv(OUT / "priority_top5_formula_rivals.csv", index=False)

    hmdb = pd.read_csv(HMDB, low_memory=False)
    summaries = []
    for row in priority.itertuples(index=False):
        candidates = top[top["family_id"] == row.family_id].sort_values("rank")
        if candidates.empty or int(candidates.iloc[0]["rank"]) != 1:
            raise RuntimeError(f"missing top candidate for family {row.family_id}")
        lead = candidates.iloc[0]
        same = candidates[(candidates["same_formula_as_priority"]) & (candidates["rank"] > 1)]
        if same.empty:
            dreams_margin = None
            final_margin = None
            nearest = None
        else:
            rival = same.sort_values(["dreams_molecule_score", "final_molecule_score"], ascending=False).iloc[0]
            dreams_margin = float(lead["dreams_molecule_score"] - rival["dreams_molecule_score"])
            final_margin = float(lead["final_molecule_score"] - rival["final_molecule_score"])
            nearest = str(rival["name"])
        hmdb_formula = hmdb[hmdb["formula"] == row.formula]
        summaries.append({
            "family_id": int(row.family_id),
            "priority_name": row.priority_name,
            "spectral_hypothesis": row.spectral_hypothesis,
            "formula": row.formula,
            "library_candidates_reported": int(len(candidates)),
            "same_formula_library_rivals": int(len(same)),
            "nearest_same_formula_rival": nearest,
            "dreams_margin_to_same_formula_rival": dreams_margin,
            "final_margin_to_same_formula_rival": final_margin,
            "local_hmdb_formula_rows": int(len(hmdb_formula)),
            "local_hmdb_unique_smiles": int(hmdb_formula["SMILES"].nunique()),
            "formula_unique_in_local_hmdb": bool(hmdb_formula["SMILES"].nunique() == 1),
            "exact_identity_allowed": False,
            "boundary": "no spectral rival available is not chemical uniqueness" if same.empty else "same-formula spectral rivals remain",
        })
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "priority_formula_rival_summary.csv", index=False)

    report = {
        "status": "lcnec_priority_formula_rival_audit_complete",
        "formal": True,
        "priorities": int(len(summary)),
        "priorities_with_same_formula_top5_rivals": int((summary["same_formula_library_rivals"] > 0).sum()),
        "priorities_without_observed_same_formula_rival": int((summary["same_formula_library_rivals"] == 0).sum()),
        "new_exact_metabolite_claims": 0,
        "key_rivals": {row.priority_name: row.nearest_same_formula_rival for row in summary.itertuples(index=False)},
        "provenance": {"top5_sha256": sha256(TOP), "priority_ledger_sha256": sha256(LEDGER), "hmdb_sha256": sha256(HMDB)},
        "claim_limit": "The audit enumerates observed library/HMDB alternatives. It cannot prove that unobserved alternatives do not exist, and it does not upgrade any identity.",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

