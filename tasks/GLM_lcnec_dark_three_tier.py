"""Three-tier claim architecture v2 (names fixed, plausibility screen).

v1 defect (audited): top1_name was joined from the pilot-era
lcnec_dark_neighbor_identities.csv, which predates the unsorted-mz cosine
fix -- its molecules and scores are stale, and its name column held USIs.
v2 joins identifiers directly from the GNPS manifest for the CORRECT
molecule-rescan top hits, and adds a halogen plausibility screen
(F/Cl/Br/I-containing structures are flagged exogenous-until-proven in
tissue metabolomics).

Tiers (unchanged):
  A    gate passed: cosine >= 0.85 AND molecule_gap >= 0.584 (@99)
  B1   high score contested: cosine >= 0.85, gap < 0.584
  B2   sub-hit analog: 0.70 <= cosine < 0.85
  C    dark: cosine < 0.70
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESCAN = ROOT / "data/validation/GLM_track2_census/lcnec_dark_molecule_rescan.json"
MANIFEST = ROOT / "data/validation/gnps_gold_silver_10ppm_benchmark_v1/manifest.csv.gz"
OUT_JSON = ROOT / "data/validation/GLM_track2_census/lcnec_dark_three_tier.json"
OUT_CSV = ROOT / "data/validation/GLM_track2_census/lcnec_dark_three_tier.csv"

GATE_99, GATE_995, HIT, SUB = 0.584, 0.7407, 0.85, 0.70


def formula_elements(formula: str) -> dict[str, int]:
    out = {}
    for el, cnt in re.findall(r"([A-Z][a-z]?)(\d*)", str(formula)):
        if el:
            out[el] = out.get(el, 0) + (int(cnt) if cnt else 1)
    return out


def main() -> None:
    modules = json.loads(RESCAN.read_text(encoding="utf-8"))["modules"]
    manifest = pd.read_csv(MANIFEST)

    def manifest_row_for(usi_tail: str | None, ik14: str):
        if usi_tail:
            hit = manifest[manifest["usi"].astype(str).str.endswith(usi_tail)]
            if len(hit):
                return hit.iloc[0]
        hit = manifest[manifest["ik14"].astype(str) == str(ik14)]
        return hit.iloc[0] if len(hit) else None

    rows = []
    for m in modules:
        top = m["top_molecules"][0] if m["top_molecules"] else None
        second = m["top_molecules"][1] if len(m["top_molecules"]) > 1 else None
        cos, gap = m["best_cosine"], m["molecule_gap"]
        if cos >= HIT and gap >= GATE_99:
            tier = "A_gate_passed"
        elif cos >= HIT:
            tier = "B1_high_score_contested"
        elif cos >= SUB:
            tier = "B2_sub_hit_analog"
        else:
            tier = "C_dark"

        def enrich(mol):
            if mol is None:
                return {}
            row = manifest_row_for(mol.get("usi_tail"), mol["ik14"])
            els = formula_elements(mol.get("formula"))
            halogens = {e: els.get(e, 0) for e in ("F", "Cl", "Br", "I")
                        if els.get(e, 0)}
            return {
                "ik14": mol["ik14"], "cosine": mol["cosine"],
                "formula": mol["formula"],
                "smiles": str(row["smiles"]) if row is not None else None,
                "usi": str(row["usi"]) if row is not None else None,
                "instrument": (str(row["instrument"]) if row is not None
                               and pd.notna(row["instrument"]) else None),
                "halogens": halogens or None,
            }

        t1, t2 = enrich(top), enrich(second)
        rows.append({
            "precursor": m["precursor"], "tier": tier,
            "best_cosine": cos, "second_cosine": m["second_cosine"],
            "molecule_gap": gap, "entropy_best": m["entropy_best"],
            "n_molecules": m["n_molecules"],
            "passes_995": bool(cos >= HIT and gap >= GATE_995),
            "top1": t1, "top1_halogen_flag": bool(t1.get("halogens")),
            "top2": t2,
        })

    df = pd.DataFrame([{
        "precursor": r["precursor"], "tier": r["tier"],
        "best_cosine": r["best_cosine"], "second_cosine": r["second_cosine"],
        "molecule_gap": r["molecule_gap"], "entropy_best": r["entropy_best"],
        "passes_995": r["passes_995"], "n_molecules": r["n_molecules"],
        "top1_ik14": r["top1"].get("ik14"),
        "top1_formula": r["top1"].get("formula"),
        "top1_smiles": r["top1"].get("smiles"),
        "top1_usi": r["top1"].get("usi"),
        "top1_halogens": r["top1"].get("halogens"),
        "top1_halogen_flag": r["top1_halogen_flag"],
        "top2_ik14": r["top2"].get("ik14"),
        "top2_formula": r["top2"].get("formula"),
        "top2_cosine": r["top2"].get("cosine"),
    } for r in rows]).sort_values(["tier", "best_cosine"],
                                  ascending=[True, False])
    df.to_csv(OUT_CSV, index=False)

    counts = df["tier"].value_counts().to_dict()
    report = {
        "status": "GLM_LCNEC_DARK_THREE_TIER_V2",
        "fix": ("names/identifiers now joined from the GNPS manifest for the "
                "corrected molecule rescan; v1 joined the pilot-era "
                "neighbor_identities.csv (stale molecules, buggy cosines, "
                "USI-as-name) - do not use that table for decisions"),
        "thresholds": {"gate_99": GATE_99, "gate_995": GATE_995,
                       "hit": HIT, "sub_hit": SUB,
                       "source": ("GNPS model-blind cosine_greedy gate, "
                                  "conservative identity-panel values")},
        "tier_counts": counts,
        "boundaries": [
            "thresholds calibrated on library-vs-library spectra; LCNEC are "
            "real-sample DDA (domain shift, guarantee approximate)",
            "analog-tolerant 0.7% precursor gate: competitors include analogs",
            "tier A is a Level-2 library-match claim; Level-1 requires a "
            "standard",
            "halogen flag: F/Cl/Br/I structures are exogenous-until-proven "
            "in tissue metabolomics (chemotherapy/environmental) - gate "
            "passage does not clear the plausibility axis",
            "n = 30 modules",
        ],
        "modules": rows,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=1))
    cols = ["precursor", "tier", "best_cosine", "molecule_gap",
            "entropy_best", "top1_ik14", "top1_formula", "top1_halogens",
            "top1_halogen_flag"]
    print(df[cols].to_string(index=False))
    print(f"written: {OUT_CSV}")


if __name__ == "__main__":
    main()
