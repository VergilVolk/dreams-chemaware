"""Three-tier claim architecture for LCNEC priority dark modules.

Applies the GNPS-calibrated confidence gate (cosine top1-top2 molecule gap)
to the molecule-level reverse-search rescan, producing the actionable
per-module claim table:

  Tier A  gate-passed Level-2 candidate (best_cosine >= 0.85 AND
          molecule_gap >= 0.584, the conservative @99% threshold from the
          identity-disjoint panel; dual-metric entropy reported)
  Tier B1 high-score contested (best_cosine >= 0.85, gap < threshold)
          -> escalate: standard purchase / MSn / orthogonal evidence
  Tier B2 sub-hit candidates (0.70 <= best_cosine < 0.85) -> analog-level
  Tier C  dark (best_cosine < 0.70) -> no spectral claim

Honesty boundaries (documented, not hidden):
  - thresholds were calibrated on library-vs-library GNPS panels; LCNEC
    dark features are real-sample DDA spectra (domain shift -> the tier-A
    accuracy guarantee transfers only approximately; tier A is still a
    Level-2 library-match claim, never Level 1 without a standard);
  - the reverse-search precursor gate is analog-tolerant (0.7%), so
    competing 'molecules' include analogs, not just same-formula isomers
    (gap semantics: contested evidence, wider than 10ppm retrieval);
  - n = 30 modules.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESCAN = ROOT / "data/validation/GLM_track2_census/lcnec_dark_molecule_rescan.json"
NAMES = ROOT / "data/validation/GLM_track2_census/lcnec_dark_neighbor_identities.csv"
OUT_JSON = ROOT / "data/validation/GLM_track2_census/lcnec_dark_three_tier.json"
OUT_CSV = ROOT / "data/validation/GLM_track2_census/lcnec_dark_three_tier.csv"

GATE_99 = 0.584      # conservative (identity panel) @99% accuracy threshold
GATE_995 = 0.7407    # @99.5% threshold
HIT = 0.85
SUB = 0.70


def main() -> None:
    modules = json.loads(RESCAN.read_text(encoding="utf-8"))["modules"]
    names = pd.read_csv(NAMES)
    name_by = {}
    for _, r in names.iterrows():
        name_by.setdefault((round(float(r["dark_precursor"]), 4),
                            str(r["neighbor_ik14"])),
                           str(r["neighbor_name_field"]))

    rows = []
    for m in modules:
        top = m["top_molecules"][0] if m["top_molecules"] else None
        cos, gap = m["best_cosine"], m["molecule_gap"]
        if cos >= HIT and gap >= GATE_99:
            tier = "A_gate_passed"
        elif cos >= HIT:
            tier = "B1_high_score_contested"
        elif cos >= SUB:
            tier = "B2_sub_hit_analog"
        else:
            tier = "C_dark"
        also_995 = cos >= HIT and gap >= GATE_995
        second = m["top_molecules"][1] if len(m["top_molecules"]) > 1 else None
        rows.append({
            "precursor": m["precursor"],
            "tier": tier,
            "best_cosine": cos,
            "molecule_gap": gap,
            "passes_995": also_995,
            "entropy_best": m["entropy_best"],
            "n_molecules": m["n_molecules"],
            "top1_ik14": top["ik14"] if top else None,
            "top1_formula": top["formula"] if top else None,
            "top1_name": name_by.get((round(m["precursor"], 4),
                                      top["ik14"])) if top else None,
            "top2_ik14": second["ik14"] if second else None,
            "top2_formula": second["formula"] if second else None,
            "top2_cosine": second["cosine"] if second else None,
        })
    df = pd.DataFrame(rows).sort_values(["tier", "best_cosine"],
                                        ascending=[True, False])
    df.to_csv(OUT_CSV, index=False)
    counts = df["tier"].value_counts().to_dict()
    report = {
        "status": "GLM_LCNEC_DARK_THREE_TIER",
        "thresholds": {"gate_99": GATE_99, "gate_995": GATE_995,
                       "hit": HIT, "sub_hit": SUB,
                       "source": ("GNPS model-blind cosine_greedy gate "
                                  "(identity panel, conservative)")},
        "tier_counts": counts,
        "boundaries": [
            "thresholds calibrated on library-vs-library spectra; LCNEC are "
            "real-sample DDA (domain shift, accuracy guarantee approximate)",
            "analog-tolerant 0.7% precursor gate: competitors include analogs",
            "tier A is a Level-2 library-match claim; Level-1 requires a "
            "standard",
            "n = 30 modules",
        ],
        "modules": df.to_dict(orient="records"),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=1))
    print(df[["precursor", "tier", "best_cosine", "molecule_gap",
              "entropy_best", "top1_ik14", "top1_formula"]].to_string(
                  index=False))
    print(f"written: {OUT_CSV}")


if __name__ == "__main__":
    main()
