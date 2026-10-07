"""Forensic screen of tier A/B dark modules against the frozen EIC matrix.

For each module in tiers A/B1/B2: sample-type breakdown (blank / pooled QC /
QC dilution / study presence and areas) from the frozen dark-feature EIC
matrix, plus the module's differential-abundance effect from the priority
module table. Disposition logic:
  blank-present      -> contaminant (kill)
  blank-absent + QC-robust + study-wide -> real constituent; if the analog
      neighbor is halogenated -> drug-exposure hypothesis (medication
      records), else MSn/standard
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EIC = ROOT / "data/validation/lcnec_hsst3n_dark_eic_gate/dark_feature_eic_matrix.npz"
AUDIT = ROOT / "data/validation/lcnec_hsst3n_acquisition_gate/file_acquisition_audit.csv"
MODULES = ROOT / "data/validation/lcnec_hsst3n_priority_ms2/priority_dark_modules.csv"
TIERS = ROOT / "data/validation/GLM_track2_census/lcnec_dark_three_tier.csv"
OUT = ROOT / "data/validation/GLM_track2_census/lcnec_dark_forensics.json"

z = np.load(EIC, allow_pickle=True)
audit = pd.read_csv(AUDIT)
sid2cls = dict(zip(audit.sample_id.astype(str), audit.injection_class.astype(str)))
cls = np.array([sid2cls.get(str(s), "unknown") for s in z["sample_id"]])
mods = pd.read_csv(MODULES)
tiers = pd.read_csv(TIERS)

rows = []
for _, t in tiers[tiers.tier.str.startswith(("A", "B"))].iterrows():
    hit = mods[(mods.target_mz.astype(float) - float(t.precursor)).abs() < 0.01]
    if not len(hit):
        rows.append({"precursor": t.precursor, "tier": t.tier,
                     "forensic": "module not found in EIC module table"})
        continue
    m = hit.iloc[0]
    cols = np.where(z["family_id"] == int(m.family_id))[0]
    area = z["area"][:, cols].sum(axis=1)
    by = {}
    for c in ("blank", "pooled_qc", "qc_dilution", "study"):
        mask = cls == c
        by[c] = {"n": int(mask.sum()),
                 "present": int((area[mask] > 0).sum()),
                 "mean_area": round(float(area[mask].mean()), 1)}
    blank_killed = by["blank"]["present"] > 0
    qc_robust = by["pooled_qc"]["present"] >= 8
    study_wide = by["study"]["present"] >= 60
    if blank_killed:
        disp = "CONTAMINANT (blank-present) - drop"
    elif qc_robust and study_wide:
        disp = ("real constituent; analog neighbor "
                + ("halogenated -> drug-exposure hypothesis (medication "
                   "records) then MSn" if bool(t.top1_halogen_flag)
                   else "-> MSn / standard"))
    else:
        disp = "intermediate robustness - inspect"
    rows.append({
        "precursor": t.precursor, "tier": t.tier,
        "family_id": int(m.family_id),
        "effect_log2fc": float(m.effect_log2fc),
        "effect_q": float(m.effect_q),
        "breakdown": by, "disposition": disp,
        "halogen_flag": bool(t.top1_halogen_flag),
        "delta_ppm_vs_mh": t.top1_delta_ppm_vs_mh,
    })
    print(f"m/z {t.precursor}: blank {by['blank']['present']}/{by['blank']['n']}, "
          f"QC {by['pooled_qc']['present']}/{by['pooled_qc']['n']}, "
          f"study {by['study']['present']}/{by['study']['n']}, "
          f"log2FC {m.effect_log2fc:+.2f} (q={m.effect_q:.1e}) -> {disp}")

OUT.write_text(json.dumps(
    {"status": "GLM_LCNEC_DARK_FORENSICS",
     "note": ("frozen EIC matrix breakdown by injection class + module "
              "differential effect; disposition per tier A/B module"),
     "modules": rows}, indent=2), encoding="utf-8")
print("written:", OUT)
