"""Unified per-entity evidence table for LCNEC.

Connects ALL module outputs to the same experimental entity axis:
  Step 1  experimental feature archiving (QC/blank/forensics)  [exists]
  Step 2  spectral coordinate retrieval (Noise/WSE/cosine)     [this builds]
  Step 3  local competition + chemical explanation             [this builds]
  Step 4  conditional sample evidence (BioAware)               [slot]
  Step 5  tiered output + confidence + disposition             [this builds]

Output: one CSV row per LCNEC entity with every module's evidence,
applicability, and the system's final tiered decision. This is the
artifact the paper's application section is built on.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
sys.path.insert(0, str(ROOT))

ENTITIES = ROOT / "data/validation/GLM_frozen_entity_dictionary/frozen_entities.csv"
THREE_TIER = ROOT / "data/validation/GLM_track2_census/lcnec_dark_three_tier.csv"
FORENSICS = ROOT / "data/validation/GLM_track2_census/lcnec_dark_forensics.json"
BENCH = ROOT / "data/validation/gnps_gold_silver_10ppm_benchmark_v1"
DARK_MGF = ROOT / "data/validation/lcnec_hsst3n_priority_ms2/priority_dark_modules.mgf"
OUT = ROOT / "data/validation/GLM_unified_entity_evidence"

# Gate thresholds (from GNPS calibration)
GATE_99_COS = 0.584
GATE_995_COS = 0.7407
HIT = 0.85


def load_dreams_embedding_if_available():
    """Try to load Noise V1 embeddings for LCNEC entities (server-dependent)."""
    # For now: return None (server asset); cosine from three-tier table is local
    return None


def main() -> None:
    # ---- load all existing per-entity results ----
    entities = pd.read_csv(ENTITIES)
    tier = pd.read_csv(THREE_TIER)
    forensics = json.loads(FORENSICS.read_text(encoding="utf-8"))
    forensics_map = {round(float(m["precursor"]), 4): m
                     for m in forensics.get("modules", [])}

    # ---- build unified evidence table ----
    rows = []
    for _, e in entities.iterrows():
        mz = round(float(e["mz"]), 4)
        eid = e["entity_id"]
        row = {"entity_id": eid, "mz": mz}

        # Step 1: experimental feature archiving
        f = forensics_map.get(mz, {})
        br = f.get("breakdown", {})
        row["step1_blank_present"] = br.get("blank", {}).get("present", 0) > 0
        row["step1_qc_stable"] = br.get("pooled_qc", {}).get("present", 0) >= 8
        row["step1_study_detected"] = br.get("study", {}).get("present", 0)
        row["step1_contaminant"] = row["step1_blank_present"]
        row["step1_effect_log2fc"] = f.get("effect_log2fc")
        row["step1_effect_q"] = f.get("effect_q")

        # Step 2: spectral coordinate retrieval
        t = tier[tier["precursor"].round(4) == mz]
        if len(t):
            t = t.iloc[0]
            row["step2_cosine_best"] = float(t["best_cosine"])
            row["step2_gap"] = float(t["molecule_gap"])
            row["step2_entropy_best"] = float(t["entropy_best"]) if pd.notna(t["entropy_best"]) else None
            row["step2_top1_ik14"] = t.get("top1_ik14")
            row["step2_top1_formula"] = t.get("top1_formula")
            row["step2_top1_smiles"] = t.get("top1_smiles")
            row["step2_halogen_flag"] = bool(t.get("top1_halogen_flag", False))
            row["step2_delta_ppm"] = float(t["delta_ppm"]) if pd.notna(t.get("delta_ppm")) else None
            row["step2_analog_mass"] = bool(t.get("top1_analog_mass_match", True))
            # Noise V1 / WSE slots (fill when server embeddings land)
            row["step2_noise_v1_score"] = None    # server-dependent
            row["step2_wse_score"] = None         # server-dependent
            row["step2_gate_pass"] = (row["step2_cosine_best"] >= HIT
                                      and row["step2_gap"] >= GATE_99_COS)
        else:
            row["step2_cosine_best"] = None
            for k in ("step2_gap", "step2_entropy_best", "step2_top1_ik14",
                      "step2_top1_formula", "step2_top1_smiles",
                      "step2_halogen_flag", "step2_delta_ppm",
                      "step2_analog_mass", "step2_noise_v1_score",
                      "step2_wse_score", "step2_gate_pass"):
                row[k] = None

        # Step 3: local competition + chemical explanation
        # P2b: neutral loss / fragment evidence (from three-tier neighbors)
        if len(t) and pd.notna(t.get("top1_ik14")) and pd.notna(t.get("top2_ik14")):
            row["step3_has_competition"] = bool(t["top2_ik14"])
            row["step3_top2_ik14"] = t.get("top2_ik14")
            row["step3_top2_cosine"] = float(t["top2_cosine"]) if pd.notna(t.get("top2_cosine")) else None
        else:
            row["step3_has_competition"] = False
            row["step3_top2_ik14"] = None
            row["step3_top2_cosine"] = None
        # ChemAware: chemical rules applicable (from three-tier SMILES)
        row["step3_chemaware_rules"] = None    # server-dependent
        row["step3_chemaware_applicable"] = False

        # Step 4: conditional sample evidence (BioAware)
        row["step4_bioaware_events"] = None     # server-dependent
        row["step4_bioaware_applicable"] = False

        # Step 5: tiered output + disposition
        if row["step1_contaminant"]:
            row["step5_tier"] = "DROP_CONTAMINANT"
            row["step5_disposition"] = "blank-positive; exclude from biology"
        elif row.get("step2_gate_pass"):
            if row.get("step2_halogen_flag"):
                row["step5_tier"] = "A_GATE_PASS_FLAGGED"
                row["step5_disposition"] = "drug-exposure check then MSn"
            else:
                row["step5_tier"] = "A_GATE_PASS"
                row["step5_disposition"] = "Level-2 candidate; confirm with standard"
        elif row.get("step2_cosine_best", 0) >= HIT:
            row["step5_tier"] = "B1_CONTESTED"
            row["step5_disposition"] = "high score but small gap; orthogonal evidence"
        elif row.get("step2_cosine_best", 0) >= 0.70:
            row["step5_tier"] = "B2_ANALOG"
            row["step5_disposition"] = "analog-level candidate"
        else:
            row["step5_tier"] = "C_DARK"
            row["step5_disposition"] = "no spectral claim; track as entity"

        # Module applicability ledger (which modules have something to say)
        applicable = []
        if row.get("step2_cosine_best") is not None:
            applicable.append("cosine")
        if row.get("step2_entropy_best") is not None:
            applicable.append("WSE_local")
        if row.get("step3_has_competition"):
            applicable.append("P2b_competition")
        if row.get("step2_top1_smiles") and pd.notna(row.get("step2_top1_smiles")):
            applicable.append("ChemAware_rules_possible")
        row["modules_applicable"] = "; ".join(applicable) if applicable else "none"
        row["modules_count"] = len(applicable)

        rows.append(row)

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "unified_entity_evidence.csv", index=False)

    # Summary
    tier_counts = df["step5_tier"].value_counts().to_dict()
    contam = int(df["step1_contaminant"].sum())
    real = len(df) - contam
    has_effect = int((df["step1_effect_q"].notna() &
                      (df["step1_effect_q"] < 1e-10)).sum())
    gate_pass = int(df["step2_gate_pass"].fillna(False).sum())

    report = {
        "status": "GLM_UNIFIED_ENTITY_EVIDENCE_TABLE",
        "n_entities": len(df),
        "tier_counts": tier_counts,
        "contaminants": contam,
        "real_entities": real,
        "with_phenotype_effect": has_effect,
        "gate_passed": gate_pass,
        "modules_with_evidence": df["modules_count"].value_counts().to_dict(),
        "slots_pending_server": [
            "step2_noise_v1_score", "step2_wse_score",
            "step3_chemaware_rules", "step4_bioaware_events",
        ],
        "pipeline_steps_implemented": [
            "step1: QC/blank/forensics (complete)",
            "step2: cosine + gate (complete); Noise/WSE (server slot)",
            "step3: competition detection (local); ChemAware (server slot)",
            "step4: BioAware (server slot)",
            "step5: tiered disposition (complete)",
        ],
    }
    (OUT / "evidence_table_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1))
    print(f"\nwritten: {OUT / 'unified_entity_evidence.csv'}")

    # Show the table
    show_cols = ["entity_id", "step5_tier", "step2_cosine_best", "step2_gap",
                 "step1_effect_log2fc", "modules_count", "step5_disposition"]
    print("\n", df[show_cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
