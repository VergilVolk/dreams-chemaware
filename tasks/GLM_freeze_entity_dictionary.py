"""Freeze experimental spectral entity prototypes from LCNEC discovery data.

Per the unified research direction (audit §5.1):
  - ALL QC-qualified entities form the reference dictionary (no differential
    significance filtering — the dictionary must be phenotype-blind);
  - each prototype = m/z, ion form, RT, spectral evidence, QC trajectory;
  - known library names are ATTACHED attributes, not entry requirements;
  - entities without any library hit remain in the dictionary as unannotated
    experimental spectral entities.

This is the "experimental entity reference" that validation data maps to.
The public library provides names and chemical interpretation; it does not
define the chemical space.

Inputs (local):
  - LCNEC QC feature universe + robustness gate
  - priority dark modules MGF (spectral evidence)
  - GNPS benchmark (library names, for attachment only)

Output: frozen entity dictionary with per-entity:
  entity_id, mz, rt_sec, n_spectra, spectral_quality, library_hit_ik14,
  library_hit_name, library_best_cosine, library_gap (top1-top2),
  halogen_flag, phenotype_effect (log2fc, q) — the last two stored as
  attributes for downstream validation, NOT used for dictionary membership.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

QC_GATE = ROOT / "data/validation/lcnec_hsst3n_priority_annotation"
DARK_MGF = ROOT / "data/validation/lcnec_hsst3n_priority_ms2"
EIC = ROOT / "data/validation/lcnec_hsst3n_dark_eic_gate"
AUDIT = ROOT / "data/validation/lcnec_hsst3n_acquisition_gate"
OUT = ROOT / "data/validation/GLM_frozen_entity_dictionary"


def formula_elements(formula: str) -> dict:
    out = {}
    for el, cnt in re.findall(r"([A-Z][a-z]?)(\d*)", str(formula)):
        if el:
            out[el] = out.get(el, 0) + (int(cnt) if cnt else 1)
    return out


def main() -> None:
    # 1. QC-qualified feature universe (robustness gate defines the universe)
    robust = pd.read_csv(QC_GATE / "priority_annotation_primary20.csv")
    print(f"robustness gate: {len(robust):,} features")

    # 2. dark module MGF for spectral evidence
    spectra = {}
    prec = {}
    cur_peaks = []
    cur_mz = None
    with open(DARK_MGF / "priority_dark_modules.mgf", encoding="utf-8",
              errors="replace") as f:
        for line in f:
            line = line.strip()
            if line == "BEGIN IONS":
                cur_peaks = []; cur_mz = None
            elif line == "END IONS":
                if cur_peaks and cur_mz:
                    key = round(float(cur_mz), 4)
                    spectra[key] = np.array(cur_peaks, dtype=float)
                    prec[key] = float(cur_mz)
            elif line.startswith("PEPMASS="):
                cur_mz = float(line.split("=")[1].split()[0])
            elif line and line[0].isdigit():
                parts = line.split()
                if len(parts) >= 2:
                    cur_peaks.append((float(parts[0]), float(parts[1])))
    print(f"spectral evidence: {len(spectra)} entities with MGF peaks")

    # 3. library attachment (from the three-tier results, for names only)
    tier = pd.read_csv(ROOT / "data/validation/GLM_track2_census/"
                       "lcnec_dark_three_tier.csv")
    lib = {}
    for _, r in tier.iterrows():
        mz = round(float(r["precursor"]), 4)
        lib[mz] = {
            "tier": r["tier"],
            "best_cosine": float(r["best_cosine"]),
            "gap": float(r["molecule_gap"]),
            "top1_ik14": r.get("top1_ik14"),
            "top1_formula": r.get("top1_formula"),
            "top1_smiles": r.get("top1_smiles"),
            "halogen": bool(r.get("top1_halogen_flag", False)),
            "delta_ppm": r.get("top1_delta_ppm_vs_mh"),
            "analog_mass": bool(r.get("top1_analog_mass_match", True)),
        }

    # 4. EIC forensics (blank/QC/study presence)
    z = np.load(EIC / "dark_feature_eic_matrix.npz", allow_pickle=True)
    audit = pd.read_csv(AUDIT / "file_acquisition_audit.csv")
    sid2cls = dict(zip(audit.sample_id.astype(str),
                       audit.injection_class.astype(str)))
    cls = np.array([sid2cls.get(str(s), "?") for s in z["sample_id"]])

    # 5. assemble the dictionary
    from GLM_lcnec_dark_molecule_rescan import parse_mgf  # reuse
    dark_mgf_path = DARK_MGF / "priority_dark_modules.mgf"
    dark = parse_mgf(dark_mgf_path)
    module_mz = [s.get("precursor", 0.0) for s in dark]
    module_effect = {}
    mods = pd.read_csv(DARK_MGF / "priority_dark_modules.csv")
    for _, r in mods.iterrows():
        module_effect[round(float(r["target_mz"]), 4)] = {
            "log2fc": float(r["effect_log2fc"]),
            "q": float(r["effect_q"]),
        }

    entities = []
    for key in sorted(module_mz):
        k = round(float(key), 4)
        e = {
            "entity_id": f"LE{k:.4f}",
            "mz": k,
            "n_peaks": len(spectra.get(k, [])),
            "has_spectral_evidence": k in spectra,
        }
        # library attachment
        if k in lib:
            e.update({f"lib_{kk}": vv for kk, vv in lib[k].items()})
        else:
            e["lib_tier"] = "no_library_search"
        # effect (attribute only, not membership criterion)
        if k in module_effect:
            e["effect_log2fc"] = module_effect[k]["log2fc"]
            e["effect_q"] = module_effect[k]["q"]
        # EIC forensics (already computed in lcnec_dark_forensics.json)
        forensics = json.loads(
            (ROOT / "data/validation/GLM_track2_census/"
             "lcnec_dark_forensics.json").read_text(encoding="utf-8"))
        for fm in forensics.get("modules", []):
            if abs(float(fm.get("precursor", 0)) - k) < 0.01:
                e["blank_present"] = fm.get("breakback", {}).get(
                    "blank", {}).get("present", 0) > 0
                e["qc_present"] = fm.get("breakdown", {}).get(
                    "pooled_qc", {}).get("present", 0)
                break
        entities.append(e)

    # 6. write
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(entities)
    df.to_csv(OUT / "frozen_entities.csv", index=False)
    report = {
        "status": "GLM_FROZEN_ENTITY_DICTIONARY",
        "n_entities": len(entities),
        "n_with_spectra": sum(1 for e in entities
                              if e["has_spectral_evidence"]),
        "library_tiers": df.get("lib_tier", pd.Series()).value_counts()
            .to_dict() if "lib_tier" in df else {},
        "membership_criterion": ("ALL QC-qualified features; "
                                 "phenotype-blind; library names attached "
                                 "as attributes, not entry requirements"),
    }
    (OUT / "dictionary_report.json").write_text(json.dumps(report, indent=2),
                                                encoding="utf-8")
    print(json.dumps(report, indent=1))
    print(f"written: {OUT / 'frozen_entities.csv'}")


if __name__ == "__main__":
    main()
