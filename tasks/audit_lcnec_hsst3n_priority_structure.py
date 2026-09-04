"""Orthogonal formula and fragment audit for the four priority LCNEC hypotheses."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tasks"))
from audit_e0_observability_residual import greedy_matches, matched_metrics, peaks  # noqa: E402
from encode_mona_neg_library import parse_mgf  # noqa: E402


PRIORITY = {
    "XTWYTFMLZFPYCI": "adenosine_diphosphate_family",
    "SRNWOUGRCWSEMX": "adenosine_diphosphoribose_family",
    "GJAWHXHKYYXBSV": "quinolinate",
    "CIWBSHSKHKDKBQ": "ascorbate",
}
PROTON_MASS = 1.007276466621


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations", type=Path,
        default=ROOT / "data/validation/lcnec_hsst3n_all_robust_annotation/priority_annotation_primary20.csv",
    )
    parser.add_argument(
        "--query-mgf", type=Path,
        default=ROOT / "data/validation/lcnec_hsst3n_all_robust_ms2/priority_dark_modules.mgf",
    )
    parser.add_argument(
        "--library-mgf", type=Path, default=ROOT / "data/models/mona_neg_full.mgf",
    )
    parser.add_argument(
        "--library-manifest", type=Path,
        default=ROOT / "data/models/mona_neg_dreams_emb/manifest.csv",
    )
    parser.add_argument(
        "--top5", type=Path,
        default=ROOT / "data/validation/lcnec_hsst3n_all_robust_annotation/priority_annotation_top5.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/validation/lcnec_hsst3n_priority_structure",
    )
    parser.add_argument("--fragment-tolerance", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotations = pd.read_csv(args.annotations)
    annotations = annotations[
        annotations["p2b_top_ik14"].isin(PRIORITY)
        & annotations["annotation_confidence"].str.contains("consistency", na=False)
    ].copy()
    if set(annotations["p2b_top_ik14"]) != set(PRIORITY):
        raise RuntimeError("the four preregistered priority hypotheses are not all present")
    queries = parse_mgf(args.query_mgf)
    references = parse_mgf(args.library_mgf)
    library = pd.read_csv(args.library_manifest)
    top5 = pd.read_csv(args.top5)
    if len(references) != len(library):
        raise RuntimeError("reference MGF/manifest alignment failure")

    rows = []
    matched_rows = []
    for annotation in annotations.itertuples(index=False):
        qidx = int(annotation.query_index)
        ik14 = str(annotation.p2b_top_ik14)
        selected = top5[
            top5["query_index"].eq(qidx)
            & top5["ppm_window"].eq(20)
            & top5["rank"].eq(1)
        ]
        if len(selected) != 1:
            raise RuntimeError(f"no unique selected reference for query {qidx}")
        ref_index = int(selected.iloc[0]["reference_index"])
        query_mz, query_intensity = peaks(queries[qidx]["peaks"])
        ref_mz, ref_intensity = peaks(references[ref_index]["peaks"])
        matches = greedy_matches(query_mz, ref_mz, args.fragment_tolerance)
        metrics = matched_metrics(
            query_mz, query_intensity, ref_mz, ref_intensity, args.fragment_tolerance,
        )
        for i, j in matches:
            matched_rows.append({
                "family_id": int(annotation.family_id), "ik14": ik14,
                "query_fragment_mz": float(query_mz[i]),
                "reference_fragment_mz": float(ref_mz[j]),
                "mass_difference_da": float(query_mz[i] - ref_mz[j]),
                "query_relative_intensity": float(query_intensity[i] / query_intensity.sum()),
                "reference_relative_intensity": float(ref_intensity[j] / ref_intensity.sum()),
            })

        smiles = str(annotation.p2b_top_smiles)
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise RuntimeError(f"RDKit cannot parse selected SMILES for {ik14}")
        formula = rdMolDescriptors.CalcMolFormula(molecule)
        neutral_mass = float(Descriptors.ExactMolWt(molecule))
        theoretical_mh = neutral_mass - PROTON_MASS
        formula_ppm = abs(float(annotation.target_mz) - theoretical_mh) / theoretical_mh * 1e6
        rows.append({
            "family_id": int(annotation.family_id), "ik14": ik14,
            "priority_name": PRIORITY[ik14],
            "spectral_hypothesis": str(annotation.p2b_top_name),
            "annotation_confidence": str(annotation.annotation_confidence),
            "target_mz": float(annotation.target_mz),
            "target_rt_sec": float(annotation.target_rt_sec),
            "effect_log2fc": float(annotation.effect_log2fc),
            "effect_q": float(annotation.effect_q),
            "formula": formula, "neutral_exact_mass": neutral_mass,
            "theoretical_m_minus_h": theoretical_mh,
            "formula_mass_error_ppm": formula_ppm,
            "dreams_score": float(annotation.dreams_top_score),
            "dreams_margin": float(annotation.dreams_margin),
            "reference_spectra": int(annotation.selected_reference_spectra),
            "full_inchikey_count": int(annotation.selected_full_inchikey_count),
            "query_peaks": len(query_mz), "reference_peaks": len(ref_mz),
            "matched_fragments": len(matches),
            "matched_peak_fraction_min": float(metrics["matched_peak_fraction_min"]),
            "query_intensity_coverage": float(metrics["query_intensity_coverage"]),
            "reference_intensity_coverage": float(metrics["candidate_intensity_coverage"]),
            "sqrt_cosine": float(metrics["sqrt_cosine"]),
            "entropy_similarity": float(metrics["entropy_similarity"]),
            "formula_mass_pass_5ppm": bool(formula_ppm <= 5),
            "fragment_support_pass": bool(
                len(matches) >= 3 and metrics["query_intensity_coverage"] >= 0.5
            ),
        })
    ledger = pd.DataFrame(rows)
    report = {
        "status": "lcnec_hsst3n_priority_structure_audit_complete",
        "formal": True,
        "hypotheses": len(ledger),
        "formula_mass_pass_5ppm": int(ledger["formula_mass_pass_5ppm"].sum()),
        "fragment_support_pass": int(ledger["fragment_support_pass"].sum()),
        "all_formula_mass_pass": bool(ledger["formula_mass_pass_5ppm"].all()),
        "all_fragment_support_pass": bool(ledger["fragment_support_pass"].all()),
        "rows": ledger[[
            "priority_name", "formula", "formula_mass_error_ppm", "dreams_score",
            "matched_fragments", "query_intensity_coverage", "sqrt_cosine",
            "annotation_confidence",
        ]].to_dict("records"),
        "decision": (
            "Passes to manuscript-level hypothesis synthesis only if exact formula mass and direct fragment support pass. "
            "ADP and ADP-ribose remain connectivity-family assignments when multiple full InChIKeys are present."
        ),
        "claim_limit": "Library-spectrum and exact-formula support only; no authentic-standard RT confirmation.",
    }
    report["pass"] = report["all_formula_mass_pass"] and report["all_fragment_support_pass"]
    ledger.to_csv(args.output_dir / "priority_structure_ledger.csv", index=False)
    pd.DataFrame(matched_rows).to_csv(args.output_dir / "matched_fragments.csv", index=False)
    (args.output_dir / "structure_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
