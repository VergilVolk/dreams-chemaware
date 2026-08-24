"""Audit agreement between frozen DreaMS and classical MS2 retrieval.

This is deliberately an *external, study-specific calibration audit*, not a
new model and not a claim that agreement proves a structure.  Both source
tables must have been evaluated against the identical mass-constrained
reference library and the identical published connectivity truth panel.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


KEY = ["metabolite", "query_source_file", "query_spectrum_id", "truth_ik14"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dreams", type=Path, default=Path("data/external/MTBLS1905/evaluation/official_dreams_blind_retrieval.tsv"))
    parser.add_argument("--classical", type=Path, default=Path("data/external/MTBLS1905/evaluation/classical_cosine_blind_retrieval.tsv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/external/MTBLS1905/evaluation"))
    args = parser.parse_args()

    dreams = pd.read_csv(args.dreams, sep="\t")
    classical = pd.read_csv(args.classical, sep="\t")
    merged = dreams.merge(classical, on=KEY, validate="one_to_one", suffixes=("_dreams", "_classical"))
    merged["top1_agree_connectivity"] = merged["top1_ik14_dreams"] == merged["top1_ik14_classical"]
    merged["both_top1_correct"] = merged["top1_correct_connectivity_dreams"] & merged["top1_correct_connectivity_classical"]
    merged["dreams_only_correct"] = merged["top1_correct_connectivity_dreams"] & ~merged["top1_correct_connectivity_classical"]
    merged["classical_only_correct"] = ~merged["top1_correct_connectivity_dreams"] & merged["top1_correct_connectivity_classical"]
    merged["both_wrong"] = ~merged["top1_correct_connectivity_dreams"] & ~merged["top1_correct_connectivity_classical"]

    audit = (merged.groupby("top1_agree_connectivity", dropna=False)
        .agg(
            spectra=("metabolite", "size"),
            dreams_top1_accuracy=("top1_correct_connectivity_dreams", "mean"),
            classical_top1_accuracy=("top1_correct_connectivity_classical", "mean"),
            both_correct=("both_top1_correct", "sum"),
            dreams_only_correct=("dreams_only_correct", "sum"),
            classical_only_correct=("classical_only_correct", "sum"),
            both_wrong=("both_wrong", "sum"),
        ).reset_index())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_dir / "official_dreams_classical_consensus_per_spectrum.tsv", sep="\t", index=False)
    audit.to_csv(args.out_dir / "official_dreams_classical_consensus_audit.tsv", sep="\t", index=False)
    payload = {
        "study": "MTBLS1905",
        "purpose": "external blinded retrieval calibration; not structure confirmation",
        "shared_protocol": "36 QC-DDA query spectra, 18 published connectivity targets, identical positive-mode reference library restricted to precursor mass +/-10 ppm",
        "total_spectra": int(len(merged)),
        "agreement_spectra": int(merged["top1_agree_connectivity"].sum()),
        "agreement_both_correct": int(merged.loc[merged["top1_agree_connectivity"], "both_top1_correct"].sum()),
        "disagreement_spectra": int((~merged["top1_agree_connectivity"]).sum()),
        "dreams_only_correct": int(merged["dreams_only_correct"].sum()),
        "classical_only_correct": int(merged["classical_only_correct"].sum()),
        "limitation": "The panel is small and used for calibration. It cannot establish universal precision or be used to tune on and then claim independent performance.",
    }
    (args.out_dir / "official_dreams_classical_consensus_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
