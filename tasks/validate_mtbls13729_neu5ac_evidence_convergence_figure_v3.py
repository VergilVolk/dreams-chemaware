#!/usr/bin/env python3
"""Validate the integrated Neu5Ac evidence-convergence figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "data/mtbls13729/neu5ac_evidence_convergence_figure_v3_final"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", nargs="?", type=Path, default=DEFAULT)
    args = parser.parse_args()
    required = {
        "neu5ac_evidence_convergence_figure_v3.png",
        "neu5ac_evidence_convergence_figure_v3.pdf",
        "same_patient_donor_deltas.csv",
        "independent_transcript_composition_models.csv",
        "independent_proteomics_context.csv",
        "report.json",
        "README.md",
    }
    missing = sorted(name for name in required if not (args.output_dir / name).is_file())
    if missing:
        raise RuntimeError(f"missing figure outputs: {missing}")
    report = json.loads((args.output_dir / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "mtbls13729_neu5ac_evidence_convergence_figure_v3_complete":
        raise RuntimeError("status mismatch")
    if report.get("formal") is not False or "not a causal" not in report.get("figure_role", ""):
        raise RuntimeError("figure role boundary missing")
    donor = pd.read_csv(args.output_dir / "same_patient_donor_deltas.csv")
    if donor["patient"].nunique() != 10 or set(donor["node"]) != {"free_neu5ac", "cmp_neu5ac", "udp_glcnac"}:
        raise RuntimeError("same-patient donor panel changed")
    transcript = pd.read_csv(args.output_dir / "independent_transcript_composition_models.csv")
    if len(transcript) != 21 or transcript["endpoint"].nunique() != 7:
        raise RuntimeError("transcript model panel changed")
    protein = pd.read_csv(args.output_dir / "independent_proteomics_context.csv")
    if set(protein["name"]) != {"AGR2", "GNE", "NANS", "CMAS", "SIAE"}:
        raise RuntimeError("protein panel changed")
    if "does not establish" not in report.get("claim_limit", ""):
        raise RuntimeError("claim boundary missing")
    print("[validate_mtbls13729_neu5ac_evidence_convergence_figure_v3] PASS")


if __name__ == "__main__":
    main()
