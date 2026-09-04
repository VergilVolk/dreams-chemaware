#!/usr/bin/env python
"""Descriptively localize proline and sialic-acid programs in GSE236697."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import analyze_gse236697_spatial_metabolic_axes as engine


AXES = {
    "proline_synthesis": ["ALDH18A1", "PYCR1", "PYCR2", "PYCRL", "OAT"],
    "proline_catabolism": ["PRODH", "ALDH4A1"],
    "glutamate_supply": ["GLS", "GLS2", "GLUD1", "GLUD2", "GOT1", "GOT2"],
    "sialic_acid_synthesis_transport": ["GNE", "NANS", "NANP", "CMAS", "SLC35A1"],
    "sialic_acid_remodeling": ["NEU1", "NEU2", "NEU3", "NEU4", "SIAE", "CASD1"],
    "mucin_sialylation": [
        "ST3GAL1", "ST3GAL2", "ST3GAL3", "ST3GAL4", "ST6GAL1",
        "ST6GALNAC1", "ST6GALNAC2", "ST6GALNAC4",
    ],
    "secretory_mucin_program": ["MUC2", "MUC5AC", "TFF3", "SPDEF", "AGR2", "FCGBP"],
    "collagen_proline_context": ["COL1A1", "COL1A2", "P4HA1", "P4HA2", "P4HB"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    output = Path("data/external/GSE236697/spatial_proline_sialic_v1")
    engine.AXES = AXES
    engine.SINGLE_GENES = ["MUC1", "MUC2", "MUC5AC", "PYCR1", "GNE", "CMAS", "ST6GAL1", "NEU1"]
    sys.argv = [sys.argv[0], "--output-dir", str(output)]
    engine.main()
    report_path = output / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["status"] = "gse236697_spatial_proline_sialic_complete"
    report["analysis_definition"] = {
        "axes": AXES,
        "interpretation_boundary": (
            "One paired case provides spatial localization, not population replication. "
            "Spot correlations do not establish metabolic flux or glycan structure."
        ),
    }
    report["provenance"]["targeted_wrapper"] = sha256(Path(__file__))
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
