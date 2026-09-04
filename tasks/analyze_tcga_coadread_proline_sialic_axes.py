#!/usr/bin/env python
"""Run the frozen TCGA analysis engine on proline and sialic-acid axes."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import analyze_tcga_coadread_mucinous_axes as engine


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
    "secretory_mucin_program": ["MUC2", "TFF3", "SPDEF", "AGR2", "FCGBP"],
    "collagen_proline_context": ["COL1A1", "COL1A2", "P4HA1", "P4HA2", "P4HB"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    output = Path("data/external/TCGA_COADREAD_Xena_20260830/proline_sialic_axes_v1")
    engine.AXES = AXES
    sys.argv = [sys.argv[0], "--output-dir", str(output)]
    engine.main()
    summary_path = output / "summary.json"
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    report["status"] = "tcga_coadread_proline_sialic_axes_complete"
    report["analysis_definition"] = {
        "axes": AXES,
        "primary_endpoints": [
            "paired primary-tumour minus adjacent-normal change",
            "mucinous minus conventional primary-tumour contrast adjusted for side, stage, age and sex",
        ],
        "interpretation_boundary": (
            "Bulk RNA context does not establish cell of origin, metabolite identity, glycan linkage, "
            "enzyme activity, or metabolic flux."
        ),
    }
    report["provenance"]["targeted_wrapper_sha256"] = sha256(Path(__file__))
    summary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "samples": report["samples"],
        "paired_patients": report["paired_tumor_normal"]["patients"], "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
