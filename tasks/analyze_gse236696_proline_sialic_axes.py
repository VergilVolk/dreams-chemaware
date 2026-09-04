#!/usr/bin/env python
"""Targeted lineage-resolved audit of proline and sialic-acid biology.

This reuses the frozen GSE236696 patient-paired pseudobulk and conservative
lineage-gating implementation.  Only the prespecified gene sets differ.  The
analysis is contextual evidence for metabolite hypotheses, not a test of
metabolic flux or glycan structure.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import analyze_gse236696_mucinous_axes_by_lineage as engine


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
    output = Path("data/external/GSE236696/proline_sialic_by_lineage_v1")
    engine.AXES = AXES
    engine.TARGET_GENES = sorted({gene for genes in AXES.values() for gene in genes})
    sys.argv = [sys.argv[0], "--output-dir", str(output)]
    engine.main()

    summary_path = output / "summary.json"
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    report["status"] = "gse236696_proline_sialic_by_lineage_complete"
    report["analysis_definition"] = {
        "axes": AXES,
        "primary_question": (
            "Which broad cell lineage carries paired tumour-normal expression changes "
            "consistent with the orthogonally recovered proline/glutamate and Neu5Ac signals?"
        ),
        "interpretation_boundary": (
            "Gene-expression concordance supports cellular context only. It does not infer "
            "metabolic flux, free Neu5Ac origin, glycan linkage, or net tissue sialylation."
        ),
    }
    report["provenance"]["targeted_wrapper_sha256"] = sha256(Path(__file__))
    summary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "patients": report["patients"],
        "axis_results": len(report["axis_results"]),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
