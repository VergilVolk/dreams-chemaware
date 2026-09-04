#!/usr/bin/env python
"""Pooled independent mucinous-CRC proteomic context for proline/sialic axes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import analyze_mucinous_crc_proteomic_axes as engine
from analyze_tcga_coadread_proline_sialic_axes import AXES


def main() -> None:
    output = Path("data/external/mucinous_crc_proteomics_2021/proline_sialic_reanalysis_v1")
    engine.AXES = AXES
    engine.REQUIRED = {"PYCR1", "GNE", "NANS", "CMAS", "MUC2", "COL1A1"}
    sys.argv = [sys.argv[0], "--output-dir", str(output)]
    engine.main()
    report_path = output / "summary.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["status"] = "mucinous_crc_proline_sialic_proteomic_reanalysis_complete"
    report["analysis_definition"] = {
        "axes": AXES,
        "primary_readout": "descriptive pooled LMC/RMC versus normal-colon protein ratios",
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
