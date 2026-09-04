#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=Path("data/external/GSE178341_mucinous_secretory_audit/independent_proteomics_fixed_panel_v1/result.json"))
    args = parser.parse_args()
    report = json.loads(args.result.read_text(encoding="utf-8"))
    if report["status"] != "independent_mucinous_crc_proteomics_fixed_panel_complete" or not report["formal"]:
        raise RuntimeError("result status/formal gate failed")
    if report["cohort"] != {"Normal colon": 16, "Adenocarcinoma not otherwise specified": 15, "Mucinous adenocarcinoma": 15}:
        raise RuntimeError("cohort counts changed")
    if len(report["proteins"]) != 8 or len(report["modules"]) != 2:
        raise RuntimeError("fixed panel/module count changed")
    if set(report["prespecified_unavailable"]) != {"NXPE1", "SPDEF", "SLC35A1", "CASD1"}:
        raise RuntimeError("unavailable fixed panel changed")
    for protein in report["proteins"]:
        if protein["n_mc"] != 15 or protein["n_ac"] != 15:
            raise RuntimeError(f"patient count changed for {protein['name']}")
        if not (0 <= protein["permutation_p"] <= 1 and 0 <= protein["permutation_bh_q_across_8"] <= 1):
            raise RuntimeError(f"invalid p/q for {protein['name']}")
    print(f"[validate] PASS proteins={len(report['proteins'])} modules={len(report['modules'])}")


if __name__ == "__main__":
    main()
