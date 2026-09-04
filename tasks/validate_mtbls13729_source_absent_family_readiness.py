"""Validate the frozen source-table-absent family readiness package."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/mtbls13729/source_absent_family_readiness_v1"


def main() -> None:
    required = [
        OUT / "report.json",
        OUT / "REPORT.md",
        OUT / "source_absent_family_readiness.csv",
        OUT / "module_readiness.csv",
        OUT / "source_absent_family_readiness.png",
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"missing or empty outputs: {missing}")

    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    data = pd.read_csv(OUT / "source_absent_family_readiness.csv")
    modules = pd.read_csv(OUT / "module_readiness.csv")
    if set(data.feature_id) != {150, 1597, 1717, 3019, 3222}:
        raise RuntimeError("unexpected feature set")
    if len(modules) != 3:
        raise RuntimeError("related ions were not collapsed to exactly three modules")
    if report["exact_metabolite_claims"] != 0 or data.exact_metabolite_claim_permitted.any():
        raise RuntimeError("an unsupported exact metabolite claim escaped the gate")
    if report["full_untargeted_exact_fdr10_pass"] != 0:
        raise RuntimeError("source-absent candidates unexpectedly pass full-space FDR10")
    if report["abundance_protocol_discordant_features"] != [1717]:
        raise RuntimeError("abundance-protocol discrepancy audit changed")
    if report["unresolved_abundance_protocol_discordance"] != []:
        raise RuntimeError("a candidate abundance discrepancy remains unresolved")
    if data.loc[data.feature_id.eq(1717), "manuscript_placement"].item() != "SECONDARY_FAMILY_HYPOTHESIS":
        raise RuntimeError("feature 1717 placement changed")
    print(json.dumps({"status": "mtbls13729_source_absent_family_readiness_validation_passed", "features": len(data), "modules": len(modules)}, indent=2))


if __name__ == "__main__":
    main()
