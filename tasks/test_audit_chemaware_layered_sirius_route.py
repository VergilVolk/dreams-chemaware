"""Contracts for the local layered-rule/SIRIUS execution boundary."""
from __future__ import annotations

import argparse
from pathlib import Path

from audit_chemaware_layered_sirius_route import ROOT, build_report


def main() -> None:
    report = build_report(argparse.Namespace(
        rule_corpus=ROOT / "data/validation/chemaware_layered_rule_corpus_v5_20260928",
        static_source=(
            ROOT / "data/validation/chemaware_layered_rule_source_qualified_v9_profiled_collision_20260928"
        ),
        static_capacity=(
            ROOT / "data/validation/chemaware_layered_static_multisource_official_geometry_capacity_v6_profiled_20260928"
        ),
        sirius_panel=(
            ROOT / "data/validation/chemaware_sirius_source_panel_v9_profiled_collision_20260928"
        ),
        output=Path("unused"),
    ))
    assert report["formal_training_authorized"] is False
    assert report["qualified_static_relationships"]["rows"] == 33_574
    assert report["qualified_static_relationships"][
        "incremental_triplets_under_official_geometry_standin"
    ] == 0
    assert all(report["gates"].values())
    print("PASS: ChemAware layered SIRIUS route audit contracts", flush=True)


if __name__ == "__main__":
    main()
