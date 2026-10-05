#!/usr/bin/env python
"""Fail-closed validation for the B47 current-context readiness closure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT
        / "data/validation/bioaware_b47_current_context_readiness_20260913_v1/report.json",
    )
    args = parser.parse_args()
    with args.report.open("r", encoding="utf-8") as handle:
        report = json.load(handle)

    assert report["status"] == "bioaware_b47_current_context_readiness_complete"
    assert report["formal"] is True
    assert report["models_fitted"] is False
    assert report["sealed_outcomes_read"] is False
    assert report["prospective_unknown_sources"] == 0
    assert report["event_specific_evidence"] == {
        "coabundance_rows": 0,
        "spectral_rows": 0,
    }
    assert report["sample_local_counterfactual_audit"]["fraction"] == 1.0
    assert report["current_assets_ready_for_prospective_context_claim"] is False
    assert not any(report["gates"].values())
    assert report["decision"]["fit_another_model_on_current_context_ledgers"] is False
    assert report["decision"]["distil_current_context_actions_into_embedding"] is False
    assert len(report["provenance"]) == 5
    assert all(len(item["sha256"]) == 64 for item in report["provenance"].values())
    print(
        "[validate_bioaware_b47_current_context_readiness] PASS; "
        "current assets are formally closed for prospective-context claims"
    )


if __name__ == "__main__":
    main()
