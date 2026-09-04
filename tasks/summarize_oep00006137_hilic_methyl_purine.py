#!/usr/bin/env python
"""Summarize frozen raw HILIC methyl-donor/purine evidence and censoring bounds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from summarize_oep00006137_modified_guanosine_sensitivity import (
    published_effects,
    sensitivity_views,
    sha256sum,
)


TARGETS = {
    "M150T308": "Methionine",
    "M282T290": "Guanosine",
    "M385T405": "S-Adenosylhomocysteine",
    "M399T41": "S-Adenosylmethionine",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--primary-dir",
        type=Path,
        default=Path(
            "data/external/OEP00006137_raw/hilic_methyl_purine_raw_reextraction_v1"
        ),
    )
    parser.add_argument(
        "--supplement",
        type=Path,
        default=Path(
            "data/external/OEP00006137_support/modified_guanosine_level1_rows.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/external/OEP00006137_raw/hilic_methyl_purine_summary_v1"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eic_path = args.primary_dir / "target_eic.csv.gz"
    summary_path = args.primary_dir / "summary.json"
    frame = pd.read_csv(eic_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    supplement = json.loads(args.supplement.read_text(encoding="utf-8"))
    results = {}
    for target_id, name in TARGETS.items():
        target = frame.loc[
            frame["target_id"].eq(target_id) & frame["subtype"].notna()
        ].copy()
        results[target_id] = {
            "assignment": name,
            "raw_detected_biological_samples": int((target["area"] > 0).sum()),
            "published_vs_raw": summary["published_vs_reextracted"][target_id],
            "raw_censoring_sensitivity": {
                subtype: sensitivity_views(target, subtype)
                for subtype in ("MSI-H", "MSS")
            },
            "published_supplement": {
                subtype: published_effects(supplement, target_id, subtype)
                for subtype in ("MSI-H", "MSS")
            },
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "OEP00006137_hilic_methyl_purine_summary_complete",
        "formal": True,
        "targets": results,
        "decision": {
            "SAH": (
                "raw-reproducible strong increase in both MSI-H and MSS; supports a perturbed "
                "methyl-donor-product environment but not methylation flux"
            ),
            "SAM": (
                "not raw-reproducible under the frozen peak coordinate; do not compute or claim "
                "a raw SAM/SAH ratio"
            ),
            "methionine_and_guanosine": (
                "raw-reproducible peak quantification without a stable paired tumor direction"
            ),
        },
        "provenance": {
            "primary_summary_sha256": sha256sum(summary_path),
            "primary_eic_sha256": sha256sum(eic_path),
            "supplement_sha256": sha256sum(args.supplement),
        },
        "claim_limit": (
            "SAH abundance is a static pool-size observation. It does not establish SAM-cycle "
            "flux, methyltransferase activity, or causal production of modified guanosines."
        ),
    }
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
