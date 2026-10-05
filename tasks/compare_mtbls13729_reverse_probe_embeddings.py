"""Pair official and experimental reverse-probe results without outcome tuning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--experimental-dir", type=Path, required=True)
    parser.add_argument("--experimental-method", default="e6_fixed_v2_sw2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    panels = {}
    for panel in ("neg_rp", "pos_rp"):
        official = pd.read_csv(args.official_dir / f"{panel}__official_dreams__identity_sample.csv.gz")
        experimental = pd.read_csv(
            args.experimental_dir / f"{panel}__{args.experimental_method}__identity_sample.csv.gz"
        )
        keys = ["panel", "ik14", "sample_id"]
        paired = official.merge(
            experimental, on=keys, suffixes=("_official", "_experimental"), validate="one_to_one"
        )
        calibration = paired[
            paired.calibration_panel_official
        ]
        corrected = int((~calibration.calibration_feature_match_official & calibration.calibration_feature_match_experimental).sum())
        introduced = int((calibration.calibration_feature_match_official & ~calibration.calibration_feature_match_experimental).sum())
        discordant = corrected + introduced
        pvalue = float(binomtest(min(corrected, introduced), discordant, 0.5).pvalue) if discordant else 1.0
        panels[panel] = {
            "paired_calibration_pairs": int(len(calibration)),
            "official_ms1_link_fraction": float(calibration.linked_feature_id_official.notna().mean()),
            "experimental_ms1_link_fraction": float(calibration.linked_feature_id_experimental.notna().mean()),
            "official_feature_match_fraction": float(calibration.calibration_feature_match_official.mean()),
            "experimental_feature_match_fraction": float(calibration.calibration_feature_match_experimental.mean()),
            "delta": float(
                calibration.calibration_feature_match_experimental.mean()
                - calibration.calibration_feature_match_official.mean()
            ),
            "corrected": corrected,
            "introduced": introduced,
            "mcnemar_exact_p": pvalue,
        }
        paired.to_csv(args.output.with_name(f"{args.output.stem}__{panel}.csv.gz"), index=False, compression="gzip")
    report = {
        "status": "mtbls13729_reverse_probe_embedding_comparison_complete",
        "formal": False,
        "experimental_method": args.experimental_method,
        "panels": panels,
        "pass": all(value["delta"] >= 0 and value["corrected"] >= value["introduced"] for value in panels.values()),
        "claim_limit": "same-cohort Level-1 feature recovery calibration; not new-metabolite truth or phenotype validation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
