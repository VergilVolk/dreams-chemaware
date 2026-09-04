"""Audit algorithmic value on the frozen MTBLS13729 biology candidates.

Published source-table Level-1 and Level-2 assignments are reported separately.
Level-1 concordance is the stronger reference; Level-2 is never promoted to
ground truth.  Family-like candidates without source InChIKeys are descriptive
only.  The script consumes frozen three-way outputs and never refits a model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = {
    "official_dreams": "dreams",
    "noise_shared_embedding_e6": "e6_fixed_v2_sw2",
    "frozen_p2b": "p2b",
}


def normalized_ik14(value) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    return text[:14] if len(text) >= 14 else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=Path("data/mtbls13729/integrated_biology_ledger_v1/integrated_candidate_ledger.csv"))
    parser.add_argument("--threeway-dir", type=Path, default=Path("data/mtbls13729/threeway_application_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/mtbls13729/integrated_candidate_algorithm_audit_v1"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger = pd.read_csv(args.ledger)
    frames = []
    for panel in ("neg_rp", "pos_rp"):
        path = args.threeway_dir / f"{panel}__threeway_features.csv.gz"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if frame["feature_id"].duplicated().any():
            raise RuntimeError(f"{panel}: duplicate feature_id in frozen three-way table")
        frame.insert(0, "audit_panel", panel)
        frames.append(frame)
    threeway = pd.concat(frames, ignore_index=True)

    candidates = ledger[ledger["discovery_panel"].isin(["neg_rp", "pos_rp"])].copy()
    merged = candidates.merge(
        threeway,
        left_on=["discovery_panel", "feature_id"],
        right_on=["audit_panel", "feature_id"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        missing = merged.loc[merged["_merge"].ne("both"), ["discovery_panel", "feature_id", "label"]]
        raise RuntimeError(f"integrated candidates missing from frozen candidate graph:\n{missing.to_string(index=False)}")
    merged.drop(columns="_merge", inplace=True)
    merged["source_ik14"] = merged["source_inchikey"].map(normalized_ik14)
    merged["source_reference_tier"] = np.select(
        [
            merged["published_source_msi"].eq("Level 1") & merged["source_ik14"].str.len().eq(14),
            merged["published_source_msi"].eq("Level 2") & merged["source_ik14"].str.len().eq(14),
        ],
        ["published_level1_reference", "published_level2_concordance_only"],
        default="family_candidate_no_identity_reference",
    )

    for method, prefix in METHODS.items():
        ik_col = f"{prefix}_ik14"
        if ik_col not in merged:
            raise RuntimeError(f"missing method column: {ik_col}")
        predicted = merged[ik_col].map(normalized_ik14)
        merged[f"{method}_source_relation"] = np.select(
            [
                merged["source_ik14"].str.len().eq(14) & predicted.eq(merged["source_ik14"]),
                merged["source_ik14"].str.len().eq(14) & predicted.str.len().eq(14),
                merged["source_ik14"].str.len().eq(14) & predicted.str.len().ne(14),
            ],
            ["source_concordant", "alternative_identity", "abstained"],
            default="no_identity_reference",
        )
        merged[f"{method}_changed_from_official"] = (
            method != "official_dreams"
        ) and predicted.ne(merged["dreams_ik14"].map(normalized_ik14))

    merged.to_csv(args.output_dir / "candidate_level_algorithm_audit.csv", index=False)

    method_reports = {}
    for method in METHODS:
        relation = f"{method}_source_relation"
        level1 = merged[merged["source_reference_tier"].eq("published_level1_reference")]
        level2 = merged[merged["source_reference_tier"].eq("published_level2_concordance_only")]
        method_reports[method] = {
            "level1_candidates": int(len(level1)),
            "level1_source_concordant": int(level1[relation].eq("source_concordant").sum()),
            "level1_alternative": int(level1[relation].eq("alternative_identity").sum()),
            "level1_abstained": int(level1[relation].eq("abstained").sum()),
            "level2_candidates": int(len(level2)),
            "level2_source_concordant": int(level2[relation].eq("source_concordant").sum()),
            "level2_alternative": int(level2[relation].eq("alternative_identity").sum()),
            "level2_abstained": int(level2[relation].eq("abstained").sum()),
        }

    report = {
        "status": "mtbls13729_integrated_candidate_algorithm_audit_complete",
        "formal": False,
        "candidate_rows": int(len(merged)),
        "published_level1_references": int(merged["source_reference_tier"].eq("published_level1_reference").sum()),
        "published_level2_concordance_rows": int(merged["source_reference_tier"].eq("published_level2_concordance_only").sum()),
        "family_candidates_without_identity_reference": int(merged["source_reference_tier"].eq("family_candidate_no_identity_reference").sum()),
        "methods": method_reports,
        "claim_limit": (
            "This is a small, same-cohort candidate audit. Published Level-1 rows are stronger references but were "
            "not newly re-injected by this project; Level-2 rows are concordance only. Family candidates have no "
            "identity truth. Method changes are not corrections outside these source-reference rows."
        ),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
