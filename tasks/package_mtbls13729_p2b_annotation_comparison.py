"""Create a truth-aware annotation comparison package for MTBLS13729.

The package separates the validated engineering benchmark from the unlabeled
cohort application.  Application changes are prioritized for manual/chemical
validation using multi-sample consensus and official-DreaMS similarity; they
are never called corrections without an external identity reference.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


TIER_ORDER = {
    "unassigned": 0,
    "Level 3-candidate": 1,
    "Level 2a-single/ambiguous": 2,
    "Level 2a-supported": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/mtbls13729/p2b_application_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/mtbls13729/p2b_application_v1/comparison"))
    parser.add_argument("--panels", nargs="+", default=["neg_rp", "pos_rp"])
    parser.add_argument("--anchor-panel", default="pos_rp")
    parser.add_argument("--targets-dir", type=Path, default=Path("data/mtbls13729/ms1_consensus"))
    parser.add_argument("--anchor-mz", type=float, default=448.339483)
    parser.add_argument("--anchor-rt-sec", type=float, default=630.436)
    parser.add_argument("--anchor-ppm", type=float, default=5.0)
    parser.add_argument("--anchor-rt-tolerance", type=float, default=15.0)
    return parser.parse_args()


def compare_panel(input_dir: Path, panel: str) -> tuple[pd.DataFrame, dict]:
    dreams_path = input_dir / f"{panel}__dreams_feature_annotations.csv.gz"
    p2b_path = input_dir / f"{panel}__p2b_feature_annotations.csv.gz"
    query_path = input_dir / f"{panel}__per_query.csv.gz"
    for path in (dreams_path, p2b_path, query_path):
        if not path.exists():
            raise FileNotFoundError(path)
    dreams = pd.read_csv(dreams_path).add_prefix("dreams_")
    p2b = pd.read_csv(p2b_path).add_prefix("p2b_")
    frame = dreams.merge(
        p2b, left_on="dreams_feature_id", right_on="p2b_feature_id", how="outer", validate="one_to_one"
    )
    frame["feature_id"] = frame.dreams_feature_id.fillna(frame.p2b_feature_id).astype(int)
    dkey = frame.dreams_ik14.fillna("").astype(str)
    pkey = frame.p2b_ik14.fillna("").astype(str)
    frame["transition"] = np.select(
        [
            (dkey.str.len() == 14) & (pkey.str.len() == 14) & (dkey == pkey),
            (dkey.str.len() == 14) & (pkey.str.len() == 14) & (dkey != pkey),
            (dkey.str.len() != 14) & (pkey.str.len() == 14),
            (dkey.str.len() == 14) & (pkey.str.len() != 14),
        ],
        ["retained", "changed", "p2b_only", "dreams_only"],
        default="both_abstained",
    )
    frame["dreams_tier_order"] = frame.dreams_annotation_evidence_tier.map(TIER_ORDER).fillna(0).astype(int)
    frame["p2b_tier_order"] = frame.p2b_annotation_evidence_tier.map(TIER_ORDER).fillna(0).astype(int)
    frame["tier_delta"] = frame.p2b_tier_order - frame.dreams_tier_order
    frame["agreement_delta"] = frame.p2b_agreement_fraction.fillna(0) - frame.dreams_agreement_fraction.fillna(0)
    frame["strong_p2b_candidate"] = (
        (frame.p2b_maximum_dreams_similarity >= 0.8)
        & (frame.p2b_n_support_spectra >= 2)
        & (frame.p2b_agreement_fraction >= 0.6)
    )
    frame["review_priority"] = np.select(
        [
            frame.transition.isin(["changed", "p2b_only"]) & frame.strong_p2b_candidate & (frame.tier_delta >= 0),
            frame.transition.eq("changed") & frame.strong_p2b_candidate,
            frame.transition.eq("changed"),
        ],
        ["A_new_strong_consensus", "B_strong_conflict", "C_changed_needs_validation"],
        default="D_stable_or_low_priority",
    )
    queries = pd.read_csv(query_path, usecols=["feature_id", "decision", "p2b_intervened"])
    report = {
        "panel": panel,
        "features": int(len(frame)),
        "transitions": {str(k): int(v) for k, v in frame.transition.value_counts().items()},
        "tier_gained": int((frame.tier_delta > 0).sum()),
        "tier_lost": int((frame.tier_delta < 0).sum()),
        "agreement_improved": int((frame.agreement_delta > 0).sum()),
        "agreement_worsened": int((frame.agreement_delta < 0).sum()),
        "strong_changed_candidates": int((frame.transition.eq("changed") & frame.strong_p2b_candidate).sum()),
        "query_level_intervention_rate": float(queries.p2b_intervened.mean()),
        "query_level_decisions": {str(k): int(v) for k, v in queries.decision.value_counts().items()},
    }
    return frame, report


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty comparison directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    all_priority = []
    anchor = None
    anchor_feature = None
    target_path = args.targets_dir / f"{args.anchor_panel}__requantification_targets.csv.gz"
    targets = pd.read_csv(target_path)
    targets["anchor_ppm"] = np.abs(targets.mz - args.anchor_mz) / args.anchor_mz * 1e6
    targets["anchor_drt_sec"] = np.abs(targets.rt_sec - args.anchor_rt_sec)
    eligible = targets[
        (targets.anchor_ppm <= args.anchor_ppm)
        & (targets.anchor_drt_sec <= args.anchor_rt_tolerance)
    ].sort_values(["anchor_ppm", "anchor_drt_sec", "feature_id"], kind="stable")
    if len(eligible):
        anchor_feature = int(eligible.iloc[0].feature_id)
    for panel in args.panels:
        frame, report = compare_panel(args.input_dir, panel)
        frame.insert(0, "panel", panel)
        frame.to_csv(args.output_dir / f"{panel}__feature_transitions.csv.gz", index=False)
        priority = frame[frame.review_priority != "D_stable_or_low_priority"].copy()
        all_priority.append(priority)
        reports[panel] = report
        if panel == args.anchor_panel:
            selected = frame[frame.feature_id == anchor_feature] if anchor_feature is not None else frame.iloc[0:0]
            anchor = selected.iloc[0].replace({np.nan: None}).to_dict() if len(selected) else None
    review = pd.concat(all_priority, ignore_index=True) if all_priority else pd.DataFrame()
    review = review.sort_values(
        ["review_priority", "p2b_n_support_samples", "p2b_agreement_fraction", "p2b_maximum_dreams_similarity"],
        ascending=[True, False, False, False], kind="stable",
    )
    review.to_csv(args.output_dir / "priority_annotation_validation.csv.gz", index=False)
    overall = {
        "status": "mtbls13729_p2b_annotation_comparison_packaged",
        "formal": True,
        "panels": reports,
        "priority_candidates": int(len(review)),
        "predeclared_biology_anchor": {
            "panel": args.anchor_panel,
            "anchor_mz": args.anchor_mz,
            "anchor_rt_sec": args.anchor_rt_sec,
            "resolved_feature_id": anchor_feature,
            "result": anchor,
        },
        "validation_ladder": [
            "candidate graph and frozen-artifact reproduction",
            "multi-sample MS2 consensus and official-DreaMS similarity",
            "diagnostic fragments / classical library evidence for changed candidates",
            "author m/z-RT feature overlap where available",
            "authentic-standard RT/MS2 only for MSI Level 1",
        ],
        "claim_limit": (
            "MTBLS13729 application transitions are unlabeled. Only the sealed P3 benchmark supports an "
            "accuracy claim; biology candidates require orthogonal spectral and, where possible, standard validation."
        ),
    }
    (args.output_dir / "report.json").write_text(json.dumps(overall, indent=2, default=str), encoding="utf-8")
    print(json.dumps(overall, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
