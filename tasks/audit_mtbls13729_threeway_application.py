"""Post-hoc descriptive audit of the completed MTBLS13729 three-way run.

This script consumes only frozen application outputs.  It does not refit,
rerank or inspect phenotype labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TIER_ORDER = {
    "unassigned": 0, "Level 3-candidate": 1,
    "Level 2a-single/ambiguous": 2, "Level 2a-supported": 3,
}


def annotation_rate(n: int, denominator: int) -> float:
    return float(n / denominator) if denominator else float("nan")


def transition(left: pd.Series, right: pd.Series) -> pd.Series:
    left = left.fillna("").astype(str)
    right = right.fillna("").astype(str)
    return pd.Series(np.select(
        [
            left.str.len().eq(14) & right.str.len().eq(14) & left.eq(right),
            left.str.len().eq(14) & right.str.len().eq(14) & left.ne(right),
            left.str.len().ne(14) & right.str.len().eq(14),
            left.str.len().eq(14) & right.str.len().ne(14),
        ],
        ["retained", "changed", "right_only", "left_only"],
        default="both_abstained",
    ), index=left.index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2b-dir", type=Path, default=Path("data/mtbls13729/p2b_application_v1"))
    parser.add_argument("--e6-dir", type=Path, default=Path("data/mtbls13729/e6_embedding_application_v1"))
    parser.add_argument("--threeway-dir", type=Path, default=Path("data/mtbls13729/threeway_application_v1"))
    parser.add_argument("--p3", type=Path, default=Path("data/validation/g8r_p2b_p3_final.json"))
    parser.add_argument("--targets-dir", type=Path, default=Path("data/mtbls13729/ms1_consensus"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/mtbls13729/threeway_application_audit_v1"))
    parser.add_argument("--e6-method", default="e6_fixed_v2_sw2")
    parser.add_argument("--anchor-mz", type=float, default=448.339483)
    parser.add_argument("--anchor-rt-sec", type=float, default=630.436)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty audit: {out}")
    out.mkdir(parents=True, exist_ok=True)
    p3 = json.loads(args.p3.read_text(encoding="utf-8"))
    panel_reports = {}
    queues = []
    anchor_report = None
    for panel in ("neg_rp", "pos_rp"):
        base_report = json.loads((args.p2b_dir / f"{panel}__report.json").read_text(encoding="utf-8"))
        features = pd.read_csv(args.threeway_dir / f"{panel}__threeway_features.csv.gz")
        per_query = pd.read_csv(args.threeway_dir / f"{panel}__threeway_per_query.csv.gz")
        denominator = int(base_report["features_with_candidates"])
        methods = ("dreams", "p2b", args.e6_method)
        counts = {}
        for method in methods:
            present = features[f"{method}_ik14"].fillna("").astype(str).str.len().eq(14)
            tier = features[f"{method}_annotation_evidence_tier"].fillna("unassigned")
            counts[method] = {
                "annotated_features": int(present.sum()),
                "annotation_rate": annotation_rate(int(present.sum()), denominator),
                "level2a_supported": int(tier.eq("Level 2a-supported").sum()),
                "level2a_any": int(tier.isin(["Level 2a-supported", "Level 2a-single/ambiguous"]).sum()),
            }
        features["official_vs_e6"] = transition(features.dreams_ik14, features[f"{args.e6_method}_ik14"])
        features["official_vs_p2b"] = transition(features.dreams_ik14, features.p2b_ik14)
        e6_tier = features[f"{args.e6_method}_annotation_evidence_tier"].map(TIER_ORDER).fillna(0)
        official_tier = features.dreams_annotation_evidence_tier.map(TIER_ORDER).fillna(0)
        features["e6_tier_delta"] = e6_tier - official_tier
        e6_priority = features[
            features.official_vs_e6.isin(["changed", "right_only"])
            & (features[f"{args.e6_method}_n_support_samples"].fillna(0) >= 2)
            & (features[f"{args.e6_method}_agreement_fraction"].fillna(0) >= 0.6)
            & (features[f"{args.e6_method}_maximum_dreams_similarity"].fillna(-1) >= 0.8)
        ].copy()
        e6_priority.insert(0, "priority_source", "E6_changed_strong_consensus")
        e6_priority.insert(0, "panel", panel)
        p2b_additions = features[
            features.official_vs_p2b.eq("right_only")
            & (features.p2b_n_support_samples.fillna(0) >= 2)
            & (features.p2b_agreement_fraction.fillna(0) >= 0.6)
            & (features.p2b_maximum_dreams_similarity.fillna(-1) >= 0.8)
        ].copy()
        p2b_additions.insert(0, "priority_source", "P2b_only_strong_consensus")
        p2b_additions.insert(0, "panel", panel)
        queues.extend((e6_priority, p2b_additions))
        features.to_csv(out / f"{panel}__audited_features.csv.gz", index=False, compression="gzip")
        panel_reports[panel] = {
            "features_with_mass_candidates": denominator,
            "systems": counts,
            "feature_official_vs_e6": features.official_vs_e6.value_counts().to_dict(),
            "feature_official_vs_p2b": features.official_vs_p2b.value_counts().to_dict(),
            "query_official_vs_e6": per_query.official_vs_e6.value_counts().to_dict(),
            "query_official_vs_p2b": per_query.official_vs_p2b.value_counts().to_dict(),
            "threeway_consensus_features": int(features.threeway_consensus.sum()),
            "e6_tier_gained": int((features.e6_tier_delta > 0).sum()),
            "e6_tier_lost": int((features.e6_tier_delta < 0).sum()),
            "e6_strong_changed_queue": int(len(e6_priority)),
            "p2b_strong_addition_queue": int(len(p2b_additions)),
        }
        if panel == "pos_rp":
            targets = pd.read_csv(args.targets_dir / "pos_rp__requantification_targets.csv.gz")
            targets["ppm"] = np.abs(targets.mz - args.anchor_mz) / args.anchor_mz * 1e6
            targets["drt_sec"] = np.abs(targets.rt_sec - args.anchor_rt_sec)
            anchor_target = targets.sort_values(["ppm", "drt_sec", "feature_id"], kind="stable").iloc[0]
            if anchor_target.ppm > 5 or anchor_target.drt_sec > 15:
                raise RuntimeError("predeclared C20:4 coordinate has no current consensus feature")
            selected = features[features.feature_id.eq(int(anchor_target.feature_id))]
            anchor_report = {
                "predeclared_mz": args.anchor_mz, "predeclared_rt_sec": args.anchor_rt_sec,
                "resolved_feature_id": int(anchor_target.feature_id),
                "resolved_mz": float(anchor_target.mz), "resolved_rt_sec": float(anchor_target.rt_sec),
                "ppm": float(anchor_target.ppm), "drt_sec": float(anchor_target.drt_sec),
                "annotation": (
                    selected.iloc[0].replace({np.nan: None}).to_dict() if len(selected) else None
                ),
            }
    nonempty = [frame for frame in queues if len(frame)]
    queue = pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()
    queue.to_csv(out / "orthogonal_validation_queue.csv.gz", index=False, compression="gzip")
    report = {
        "status": "mtbls13729_threeway_application_audit_complete",
        "formal": False,
        "panels": panel_reports,
        "engineering_benchmark": {
            "primary": p3["panels"]["P3-main-real-pristine"],
            "near": p3["panels"]["P3-near-core-real-pristine"],
        },
        "predeclared_c20_4_anchor": anchor_report,
        "validation_queue_rows": int(len(queue)),
        "claim_limit": (
            "Descriptive application comparison without structure truth. E6 is one-fold experimental; "
            "P2b improves P3 main but degrades P3 near-core."
        ),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()

