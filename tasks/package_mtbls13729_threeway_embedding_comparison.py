"""Package official DreaMS, E6 shared embedding and frozen-P2b annotations.

No application observation is treated as structural truth.  The package reports
agreement, abstention and changed hypotheses, ready for later standards/manual
fragment-evidence review.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def transition(left: pd.Series, right: pd.Series) -> pd.Series:
    left = left.fillna("").astype(str)
    right = right.fillna("").astype(str)
    return pd.Series(
        [
            "abstained" if len(a) != 14 or len(b) != 14
            else ("retained" if a == b else "changed")
            for a, b in zip(left, right)
        ],
        index=left.index,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2b-dir", type=Path, required=True)
    parser.add_argument("--e6-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--e6-method", default="e6_fixed_v2_sw2")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty comparison output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    reports = {}
    for panel in ("neg_rp", "pos_rp"):
        p2b_path = args.p2b_dir / f"{panel}__per_query.csv.gz"
        e6_path = args.e6_dir / f"{panel}__{args.e6_method}__per_query.csv.gz"
        for path in (p2b_path, e6_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        p2b = pd.read_csv(p2b_path)
        e6 = pd.read_csv(e6_path)
        key = ["panel", "query_idx", "feature_id"]
        e6_columns = key + [
            column for column in e6.columns
            if column.startswith(args.e6_method + "_")
        ]
        merged = p2b.merge(e6[e6_columns], on=key, how="inner", validate="one_to_one")
        if len(merged) != len(p2b) or len(merged) != len(e6):
            raise RuntimeError(f"official/P2b/E6 query graph mismatch for {panel}")
        merged["official_vs_p2b"] = transition(merged.dreams_ik14, merged.p2b_ik14)
        merged["official_vs_e6"] = transition(
            merged.dreams_ik14, merged[f"{args.e6_method}_ik14"]
        )
        merged["p2b_vs_e6"] = transition(
            merged.p2b_ik14, merged[f"{args.e6_method}_ik14"]
        )
        merged["threeway_consensus"] = (
            merged.dreams_ik14.fillna("").astype(str).str.len().eq(14)
            & merged.dreams_ik14.eq(merged.p2b_ik14)
            & merged.dreams_ik14.eq(merged[f"{args.e6_method}_ik14"])
        )
        merged.to_csv(out / f"{panel}__threeway_per_query.csv.gz", index=False, compression="gzip")

        feature_frames = []
        for method, path in (
            ("dreams", args.p2b_dir / f"{panel}__dreams_feature_annotations.csv.gz"),
            ("p2b", args.p2b_dir / f"{panel}__p2b_feature_annotations.csv.gz"),
            (args.e6_method, args.e6_dir / f"{panel}__{args.e6_method}__feature_annotations.csv.gz"),
        ):
            frame = pd.read_csv(path)
            keep = [
                "feature_id", "ik14", "inchikey", "name", "smiles",
                "n_support_spectra", "n_support_samples", "agreement_fraction",
                "maximum_dreams_similarity", "median_dreams_similarity",
                "annotation_evidence_tier",
            ]
            frame = frame[[column for column in keep if column in frame]].copy()
            frame = frame.rename(columns={
                column: f"{method}_{column}" for column in frame.columns if column != "feature_id"
            })
            feature_frames.append(frame)
        features = feature_frames[0]
        for frame in feature_frames[1:]:
            features = features.merge(frame, on="feature_id", how="outer", validate="one_to_one")
        features["official_vs_p2b"] = transition(features.dreams_ik14, features.p2b_ik14)
        features["official_vs_e6"] = transition(
            features.dreams_ik14, features[f"{args.e6_method}_ik14"]
        )
        features["threeway_consensus"] = (
            features.dreams_ik14.fillna("").astype(str).str.len().eq(14)
            & features.dreams_ik14.eq(features.p2b_ik14)
            & features.dreams_ik14.eq(features[f"{args.e6_method}_ik14"])
        )
        features.to_csv(out / f"{panel}__threeway_features.csv.gz", index=False, compression="gzip")
        reports[panel] = {
            "queries": int(len(merged)),
            "features_union": int(len(features)),
            "query_threeway_consensus": int(merged.threeway_consensus.sum()),
            "feature_threeway_consensus": int(features.threeway_consensus.sum()),
            "official_vs_p2b": merged.official_vs_p2b.value_counts().to_dict(),
            "official_vs_e6": merged.official_vs_e6.value_counts().to_dict(),
            "p2b_vs_e6": merged.p2b_vs_e6.value_counts().to_dict(),
        }
    report = {
        "status": "mtbls13729_threeway_annotation_comparison_complete",
        "formal": False,
        "methods": ["official_dreams", "experimental_e6_fixed_v2_sw2", "frozen_p2b"],
        "panels": reports,
        "interpretation": (
            "P2b is a downstream candidate expert; E6 is a changed shared embedding. "
            "Changes are hypotheses, not corrections, because application truth is absent."
        ),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

