"""Build a denominator-safe source-vs-DreaMS-vs-full-tool comparison for LCNEC."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-overlap",
        type=Path,
        default=Path(
            "data/validation/lcnec_hsst3n_author_overlap_gate/qualified_family_author_overlap.csv"
        ),
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=Path(
            "data/validation/lcnec_hsst3n_all_qc_annotation_v1/annotation/priority_annotation_primary20.csv"
        ),
    )
    parser.add_argument(
        "--biology-report",
        type=Path,
        default=Path("data/validation/lcnec_hsst3n_annotation_biology/biology_report.json"),
    )
    parser.add_argument(
        "--benchmark-report",
        type=Path,
        default=Path("data/validation/lcnec_hsst3n_annotation_benchmark_v1/report.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/validation/lcnec_hsst3n_same_universe_comparison_v1"),
    )
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(args.source_overlap)
    annotation = pd.read_csv(args.annotation)
    biology = json.loads(args.biology_report.read_text(encoding="utf-8"))
    benchmark = json.loads(args.benchmark_report.read_text(encoding="utf-8"))
    if len(source) != 263 or source["family_id"].nunique() != 263:
        raise RuntimeError("source overlap ledger must contain 263 unique QC-qualified families")
    if len(annotation) != 263 or annotation["family_id"].nunique() != 263:
        raise RuntimeError("annotation ledger must contain the same 263 families")

    ledger = source.merge(annotation, on="family_id", validate="one_to_one", suffixes=("_source", "_tool"))
    ledger["official_dreams_candidate"] = ledger["candidate_molecules"].gt(0)
    ledger["dreams_p2b_agreement"] = ledger["official_dreams_candidate"] & ledger[
        "dreams_p2b_agree"
    ].astype(bool)
    ledger["multi_evidence_retained"] = ledger["annotation_confidence"].str.startswith(
        ("high", "moderate")
    )
    ledger["high_consistency"] = ledger["annotation_confidence"].str.startswith("high")
    ledger.to_csv(args.output_dir / "per_family_comparison.csv", index=False)

    n = len(ledger)
    source_count = int(ledger["author_matched"].sum())
    candidate_count = int(ledger["official_dreams_candidate"].sum())
    agreement_count = int(ledger["dreams_p2b_agreement"].sum())
    retained_count = int(ledger["multi_evidence_retained"].sum())
    high_count = int(ledger["high_consistency"].sum())

    strata_rows = []
    for author_matched, label in [(True, "source_matched_positive_control"), (False, "source_table_absent")]:
        subset = ledger[ledger["author_matched"].eq(author_matched)]
        strata_rows.append(
            {
                "stratum": label,
                "families": len(subset),
                "official_dreams_candidates": int(subset["official_dreams_candidate"].sum()),
                "official_dreams_candidate_rate": float(subset["official_dreams_candidate"].mean()),
                "dreams_p2b_agreement": int(subset["dreams_p2b_agreement"].sum()),
                "dreams_p2b_agreement_rate": float(subset["dreams_p2b_agreement"].mean()),
                "multi_evidence_retained": int(subset["multi_evidence_retained"].sum()),
                "multi_evidence_retained_rate": float(subset["multi_evidence_retained"].mean()),
                "high_consistency": int(subset["high_consistency"].sum()),
                "high_consistency_rate": float(subset["high_consistency"].mean()),
            }
        )
    strata = pd.DataFrame(strata_rows)
    strata.to_csv(args.output_dir / "stratified_summary.csv", index=False)

    comparison = pd.DataFrame(
        [
            {
                "stage": "source-table feature overlap",
                "count": source_count,
                "denominator": n,
                "rate": source_count / n,
                "meaning": "reconstructed QC families with an m/z-RT match to the published HSST3n table",
                "boundary": "not the source study's annotation rate",
            },
            {
                "stage": "official DreaMS constrained candidate coverage",
                "count": candidate_count,
                "denominator": n,
                "rate": candidate_count / n,
                "meaning": "families with at least one 20-ppm MoNA candidate",
                "boundary": "candidate coverage, not correct identity",
            },
            {
                "stage": "official DreaMS plus frozen P2b agreement",
                "count": agreement_count,
                "denominator": n,
                "rate": agreement_count / n,
                "meaning": "top-candidate agreement on the same 263 families",
                "boundary": "agreement, not truth",
            },
            {
                "stage": "full multi-evidence retained",
                "count": retained_count,
                "denominator": n,
                "rate": retained_count / n,
                "meaning": "high/moderate Level-2 or connectivity-family hypotheses",
                "boundary": "evidence-calibrated yield, not accuracy",
            },
            {
                "stage": "high-consistency retained",
                "count": high_count,
                "denominator": n,
                "rate": high_count / n,
                "meaning": "high-consistency Level-2 or connectivity-family hypotheses",
                "boundary": "authentic-standard RT remains absent",
            },
        ]
    )
    comparison.to_csv(args.output_dir / "same_universe_comparison.csv", index=False)

    absent = strata.set_index("stratum").loc["source_table_absent"]
    report = {
        "status": "lcnec_hsst3n_same_universe_annotation_comparison_complete",
        "formal": True,
        "shared_universe": "263 phenotype-blind QC/blank/dilution-qualified precursor-RT families",
        "source_annotation_rate_available": False,
        "same_universe": {
            "families": n,
            "source_table_feature_overlap": source_count,
            "source_table_feature_overlap_rate": source_count / n,
            "official_dreams_candidate_coverage": candidate_count,
            "official_dreams_candidate_coverage_rate": candidate_count / n,
            "dreams_p2b_agreement": agreement_count,
            "dreams_p2b_agreement_rate": agreement_count / n,
            "full_multi_evidence_retained": retained_count,
            "full_multi_evidence_retained_rate": retained_count / n,
            "high_consistency_retained": high_count,
            "high_consistency_retained_rate": high_count / n,
        },
        "source_matched_positive_control": strata_rows[0],
        "source_table_absent_headroom": {
            **strata_rows[1],
            "biology_robust_author_unreported": int(benchmark["frozen_dark_universe"]["author_unreported_hypotheses"]),
            "priority_author_unreported": int(benchmark["frozen_dark_universe"]["priority_hypotheses"]),
        },
        "interpretation": (
            "Official DreaMS expands auditable candidate coverage on the same 263-family universe; "
            "the full workflow trades coverage for evidence calibration. Source-table overlap, candidate "
            "coverage, model agreement and Level-2/family yield are distinct endpoints, not accuracy estimates."
        ),
        "claim_limit": (
            "The source paper has no detected-feature denominator, so its annotation rate cannot be "
            "reconstructed. Without authentic-standard truth, none of these rates is annotation accuracy."
        ),
        "provenance": {
            "source_overlap_sha256": sha256(args.source_overlap),
            "annotation_sha256": sha256(args.annotation),
            "biology_report_sha256": sha256(args.biology_report),
            "benchmark_report_sha256": sha256(args.benchmark_report),
            "script_sha256": sha256(Path(__file__)),
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.3, 5.4), gridspec_kw={"width_ratios": [1.05, 1.25]})
    labels = ["Source-table\noverlap", "Official DreaMS\ncandidate", "DreaMS+P2b\nagreement", "Full evidence\nretained"]
    counts = [source_count, candidate_count, agreement_count, retained_count]
    colors = ["#9E9E9E", "#3B78A8", "#76539A", "#2C8063"]
    bars = axes[0].bar(np.arange(4), np.asarray(counts) / n * 100, color=colors, width=0.72)
    axes[0].set_xticks(np.arange(4), labels)
    axes[0].set_ylabel("Share of the same 263-family universe (%)")
    axes[0].set_title("A  Denominator-safe analytical comparison", loc="left", fontweight="bold")
    axes[0].set_ylim(0, 70)
    axes[0].grid(axis="y", color="#E6E6E6", linewidth=0.7)
    for bar, count in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2, f"{count}/263", ha="center", fontsize=9)

    funnel_labels = ["Source-table absent", "DreaMS candidate", "DreaMS+P2b agree", "Multi-evidence retained", "Biology-robust", "Priority"]
    funnel_counts = [221, int(absent["official_dreams_candidates"]), int(absent["dreams_p2b_agreement"]), int(absent["multi_evidence_retained"]), int(benchmark["frozen_dark_universe"]["author_unreported_hypotheses"]), int(benchmark["frozen_dark_universe"]["priority_hypotheses"])]
    y = np.arange(len(funnel_labels))[::-1]
    axes[1].barh(y, funnel_counts, color=["#D7D7D7", "#3B78A8", "#76539A", "#2C8063", "#D58A27", "#B74343"])
    axes[1].set_yticks(y, funnel_labels)
    axes[1].set_xlabel("Families / hypotheses")
    axes[1].set_title("B  Source-table-absent evidence funnel", loc="left", fontweight="bold")
    axes[1].grid(axis="x", color="#E6E6E6", linewidth=0.7)
    for position, count in zip(y, funnel_counts):
        axes[1].text(count + 3, position, str(count), va="center", fontsize=9)
    axes[1].set_xlim(0, 240)

    fig.suptitle("LCNEC annotation recovery: coverage is not accuracy", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.015,
        "The source study lacks a detected-feature denominator. All percentages here use our reconstructed, phenotype-blind 263-family analytical universe; identity remains Level 2/family without standards.",
        ha="center",
        fontsize=8.8,
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.99, 0.94))
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"same_universe_annotation_comparison.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
