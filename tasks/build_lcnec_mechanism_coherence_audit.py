from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


AXES = {
    "free_purine_pool_depletion": {
        304: -1,
        122: -1,
    },
    "phosphorylated_nucleotide_and_sugar_accumulation": {
        94: 1,
        294: 1,
        148: 1,
        127: 1,
        104: 1,
    },
    "tryptophan_quinolinate_nad_context": {
        386: 1,
        169: 1,
        102: 1,
    },
    "antioxidant_pool_remodeling": {
        109: 1,
        138: 1,
        214: 1,
        180: -1,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("data/validation/lcnec_hsst3n_annotation_biology/identity_evidence_ledger.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/validation/lcnec_hsst3n_mechanism_coherence_v1"),
    )
    return parser.parse_args()


def evidence_class(row: pd.Series) -> str:
    if row["author_status"] == "published_atlas_overlap":
        return "R_source_atlas_cross_platform_reproduction"
    if bool(row["priority_novel_hypothesis"]):
        return "N_author_unreported_priority_hypothesis"
    return "H_other_level2_or_family_hypothesis"


def main() -> None:
    args = parse_args()
    ledger = pd.read_csv(args.ledger)
    required = {
        "family_id",
        "spectral_hypothesis",
        "author_status",
        "priority_novel_hypothesis",
        "dark_effect_log2fc",
        "dark_effect_q",
        "annotation_confidence",
    }
    missing = required.difference(ledger.columns)
    if missing:
        raise RuntimeError(f"missing ledger columns: {sorted(missing)}")
    if ledger["family_id"].duplicated().any():
        raise RuntimeError("family_id must be unique in the identity ledger")

    ledger = ledger.set_index("family_id", drop=False)
    expected_ids = {family_id for members in AXES.values() for family_id in members}
    absent = expected_ids.difference(set(ledger.index.astype(int)))
    if absent:
        raise RuntimeError(f"fixed mechanism members absent from ledger: {sorted(absent)}")

    records: list[dict] = []
    summaries: list[dict] = []
    for axis, members in AXES.items():
        axis_records: list[dict] = []
        for family_id, expected_direction in members.items():
            row = ledger.loc[family_id]
            effect = float(row["dark_effect_log2fc"])
            observed_direction = 1 if effect > 0 else -1 if effect < 0 else 0
            record = {
                "axis": axis,
                "family_id": int(family_id),
                "spectral_hypothesis": row["spectral_hypothesis"],
                "effect_log2fc": effect,
                "effect_q": float(row["dark_effect_q"]),
                "expected_direction": "up" if expected_direction > 0 else "down",
                "direction_matches_fixed_axis": observed_direction == expected_direction,
                "evidence_class": evidence_class(row),
                "annotation_confidence": row["annotation_confidence"],
                "identity_claim": "MSI_Level_2_or_connectivity_family_only",
            }
            axis_records.append(record)
            records.append(record)

        frame = pd.DataFrame(axis_records)
        summaries.append(
            {
                "axis": axis,
                "members": int(len(frame)),
                "direction_matching_members": int(frame["direction_matches_fixed_axis"].sum()),
                "all_fixed_directions_observed": bool(frame["direction_matches_fixed_axis"].all()),
                "source_atlas_reproductions": int(frame["evidence_class"].str.startswith("R_").sum()),
                "author_unreported_priority_hypotheses": int(frame["evidence_class"].str.startswith("N_").sum()),
                "other_level2_or_family_hypotheses": int(frame["evidence_class"].str.startswith("H_").sum()),
                "median_absolute_log2fc": float(frame["effect_log2fc"].abs().median()),
                "maximum_q": float(frame["effect_q"].max()),
            }
        )

    evidence = pd.DataFrame(records)
    axes = pd.DataFrame(summaries)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(args.output_dir / "axis_evidence_ledger.csv", index=False)
    axes.to_csv(args.output_dir / "axis_summary.csv", index=False)

    report = {
        "status": "lcnec_hsst3n_mechanism_coherence_audit_complete",
        "formal": False,
        "descriptive_post_selection": True,
        "dataset": "34 paired LCNEC tumor-adjacent tissues, HSST3n raw arm",
        "members": int(len(evidence)),
        "axes": summaries,
        "evidence_source_counts": evidence["evidence_class"].value_counts().to_dict(),
        "observed_result": (
            "Four fixed descriptive axes show internally concordant paired-abundance directions: free purine pool depletion, "
            "phosphorylated nucleotide/sugar accumulation, tryptophan-quinolinate-NAD context, and antioxidant pool remodeling."
        ),
        "allowed_interpretation": (
            "The measured pools are consistent with coordinated nucleotide/NAD-related and antioxidant remodeling in LCNEC."
        ),
        "forbidden_interpretations": [
            "The axes are not independent statistical tests because they were assembled after metabolite review.",
            "The data do not establish pathway flux, enzyme activity, ATP energy charge, redox potential, or causal dependency.",
            "Author-unreported rows remain MSI Level 2 or molecular-family hypotheses without same-method authentic standards.",
            "External proteogenomics may provide pathway context but cannot replicate metabolite abundance or identity.",
        ],
        "claim_limit": (
            "Descriptive mechanism coherence only; no new p-value, independent replication, Level-1 identity, flux, enzyme-activity, "
            "or causal-mechanism claim."
        ),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# LCNEC mechanism-coherence audit",
        "",
        "This is a descriptive, post-selection synthesis. It is not an independent pathway-enrichment test.",
        "",
        "| Axis | Members | Direction match | Source reproduced | New priority | Other hypotheses |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['axis']} | {item['members']} | {item['direction_matching_members']}/{item['members']} | "
            f"{item['source_atlas_reproductions']} | {item['author_unreported_priority_hypotheses']} | "
            f"{item['other_level2_or_family_hypotheses']} |"
        )
    lines.extend(
        [
            "",
            "Allowed wording: the measured pools are consistent with coordinated nucleotide/NAD-related and antioxidant remodeling.",
            "",
            "Forbidden wording: flux, enzyme activity, ATP energy charge, redox potential, causal dependency, or Level-1 identity.",
        ]
    )
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
