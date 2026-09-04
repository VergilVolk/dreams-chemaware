from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xml",
        type=Path,
        default=Path("data/external/LCNEC_proteogenomic_2026/PMC13464647_fulltext.xml"),
    )
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path("data/external/LCNEC_proteogenomic_2026/fixed_panel_preregistration_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/external/LCNEC_proteogenomic_2026/text_context_audit_v1"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    if not prereg["formal"] or prereg["outcomes_inspected_before_freeze"]:
        raise RuntimeError("fixed panel was not frozen before outcome inspection")
    genes = [gene for axis in prereg["axes"].values() for gene in axis]
    root = ET.parse(args.xml).getroot()
    paragraphs = [" ".join("".join(node.itertext()).split()) for node in root.findall(".//p")]

    records = []
    for axis, axis_genes in prereg["axes"].items():
        for gene in axis_genes:
            pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(gene)}(?![A-Za-z0-9])", re.IGNORECASE)
            matching = [text for text in paragraphs if pattern.search(text)]
            records.append(
                {
                    "axis": axis,
                    "gene": gene,
                    "paragraph_mentions": len(matching),
                    "mentioned_in_article_text": bool(matching),
                }
            )
    ledger = pd.DataFrame(records)
    axis_summary = []
    for axis, frame in ledger.groupby("axis", sort=False):
        axis_summary.append(
            {
                "axis": axis,
                "fixed_genes": int(len(frame)),
                "genes_mentioned_in_article_text": int(frame["mentioned_in_article_text"].sum()),
                "mentioned_genes": frame.loc[frame["mentioned_in_article_text"], "gene"].tolist(),
            }
        )

    expected_narrative = {
        "combined_lcnec_ppp": {
            "stratum": "combined LCNEC with NSCLC versus pure LCNEC",
            "reported_direction": "up",
            "fixed_panel_genes_named": [gene for gene in ["G6PD", "PGD", "TKT", "TALDO1"] if gene in genes],
            "claim": "secondary protein-context prior only",
        },
        "keap1_ppp": {
            "stratum": "KEAP1-mutant versus KEAP1-wild tumors",
            "reported_direction": "up",
            "fixed_panel_genes_named": ["G6PD"],
            "claim": "secondary modifier prior only",
        },
        "apobec_ido1": {
            "stratum": "high versus low APOBEC signature",
            "reported_direction": "up",
            "fixed_panel_genes_named": ["IDO1"],
            "claim": "immune-context observation, not quinolinate flux",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(args.output_dir / "fixed_gene_text_mentions.csv", index=False)
    report = {
        "status": "lcnec_independent_proteogenomic_text_context_audit_complete",
        "formal": False,
        "fixed_panel_frozen_before_text_outcome_audit": True,
        "article": {
            "pmid": prereg["cohort"]["pmid"],
            "pmcid": prereg["cohort"]["pmcid"],
            "paired_patients": prereg["cohort"]["paired_patients"],
            "quantified_protein_pairs_reported_in_text": 103,
        },
        "axis_gene_mentions": axis_summary,
        "article_reported_context": expected_narrative,
        "next_test": (
            "Use the frozen patient-level processed matrix to compute pure-LCNEC paired tumor-minus-NAT effects for every fixed-panel protein, "
            "then evaluate combined/pure and KEAP1 strata separately."
        ),
        "claim_limit": (
            "Text mentions and author-reported subgroup statements are priors and context only. They do not pass the fixed protein gate, "
            "replicate metabolite abundance, validate identity, or establish flux."
        ),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
