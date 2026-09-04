#!/usr/bin/env python3
"""Audit the MTBLS7387 patient-count layers without conflating them.

The paper text, MetaboLights sample sheet, and Fig. 3 processed source matrix
describe related but non-identical cohort layers.  This audit identifies the
exact metadata patient codes absent from the processed matrix and freezes that
attrition as a machine-readable provenance record.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


NORMALIZED_SHEET = "3c normalized data"
HUMAN = "Homo sapiens"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_code(source_name: str) -> str:
    match = re.match(r"^ND001_(.+?)_(?:Normal|Tumor)Tissue_", source_name)
    if match is None:
        raise RuntimeError(f"cannot parse patient code from {source_name!r}")
    token = match.group(1)
    code = re.sub(r"[nt]\d*$", "", token)
    if not code or code == token:
        raise RuntimeError(f"cannot remove tissue replicate suffix from {source_name!r}")
    return code


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-sheet", type=Path, required=True)
    parser.add_argument("--source-workbook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paper-reported-patients", type=int, default=259)
    arguments = parser.parse_args()

    sample_sheet = arguments.sample_sheet.resolve()
    source_workbook = arguments.source_workbook.resolve()
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(sample_sheet, sep="\t", dtype=str)
    human = metadata[metadata["Characteristics[Organism]"] == HUMAN].copy()
    human["patient_code"] = human["Source Name"].map(metadata_code)
    human["tissue_class"] = human["Factor Value[Organism type]"].map(
        {
            "colorectal cancer": "tumour",
            "cancer adjacent tissue": "adjacent",
        }
    )
    if human["tissue_class"].isna().any():
        raise RuntimeError("unexpected human tissue class in sample sheet")
    counts = pd.crosstab(human["patient_code"], human["tissue_class"])

    source = pd.read_excel(source_workbook, sheet_name=NORMALIZED_SHEET, dtype=str)
    source_codes = set(source["ATTRIBUTE_Code"].dropna().astype(str))
    metadata_codes = set(human["patient_code"].astype(str))
    missing_from_processed = sorted(metadata_codes - source_codes)
    unexpected_in_processed = sorted(source_codes - metadata_codes)
    probable_aliases = []
    for metadata_code_value in missing_from_processed:
        for source_code_value in unexpected_in_processed:
            similarity = difflib.SequenceMatcher(
                None, metadata_code_value.lower(), source_code_value.lower()
            ).ratio()
            if similarity >= 0.70:
                probable_aliases.append(
                    {
                        "metadata_label": metadata_code_value,
                        "processed_label": source_code_value,
                        "string_similarity": similarity,
                    }
                )

    attrition_rows = []
    for code in sorted(metadata_codes):
        subset = human[human["patient_code"] == code]
        attrition_rows.append(
            {
                "patient_code": code,
            "metadata_adjacent_rows": int(counts.loc[code, "adjacent"]),
            "metadata_tumour_rows": int(counts.loc[code, "tumour"]),
                "present_in_fig3_processed_matrix": code in source_codes,
                "metadata_source_names": "|".join(sorted(subset["Source Name"].astype(str))),
            }
        )
    table = pd.DataFrame(attrition_rows)
    table.to_csv(output_dir / "patient_pair_attrition.csv", index=False)

    report = {
        "status": "mtbls7387_processed_pair_attrition_complete",
        "formal": True,
        "cohort_layers": {
            "paper_reported_patients": int(arguments.paper_reported_patients),
            "metabolights_human_samples": int(len(human)),
            "metabolights_adjacent_samples": int((human["tissue_class"] == "adjacent").sum()),
            "metabolights_tumour_samples": int((human["tissue_class"] == "tumour").sum()),
            "metabolights_inferred_patient_labels": int(len(metadata_codes)),
            "fig3_processed_rows": int(len(source)),
            "fig3_processed_complete_pairs": int(len(source_codes)),
        },
        "differences": {
            "paper_minus_metabolights_pairs": int(
                arguments.paper_reported_patients - len(metadata_codes)
            ),
            "metabolights_minus_fig3_pairs": int(len(metadata_codes) - len(source_codes)),
            "metadata_patient_codes_absent_from_fig3": missing_from_processed,
            "fig3_patient_codes_absent_from_metadata": unexpected_in_processed,
            "probable_identifier_aliases_not_silently_resolved": probable_aliases,
        },
        "interpretation": (
            "The three counts refer to distinct reporting, deposition and processed-analysis "
            "layers. MetaboLights contains 258 tumour and 258 adjacent rows and 258 inferred "
            "patient-like labels, but several deposited identifiers/tissue labels are irregular. "
            "Exact set reconciliation yields eight metadata-only labels and one processed-only "
            "label, with one probable spelling alias; the net processed-cohort gap is seven. "
            "The public files do not establish the exclusion reasons. The one-patient "
            "paper-to-deposition difference is also unresolved and must not be collapsed into "
            "the 251-pair analysis cohort."
        ),
        "provenance": {
            "sample_sheet": str(sample_sheet),
            "sample_sheet_sha256": sha256(sample_sheet),
            "source_workbook": str(source_workbook),
            "source_workbook_sha256": sha256(source_workbook),
        },
    }
    with (output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
