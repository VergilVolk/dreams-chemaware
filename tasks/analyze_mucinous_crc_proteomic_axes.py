#!/usr/bin/env python
"""Reanalyse independent mucinous CRC tissue proteomics for fixed metabolic axes.

The source study used pooled TMT channels for five tissue groups.  Therefore
this program reports descriptive group-level log2 ratios only; it deliberately
does not manufacture patient-level p-values from pooled measurements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openpyxl


RATIO_COLUMNS = {
    "LMC_vs_NC": "Abundance Ratio (log2): (F1, 128) / (F1, 126)",
    "RMC_vs_NC": "Abundance Ratio (log2): (F1, 129) / (F1, 126)",
    "LNMC_vs_NC": "Abundance Ratio (log2): (F1, 130) / (F1, 126)",
    "RNMC_vs_NC": "Abundance Ratio (log2): (F1, 131) / (F1, 126)",
    "LMC_vs_LNMC": "Abundance Ratio (log2): (F1, 128) / (F1, 130)",
    "RMC_vs_RNMC": "Abundance Ratio (log2): (F1, 129) / (F1, 131)",
}

AXES = {
    "modified_nucleoside_processing": [
        "METTL1", "WDR4", "RNMT", "CMTR1", "CMTR2", "TRMT1", "TRMT5",
        "TRMT10C", "TGS1", "THUMPD3", "NUDT16", "DCP2",
    ],
    "purine_synthesis_salvage": [
        "HPRT1", "PNP", "GMPS", "IMPDH1", "IMPDH2", "GDA", "APRT",
        "XDH", "ADA", "ADK",
    ],
    "methionine_sah_cycle": [
        "AHCY", "MAT2A", "MAT2B", "MTAP", "MTR", "MTHFR", "BHMT",
    ],
    "carnitine_long_chain_fao": [
        "CPT1A", "CPT2", "SLC25A20", "ACADVL", "ACADM", "ACADS", "HADHA",
        "HADHB", "ETFA", "ETFB", "ETFDH", "CRAT", "CROT",
    ],
    "sphingolipid_metabolism": [
        "CERS2", "CERS4", "CERS5", "CERS6", "SPTLC1", "SPTLC2", "SPTLC3",
        "SGMS1", "SGMS2", "SPHK1", "SPHK2", "ASAH1", "ASAH2", "SMPD1",
        "SMPD2", "SMPD3", "SMPD4", "UGCG",
    ],
}

REQUIRED = {
    "METTL1", "TRMT1", "HPRT1", "PNP", "GMPS", "IMPDH1", "IMPDH2",
    "CPT1A", "CPT2", "ACADVL", "HADHA", "HADHB", "ETFDH", "CERS6",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def read_source(path: Path):
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = workbook.active
    iterator = sheet.iter_rows(values_only=True)
    header = list(next(iterator))
    index = {name: position for position, name in enumerate(header)}
    required_columns = {
        "Accession", "Description", "Gene Symbol", "Exp. q-value: Combined", "# PSMs",
        *RATIO_COLUMNS.values(),
    }
    missing = sorted(required_columns - set(index))
    if missing:
        raise RuntimeError(f"source workbook is missing required columns: {missing}")

    requested = {gene for genes in AXES.values() for gene in genes}
    rows_by_gene = defaultdict(list)
    for values in iterator:
        gene = str(values[index["Gene Symbol"]] or "").strip().upper()
        if gene not in requested:
            continue
        rows_by_gene[gene].append(values)

    selected = {}
    for gene, rows in rows_by_gene.items():
        # Deterministic protein-group resolution: smallest experimental q-value,
        # then largest PSM count, then accession. No phenotype ratio enters selection.
        selected[gene] = min(
            rows,
            key=lambda row: (
                finite(row[index["Exp. q-value: Combined"]])
                if finite(row[index["Exp. q-value: Combined"]]) is not None else float("inf"),
                -(finite(row[index["# PSMs"]]) or 0.0),
                str(row[index["Accession"]] or ""),
            ),
        )

    records = []
    for axis, genes in AXES.items():
        for gene in genes:
            row = selected.get(gene)
            if row is None:
                continue
            record = {
                "axis": axis,
                "gene": gene,
                "accession": str(row[index["Accession"]] or ""),
                "description": str(row[index["Description"]] or ""),
                "experimental_q_value": finite(row[index["Exp. q-value: Combined"]]),
                "psms": int(finite(row[index["# PSMs"]]) or 0),
            }
            for label, source_column in RATIO_COLUMNS.items():
                record[label] = finite(row[index[source_column]])
            records.append(record)
    return records, sorted(requested - set(selected)), {
        gene: len(rows) for gene, rows in rows_by_gene.items() if len(rows) > 1
    }


def summarize(records):
    summaries = []
    by_axis = defaultdict(list)
    for record in records:
        by_axis[record["axis"]].append(record)
    for axis, requested_genes in AXES.items():
        axis_rows = by_axis[axis]
        summary = {
            "axis": axis,
            "genes_requested": len(requested_genes),
            "genes_detected": len(axis_rows),
        }
        for label in RATIO_COLUMNS:
            values = np.asarray(
                [row[label] for row in axis_rows if row[label] is not None], dtype=float
            )
            summary[f"{label}__n"] = int(values.size)
            summary[f"{label}__median_log2"] = (
                float(np.median(values)) if values.size else None
            )
            summary[f"{label}__positive_fraction"] = (
                float(np.mean(values > 0)) if values.size else None
            )
        summaries.append(summary)
    return summaries


def write_csv(path: Path, rows):
    if not rows:
        raise RuntimeError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_heatmap(path: Path, records):
    priority = [
        "METTL1", "RNMT", "TRMT1", "TRMT10C", "HPRT1", "PNP", "GMPS",
        "IMPDH1", "IMPDH2", "GDA", "AHCY", "MAT2A", "CPT1A", "CPT2",
        "SLC25A20", "ACADVL", "HADHA", "HADHB", "ETFA", "ETFB", "ETFDH",
        "CERS6",
    ]
    by_gene = {record["gene"]: record for record in records}
    genes = [gene for gene in priority if gene in by_gene]
    genes.extend(gene for gene in sorted(by_gene) if gene not in genes)
    columns = list(RATIO_COLUMNS)
    matrix = np.asarray([[by_gene[g][column] for column in columns] for g in genes], dtype=float)
    limit = max(1.0, float(np.nanmax(np.abs(matrix))))

    fig, axis = plt.subplots(figsize=(11.5, 8.2))
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(range(len(columns)), [c.replace("_vs_", " / ") for c in columns], rotation=35, ha="right")
    axis.set_yticks(range(len(genes)), genes)
    axis.set_title("Independent mucinous CRC tissue proteomics: fixed metabolic targets")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(column_index, row_index, f"{matrix[row_index, column_index]:+.2f}",
                      ha="center", va="center", fontsize=7,
                      color="white" if abs(matrix[row_index, column_index]) > 0.65 * limit else "black")
    colorbar = fig.colorbar(image, ax=axis, shrink=0.82)
    colorbar.set_label("pooled TMT log2 abundance ratio")
    fig.text(0.5, 0.01,
             "Descriptive pooled-group evidence only; not patient-level statistical replication.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/external/mucinous_crc_proteomics_2021/supplement/Table S2.xlsx"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/external/mucinous_crc_proteomics_2021/axis_reanalysis_v1"),
    )
    arguments = parser.parse_args()
    if not arguments.source.is_file():
        raise FileNotFoundError(arguments.source)
    if arguments.output_dir.exists() and any(arguments.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {arguments.output_dir}")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    records, missing, duplicates = read_source(arguments.source)
    detected = {record["gene"] for record in records}
    missing_required = sorted(REQUIRED - detected)
    if missing_required:
        raise RuntimeError(f"required sentinel proteins were not recovered: {missing_required}")
    summaries = summarize(records)

    protein_csv = arguments.output_dir / "target_proteins.csv"
    summary_csv = arguments.output_dir / "axis_summary.csv"
    figure = arguments.output_dir / "target_protein_heatmap.png"
    write_csv(protein_csv, records)
    write_csv(summary_csv, summaries)
    render_heatmap(figure, records)

    report = {
        "status": "mucinous_crc_proteomic_axis_reanalysis_complete",
        "formal": True,
        "study_design": {
            "patients": 29,
            "groups": {"LMC": 6, "LNMC": 8, "RMC": 7, "RNMC": 8, "NC": 25},
            "measurement": "pooled group-level TMT ratios",
            "patient_level_inference_available": False,
        },
        "targets": {
            "genes_requested": sum(len(genes) for genes in AXES.values()),
            "genes_detected": len(records),
            "missing_genes": missing,
            "duplicate_gene_rows_resolved_without_phenotype_ratios": duplicates,
        },
        "axis_summaries": summaries,
        "outputs": {
            "target_proteins": str(protein_csv),
            "axis_summary": str(summary_csv),
            "heatmap": str(figure),
        },
        "provenance": {
            "source_workbook": str(arguments.source),
            "source_workbook_sha256": sha256(arguments.source),
            "script_sha256": sha256(Path(__file__)),
            "study_doi": "10.3390/curroncol28050305",
        },
        "claim_limit": (
            "This is independent, histology-stratified, pooled TMT tissue proteomics. "
            "It can provide orthogonal directional support for pathway hypotheses but "
            "cannot replicate metabolite identity, patient-level variance, flux, or causality."
        ),
    }
    report_path = arguments.output_dir / "summary.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
