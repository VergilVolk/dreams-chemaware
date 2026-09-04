#!/usr/bin/env python
"""Conservative lineage-resolved paired pseudobulk screen for GSE236696.

The statistical unit is the patient.  Cells are QC-filtered using the source
paper's thresholds and assigned to broad lineages with a fixed, conservative
marker gate.  Ambiguous cells are explicitly left unassigned.  The analysis
is a composition-sensitivity follow-up, not a replacement for the authors'
Seurat clustering or a claim of malignant-cell identity.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import mmread


AXES = {
    "modified_nucleoside_processing": [
        "METTL1", "WDR4", "RNMT", "CMTR1", "CMTR2", "TRMT1", "TRMT5",
        "TRMT10C", "TGS1", "THUMPD3", "NUDT16", "DCP2",
    ],
    "purine_synthesis_salvage": [
        "HPRT1", "PNP", "GMPS", "IMPDH1", "IMPDH2", "GDA", "APRT",
        "XDH", "ADA", "ADK",
    ],
    "methionine_sah_cycle": ["AHCY", "MAT2A", "MAT2B", "MTAP", "MTR", "MTHFR", "BHMT"],
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

# The source paper explicitly lists the first markers in each set.  A small
# number of canonical broad-lineage anchors are appended to make the gate
# robust to dropout; none overlap a tested metabolic axis.
LINEAGE_MARKERS = {
    "epithelial": [
        "CLDN4", "KRT20", "CLDN7", "DEFA6", "ITPR2", "PSCA", "PTTG1", "STMN1",
        "EPCAM", "KRT8", "KRT18", "KRT19", "VIL1",
    ],
    "myeloid": ["CD68", "S100A8", "S100A9", "C1QC", "IL1B", "SPP1", "LYZ", "FCER1G"],
    "b_plasma": ["MS4A1", "IGHD", "CD19", "MZB1", "JCHAIN", "XBP1", "CD79A", "CD37"],
    "t_nk": [
        "CD4", "KRT86", "KRT81", "CTLA4", "ANXA1", "IL7R", "GNLY", "NKG7", "KLRD1",
        "CD3D", "CD3E", "TRBC1",
    ],
    "endothelial": ["CDH5", "PECAM1", "CD34", "VWF", "EMCN", "KDR"],
    "fibroblast": [
        "MYL9", "TPM2", "TPM1", "BMP5", "SOX6", "ENHO", "DPT", "CFD", "CXCL12",
        "COL1A1", "COL1A2", "DCN",
    ],
}

TARGET_GENES = sorted({gene for genes in AXES.values() for gene in genes})
ALL_MARKERS = sorted({gene for genes in LINEAGE_MARKERS.values() for gene in genes})
SAMPLE_PATTERN = re.compile(r"^(GSM\d+)_P([1-6])([NT])\.(barcodes|features|matrix)\.(tsv|mtx)\.gz$")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_sign_flip_p(values: np.ndarray) -> float:
    observed = abs(float(np.mean(values)))
    exceed = 0
    for mask in range(1 << len(values)):
        signs = np.asarray([1.0 if mask & (1 << bit) else -1.0 for bit in range(len(values))])
        exceed += abs(float(np.mean(values * signs))) >= observed - 1e-12
    return exceed / (1 << len(values))


def patient_bootstrap(values: np.ndarray, seed: int, resamples: int = 10000):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    draws = np.mean(values[indices], axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def discover_triplets(path: Path):
    samples = defaultdict(dict)
    for item in path.glob("*.gz"):
        match = SAMPLE_PATTERN.match(item.name)
        if match:
            _, patient, condition, kind, _ = match.groups()
            samples[f"P{patient}{condition}"][kind] = item
    expected = {f"P{patient}{condition}" for patient in range(1, 7) for condition in ("N", "T")}
    if set(samples) != expected:
        raise RuntimeError(f"sample mismatch: missing={sorted(expected-set(samples))}")
    for sample in expected:
        if set(samples[sample]) != {"barcodes", "features", "matrix"}:
            raise RuntimeError(f"incomplete triplet for {sample}: {sorted(samples[sample])}")
    return samples


def gene_row_map(symbols):
    rows = defaultdict(list)
    wanted = set(TARGET_GENES) | set(ALL_MARKERS) | {"PTPRC"}
    for index, symbol in enumerate(symbols):
        if symbol in wanted:
            rows[symbol].append(index)
    return rows


def sum_rows(matrix, rows):
    if not rows:
        return np.zeros(matrix.shape[1], dtype=np.float64)
    return np.asarray(matrix[rows].sum(axis=0)).ravel().astype(np.float64)


def assign_lineages(
    matrix, total_umi, gene_rows, qc_mask, margin: float,
    epithelial_minimum_markers: int,
):
    scores = []
    detections = []
    lineages = list(LINEAGE_MARKERS)
    for lineage in lineages:
        values = []
        detected = np.zeros(matrix.shape[1], dtype=np.int16)
        for gene in LINEAGE_MARKERS[lineage]:
            counts = sum_rows(matrix, gene_rows.get(gene, []))
            values.append(np.log1p(counts * 10_000.0 / np.maximum(total_umi, 1.0)))
            detected += counts > 0
        scores.append(np.mean(values, axis=0))
        detections.append(detected)
    scores = np.vstack(scores)
    detections = np.vstack(detections)
    best = np.argmax(scores, axis=0)
    ordered = np.sort(scores, axis=0)
    score_margin = ordered[-1] - ordered[-2]
    assigned = np.full(matrix.shape[1], "unassigned", dtype=object)
    ptprc = sum_rows(matrix, gene_rows.get("PTPRC", []))
    epithelial_index = lineages.index("epithelial")
    # Epithelial cells are the prespecified compartment of primary interest.
    # Give this highly specific PTPRC-negative gate priority instead of forcing
    # epithelial cells to win a six-lineage competition that is sensitive to
    # ambient immunoglobulin and stress transcripts.
    epithelial = (
        qc_mask
        & (detections[epithelial_index] >= epithelial_minimum_markers)
        & (scores[epithelial_index] > 0)
        & (ptprc == 0)
    )
    assigned[epithelial] = "epithelial"
    for index, lineage in enumerate(lineages):
        if lineage == "epithelial":
            continue
        selected = (
            qc_mask
            & (assigned == "unassigned")
            & (best == index)
            & (detections[index] >= 2)
            & (score_margin >= margin)
        )
        assigned[selected] = lineage
    return assigned, scores, detections, lineages


def write_csv(path: Path, rows):
    if not rows:
        raise RuntimeError(f"refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/external/GSE236696/raw_files"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/external/GSE236696/paired_axis_by_lineage_v2"),
    )
    parser.add_argument("--minimum-cells", type=int, default=50)
    parser.add_argument("--lineage-score-margin", type=float, default=0.15)
    parser.add_argument("--epithelial-minimum-markers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260830)
    arguments = parser.parse_args()
    if arguments.output_dir.exists() and any(arguments.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {arguments.output_dir}")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    samples = discover_triplets(arguments.raw_dir)

    cell_reports = []
    pseudobulk = []
    raw_manifest = []
    for sample in sorted(samples, key=lambda value: (int(value[1]), value[2])):
        members = samples[sample]
        for kind, path in sorted(members.items()):
            raw_manifest.append({"sample": sample, "kind": kind, "name": path.name,
                                 "bytes": path.stat().st_size, "sha256": file_sha256(path)})
        with gzip.open(members["features"], "rt", encoding="utf-8") as handle:
            features = [line.rstrip("\n").split("\t") for line in handle]
        symbols = [row[1].upper() if len(row) > 1 else row[0].upper() for row in features]
        matrix = mmread(members["matrix"]).tocsc()
        with gzip.open(members["barcodes"], "rt", encoding="utf-8") as handle:
            n_barcodes = sum(1 for _ in handle)
        if matrix.shape != (len(symbols), n_barcodes):
            raise RuntimeError(f"shape mismatch for {sample}: {matrix.shape}")
        total_umi = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64)
        detected_genes = np.asarray(matrix.getnnz(axis=0)).ravel()
        mitochondrial_rows = [index for index, symbol in enumerate(symbols) if symbol.startswith("MT-")]
        mitochondrial_umi = sum_rows(matrix, mitochondrial_rows)
        mitochondrial_fraction = mitochondrial_umi / np.maximum(total_umi, 1.0)
        qc_mask = (detected_genes > 200) & (mitochondrial_fraction < 0.25) & (total_umi > 0)
        rows = gene_row_map(symbols)
        assigned, scores, detections, lineages = assign_lineages(
            matrix, total_umi, rows, qc_mask, arguments.lineage_score_margin,
            arguments.epithelial_minimum_markers,
        )

        patient = int(sample[1])
        condition = "normal" if sample.endswith("N") else "tumor"
        counts_by_lineage = {lineage: int(np.sum(assigned == lineage)) for lineage in lineages}
        cell_reports.append({
            "sample": sample,
            "patient": patient,
            "condition": condition,
            "raw_cells": int(matrix.shape[1]),
            "qc_cells": int(np.sum(qc_mask)),
            "unassigned_qc_cells": int(np.sum(qc_mask & (assigned == "unassigned"))),
            **{f"{lineage}_cells": counts_by_lineage[lineage] for lineage in lineages},
        })
        for lineage in lineages:
            mask = assigned == lineage
            n_cells = int(np.sum(mask))
            if n_cells < arguments.minimum_cells:
                continue
            lineage_total = float(total_umi[mask].sum())
            for gene in TARGET_GENES:
                counts = float(sum_rows(matrix, rows.get(gene, []))[mask].sum())
                cpm = counts * 1_000_000.0 / lineage_total
                pseudobulk.append({
                    "sample": sample,
                    "patient": patient,
                    "condition": condition,
                    "lineage": lineage,
                    "cells": n_cells,
                    "gene": gene,
                    "counts": counts,
                    "total_umi": lineage_total,
                    "log2_cpm_plus1": float(np.log2(cpm + 1.0)),
                })
        print(f"[lineage] {sample}: raw={matrix.shape[1]:,} qc={np.sum(qc_mask):,} "
              + " ".join(f"{key}={value:,}" for key, value in counts_by_lineage.items()))

    lookup = {
        (row["sample"], row["lineage"], row["gene"]): row["log2_cpm_plus1"]
        for row in pseudobulk
    }
    axis_results = []
    gene_results = []
    for lineage in LINEAGE_MARKERS:
        for gene in TARGET_GENES:
            deltas = []
            complete = True
            for patient in range(1, 7):
                key_n = (f"P{patient}N", lineage, gene)
                key_t = (f"P{patient}T", lineage, gene)
                if key_n not in lookup or key_t not in lookup:
                    complete = False
                    break
                deltas.append(lookup[key_t] - lookup[key_n])
            if not complete:
                continue
            values = np.asarray(deltas)
            interval = patient_bootstrap(values, arguments.seed)
            gene_results.append({
                "lineage": lineage,
                "gene": gene,
                "mean_paired_delta": float(np.mean(values)),
                "median_paired_delta": float(np.median(values)),
                "tumor_higher_pairs": int(np.sum(values > 0)),
                "tumor_lower_pairs": int(np.sum(values < 0)),
                "exact_sign_flip_p": exact_sign_flip_p(values),
                "patient_bootstrap_ci_low": interval[0],
                "patient_bootstrap_ci_high": interval[1],
                "paired_deltas": ";".join(f"{value:.6g}" for value in values),
            })
        for axis, genes in AXES.items():
            deltas = []
            complete = True
            for patient in range(1, 7):
                keys_n = [(f"P{patient}N", lineage, gene) for gene in genes]
                keys_t = [(f"P{patient}T", lineage, gene) for gene in genes]
                if any(key not in lookup for key in keys_n + keys_t):
                    complete = False
                    break
                score_n = float(np.median([lookup[key] for key in keys_n]))
                score_t = float(np.median([lookup[key] for key in keys_t]))
                deltas.append(score_t - score_n)
            if not complete:
                continue
            values = np.asarray(deltas)
            axis_results.append({
                "lineage": lineage,
                "axis": axis,
                "patients": len(values),
                "mean_paired_delta": float(np.mean(values)),
                "median_paired_delta": float(np.median(values)),
                "tumor_higher_pairs": int(np.sum(values > 0)),
                "tumor_lower_pairs": int(np.sum(values < 0)),
                "exact_sign_flip_p": exact_sign_flip_p(values),
                "patient_bootstrap_ci_low": patient_bootstrap(values, arguments.seed)[0],
                "patient_bootstrap_ci_high": patient_bootstrap(values, arguments.seed)[1],
                "paired_deltas": ";".join(f"{value:.6g}" for value in values),
            })

    write_csv(arguments.output_dir / "raw_file_manifest.csv", raw_manifest)
    write_csv(arguments.output_dir / "cell_assignment_summary.csv", cell_reports)
    write_csv(arguments.output_dir / "lineage_gene_pseudobulk.csv", pseudobulk)
    write_csv(arguments.output_dir / "lineage_gene_paired_results.csv", gene_results)
    write_csv(arguments.output_dir / "lineage_axis_paired_results.csv", axis_results)

    preferred_key_axes = [
        "modified_nucleoside_processing", "purine_synthesis_salvage",
        "carnitine_long_chain_fao",
    ]
    key_axes = [axis_name for axis_name in preferred_key_axes if axis_name in AXES]
    if len(key_axes) < min(3, len(AXES)):
        key_axes.extend(axis_name for axis_name in AXES if axis_name not in key_axes)
    key_axes = key_axes[:3]
    lineages_with_results = [
        lineage for lineage in LINEAGE_MARKERS
        if any(row["lineage"] == lineage for row in axis_results)
    ]
    figure, axis = plt.subplots(figsize=(11.0, 5.8))
    width = 0.23
    x = np.arange(len(lineages_with_results))
    colors = ["#7b3294", "#008837", "#2166ac"][:len(key_axes)]
    for offset, (tested_axis, color) in enumerate(zip(key_axes, colors)):
        values = []
        for lineage in lineages_with_results:
            match = next(row for row in axis_results if row["lineage"] == lineage and row["axis"] == tested_axis)
            values.append(match["mean_paired_delta"])
        axis.bar(x + (offset - 1) * width, values, width, color=color, label=tested_axis)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x, lineages_with_results, rotation=20, ha="right")
    axis.set_ylabel("mean paired tumor-normal delta\n(lineage pseudobulk axis score)")
    axis.set_title("GSE236696: patient-paired metabolic axes by conservative marker-gated lineage")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(arguments.output_dir / "lineage_axis_deltas.png", dpi=220)
    plt.close(figure)

    report = {
        "status": "gse236696_mucinous_axis_by_lineage_complete",
        "formal": True,
        "patients": 6,
        "qc_contract": "source-paper thresholds: >200 detected genes and mitochondrial fraction <25%",
        "assignment_contract": (
            "fixed broad-lineage markers; epithelial first with at least two markers and zero PTPRC; "
            "remaining lineages require at least two markers and a top-score margin; ambiguous cells abstain"
        ),
        "minimum_cells_per_sample_lineage": arguments.minimum_cells,
        "lineage_score_margin": arguments.lineage_score_margin,
        "epithelial_minimum_markers": arguments.epithelial_minimum_markers,
        "cell_assignment_summary": cell_reports,
        "axis_results": axis_results,
        "provenance": {
            "geo": "GSE236696",
            "raw_manifest_sha256": file_sha256(arguments.output_dir / "raw_file_manifest.csv"),
            "script_sha256": file_sha256(Path(__file__)),
            "source_paper": "10.1002/ctm2.1701",
        },
        "claim_limit": (
            "Patient-paired broad-lineage pseudobulk sensitivity analysis. Marker gating is deliberately "
            "conservative and is not an exact reproduction of the authors' Seurat clusters. It cannot "
            "establish malignant-cell identity, metabolite identity, enzyme activity, or metabolic flux."
        ),
    }
    (arguments.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
