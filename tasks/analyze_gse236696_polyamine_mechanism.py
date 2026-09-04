"""Patient-paired scRNA pseudobulk audit of the acetylated-polyamine hypothesis.

This analysis follows the already frozen broad-lineage marker gate used for the
GSE236696 composition sensitivity analysis.  The patient, not the cell, is the
statistical unit.  The polyamine/acidity/myeloid gene sets are fixed below from
biochemical pathway roles and the external SAT1-acetylspermidine literature.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.io import mmread
from scipy.stats import spearmanr

from analyze_gse236696_mucinous_axes_by_lineage import (
    LINEAGE_MARKERS,
    assign_lineages,
    discover_triplets,
    exact_sign_flip_p,
    patient_bootstrap,
    sum_rows,
)


AXES = {
    "polyamine_acetylation_catabolism": ["SAT1", "PAOX", "SMOX"],
    "polyamine_synthesis": ["ODC1", "AMD1", "SRM", "SMS", "DHPS", "DOHH"],
    "acidity_lactate_response": ["CA9", "LDHA", "SLC16A3", "SLC2A1", "PDK1", "HIF1A"],
    "neutrophil_recruitment": ["CXCL1", "CXCL2", "CXCL5", "CXCL8", "CSF3"],
    "purine_synthesis_salvage": [
        "HPRT1", "PNP", "GMPS", "IMPDH1", "IMPDH2", "GDA", "APRT", "XDH", "ADA", "ADK"
    ],
}
SINGLE_GENES = ["MUC1", "SAT1", "PAOX", "SMOX", "MS4A4A", "FGF7", "THBS1"]
WANTED = (
    {gene for genes in AXES.values() for gene in genes}
    | set(SINGLE_GENES)
    | {gene for genes in LINEAGE_MARKERS.values() for gene in genes}
    | {"PTPRC"}
)


def row_map(symbols: list[str]) -> dict[str, list[int]]:
    result = defaultdict(list)
    for index, symbol in enumerate(symbols):
        if symbol in WANTED:
            result[symbol].append(index)
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty output: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def exact_permutation_spearman(x: np.ndarray, y: np.ndarray) -> float:
    # n=6 for the complete paired epithelial panel: exhaustive 6! permutations.
    from itertools import permutations

    observed = abs(float(spearmanr(x, y).statistic))
    exceed = 0
    total = 0
    for order in permutations(range(len(y))):
        statistic = abs(float(spearmanr(x, y[list(order)]).statistic))
        exceed += statistic >= observed - 1e-12
        total += 1
    return exceed / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/external/GSE236696/raw_files"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/external/GSE236696/polyamine_mechanism_v1"),
    )
    parser.add_argument("--minimum-cells", type=int, default=50)
    parser.add_argument("--lineage-score-margin", type=float, default=0.15)
    parser.add_argument("--epithelial-minimum-markers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = discover_triplets(args.raw_dir)
    pseudobulk_rows = []
    cell_rows = []
    for sample in sorted(samples, key=lambda value: (int(value[1]), value[2])):
        members = samples[sample]
        with gzip.open(members["features"], "rt", encoding="utf-8") as handle:
            features = [line.rstrip("\n").split("\t") for line in handle]
        symbols = [(row[1] if len(row) > 1 else row[0]).upper() for row in features]
        matrix = mmread(members["matrix"]).tocsc()
        with gzip.open(members["barcodes"], "rt", encoding="utf-8") as handle:
            barcode_count = sum(1 for _ in handle)
        if matrix.shape != (len(symbols), barcode_count):
            raise RuntimeError(f"shape mismatch for {sample}: {matrix.shape}")
        total_umi = np.asarray(matrix.sum(axis=0)).ravel().astype(float)
        detected = np.asarray(matrix.getnnz(axis=0)).ravel()
        mitochondrial = [index for index, symbol in enumerate(symbols) if symbol.startswith("MT-")]
        mitochondrial_fraction = sum_rows(matrix, mitochondrial) / np.maximum(total_umi, 1.0)
        qc = (detected > 200) & (mitochondrial_fraction < 0.25) & (total_umi > 0)
        rows = row_map(symbols)
        assigned, _, _, lineages = assign_lineages(
            matrix, total_umi, rows, qc, args.lineage_score_margin,
            args.epithelial_minimum_markers,
        )
        patient = int(sample[1])
        condition = "normal" if sample.endswith("N") else "tumour"
        cell_rows.append({
            "sample": sample,
            "patient": patient,
            "condition": condition,
            "qc_cells": int(qc.sum()),
            **{f"{lineage}_cells": int(np.sum(assigned == lineage)) for lineage in lineages},
        })
        for lineage in lineages:
            mask = assigned == lineage
            cells = int(mask.sum())
            if cells < args.minimum_cells:
                continue
            lineage_library = float(total_umi[mask].sum())
            gene_values = {}
            for gene in sorted(WANTED - {"PTPRC"}):
                count = float(sum_rows(matrix, rows.get(gene, []))[mask].sum())
                gene_values[gene] = float(np.log2(1 + count * 1_000_000.0 / lineage_library))
            row = {
                "sample": sample, "patient": patient, "condition": condition,
                "lineage": lineage, "cells": cells,
            }
            for axis, genes in AXES.items():
                present = [gene for gene in genes if gene in rows]
                row[axis] = float(np.mean([gene_values[gene] for gene in present]))
                row[f"{axis}__present_genes"] = ";".join(present)
            for gene in SINGLE_GENES:
                row[gene] = gene_values.get(gene, float("nan"))
            pseudobulk_rows.append(row)

    results = []
    delta_rows = []
    lineages = sorted({row["lineage"] for row in pseudobulk_rows})
    endpoints = list(AXES) + SINGLE_GENES
    for lineage in lineages:
        for endpoint in endpoints:
            values = defaultdict(dict)
            for row in pseudobulk_rows:
                if row["lineage"] == lineage:
                    values[row["patient"]][row["condition"]] = float(row[endpoint])
            complete = sorted(
                patient for patient, pair in values.items()
                if set(pair) == {"normal", "tumour"} and np.all(np.isfinite(list(pair.values())))
            )
            if len(complete) < 3:
                continue
            deltas = np.asarray([
                values[patient]["tumour"] - values[patient]["normal"]
                for patient in complete
            ])
            for patient, delta in zip(complete, deltas):
                delta_rows.append({
                    "lineage": lineage, "endpoint": endpoint,
                    "patient": patient, "tumour_minus_normal": float(delta),
                })
            results.append({
                "lineage": lineage,
                "endpoint": endpoint,
                "paired_patients": len(complete),
                "mean_tumour_minus_normal": float(np.mean(deltas)),
                "median_tumour_minus_normal": float(np.median(deltas)),
                "positive_patients": int(np.sum(deltas > 0)),
                "negative_patients": int(np.sum(deltas < 0)),
                "exact_sign_flip_p": exact_sign_flip_p(deltas),
                "patient_bootstrap_95ci_low": patient_bootstrap(deltas, args.seed)[0],
                "patient_bootstrap_95ci_high": patient_bootstrap(deltas, args.seed)[1],
            })

    epithelial_deltas = defaultdict(dict)
    for row in delta_rows:
        if row["lineage"] == "epithelial":
            epithelial_deltas[row["endpoint"]][row["patient"]] = row["tumour_minus_normal"]
    correlation_rows = []
    exposure = "polyamine_acetylation_catabolism"
    for endpoint in [
        "acidity_lactate_response", "neutrophil_recruitment", "MUC1", "SAT1",
        "purine_synthesis_salvage",
    ]:
        common = sorted(set(epithelial_deltas[exposure]) & set(epithelial_deltas[endpoint]))
        if len(common) < 4:
            continue
        x = np.asarray([epithelial_deltas[exposure][patient] for patient in common])
        y = np.asarray([epithelial_deltas[endpoint][patient] for patient in common])
        rho = float(spearmanr(x, y).statistic)
        correlation_rows.append({
            "exposure": exposure, "outcome": endpoint, "patients": len(common),
            "spearman_rho_of_paired_deltas": rho,
            "exact_permutation_p": exact_permutation_spearman(x, y) if len(common) <= 8 else float("nan"),
        })

    write_csv(args.output_dir / "cell_assignment_summary.csv", cell_rows)
    write_csv(args.output_dir / "lineage_pseudobulk.csv", pseudobulk_rows)
    write_csv(args.output_dir / "paired_results.csv", results)
    write_csv(args.output_dir / "paired_deltas.csv", delta_rows)
    if correlation_rows:
        write_csv(args.output_dir / "epithelial_delta_correlations.csv", correlation_rows)
    report = {
        "status": "gse236696_polyamine_mechanism_complete",
        "formal": True,
        "patients": 6,
        "results": results,
        "epithelial_delta_correlations": correlation_rows,
        "contracts": {
            "statistical_unit": "patient",
            "lineage_assignment": "previously frozen conservative marker gate",
            "claim_limit": (
                "paired transcript programs may support a SAT1/polyamine hypothesis; "
                "they do not identify feature 1717 or establish metabolic flux"
            ),
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"], "results": len(results),
        "output": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
