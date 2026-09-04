#!/usr/bin/env python
"""Patient-level epithelial NXPE1/secretory context in six mucinous CRC pairs."""

from __future__ import annotations

import gzip
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.stats import spearmanr

from analyze_gse236696_mucinous_axes_by_lineage import (
    LINEAGE_MARKERS,
    assign_lineages,
    discover_triplets,
    exact_sign_flip_p,
    file_sha256,
    patient_bootstrap,
    sum_rows,
)


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/external/GSE236696/raw_files"
OUT = ROOT / "data/external/GSE236696/nxpe1_secretory_epithelial_v2"
TARGETS = ("NXPE1", "CASD1", "SIAE", "MUC2", "TFF3", "SPDEF", "FCGBP", "AGR2")
SECRETORY = ("MUC2", "TFF3", "SPDEF", "FCGBP", "AGR2")


def row_map(symbols: list[str]) -> dict[str, list[int]]:
    wanted = set(TARGETS) | {gene for genes in LINEAGE_MARKERS.values() for gene in genes} | {"PTPRC"}
    rows: dict[str, list[int]] = defaultdict(list)
    for index, symbol in enumerate(symbols):
        if symbol in wanted:
            rows[symbol].append(index)
    return rows


def exact_spearman_permutation(x: np.ndarray, y: np.ndarray) -> float:
    observed = abs(float(spearmanr(x, y).statistic))
    exceed = 0
    total = 0
    for permutation in itertools.permutations(range(len(y))):
        statistic = abs(float(spearmanr(x, y[list(permutation)]).statistic))
        exceed += statistic >= observed - 1e-12
        total += 1
    return exceed / total


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {OUT}")
    OUT.mkdir(parents=True, exist_ok=False)
    samples = discover_triplets(RAW)
    pseudobulk: list[dict] = []
    sample_audit: list[dict] = []
    manifest: list[dict] = []
    feature_availability: dict[str, int] = {gene: 0 for gene in TARGETS}

    for sample in sorted(samples, key=lambda value: (int(value[1]), value[2])):
        members = samples[sample]
        for kind, source in sorted(members.items()):
            manifest.append({"sample": sample, "kind": kind, "name": source.name, "sha256": file_sha256(source)})
        with gzip.open(members["features"], "rt", encoding="utf-8") as handle:
            features = [line.rstrip("\n").split("\t") for line in handle]
        symbols = [row[1].upper() if len(row) > 1 else row[0].upper() for row in features]
        matrix = mmread(members["matrix"]).tocsc()
        with gzip.open(members["barcodes"], "rt", encoding="utf-8") as handle:
            n_barcodes = sum(1 for _ in handle)
        if matrix.shape != (len(symbols), n_barcodes):
            raise RuntimeError(f"shape mismatch for {sample}")

        total_umi = np.asarray(matrix.sum(axis=0)).ravel().astype(float)
        detected = np.asarray(matrix.getnnz(axis=0)).ravel()
        mitochondrial = [i for i, symbol in enumerate(symbols) if symbol.startswith("MT-")]
        mito_fraction = sum_rows(matrix, mitochondrial) / np.maximum(total_umi, 1.0)
        qc = (detected > 200) & (mito_fraction < 0.25) & (total_umi > 0)
        rows = row_map(symbols)
        for gene in TARGETS:
            feature_availability[gene] += int(bool(rows.get(gene)))
        assigned, _, _, _ = assign_lineages(matrix, total_umi, rows, qc, 0.15, 2)
        epithelial = assigned == "epithelial"
        secretory_detected = np.zeros(matrix.shape[1], dtype=np.int16)
        available_secretory = [gene for gene in SECRETORY if rows.get(gene)]
        for gene in available_secretory:
            secretory_detected += sum_rows(matrix, rows.get(gene, [])) > 0
        high_secretory = epithelial & (secretory_detected >= 2)
        masks = {"broad_epithelial": epithelial, "secretory_epithelial": high_secretory}
        patient = int(sample[1])
        condition = "normal" if sample.endswith("N") else "tumour"
        sample_audit.append({
            "sample": sample,
            "patient": patient,
            "condition": condition,
            "qc_cells": int(qc.sum()),
            "epithelial_cells": int(epithelial.sum()),
            "secretory_epithelial_cells": int(high_secretory.sum()),
            "nxpe1_positive_epithelial_cells": int(np.sum(epithelial & (sum_rows(matrix, rows.get("NXPE1", [])) > 0))),
        })
        for compartment, mask in masks.items():
            n_cells = int(mask.sum())
            if n_cells < 30:
                continue
            library = float(total_umi[mask].sum())
            for gene in TARGETS:
                if not rows.get(gene):
                    continue
                count = float(sum_rows(matrix, rows.get(gene, []))[mask].sum())
                positive_cells = int(np.sum(mask & (sum_rows(matrix, rows.get(gene, [])) > 0)))
                pseudobulk.append({
                    "sample": sample,
                    "patient": patient,
                    "condition": condition,
                    "compartment": compartment,
                    "cells": n_cells,
                    "gene": gene,
                    "counts": count,
                    "positive_cells": positive_cells,
                    "detection_fraction": positive_cells / n_cells,
                    "log2_cpm_plus1": float(np.log2(count * 1_000_000.0 / library + 1.0)),
                })
        print(f"[NXPE1-sc] {sample}: epithelial={epithelial.sum():,} secretory={high_secretory.sum():,}")

    table = pd.DataFrame(pseudobulk)
    table.to_csv(OUT / "pseudobulk.csv", index=False)
    pd.DataFrame(sample_audit).to_csv(OUT / "sample_audit.csv", index=False)
    pd.DataFrame(manifest).to_csv(OUT / "raw_manifest.csv", index=False)

    paired_rows: list[dict] = []
    delta_rows: list[dict] = []
    for compartment in ("broad_epithelial", "secretory_epithelial"):
        subset = table[table["compartment"].eq(compartment)]
        for gene in TARGETS:
            gene_table = subset[subset["gene"].eq(gene)]
            wide = gene_table.pivot(index="patient", columns="condition", values="log2_cpm_plus1").dropna()
            if not {"tumour", "normal"} <= set(wide.columns):
                continue
            delta = (wide["tumour"] - wide["normal"]).to_numpy(float)
            for patient, value in zip(wide.index, delta):
                delta_rows.append({"compartment": compartment, "gene": gene, "patient": int(patient), "delta": value})
            paired_rows.append({
                "compartment": compartment,
                "gene": gene,
                "n_pairs": len(delta),
                "positive_pairs": int(np.sum(delta > 0)),
                "mean_delta": float(np.mean(delta)),
                "median_delta": float(np.median(delta)),
                "bootstrap_95ci": patient_bootstrap(delta, 20260831 + len(paired_rows)),
                "exact_sign_flip_p": exact_sign_flip_p(delta),
            })

    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(OUT / "paired_deltas.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(OUT / "paired_results.csv", index=False)
    correlations: list[dict] = []
    for compartment in ("broad_epithelial", "secretory_epithelial"):
        compartment_delta = deltas[deltas["compartment"].eq(compartment)]
        nx = compartment_delta[compartment_delta["gene"].eq("NXPE1")].set_index("patient")["delta"]
        secretory_matrix = compartment_delta[compartment_delta["gene"].isin(SECRETORY)].pivot(index="patient", columns="gene", values="delta")
        common = nx.index.intersection(secretory_matrix.dropna().index)
        if len(common) < 4:
            continue
        secretory_delta = secretory_matrix.loc[common].mean(axis=1)
        statistic = float(spearmanr(nx.loc[common], secretory_delta).statistic)
        correlations.append({
            "compartment": compartment,
            "n_pairs": len(common),
            "spearman_rho": statistic,
            "exact_permutation_p": exact_spearman_permutation(nx.loc[common].to_numpy(), secretory_delta.to_numpy()),
        })

    report = {
        "status": "gse236696_nxpe1_secretory_epithelial_audit_complete",
        "formal": False,
        "patients": 6,
        "primary_compartment": "conservative broad epithelial gate",
        "feature_availability_samples": feature_availability,
        "secretory_sensitivity_gate": (
            "broad epithelial cells detecting at least two of the available TFF3/SPDEF/FCGBP/AGR2 markers; "
            "MUC2 is absent from every deposited feature index and is not interpreted as zero expression"
        ),
        "paired_results": paired_rows,
        "nxpe1_secretory_delta_correlations": correlations,
        "claim_limit": (
            "Six paired mucinous CRC samples; broad epithelial composition sensitivity only. NXPE1 is low-count, "
            "and MUC2 is absent from the deposited feature index. "
            "The gate is not a malignant-cell or goblet-cell annotation, and RNA does not establish enzyme activity."
        ),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
