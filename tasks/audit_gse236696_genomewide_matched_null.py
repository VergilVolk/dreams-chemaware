#!/usr/bin/env python
"""Genome-wide expression-matched random-set audit for GSE236696 axes.

The frozen marker gate is replayed from the lineage analysis.  All expressed
gene symbols are aggregated into epithelial pseudobulk for the 12 samples.
For every target gene, controls are chosen from nearest non-target genes in
mean expression, detection fraction, and between-sample standard deviation.
The null therefore asks whether a same-sized, similarly observable gene set
shows an equal or stronger patient-paired direction by chance.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import mmread

from analyze_gse236696_mucinous_axes_by_lineage import (
    ALL_MARKERS,
    AXES,
    TARGET_GENES,
    assign_lineages,
    discover_triplets,
    gene_row_map,
    sum_rows,
)


EXPECTED_DIRECTION = {
    "modified_nucleoside_processing": 1,
    "purine_synthesis_salvage": 1,
    "carnitine_long_chain_fao": -1,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_symbols(matrix, symbols: list[str], mask: np.ndarray):
    counts = np.asarray(matrix[:, mask].sum(axis=1)).ravel().astype(np.float64)
    aggregate: dict[str, float] = {}
    for symbol, count in zip(symbols, counts):
        if not symbol:
            continue
        aggregate[symbol] = aggregate.get(symbol, 0.0) + float(count)
    return aggregate


def axis_deltas(values: np.ndarray, gene_index: dict[str, int], genes: list[str]):
    indices = [gene_index[gene] for gene in genes]
    scores = np.median(values[:, indices], axis=1)
    return scores[1::2] - scores[0::2]


def summarize(values: np.ndarray):
    return {
        "mean_paired_delta": float(np.mean(values)),
        "median_paired_delta": float(np.median(values)),
        "positive_pairs": int(np.sum(values > 0)),
        "negative_pairs": int(np.sum(values < 0)),
        "paired_deltas": [float(value) for value in values],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/external/GSE236696/raw_files"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/external/GSE236696/epithelial_axis_genomewide_matched_null_v1"),
    )
    parser.add_argument("--random-sets", type=int, default=20000)
    parser.add_argument("--matching-neighbours", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = discover_triplets(args.raw_dir)
    sample_names = sorted(samples, key=lambda value: (int(value[1]), value[2]))
    sample_counts: list[dict[str, float]] = []
    sample_totals: list[float] = []
    epithelial_cells: dict[str, int] = {}
    raw_hashes = []

    for sample in sample_names:
        members = samples[sample]
        with gzip.open(members["features"], "rt", encoding="utf-8") as handle:
            features = [line.rstrip("\n").split("\t") for line in handle]
        symbols = [row[1].upper() if len(row) > 1 else row[0].upper() for row in features]
        matrix = mmread(members["matrix"]).tocsc()
        total_umi = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64)
        detected_genes = np.asarray(matrix.getnnz(axis=0)).ravel()
        mitochondrial_rows = [index for index, symbol in enumerate(symbols) if symbol.startswith("MT-")]
        mitochondrial_umi = sum_rows(matrix, mitochondrial_rows)
        qc = (
            (detected_genes > 200)
            & (mitochondrial_umi / np.maximum(total_umi, 1.0) < 0.25)
            & (total_umi > 0)
        )
        rows = gene_row_map(symbols)
        assigned, _, _, _ = assign_lineages(matrix, total_umi, rows, qc, 0.15, 2)
        epithelial = assigned == "epithelial"
        n_cells = int(np.sum(epithelial))
        if n_cells < 50:
            raise RuntimeError(f"{sample}: only {n_cells} epithelial cells")
        counts = aggregate_symbols(matrix, symbols, epithelial)
        sample_counts.append(counts)
        sample_totals.append(float(sum(counts.values())))
        epithelial_cells[sample] = n_cells
        raw_hashes.extend(sha256(path) for path in members.values())
        print(f"[genome-null] {sample}: epithelial={n_cells:,} symbols={len(counts):,}")

    universe = sorted(set().union(*(counts.keys() for counts in sample_counts)))
    gene_index = {gene: index for index, gene in enumerate(universe)}
    counts = np.zeros((len(sample_names), len(universe)), dtype=np.float64)
    for sample_index, mapping in enumerate(sample_counts):
        for gene, count in mapping.items():
            counts[sample_index, gene_index[gene]] = count
    totals = np.asarray(sample_totals)[:, None]
    log_cpm = np.log2(counts * 1_000_000.0 / np.maximum(totals, 1.0) + 1.0)

    mean = np.mean(log_cpm, axis=0)
    prevalence = np.mean(counts > 0, axis=0)
    deviation = np.std(log_cpm, axis=0)
    excluded = set(TARGET_GENES) | set(ALL_MARKERS) | {"PTPRC"}
    eligible = np.asarray([
        gene not in excluded
        and not gene.startswith("MT-")
        and not gene.startswith("RPL")
        and not gene.startswith("RPS")
        and prevalence[index] >= 0.25
        for index, gene in enumerate(universe)
    ])
    eligible_indices = np.flatnonzero(eligible)
    features = np.column_stack([mean, prevalence, deviation])
    scale = np.maximum(np.std(features[eligible_indices], axis=0), 1e-8)
    scaled = features / scale

    neighbour_map: dict[str, list[str]] = {}
    for gene in sorted(set(TARGET_GENES)):
        if gene not in gene_index:
            raise RuntimeError(f"target gene absent: {gene}")
        distance = np.sum((scaled[eligible_indices] - scaled[gene_index[gene]]) ** 2, axis=1)
        order = eligible_indices[np.argsort(distance)[: args.matching_neighbours]]
        neighbour_map[gene] = [universe[index] for index in order]
        if len(neighbour_map[gene]) < 50:
            raise RuntimeError(f"insufficient matched controls for {gene}")

    rng = np.random.default_rng(args.seed)
    results = {}
    for axis in EXPECTED_DIRECTION:
        genes = AXES[axis]
        direction = EXPECTED_DIRECTION[axis]
        observed_values = axis_deltas(log_cpm, gene_index, genes)
        observed = float(np.mean(observed_values))
        null_mean = np.empty(args.random_sets, dtype=np.float64)
        null_concordance = np.empty(args.random_sets, dtype=np.int16)
        for draw in range(args.random_sets):
            selected: list[str] = []
            used: set[str] = set()
            for gene in genes:
                candidates = neighbour_map[gene]
                start = int(rng.integers(0, len(candidates)))
                chosen = None
                for offset in range(len(candidates)):
                    candidate = candidates[(start + offset) % len(candidates)]
                    if candidate not in used:
                        chosen = candidate
                        break
                if chosen is None:
                    raise RuntimeError(f"unable to draw unique matched set for {axis}")
                used.add(chosen)
                selected.append(chosen)
            values = axis_deltas(log_cpm, gene_index, selected)
            null_mean[draw] = float(np.mean(values))
            null_concordance[draw] = int(np.sum(direction * values > 0))
        observed_concordance = int(np.sum(direction * observed_values > 0))
        results[axis] = {
            "observed": summarize(observed_values),
            "expected_direction": "up" if direction > 0 else "down",
            "matched_null_draws": args.random_sets,
            "null_mean": float(np.mean(null_mean)),
            "null_p05": float(np.quantile(null_mean, 0.05)),
            "null_p95": float(np.quantile(null_mean, 0.95)),
            "directional_empirical_p_mean": float(
                (1 + np.sum(direction * null_mean >= direction * observed)) / (1 + args.random_sets)
            ),
            "directional_empirical_p_concordance": float(
                (1 + np.sum(null_concordance >= observed_concordance)) / (1 + args.random_sets)
            ),
            "observed_directional_pairs": observed_concordance,
            "matched_control_neighbours_per_gene": args.matching_neighbours,
        }

    report = {
        "status": "gse236696_epithelial_axis_genomewide_matched_null_complete",
        "formal": True,
        "samples": sample_names,
        "epithelial_cells": epithelial_cells,
        "gene_universe": len(universe),
        "eligible_control_genes": int(np.sum(eligible)),
        "matching_variables": ["mean_log2_cpm_plus1", "detection_fraction", "between_sample_sd"],
        "results": results,
        "parameters": {
            "random_sets": args.random_sets,
            "matching_neighbours": args.matching_neighbours,
            "seed": args.seed,
        },
        "provenance": {
            "raw_triplet_hashes_sha256": hashlib.sha256("".join(sorted(raw_hashes)).encode()).hexdigest(),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "Expression-matched random-set specificity audit under the frozen broad epithelial marker gate. "
            "It is not an exact Seurat-cluster reproduction, a competitive pathway test accounting for all "
            "gene-gene correlation, or evidence of metabolite identity, flux, or enzyme activity."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
