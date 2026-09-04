#!/usr/bin/env python
"""Adversarial specificity audit for the GSE236696 polyamine hypothesis.

The patient is the statistical unit.  Three fixed epithelial gates are replayed
on the 12 public count matrices.  The primary pathway statistic is the mean of
gene-level log2(CPM+1), matching the first polyamine analysis.  Expression-,
detection- and variance-matched random gene sets test whether the paired signal
is more specific than generic tumour transcriptional change.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.io import mmread

from analyze_gse236696_mucinous_axes_by_lineage import (
    ALL_MARKERS,
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
        "HPRT1", "PNP", "GMPS", "IMPDH1", "IMPDH2", "GDA", "APRT", "XDH", "ADA", "ADK",
    ],
}
CANONICAL_EPITHELIAL = {"EPCAM", "KRT8", "KRT18", "KRT19", "VIL1", "KRT20", "CLDN4", "CLDN7"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_symbols(matrix, symbols: list[str], mask: np.ndarray) -> dict[str, float]:
    counts = np.asarray(matrix[:, mask].sum(axis=1)).ravel().astype(np.float64)
    result: dict[str, float] = {}
    for symbol, count in zip(symbols, counts):
        if symbol:
            result[symbol] = result.get(symbol, 0.0) + float(count)
    return result


def make_rows(symbols: list[str], wanted: set[str]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = defaultdict(list)
    for index, symbol in enumerate(symbols):
        if symbol in wanted:
            result[symbol].append(index)
    return result


def summarize_axis(values: np.ndarray, seed: int) -> dict:
    interval = patient_bootstrap(values, seed)
    return {
        "paired_patients": int(len(values)),
        "mean_tumour_minus_normal": float(np.mean(values)),
        "median_tumour_minus_normal": float(np.median(values)),
        "positive_patients": int(np.sum(values > 0)),
        "negative_patients": int(np.sum(values < 0)),
        "exact_sign_flip_p": float(exact_sign_flip_p(values)),
        "patient_bootstrap_95ci": interval,
        "paired_deltas": [float(value) for value in values],
    }


def axis_delta(log_cpm: np.ndarray, gene_index: dict[str, int], genes: list[str]) -> np.ndarray:
    indices = [gene_index[gene] for gene in genes]
    score = np.mean(log_cpm[:, indices], axis=1)
    return score[1::2] - score[0::2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/external/GSE236696/raw_files"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/external/GSE236696/polyamine_gate_specificity_v1"),
    )
    parser.add_argument("--random-sets", type=int, default=10000)
    parser.add_argument("--matching-neighbours", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = discover_triplets(args.raw_dir)
    sample_names = sorted(samples, key=lambda value: (int(value[1]), value[2]))
    gates = ["broad_frozen", "competitive", "canonical_strict"]
    counts_by_gate: dict[str, list[dict[str, float]]] = {gate: [] for gate in gates}
    cells_by_gate: dict[str, dict[str, int]] = {gate: {} for gate in gates}
    raw_hashes: list[str] = []
    marker_wanted = set(ALL_MARKERS) | {"PTPRC"}

    for sample in sample_names:
        members = samples[sample]
        with gzip.open(members["features"], "rt", encoding="utf-8") as handle:
            features = [line.rstrip("\n").split("\t") for line in handle]
        symbols = [(row[1] if len(row) > 1 else row[0]).upper() for row in features]
        matrix = mmread(members["matrix"]).tocsc()
        total = np.asarray(matrix.sum(axis=0)).ravel().astype(float)
        detected = np.asarray(matrix.getnnz(axis=0)).ravel()
        mitochondrial = [index for index, symbol in enumerate(symbols) if symbol.startswith("MT-")]
        qc = (detected > 200) & (sum_rows(matrix, mitochondrial) / np.maximum(total, 1) < 0.25) & (total > 0)
        rows = make_rows(symbols, marker_wanted)
        assigned, scores, marker_detections, lineages = assign_lineages(matrix, total, rows, qc, 0.15, 2)
        epi_index = lineages.index("epithelial")
        other_max = np.max(np.delete(scores, epi_index, axis=0), axis=0)
        ptprc = sum_rows(matrix, rows.get("PTPRC", []))
        canonical_detected = np.zeros(matrix.shape[1], dtype=np.int16)
        for gene in CANONICAL_EPITHELIAL:
            canonical_detected += sum_rows(matrix, rows.get(gene, [])) > 0
        masks = {
            "broad_frozen": assigned == "epithelial",
            "competitive": (
                qc & (ptprc == 0) & (marker_detections[epi_index] >= 2)
                & (scores[epi_index] - other_max >= 0.15)
            ),
            "canonical_strict": (
                qc & (ptprc == 0) & (marker_detections[epi_index] >= 3)
                & (canonical_detected >= 2) & (scores[epi_index] - other_max >= 0.05)
            ),
        }
        for gate, mask in masks.items():
            n_cells = int(np.sum(mask))
            cells_by_gate[gate][sample] = n_cells
            # The strict gate is intentionally not relaxed to manufacture a
            # complete panel.  Ten cells is only a fail-closed pseudobulk floor;
            # low-count pairs are explicitly reported and interpreted as a
            # sensitivity analysis rather than primary evidence.
            if n_cells < 10:
                raise RuntimeError(f"{gate}/{sample}: only {n_cells} cells")
            counts_by_gate[gate].append(aggregate_symbols(matrix, symbols, mask))
        raw_hashes.extend(sha256(path) for path in members.values())
        print("[gate]", sample, {gate: cells_by_gate[gate][sample] for gate in gates})

    target_genes = {gene for genes in AXES.values() for gene in genes}
    results: dict[str, dict] = {}
    rng = np.random.default_rng(args.seed)
    for gate in gates:
        mappings = counts_by_gate[gate]
        universe = sorted(set().union(*(mapping.keys() for mapping in mappings)))
        gene_index = {gene: index for index, gene in enumerate(universe)}
        count_matrix = np.zeros((len(sample_names), len(universe)), dtype=float)
        for sample_index, mapping in enumerate(mappings):
            for gene, value in mapping.items():
                count_matrix[sample_index, gene_index[gene]] = value
        totals = np.sum(count_matrix, axis=1, keepdims=True)
        log_cpm = np.log2(count_matrix * 1_000_000 / np.maximum(totals, 1) + 1)
        mean = np.mean(log_cpm, axis=0)
        prevalence = np.mean(count_matrix > 0, axis=0)
        deviation = np.std(log_cpm, axis=0)
        excluded = target_genes | set(ALL_MARKERS) | {"PTPRC"}
        eligible = np.asarray([
            gene not in excluded and not gene.startswith(("MT-", "RPL", "RPS")) and prevalence[index] >= 0.25
            for index, gene in enumerate(universe)
        ])
        eligible_indices = np.flatnonzero(eligible)
        matching_features = np.column_stack([mean, prevalence, deviation])
        scale = np.maximum(np.std(matching_features[eligible_indices], axis=0), 1e-8)
        scaled = matching_features / scale
        neighbours: dict[str, list[str]] = {}
        for gene in sorted(target_genes):
            if gene not in gene_index:
                raise RuntimeError(f"{gate}: target gene absent: {gene}")
            distance = np.sum((scaled[eligible_indices] - scaled[gene_index[gene]]) ** 2, axis=1)
            order = eligible_indices[np.argsort(distance)[: args.matching_neighbours]]
            neighbours[gene] = [universe[index] for index in order]

        gate_results: dict[str, dict] = {}
        for axis, genes in AXES.items():
            observed_delta = axis_delta(log_cpm, gene_index, genes)
            observed_mean = float(np.mean(observed_delta))
            null_means = np.empty(args.random_sets, dtype=float)
            null_concordance = np.empty(args.random_sets, dtype=np.int16)
            for draw in range(args.random_sets):
                selected: list[str] = []
                used: set[str] = set()
                for gene in genes:
                    candidates = neighbours[gene]
                    start = int(rng.integers(0, len(candidates)))
                    chosen = next(
                        (candidates[(start + offset) % len(candidates)] for offset in range(len(candidates))
                         if candidates[(start + offset) % len(candidates)] not in used),
                        None,
                    )
                    if chosen is None:
                        raise RuntimeError(f"unable to draw unique controls for {gate}/{axis}")
                    selected.append(chosen)
                    used.add(chosen)
                null_delta = axis_delta(log_cpm, gene_index, selected)
                null_means[draw] = np.mean(null_delta)
                null_concordance[draw] = np.sum(null_delta > 0)
            summary = summarize_axis(observed_delta, args.seed)
            summary["matched_null"] = {
                "draws": args.random_sets,
                "null_mean": float(np.mean(null_means)),
                "null_p95": float(np.quantile(null_means, 0.95)),
                "directional_empirical_p_mean": float(
                    (1 + np.sum(null_means >= observed_mean)) / (1 + args.random_sets)
                ),
                "directional_empirical_p_concordance": float(
                    (1 + np.sum(null_concordance >= np.sum(observed_delta > 0))) / (1 + args.random_sets)
                ),
            }
            gate_results[axis] = summary
        results[gate] = {
            "epithelial_cells": cells_by_gate[gate],
            "minimum_cells": min(cells_by_gate[gate].values()),
            "gene_universe": len(universe),
            "eligible_control_genes": int(np.sum(eligible)),
            "axes": gate_results,
        }

    report = {
        "status": "gse236696_polyamine_gate_specificity_complete",
        "formal": True,
        "statistical_unit": "patient",
        "samples": sample_names,
        "primary_axis_statistic": "mean gene-level log2(CPM+1)",
        "gates": {
            "broad_frozen": "previous frozen priority gate",
            "competitive": "epithelial score exceeds every other lineage by 0.15; >=2 epithelial markers; PTPRC=0",
            "canonical_strict": ">=3 epithelial markers including >=2 canonical anchors; score advantage >=0.05; PTPRC=0",
        },
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
            "Cross-gate paired transcript specificity can support a compartment-level hypothesis only. "
            "It cannot identify feature 1717, establish malignant-cell identity, metabolic flux, or causal recruitment."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output_dir),
        "minimum_cells": {gate: results[gate]["minimum_cells"] for gate in gates},
    }, indent=2))


if __name__ == "__main__":
    main()
