#!/usr/bin/env python
"""Cell-count balance audit for GSE236696 epithelial metabolic axes."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import mmread

from analyze_gse236696_mucinous_axes_by_lineage import (
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/external/GSE236696/raw_files"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/external/GSE236696/epithelial_axis_cell_balance_v1"),
    )
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = discover_triplets(args.raw_dir)
    sample_names = sorted(samples, key=lambda value: (int(value[1]), value[2]))
    data = {}
    for sample in sample_names:
        members = samples[sample]
        with gzip.open(members["features"], "rt", encoding="utf-8") as handle:
            features = [line.rstrip("\n").split("\t") for line in handle]
        symbols = [row[1].upper() if len(row) > 1 else row[0].upper() for row in features]
        matrix = mmread(members["matrix"]).tocsc()
        total_umi = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64)
        detected = np.asarray(matrix.getnnz(axis=0)).ravel()
        mitochondrial = sum_rows(
            matrix, [index for index, symbol in enumerate(symbols) if symbol.startswith("MT-")]
        )
        qc = (detected > 200) & (mitochondrial / np.maximum(total_umi, 1.0) < 0.25) & (total_umi > 0)
        rows = gene_row_map(symbols)
        assigned, _, _, _ = assign_lineages(matrix, total_umi, rows, qc, 0.15, 2)
        selected = np.flatnonzero(assigned == "epithelial")
        target = np.vstack([sum_rows(matrix, rows.get(gene, []))[selected] for gene in TARGET_GENES])
        data[sample] = {"indices": selected, "counts": target, "total": total_umi[selected]}
        print(f"[cell-balance] {sample}: epithelial={len(selected):,}")

    gene_index = {gene: index for index, gene in enumerate(TARGET_GENES)}
    pair_sizes = {
        patient: min(len(data[f"P{patient}N"]["indices"]), len(data[f"P{patient}T"]["indices"]))
        for patient in range(1, 7)
    }
    global_size = min(pair_sizes.values())
    rng = np.random.default_rng(args.seed)

    def run(protocol: str):
        means = {axis: np.empty(args.resamples) for axis in EXPECTED_DIRECTION}
        concordance = {axis: np.empty(args.resamples, dtype=np.int16) for axis in EXPECTED_DIRECTION}
        for repeat in range(args.resamples):
            scores = {}
            for patient in range(1, 7):
                size = pair_sizes[patient] if protocol == "patient_pair_balanced" else global_size
                for condition in "NT":
                    sample = f"P{patient}{condition}"
                    record = data[sample]
                    chosen = rng.choice(record["counts"].shape[1], size=size, replace=False)
                    total = float(np.sum(record["total"][chosen]))
                    cpm = np.sum(record["counts"][:, chosen], axis=1) * 1_000_000.0 / max(total, 1.0)
                    log_cpm = np.log2(cpm + 1.0)
                    for axis, genes in AXES.items():
                        if axis not in EXPECTED_DIRECTION:
                            continue
                        scores[(sample, axis)] = float(np.median([log_cpm[gene_index[gene]] for gene in genes]))
            for axis, direction in EXPECTED_DIRECTION.items():
                delta = np.asarray([
                    scores[(f"P{patient}T", axis)] - scores[(f"P{patient}N", axis)]
                    for patient in range(1, 7)
                ])
                means[axis][repeat] = float(np.mean(delta))
                concordance[axis][repeat] = int(np.sum(direction * delta > 0))
        result = {}
        for axis, direction in EXPECTED_DIRECTION.items():
            result[axis] = {
                "mean_delta_median": float(np.median(means[axis])),
                "mean_delta_p025": float(np.quantile(means[axis], 0.025)),
                "mean_delta_p975": float(np.quantile(means[axis], 0.975)),
                "fraction_expected_mean_direction": float(np.mean(direction * means[axis] > 0)),
                "median_expected_direction_pairs": float(np.median(concordance[axis])),
                "fraction_all_six_pairs_expected_direction": float(np.mean(concordance[axis] == 6)),
                "fraction_at_least_five_pairs_expected_direction": float(np.mean(concordance[axis] >= 5)),
            }
        return result

    report = {
        "status": "gse236696_epithelial_axis_cell_balance_audit_complete",
        "formal": True,
        "epithelial_cells": {sample: int(len(record["indices"])) for sample, record in data.items()},
        "patient_pair_balanced_sizes": pair_sizes,
        "global_balanced_size": global_size,
        "resamples": args.resamples,
        "patient_pair_balanced": run("patient_pair_balanced"),
        "global_91_cell_balanced": run("global_balanced"),
        "provenance": {"script_sha256": sha256(Path(__file__)), "seed": args.seed},
        "claim_limit": (
            "Repeated cell subsampling under the frozen broad epithelial marker gate. This tests cell-count "
            "imbalance and within-gate sampling sensitivity, not exact source-cluster identity or biological "
            "replication beyond the six patients."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
