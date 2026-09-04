#!/usr/bin/env python
"""Paired pseudobulk screen of fixed metabolic axes in GSE236696.

This first-pass analysis uses each patient's entire scRNA-seq library as one
pseudobulk sample.  Cells are never treated as biological replicates.  A later
cell-type-resolved analysis is required to separate epithelial regulation from
composition changes.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import tarfile
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
TARGETS = sorted({gene for genes in AXES.values() for gene in genes})
SAMPLE_PATTERN = re.compile(r"^(GSM\d+)_P([1-6])([NT])\.(barcodes|features|matrix)\.(tsv|mtx)\.gz$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_sign_flip_p(values: np.ndarray) -> float:
    observed = abs(float(np.mean(values)))
    exceed = 0
    total = 1 << len(values)
    for mask in range(total):
        signs = np.asarray([1.0 if mask & (1 << bit) else -1.0 for bit in range(len(values))])
        exceed += abs(float(np.mean(values * signs))) >= observed - 1e-12
    return exceed / total


def patient_bootstrap(values: np.ndarray, seed: int, resamples: int = 10000):
    rng = np.random.default_rng(seed)
    draws = np.mean(values[rng.integers(0, len(values), size=(resamples, len(values)))], axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def parse_archive(path: Path):
    members_by_sample = defaultdict(dict)
    with tarfile.open(path, "r") as archive:
        for member in archive.getmembers():
            name = Path(member.name).name
            match = SAMPLE_PATTERN.match(name)
            if match:
                gsm, patient, condition, kind, _ = match.groups()
                sample = f"P{patient}{condition}"
                members_by_sample[sample][kind] = member.name
        expected = {f"P{patient}{condition}" for patient in range(1, 7) for condition in ("N", "T")}
        if set(members_by_sample) != expected:
            raise RuntimeError(
                f"archive sample mismatch: missing={sorted(expected-set(members_by_sample))}, "
                f"extra={sorted(set(members_by_sample)-expected)}"
            )
        records = []
        for sample in sorted(expected, key=lambda value: (int(value[1]), value[2])):
            members = members_by_sample[sample]
            if set(members) != {"barcodes", "features", "matrix"}:
                raise RuntimeError(f"incomplete 10x triplet for {sample}: {sorted(members)}")

            feature_stream = archive.extractfile(members["features"])
            matrix_stream = archive.extractfile(members["matrix"])
            barcode_stream = archive.extractfile(members["barcodes"])
            if feature_stream is None or matrix_stream is None or barcode_stream is None:
                raise RuntimeError(f"failed to open archive members for {sample}")

            with gzip.GzipFile(fileobj=feature_stream) as handle:
                features = [line.decode("utf-8").rstrip("\n").split("\t") for line in handle]
            symbols = [row[1].upper() if len(row) > 1 else row[0].upper() for row in features]
            with gzip.GzipFile(fileobj=barcode_stream) as handle:
                n_barcodes = sum(1 for _ in handle)
            with gzip.GzipFile(fileobj=matrix_stream) as handle:
                # mmread requires a seekable stream in some SciPy builds.
                matrix_bytes = io.BytesIO(handle.read())
                matrix = mmread(matrix_bytes).tocsr()
            if matrix.shape != (len(features), n_barcodes):
                raise RuntimeError(
                    f"shape mismatch for {sample}: matrix={matrix.shape}, "
                    f"features={len(features)}, barcodes={n_barcodes}"
                )
            total_umi = float(matrix.sum())
            if total_umi <= 0:
                raise RuntimeError(f"zero total UMI for {sample}")
            symbol_rows = defaultdict(list)
            for row_index, symbol in enumerate(symbols):
                if symbol in TARGETS:
                    symbol_rows[symbol].append(row_index)
            patient = int(sample[1])
            condition = "normal" if sample.endswith("N") else "tumor"
            for gene in TARGETS:
                row_indices = symbol_rows.get(gene, [])
                counts = float(matrix[row_indices].sum()) if row_indices else 0.0
                cpm = counts * 1_000_000.0 / total_umi
                records.append({
                    "sample": sample,
                    "patient": patient,
                    "condition": condition,
                    "gene": gene,
                    "counts": counts,
                    "total_umi": total_umi,
                    "log2_cpm_plus1": float(np.log2(cpm + 1.0)),
                    "feature_rows": len(row_indices),
                })
            print(f"[GSE236696] {sample}: cells={n_barcodes:,} total_umi={total_umi:,.0f}")
    return records


def parse_directory(path: Path):
    """Read the same 10x triplets after direct per-sample GEO download."""
    members_by_sample = defaultdict(dict)
    for item in path.glob("*.gz"):
        match = SAMPLE_PATTERN.match(item.name)
        if match:
            _, patient, condition, kind, _ = match.groups()
            members_by_sample[f"P{patient}{condition}"][kind] = item
    expected = {f"P{patient}{condition}" for patient in range(1, 7) for condition in ("N", "T")}
    if set(members_by_sample) != expected:
        raise RuntimeError(
            f"directory sample mismatch: missing={sorted(expected-set(members_by_sample))}, "
            f"extra={sorted(set(members_by_sample)-expected)}"
        )
    records = []
    for sample in sorted(expected, key=lambda value: (int(value[1]), value[2])):
        members = members_by_sample[sample]
        if set(members) != {"barcodes", "features", "matrix"}:
            raise RuntimeError(f"incomplete 10x triplet for {sample}: {sorted(members)}")
        with gzip.open(members["features"], "rt", encoding="utf-8") as handle:
            features = [line.rstrip("\n").split("\t") for line in handle]
        symbols = [row[1].upper() if len(row) > 1 else row[0].upper() for row in features]
        with gzip.open(members["barcodes"], "rt", encoding="utf-8") as handle:
            n_barcodes = sum(1 for _ in handle)
        matrix = mmread(members["matrix"]).tocsr()
        if matrix.shape != (len(features), n_barcodes):
            raise RuntimeError(
                f"shape mismatch for {sample}: matrix={matrix.shape}, "
                f"features={len(features)}, barcodes={n_barcodes}"
            )
        total_umi = float(matrix.sum())
        if total_umi <= 0:
            raise RuntimeError(f"zero total UMI for {sample}")
        symbol_rows = defaultdict(list)
        for row_index, symbol in enumerate(symbols):
            if symbol in TARGETS:
                symbol_rows[symbol].append(row_index)
        patient = int(sample[1])
        condition = "normal" if sample.endswith("N") else "tumor"
        for gene in TARGETS:
            row_indices = symbol_rows.get(gene, [])
            counts = float(matrix[row_indices].sum()) if row_indices else 0.0
            cpm = counts * 1_000_000.0 / total_umi
            records.append({
                "sample": sample,
                "patient": patient,
                "condition": condition,
                "gene": gene,
                "counts": counts,
                "total_umi": total_umi,
                "log2_cpm_plus1": float(np.log2(cpm + 1.0)),
                "feature_rows": len(row_indices),
            })
        print(f"[GSE236696] {sample}: cells={n_barcodes:,} total_umi={total_umi:,.0f}")
    return records


def summarize(records, seed: int):
    lookup = {(row["sample"], row["gene"]): row["log2_cpm_plus1"] for row in records}
    gene_results = []
    for gene in TARGETS:
        deltas = np.asarray([
            lookup[(f"P{patient}T", gene)] - lookup[(f"P{patient}N", gene)]
            for patient in range(1, 7)
        ])
        gene_results.append({
            "gene": gene,
            "mean_paired_delta": float(np.mean(deltas)),
            "median_paired_delta": float(np.median(deltas)),
            "tumor_higher_pairs": int(np.sum(deltas > 0)),
            "tumor_lower_pairs": int(np.sum(deltas < 0)),
            "exact_sign_flip_p": exact_sign_flip_p(deltas),
            "paired_deltas": ";".join(f"{value:.6g}" for value in deltas),
        })

    axis_results = []
    for axis, genes in AXES.items():
        sample_scores = {}
        for patient in range(1, 7):
            for condition in ("N", "T"):
                values = [lookup[(f"P{patient}{condition}", gene)] for gene in genes]
                sample_scores[f"P{patient}{condition}"] = float(np.median(values))
        deltas = np.asarray([
            sample_scores[f"P{patient}T"] - sample_scores[f"P{patient}N"]
            for patient in range(1, 7)
        ])
        axis_results.append({
            "axis": axis,
            "genes": len(genes),
            "mean_paired_delta": float(np.mean(deltas)),
            "median_paired_delta": float(np.median(deltas)),
            "tumor_higher_pairs": int(np.sum(deltas > 0)),
            "tumor_lower_pairs": int(np.sum(deltas < 0)),
            "exact_sign_flip_p": exact_sign_flip_p(deltas),
            "patient_bootstrap_95ci": patient_bootstrap(deltas, seed),
            "paired_deltas": ";".join(f"{value:.6g}" for value in deltas),
        })
    return gene_results, axis_results


def write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(path: Path, axis_results):
    labels = [row["axis"].replace("_", "\n") for row in axis_results]
    deltas = [row["mean_paired_delta"] for row in axis_results]
    colors = ["#b2182b" if value > 0 else "#2166ac" for value in deltas]
    figure, axis = plt.subplots(figsize=(10.5, 5.4))
    axis.bar(range(len(labels)), deltas, color=colors)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(range(len(labels)), labels)
    axis.set_ylabel("mean paired tumor-normal delta\n(median gene log2 CPM+1)")
    axis.set_title("GSE236696: six paired mucinous CRC whole-sample pseudobulks")
    for index, row in enumerate(axis_results):
        axis.text(index, deltas[index], f"{deltas[index]:+.2f}\n{row['tumor_higher_pairs']}/6 up",
                  ha="center", va="bottom" if deltas[index] >= 0 else "top", fontsize=9)
    figure.text(0.5, 0.01, "Patient is the statistical unit; whole-sample screen remains composition-sensitive.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(path, dpi=220)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("data/external/GSE236696/GSE236696_RAW.tar"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/external/GSE236696/raw_files"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/external/GSE236696/paired_axis_screen_v1"))
    parser.add_argument("--seed", type=int, default=20260830)
    arguments = parser.parse_args()
    if not arguments.raw_dir.is_dir() and not arguments.archive.is_file():
        raise FileNotFoundError(
            f"neither direct raw directory nor archive is available: "
            f"{arguments.raw_dir}, {arguments.archive}"
        )
    if arguments.output_dir.exists() and any(arguments.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {arguments.output_dir}")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    input_mode = "direct_raw_directory" if arguments.raw_dir.is_dir() else "tar_archive"
    records = (
        parse_directory(arguments.raw_dir)
        if input_mode == "direct_raw_directory"
        else parse_archive(arguments.archive)
    )
    gene_results, axis_results = summarize(records, arguments.seed)
    write_csv(arguments.output_dir / "sample_gene_pseudobulk.csv", records)
    write_csv(arguments.output_dir / "gene_paired_results.csv", gene_results)
    write_csv(arguments.output_dir / "axis_paired_results.csv", axis_results)
    render(arguments.output_dir / "paired_axis_deltas.png", axis_results)
    report = {
        "status": "gse236696_mucinous_axis_screen_complete",
        "formal": True,
        "patients": 6,
        "samples": 12,
        "statistical_unit": "patient",
        "axis_results": axis_results,
        "provenance": {
            "geo": "GSE236696",
            "input_mode": input_mode,
            "input": str(arguments.raw_dir if input_mode == "direct_raw_directory" else arguments.archive),
            "archive_sha256": sha256(arguments.archive) if input_mode == "tar_archive" else None,
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "Paired whole-sample pseudobulk screen of a mucinous CRC scRNA-seq cohort. "
            "It does not use cells as replicates, but remains sensitive to cell-composition "
            "changes and cannot validate metabolite identity, enzyme activity, or flux."
        ),
        "next_gate": (
            "Only axes with a coherent paired direction proceed to patient-by-cell-type "
            "pseudobulk analysis, with malignant epithelial cells prioritized."
        ),
    }
    (arguments.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
