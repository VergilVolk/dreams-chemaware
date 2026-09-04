#!/usr/bin/env python
"""Adversarial robustness audit for the GSE236696 epithelial axis result.

This program does not revisit cell assignment.  It treats the frozen broad-
lineage pseudobulk table as input and asks whether the reported epithelial
effects survive alternative score definitions, multiplicity correction,
leave-one-gene-out perturbation, leave-one-patient-out perturbation, and a
target-gene-universe random-set null.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


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


def exact_sign_flip(values: np.ndarray, direction: int | None = None) -> float:
    observed = float(np.mean(values))
    permuted = []
    for mask in range(1 << len(values)):
        signs = np.asarray([1.0 if mask & (1 << bit) else -1.0 for bit in range(len(values))])
        permuted.append(float(np.mean(values * signs)))
    permuted = np.asarray(permuted)
    if direction is None:
        return float(np.mean(np.abs(permuted) >= abs(observed) - 1e-12))
    return float(np.mean(direction * permuted >= direction * observed - 1e-12))


def holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def bh(values: list[float]) -> list[float]:
    order = np.argsort(values)[::-1]
    adjusted = np.empty(len(values), dtype=float)
    running = 1.0
    for reverse_rank, index in enumerate(order):
        rank = len(values) - reverse_rank
        running = min(running, values[index] * len(values) / rank)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def read_lookup(path: Path):
    lookup = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["lineage"] != "epithelial":
                continue
            lookup[(row["sample"], row["gene"])] = float(row["log2_cpm_plus1"])
    return lookup


def patient_gene_deltas(lookup, genes):
    matrix = []
    for patient in range(1, 7):
        matrix.append([
            lookup[(f"P{patient}T", gene)] - lookup[(f"P{patient}N", gene)]
            for gene in genes
        ])
    return np.asarray(matrix, dtype=float)


def current_score_deltas(lookup, genes):
    values = []
    for patient in range(1, 7):
        normal = [lookup[(f"P{patient}N", gene)] for gene in genes]
        tumor = [lookup[(f"P{patient}T", gene)] for gene in genes]
        values.append(float(np.median(tumor) - np.median(normal)))
    return np.asarray(values)


def summarize(values, expected=None):
    result = {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "positive_pairs": int(np.sum(values > 0)),
        "negative_pairs": int(np.sum(values < 0)),
        "two_sided_exact_p": exact_sign_flip(values),
    }
    if expected is not None:
        result["directional_exact_p"] = exact_sign_flip(values, expected)
    return result


def write_csv(path: Path, rows):
    if not rows:
        raise RuntimeError(f"refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pseudobulk",
        type=Path,
        default=Path("data/external/GSE236696/paired_axis_by_lineage_v3/lineage_gene_pseudobulk.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/external/GSE236696/epithelial_axis_adversarial_audit_v1"),
    )
    parser.add_argument("--random-sets", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lookup = read_lookup(args.pseudobulk)
    required = {(f"P{p}{c}", gene) for p in range(1, 7) for c in "NT"
                for genes in AXES.values() for gene in genes}
    missing = sorted(required - set(lookup))
    if missing:
        raise RuntimeError(f"missing frozen epithelial pseudobulk cells: {missing[:5]}")

    rows = []
    details = {}
    two_sided = []
    axis_names = list(AXES)
    for axis, genes in AXES.items():
        expected = EXPECTED_DIRECTION.get(axis)
        gene_delta = patient_gene_deltas(lookup, genes)
        variants = {
            "difference_of_gene_medians": current_score_deltas(lookup, genes),
            "median_of_paired_gene_deltas": np.median(gene_delta, axis=1),
            "mean_of_paired_gene_deltas": np.mean(gene_delta, axis=1),
        }
        primary = summarize(variants["difference_of_gene_medians"], expected)
        two_sided.append(primary["two_sided_exact_p"])
        for method, values in variants.items():
            row = {"axis": axis, "score_method": method, **summarize(values, expected)}
            rows.append(row)

        loo_gene = []
        for gene in genes:
            kept = [candidate for candidate in genes if candidate != gene]
            values = current_score_deltas(lookup, kept)
            loo_gene.append({"omitted_gene": gene, **summarize(values, expected)})
        base_values = variants["difference_of_gene_medians"]
        loo_patient = []
        for patient in range(6):
            values = np.delete(base_values, patient)
            loo_patient.append({"omitted_patient": patient + 1, **summarize(values, expected)})
        details[axis] = {
            "score_variants": {method: summarize(values, expected) for method, values in variants.items()},
            "leave_one_gene_out": loo_gene,
            "leave_one_patient_out": loo_patient,
        }

    adjusted_holm = holm(two_sided)
    adjusted_bh = bh(two_sided)
    multiplicity = []
    for axis, p_value, p_holm, p_bh in zip(axis_names, two_sided, adjusted_holm, adjusted_bh):
        multiplicity.append({
            "axis": axis,
            "two_sided_exact_p": p_value,
            "holm_5_axes": p_holm,
            "bh_5_axes": p_bh,
        })
    primary_axes = list(EXPECTED_DIRECTION)
    primary_p = [two_sided[axis_names.index(axis)] for axis in primary_axes]
    primary_holm = holm(primary_p)
    directional_p = [
        details[axis]["score_variants"]["difference_of_gene_medians"]["directional_exact_p"]
        for axis in primary_axes
    ]
    directional_holm = holm(directional_p)

    rng = np.random.default_rng(args.seed)
    universe = sorted({gene for genes in AXES.values() for gene in genes})
    random_null = {}
    for axis in primary_axes:
        observed = details[axis]["score_variants"]["difference_of_gene_medians"]["mean"]
        direction = EXPECTED_DIRECTION[axis]
        null = []
        for _ in range(args.random_sets):
            genes = rng.choice(universe, size=len(AXES[axis]), replace=False).tolist()
            null.append(float(np.mean(current_score_deltas(lookup, genes))))
        null = np.asarray(null)
        random_null[axis] = {
            "universe": "union of the five prespecified metabolic target sets",
            "draws": args.random_sets,
            "observed_mean": observed,
            "null_mean": float(np.mean(null)),
            "null_p05": float(np.quantile(null, 0.05)),
            "null_p95": float(np.quantile(null, 0.95)),
            "directional_empirical_p": float((1 + np.sum(direction * null >= direction * observed)) / (1 + len(null))),
            "claim_limit": "target-gene-universe specificity check, not a genome-wide matched-expression null",
        }

    write_csv(args.output_dir / "score_method_results.csv", rows)
    write_csv(args.output_dir / "multiplicity.csv", multiplicity)
    report = {
        "status": "gse236696_epithelial_axis_adversarial_audit_complete",
        "formal": True,
        "frozen_input": str(args.pseudobulk),
        "five_axis_multiplicity": multiplicity,
        "two_primary_axis_two_sided_holm": [
            {"axis": axis, "raw_p": p, "holm_p": adj}
            for axis, p, adj in zip(primary_axes, primary_p, primary_holm)
        ],
        "two_primary_axis_directional_internal_holm": [
            {"axis": axis, "raw_p": p, "holm_p": adj}
            for axis, p, adj in zip(primary_axes, directional_p, directional_holm)
        ],
        "axis_robustness": details,
        "target_gene_universe_random_null": random_null,
        "provenance": {"input_sha256": sha256(args.pseudobulk), "script_sha256": sha256(Path(__file__))},
        "claim_limit": (
            "The directional test was internally prespecified by prior metabolomics/proteomics but was not "
            "publicly time-stamped. Marker assignment is frozen but not the authors' original Seurat annotation."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "two_primary_axis_two_sided_holm": report["two_primary_axis_two_sided_holm"],
        "two_primary_axis_directional_internal_holm": report["two_primary_axis_directional_internal_holm"],
        "random_null": random_null,
    }, indent=2))


if __name__ == "__main__":
    main()
