#!/usr/bin/env python
"""Test a structure-anchored mucinous CRC O-glycan branching model in TCGA.

The gene sets are fixed from the biosynthetic interpretation of the independent
CRC PGC-LC-MS/MS O-glycomics study (PMCID: PMC9254241), before examining the
outcomes produced by this script.  The central question is not whether CRC is
globally "sialyl-high", but whether mucinous tumours redistribute biosynthetic
capacity between donor supply, tumour-associated core-2/alpha2-3/sLeX branches,
normal-mucosal core-3/Sda branches, and specific alpha2-6 routes.

All outcomes are transcript abundance.  They do not measure glycan structures,
enzyme activity, Neu5Ac flux, or cell-surface sialylation.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_gse236696_mucinous_axes_by_lineage import LINEAGE_MARKERS
from analyze_tcga_coadread_mucinous_axes import GENE_ALIASES, bh, side, stage_group
from audit_tcga_proline_sialic_lineage_sensitivity_v1 import fit_hc3, standardize


BRANCHES = {
    "neu5ac_donor_supply_transport": ["GNE", "NANS", "NANP", "CMAS", "SLC35A1"],
    "core2_slex_biosynthesis": [
        "C1GALT1", "GCNT1", "GCNT4", "B4GALT2", "B4GALT3", "FUT4",
    ],
    "alpha23_o_glycan_sialylation": ["ST3GAL1", "ST3GAL2"],
    "normal_mucosal_core3_sda": ["B3GNT6", "B4GALNT2"],
    "secretory_mucin_program": ["MUC2", "TFF3", "SPDEF", "AGR2", "FCGBP"],
}

# These single-gene routes are deliberately not averaged together.  ST6GAL1
# mainly reports an N-glycan alpha2-6 route, whereas ST6GALNAC1/3 act on
# O-GalNAc substrates with different structural consequences.
ROUTE_GENES = ["ST6GAL1", "ST6GALNAC1", "ST6GALNAC3", "GCNT3"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_expression(path: Path, requested: set[str]) -> tuple[list[str], dict[str, np.ndarray]]:
    values: dict[str, np.ndarray] = {}
    source_to_current = {GENE_ALIASES.get(gene, gene): gene for gene in requested}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        samples = header[1:]
        for row in reader:
            source_gene = row[0].upper()
            if source_gene in source_to_current:
                values[source_to_current[source_gene]] = np.asarray(row[1:], dtype=float)
    return samples, values


def prepare_clinical(path: Path, sample_index: dict[str, int]) -> pd.DataFrame:
    clinical = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    conventional = {"Colon Adenocarcinoma", "Rectal Adenocarcinoma"}
    mucinous = {"Colon Mucinous Adenocarcinoma", "Rectal Mucinous Adenocarcinoma"}
    clinical = clinical[
        (clinical["sample_type"] == "Primary Tumor")
        & clinical["histological_type"].isin(conventional | mucinous)
        & clinical["sampleID"].isin(sample_index)
    ].copy()
    clinical["patient"] = clinical["sampleID"].str.slice(0, 12)
    clinical = clinical.sort_values("sampleID").drop_duplicates("patient", keep="first")
    clinical["mucinous"] = clinical["histological_type"].isin(mucinous).astype(int)
    clinical["side"] = clinical["anatomic_neoplasm_subdivision"].map(side)
    clinical["stage_group"] = clinical["pathologic_stage"].map(stage_group)
    clinical["age"] = pd.to_numeric(
        clinical["age_at_initial_pathologic_diagnosis"], errors="coerce",
    )
    standardized_msi = clinical["CDE_ID_3226963"].replace(
        {"": "unknown", "Indeterminate": "unknown"},
    )
    updated_msi = clinical["MSI_updated_Oct62011"].replace("", "unknown")
    legacy_msi = clinical["microsatellite_instability"].map(
        {"YES": "MSI-H", "NO": "MSS"},
    ).fillna("unknown")
    clinical["msi"] = np.where(
        standardized_msi != "unknown", standardized_msi,
        np.where(updated_msi != "unknown", updated_msi, legacy_msi),
    )
    if clinical["mucinous"].sum() < 30 or (1 - clinical["mucinous"]).sum() < 100:
        raise RuntimeError("insufficient histology groups after expression matching")
    return clinical


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clinical", type=Path, default=Path(
        "data/external/TCGA_COADREAD_Xena_20260830/COADREAD_clinicalMatrix.tsv"))
    parser.add_argument("--expression", type=Path, default=Path(
        "data/external/TCGA_COADREAD_Xena_20260830/HiSeqV2.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "data/external/TCGA_COADREAD_Xena_20260830/glycan_branching_v2"))
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_genes = {gene for genes in BRANCHES.values() for gene in genes} | set(ROUTE_GENES)
    requested = target_genes | {gene for genes in LINEAGE_MARKERS.values() for gene in genes}
    samples, expression = read_expression(args.expression, requested)
    missing_targets = sorted(target_genes - set(expression))
    if missing_targets:
        raise RuntimeError(f"missing frozen glycan genes: {missing_targets}")
    sample_index = {sample: index for index, sample in enumerate(samples)}
    clinical = prepare_clinical(args.clinical, sample_index)
    positions = np.asarray([sample_index[sample] for sample in clinical["sampleID"]])
    z = {gene: standardize(values[positions]) for gene, values in expression.items()}

    for lineage, markers in LINEAGE_MARKERS.items():
        present = sorted(set(markers) & set(z))
        if len(present) < 3:
            raise RuntimeError(f"insufficient observable {lineage} markers: {present}")

    base_continuous = ["age"]
    base_categorical = ["side", "stage_group", "gender"]
    outcomes: list[tuple[str, str, list[str]]] = []
    for branch, genes in BRANCHES.items():
        column = f"branch__{branch}"
        clinical[column] = np.mean(np.vstack([z[gene] for gene in genes]), axis=0)
        outcomes.append(("branch", branch, genes))
    # Report every component gene, not only the four routes that were kept out
    # of composite axes.  This makes opposing components visible and prevents
    # an apparently null branch mean from hiding a strong internal divergence.
    for gene in sorted(target_genes):
        column = f"gene__{gene}"
        clinical[column] = z[gene]
        outcomes.append((
            "route_gene" if gene in ROUTE_GENES else "component_gene", gene, [gene],
        ))

    # A signed, structure-anchored contrast.  Positive values indicate more
    # core-2/alpha2-3/sLeX capacity relative to the normal-mucosal core-3/Sda
    # branch.  It is a transcriptomic branching score, not a glycan ratio.
    tumour_columns = [
        "branch__core2_slex_biosynthesis", "branch__alpha23_o_glycan_sialylation",
    ]
    clinical["branch__tumour_vs_mucosal_balance"] = (
        clinical[tumour_columns].mean(axis=1) - clinical["branch__normal_mucosal_core3_sda"]
    )
    outcomes.append((
        "derived_branch_contrast", "tumour_vs_mucosal_balance",
        BRANCHES["core2_slex_biosynthesis"]
        + BRANCHES["alpha23_o_glycan_sialylation"]
        + BRANCHES["normal_mucosal_core3_sda"],
    ))

    rows: list[dict] = []
    for outcome_type, name, genes in outcomes:
        outcome = (
            f"gene__{name}"
            if outcome_type in {"route_gene", "component_gene"}
            else f"branch__{name}"
        )
        lineage_columns: list[str] = []
        excluded: dict[str, list[str]] = {}
        for lineage, markers in LINEAGE_MARKERS.items():
            usable = sorted((set(markers) & set(z)) - set(genes))
            if len(usable) < 3:
                raise RuntimeError(f"{name}: fewer than three non-overlapping {lineage} markers")
            column = f"tmp__{name}__{lineage}"
            clinical[column] = np.mean(np.vstack([z[gene] for gene in usable]), axis=0)
            lineage_columns.append(column)
            excluded[lineage] = sorted(set(markers) & set(genes))

        clinical_model = fit_hc3(clinical, outcome, base_continuous, base_categorical)
        lineage_model = fit_hc3(
            clinical, outcome, [*base_continuous, *lineage_columns], base_categorical,
        )
        msi_lineage_model = fit_hc3(
            clinical, outcome, [*base_continuous, *lineage_columns],
            [*base_categorical, "msi"],
        )
        rows.append({
            "outcome_type": outcome_type,
            "outcome": name,
            "genes": ";".join(genes),
            "clinical_beta": clinical_model["beta"],
            "clinical_p": clinical_model["p"],
            "clinical_ci_low": clinical_model["ci_low"],
            "clinical_ci_high": clinical_model["ci_high"],
            "lineage_beta": lineage_model["beta"],
            "lineage_p": lineage_model["p"],
            "lineage_ci_low": lineage_model["ci_low"],
            "lineage_ci_high": lineage_model["ci_high"],
            "msi_lineage_n": msi_lineage_model["n"],
            "msi_lineage_estimable": msi_lineage_model["estimable"],
            "msi_lineage_beta": msi_lineage_model["beta"],
            "msi_lineage_p": msi_lineage_model["p"],
            "excluded_lineage_overlap": json.dumps(excluded, sort_keys=True),
        })

    for field in ("clinical", "lineage", "msi_lineage"):
        valid = [row[f"{field}_p"] for row in rows if row[f"{field}_p"] is not None]
        qvalues = iter(bh(valid))
        for row in rows:
            row[f"{field}_bh_q"] = next(qvalues) if row[f"{field}_p"] is not None else None

    results = pd.DataFrame(rows)
    results.to_csv(args.output_dir / "glycan_branch_results.csv", index=False)
    keep = [
        "sampleID", "patient", "histological_type", "mucinous", "side",
        "stage_group", "age", "gender", "msi",
        *[column for column in clinical.columns if column.startswith(("branch__", "gene__"))],
    ]
    clinical[keep].to_csv(args.output_dir / "analysis_samples.csv", index=False)

    report = {
        "status": "tcga_mucinous_glycan_branching_audit_complete",
        "formal": False,
        "analysis_role": (
            "structure-anchored transcriptomic convergence analysis defined from an independent "
            "CRC O-glycomics biosynthetic model; not a blind glycomics replication"
        ),
        "samples": {
            "total": int(len(clinical)),
            "mucinous": int(clinical["mucinous"].sum()),
            "conventional": int((1 - clinical["mucinous"]).sum()),
            "msi_complete": int(clinical["msi"].isin(["MSS", "MSI-L", "MSI-H"]).sum()),
        },
        "frozen_branches": BRANCHES,
        "route_genes_kept_separate": ROUTE_GENES,
        "models": {
            "clinical": "HC3 OLS: outcome ~ mucinous + age + side + stage + sex",
            "lineage": "clinical model + six broad-lineage expression scores",
            "msi_lineage": "lineage model + MSI, complete cases only",
        },
        "results": rows,
        "provenance": {
            "clinical_sha256": sha256(args.clinical),
            "expression_sha256": sha256(args.expression),
            "script_sha256": sha256(Path(__file__)),
            "external_structure_source": "PMCID: PMC9254241",
        },
        "claim_limit": (
            "This analysis tests transcriptomic capacity of predefined glycan branches. It does not "
            "measure glycan abundance, linkage, carrier proteins, enzyme activity, Neu5Ac flux, or "
            "cell-surface sialylation. TCGA overlaps prior project analyses and is contextual evidence, "
            "not independent metabolomic replication."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"], "samples": report["samples"],
        "results": [
            {
                "outcome": row["outcome"], "lineage_beta": row["lineage_beta"],
                "lineage_q": row["lineage_bh_q"], "msi_beta": row["msi_lineage_beta"],
                "msi_q": row["msi_lineage_bh_q"],
            }
            for row in rows
        ],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
