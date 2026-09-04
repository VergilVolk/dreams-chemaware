#!/usr/bin/env python
"""Audit transcriptomic mechanisms that could accompany a free-Neu5Ac pool.

The mechanisms are fixed before outcome inspection from sialic-acid biochemistry:
de-novo supply, CMP activation/transport, selected sialidase release, and
O-acetylation protection/removal.  The analysis is contextual only; bulk RNA
cannot identify the source of free Neu5Ac or establish enzyme activity/flux.
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
from scipy.stats import binomtest, wilcoxon

from analyze_gse236696_mucinous_axes_by_lineage import LINEAGE_MARKERS
from analyze_tcga_coadread_mucinous_axes import GENE_ALIASES, bh, side, stage_group
from audit_tcga_proline_sialic_lineage_sensitivity_v1 import fit_hc3, standardize


ROOT = Path(__file__).resolve().parents[1]

GENES = [
    "GNE", "NANS", "NANP", "CMAS", "SLC35A1",
    "NEU1", "NEU2", "NEU3", "NEU4", "CASD1", "SIAE",
]

# Signed weights are explicit.  CASD1 adds O-acetylation and SIAE removes it;
# NEU1/NEU3 are kept together only as a deliberately narrow release axis.
AXES = {
    "de_novo_supply": {"GNE": 1.0, "NANS": 1.0, "NANP": 1.0},
    "cmp_activation_transport": {"CMAS": 1.0, "SLC35A1": 1.0},
    "selected_sialidase_release": {"NEU1": 1.0, "NEU3": 1.0},
    "o_acetyl_protection_balance": {"CASD1": 1.0, "SIAE": -1.0},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_expression(path: Path, requested: set[str]) -> tuple[list[str], dict[str, np.ndarray]]:
    values: dict[str, np.ndarray] = {}
    source_to_current = {GENE_ALIASES.get(gene, gene): gene for gene in requested}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        samples = next(reader)[1:]
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
    return clinical


def paired_samples(samples: list[str]) -> list[tuple[str, str, str]]:
    members: dict[str, dict[str, str]] = {}
    for sample in sorted(samples):
        if len(sample) >= 15 and sample[13:15] in {"01", "11"}:
            members.setdefault(sample[:12], {}).setdefault(sample[13:15], sample)
    return [
        (patient, rows["01"], rows["11"])
        for patient, rows in sorted(members.items())
        if {"01", "11"} <= set(rows)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clinical", type=Path, default=ROOT / (
        "data/external/TCGA_COADREAD_Xena_20260830/COADREAD_clinicalMatrix.tsv"))
    parser.add_argument("--expression", type=Path, default=ROOT / (
        "data/external/TCGA_COADREAD_Xena_20260830/HiSeqV2.gz"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / (
        "data/external/TCGA_COADREAD_Xena_20260830/sialic_pool_mechanisms_v1"))
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    for path in (args.clinical, args.expression):
        if not path.is_file():
            raise FileNotFoundError(path)

    requested = set(GENES) | {g for genes in LINEAGE_MARKERS.values() for g in genes}
    samples, expression = read_expression(args.expression, requested)
    missing = sorted(set(GENES) - set(expression))
    if missing:
        raise RuntimeError(f"missing prespecified genes: {missing}")
    sample_index = {sample: index for index, sample in enumerate(samples)}
    clinical = prepare_clinical(args.clinical, sample_index)
    positions = np.asarray([sample_index[sample] for sample in clinical["sampleID"]])
    z_all = {gene: standardize(values) for gene, values in expression.items()}
    z = {gene: values[positions] for gene, values in z_all.items()}

    outcomes: list[dict] = []
    outcome_members: dict[str, set[str]] = {}
    for gene in GENES:
        name = f"gene__{gene}"
        clinical[name] = z[gene]
        outcomes.append({"outcome_type": "gene", "outcome": gene, "column": name})
        outcome_members[name] = {gene}
    for axis, weights in AXES.items():
        name = f"axis__{axis}"
        denom = sum(abs(weight) for weight in weights.values())
        clinical[name] = sum(z[gene] * weight for gene, weight in weights.items()) / denom
        outcomes.append({"outcome_type": "axis", "outcome": axis, "column": name})
        outcome_members[name] = set(weights)

    base_continuous = ["age"]
    base_categorical = ["side", "stage_group", "gender"]
    pairs = paired_samples(samples)
    rows: list[dict] = []
    for item in outcomes:
        column = item["column"]
        genes = outcome_members[column]
        lineage_columns: list[str] = []
        for lineage, markers in LINEAGE_MARKERS.items():
            usable = sorted((set(markers) & set(z)) - genes)
            if len(usable) < 3:
                raise RuntimeError(f"{column}: insufficient non-overlapping {lineage} markers")
            proxy = f"tmp__{item['outcome']}__{lineage}"
            clinical[proxy] = np.mean(np.vstack([z[gene] for gene in usable]), axis=0)
            lineage_columns.append(proxy)

        clinical_model = fit_hc3(clinical, column, base_continuous, base_categorical)
        lineage_model = fit_hc3(
            clinical, column, [*base_continuous, *lineage_columns], base_categorical,
        )
        msi_model = fit_hc3(
            clinical, column, [*base_continuous, *lineage_columns],
            [*base_categorical, "msi"],
        )

        if item["outcome_type"] == "gene":
            all_values = z_all[item["outcome"]]
        else:
            weights = AXES[item["outcome"]]
            denom = sum(abs(weight) for weight in weights.values())
            all_values = sum(z_all[gene] * weight for gene, weight in weights.items()) / denom
        deltas = np.asarray([
            all_values[sample_index[tumour]] - all_values[sample_index[normal]]
            for _, tumour, normal in pairs
        ])
        nonzero = deltas[deltas != 0]
        sign_p = float(binomtest(int(np.sum(nonzero > 0)), len(nonzero), .5).pvalue)
        wilcoxon_p = float(wilcoxon(deltas, zero_method="wilcox").pvalue)
        rows.append({
            **item,
            "members": ";".join(sorted(genes)),
            "paired_n": len(deltas),
            "paired_mean_tumour_minus_normal_z": float(np.mean(deltas)),
            "paired_tumour_higher": int(np.sum(deltas > 0)),
            "paired_sign_p": sign_p,
            "paired_wilcoxon_p": wilcoxon_p,
            "clinical_beta": clinical_model["beta"],
            "clinical_p": clinical_model["p"],
            "lineage_beta": lineage_model["beta"],
            "lineage_p": lineage_model["p"],
            "lineage_ci_low": lineage_model["ci_low"],
            "lineage_ci_high": lineage_model["ci_high"],
            "msi_lineage_n": msi_model["n"],
            "msi_lineage_beta": msi_model["beta"],
            "msi_lineage_p": msi_model["p"],
        })

    for field in ("paired_wilcoxon", "clinical", "lineage", "msi_lineage"):
        qvalues = bh([row[f"{field}_p"] for row in rows])
        for row, qvalue in zip(rows, qvalues):
            row[f"{field}_bh_q"] = qvalue

    frame = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "sialic_pool_mechanism_results.csv", index=False)
    report = {
        "status": "tcga_sialic_pool_mechanism_audit_complete",
        "formal": False,
        "samples": {
            "primary_tumours": int(len(clinical)),
            "mucinous": int(clinical["mucinous"].sum()),
            "conventional": int((1 - clinical["mucinous"]).sum()),
            "paired_tumour_normal": len(pairs),
        },
        "prespecified_genes": GENES,
        "prespecified_axes": AXES,
        "multiple_testing_family": len(rows),
        "results": rows,
        "source_anchors": {
            "NXPE1_colon_o_acetylation": "Nature Communications 2025; DOI 10.1038/s41467-025-59671-9",
            "NXPE1_9_o_acetylation": "JACS 2025; DOI 10.1021/jacs.5c00769",
            "NEU3_colon_cancer": "PNAS 2002; PMID 12114515",
        },
        "provenance": {
            "clinical_sha256": sha256(args.clinical),
            "expression_sha256": sha256(args.expression),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "Prespecified bulk-RNA mechanism context. Gene expression does not establish enzyme activity, "
            "subcellular localisation, free-Neu5Ac source, glycan destination or flux. NXPE1 was absent "
            "from the legacy TCGA expression matrix and therefore was not substituted post hoc."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
