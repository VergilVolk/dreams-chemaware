#!/usr/bin/env python
"""Test the NXPE1/sialic-acid O-acetylation context in current TCGA RNA-seq.

The primary question is whether NXPE1 expression differs between mucinous and
conventional primary colorectal tumours after the same clinical and lineage
adjustments used by the existing TCGA mechanism audit.  TPM is primary and
FPKM-UQ is a processing sensitivity analysis.  Gene expression is contextual
evidence only: it cannot establish enzyme activity, glycan abundance or flux.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

try:
    import xenaPython as xena
except ImportError as exc:  # pragma: no cover - fail-closed runtime guidance
    raise RuntimeError(
        "xenaPython is required: python -m pip install "
        "'git+https://github.com/ucscXena/xenaPython'"
    ) from exc

from analyze_gse236696_mucinous_axes_by_lineage import LINEAGE_MARKERS
from analyze_tcga_coadread_mucinous_axes import side, stage_group
from audit_tcga_proline_sialic_lineage_sensitivity_v1 import fit_hc3, standardize


ROOT = Path(__file__).resolve().parents[1]
HUB = xena.PUBLIC_HUBS["gdcHub"]
COHORTS = ("COAD", "READ")
UNITS = {
    "tpm": "star_tpm.tsv",
    "fpkm_uq": "star_fpkm-uq.tsv",
}
PRIMARY_GENE = "NXPE1"
CONTEXT_GENES = ("CASD1", "SIAE")
SECRETORY_MUCIN_GENES = ("MUC2", "TFF3", "SPDEF", "FCGBP", "AGR2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalise_sample(sample: str) -> str:
    """Map current GDC aliquot-like barcodes to legacy Xena sample barcodes."""
    return str(sample)[:15]


def fetch_unit(unit: str, suffix: str, genes: list[str]) -> tuple[pd.DataFrame, dict]:
    rows: list[pd.DataFrame] = []
    metadata: dict[str, dict] = {}
    for cohort in COHORTS:
        dataset = f"TCGA-{cohort}.{suffix}"
        samples = xena.dataset_samples(HUB, dataset, None)
        if not samples:
            raise RuntimeError(f"no samples returned for {dataset}")
        results = xena.dataset_gene_probe_avg(HUB, dataset, samples, genes)
        by_gene = {str(item["gene"]).upper(): item for item in results}
        missing = sorted(set(genes) - set(by_gene))
        if missing:
            raise RuntimeError(f"{dataset}: missing requested genes {missing}")
        frame = pd.DataFrame({"gdc_sample": samples})
        frame["sampleID"] = frame["gdc_sample"].map(normalise_sample)
        frame["gdc_cohort"] = cohort
        for gene in genes:
            scores = by_gene[gene]["scores"]
            if len(scores) != 1 or len(scores[0]) != len(samples):
                raise RuntimeError(f"{dataset}: malformed score vector for {gene}")
            frame[gene] = pd.to_numeric(pd.Series(scores[0]), errors="coerce")
        # Current GDC may retain multiple sample vials (e.g. 01A/01B) while the
        # legacy clinical matrix is sample-code level.  Average these explicitly
        # and preserve the source barcodes/count instead of choosing a vial.
        if frame["sampleID"].duplicated().any():
            aggregations = {gene: "mean" for gene in genes}
            aggregations.update({
                "gdc_sample": lambda values: ";".join(sorted(map(str, values))),
                "gdc_cohort": "first",
            })
            frame = frame.groupby("sampleID", as_index=False).agg(aggregations)
            frame["source_vial_count"] = frame["gdc_sample"].str.count(";") + 1
        else:
            frame["source_vial_count"] = 1
        rows.append(frame)
        raw_metadata = xena.dataset_metadata(HUB, dataset)
        metadata[dataset] = raw_metadata[0] if raw_metadata else {}
    combined = pd.concat(rows, ignore_index=True)
    if combined["sampleID"].duplicated().any():
        raise RuntimeError(f"{unit}: COAD/READ sample overlap after normalisation")
    return combined, metadata


def prepare_clinical(path: Path, available: set[str]) -> pd.DataFrame:
    clinical = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    conventional = {"Colon Adenocarcinoma", "Rectal Adenocarcinoma"}
    mucinous = {"Colon Mucinous Adenocarcinoma", "Rectal Mucinous Adenocarcinoma"}
    clinical = clinical[
        (clinical["sample_type"] == "Primary Tumor")
        & clinical["histological_type"].isin(conventional | mucinous)
        & clinical["sampleID"].isin(available)
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


def paired_deltas(frame: pd.DataFrame, gene: str) -> np.ndarray:
    frame = frame.copy()
    frame["patient"] = frame["sampleID"].str.slice(0, 12)
    frame["sample_code"] = frame["sampleID"].str.slice(13, 15)
    deltas: list[float] = []
    for _, group in frame.groupby("patient"):
        tumour = group[group["sample_code"] == "01"]
        normal = group[group["sample_code"] == "11"]
        if len(tumour) and len(normal):
            deltas.append(float(tumour.iloc[0][gene] - normal.iloc[0][gene]))
    return np.asarray(deltas, dtype=float)


def analyse_unit(
    unit: str,
    expression: pd.DataFrame,
    clinical_path: Path,
    genes: list[str],
    analysis_scope: str,
    locked_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    clinical = prepare_clinical(clinical_path, set(expression["sampleID"]))
    if locked_ids is not None:
        missing_locked = sorted(locked_ids - set(clinical["sampleID"]))
        if missing_locked:
            raise RuntimeError(
                f"{unit}: {len(missing_locked)} locked samples unavailable in current GDC data"
            )
        clinical = clinical[clinical["sampleID"].isin(locked_ids)].copy()
    merged = clinical.merge(expression, on="sampleID", how="left", validate="one_to_one")
    if merged[genes].isna().any().any():
        raise RuntimeError(f"{unit}: missing expression in retained clinical cohort")

    # Standardise over all current GDC COAD/READ samples, as the legacy audit did
    # over all samples in its expression matrix.
    z_all = {gene: standardize(expression[gene].to_numpy(float)) for gene in genes}
    position = {sample: i for i, sample in enumerate(expression["sampleID"])}
    positions = np.asarray([position[sample] for sample in merged["sampleID"]], dtype=np.int64)
    z = {gene: values[positions] for gene, values in z_all.items()}
    for gene in genes:
        merged[f"gene__{gene}"] = z[gene]

    lineage_columns: list[str] = []
    target_genes = {PRIMARY_GENE, *CONTEXT_GENES}
    for lineage, markers in LINEAGE_MARKERS.items():
        usable = sorted(set(markers) - target_genes)
        if len(usable) < 3 or not set(usable) <= set(z):
            raise RuntimeError(f"{unit}: incomplete lineage markers for {lineage}")
        column = f"lineage__{lineage}"
        merged[column] = np.mean(np.vstack([z[gene] for gene in usable]), axis=0)
        lineage_columns.append(column)

    secretory_column = "axis__secretory_mucin_program"
    merged[secretory_column] = np.mean(
        np.vstack([z[gene] for gene in SECRETORY_MUCIN_GENES]), axis=0,
    )

    merged["axis__nxpe1_minus_siae"] = (
        merged[f"gene__{PRIMARY_GENE}"] - merged["gene__SIAE"]
    ) / 2.0
    merged["axis__casd1_minus_siae"] = (
        merged["gene__CASD1"] - merged["gene__SIAE"]
    ) / 2.0
    outcomes = [
        ("gene", PRIMARY_GENE, f"gene__{PRIMARY_GENE}"),
        ("gene", "CASD1", "gene__CASD1"),
        ("gene", "SIAE", "gene__SIAE"),
        ("axis", "nxpe1_minus_siae", "axis__nxpe1_minus_siae"),
        ("axis", "casd1_minus_siae", "axis__casd1_minus_siae"),
    ]
    rows: list[dict] = []
    for outcome_type, outcome, column in outcomes:
        clinical_model = fit_hc3(merged, column, ["age"], ["side", "stage_group", "gender"])
        lineage_model = fit_hc3(
            merged, column, ["age", *lineage_columns], ["side", "stage_group", "gender"],
        )
        msi_model = fit_hc3(
            merged, column, ["age", *lineage_columns],
            ["side", "stage_group", "gender", "msi"],
        )
        secretory_model = fit_hc3(
            merged, column, ["age", *lineage_columns, secretory_column],
            ["side", "stage_group", "gender"],
        )
        secretory_msi_model = fit_hc3(
            merged, column, ["age", *lineage_columns, secretory_column],
            ["side", "stage_group", "gender", "msi"],
        )
        rows.append({
            "analysis_scope": analysis_scope,
            "unit": unit,
            "outcome_type": outcome_type,
            "outcome": outcome,
            "clinical_beta": clinical_model["beta"],
            "clinical_p": clinical_model["p"],
            "clinical_ci_low": clinical_model["ci_low"],
            "clinical_ci_high": clinical_model["ci_high"],
            "lineage_beta": lineage_model["beta"],
            "lineage_p": lineage_model["p"],
            "lineage_ci_low": lineage_model["ci_low"],
            "lineage_ci_high": lineage_model["ci_high"],
            "msi_lineage_n": msi_model["n"],
            "msi_lineage_beta": msi_model["beta"],
            "msi_lineage_p": msi_model["p"],
            "msi_lineage_ci_low": msi_model["ci_low"],
            "msi_lineage_ci_high": msi_model["ci_high"],
            "secretory_lineage_beta": secretory_model["beta"],
            "secretory_lineage_p": secretory_model["p"],
            "secretory_lineage_ci_low": secretory_model["ci_low"],
            "secretory_lineage_ci_high": secretory_model["ci_high"],
            "secretory_msi_lineage_n": secretory_msi_model["n"],
            "secretory_msi_lineage_beta": secretory_msi_model["beta"],
            "secretory_msi_lineage_p": secretory_msi_model["p"],
            "secretory_msi_lineage_ci_low": secretory_msi_model["ci_low"],
            "secretory_msi_lineage_ci_high": secretory_msi_model["ci_high"],
        })

    nxpe1_deltas = paired_deltas(expression, PRIMARY_GENE)
    finite = nxpe1_deltas[np.isfinite(nxpe1_deltas)]
    nonzero = finite[finite != 0]
    paired = {
        "n": int(len(finite)),
        "mean_tumour_minus_normal_log_expression": float(np.mean(finite)),
        "median_tumour_minus_normal_log_expression": float(np.median(finite)),
        "tumour_higher": int(np.sum(finite > 0)),
        "tumour_lower": int(np.sum(finite < 0)),
        "sign_p": float(binomtest(int(np.sum(nonzero > 0)), len(nonzero), .5).pvalue),
        "wilcoxon_p": float(wilcoxon(finite, zero_method="wilcox").pvalue),
    }
    cohort = {
        "primary_tumours": int(len(merged)),
        "mucinous": int(merged["mucinous"].sum()),
        "conventional": int((1 - merged["mucinous"]).sum()),
        "nxpe1_paired_tumour_normal": paired,
    }
    sample_columns = [
        "sampleID", "patient", "histological_type", "mucinous", "side",
        "stage_group", "age", "gender", "msi", *[f"gene__{g}" for g in genes],
        *lineage_columns, secretory_column,
    ]
    return pd.DataFrame(rows), {"cohort": cohort, "samples": merged[sample_columns]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clinical", type=Path,
        default=ROOT / "data/external/TCGA_COADREAD_Xena_20260830/COADREAD_clinicalMatrix.tsv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/external/TCGA_COADREAD_Xena_20260830/nxpe1_free_donor_v1",
    )
    parser.add_argument(
        "--locked-samples", type=Path,
        default=ROOT / (
            "data/external/TCGA_COADREAD_Xena_20260830/"
            "mucinous_axis_analysis_v4/analysis_samples.csv"
        ),
    )
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    for required in (args.clinical, args.locked_samples):
        if not required.is_file():
            raise FileNotFoundError(required)

    locked_frame = pd.read_csv(args.locked_samples)
    if len(locked_frame) != 371 or int(locked_frame["mucinous"].sum()) != 42:
        raise RuntimeError("locked legacy cohort must contain 371 tumours including 42 mucinous")
    locked_ids = set(locked_frame["sampleID"].astype(str))

    lineage_genes = {gene for markers in LINEAGE_MARKERS.values() for gene in markers}
    genes = sorted(
        {PRIMARY_GENE, *CONTEXT_GENES, *SECRETORY_MUCIN_GENES} | lineage_genes
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)

    result_frames: list[pd.DataFrame] = []
    unit_reports: dict[str, dict] = {}
    metadata: dict[str, dict] = {}
    for unit, suffix in UNITS.items():
        expression, unit_metadata = fetch_unit(unit, suffix, genes)
        expression.to_csv(args.output_dir / f"gdc_{unit}_requested_genes.csv.gz", index=False)
        locked_results, locked_details = analyse_unit(
            unit, expression, args.clinical, genes, "locked_legacy_371", locked_ids,
        )
        extended_results, extended_details = analyse_unit(
            unit, expression, args.clinical, genes, "extended_current_gdc", None,
        )
        locked_details["samples"].to_csv(
            args.output_dir / f"analysis_samples_{unit}_locked.csv", index=False,
        )
        extended_details["samples"].to_csv(
            args.output_dir / f"analysis_samples_{unit}_extended.csv", index=False,
        )
        result_frames.extend([locked_results, extended_results])
        unit_reports[unit] = {
            "locked_legacy_371": locked_details["cohort"],
            "extended_current_gdc": extended_details["cohort"],
        }
        metadata.update(unit_metadata)

    results = pd.concat(result_frames, ignore_index=True)
    results.to_csv(args.output_dir / "nxpe1_mechanism_results.csv", index=False)
    primary = results[
        (results["analysis_scope"] == "locked_legacy_371")
        & (results["unit"] == "tpm")
        & (results["outcome"] == PRIMARY_GENE)
    ].iloc[0]
    sensitivity = results[
        (results["analysis_scope"] == "locked_legacy_371")
        & (results["unit"] == "fpkm_uq")
        & (results["outcome"] == PRIMARY_GENE)
    ].iloc[0]
    extended = results[
        (results["analysis_scope"] == "extended_current_gdc")
        & (results["unit"] == "tpm")
        & (results["outcome"] == PRIMARY_GENE)
    ].iloc[0]
    direction_consistent = bool(
        np.sign(primary["lineage_beta"]) == np.sign(sensitivity["lineage_beta"])
    )
    report = {
        "status": "tcga_nxpe1_free_donor_mechanism_audit_complete",
        "formal": False,
        "hypothesis": (
            "NXPE1 is a mucin-associated sialic-acid O-acetyltransferase with reported free- "
            "and CMP-Neu5Ac acceptor contexts in vitro; mucinous-relative expression is tested "
            "as carrier-state context for the local free-Neu5Ac pool, not as substrate proof."
        ),
        "secretory_mucin_sensitivity_genes": list(SECRETORY_MUCIN_GENES),
        "primary_endpoint": (
            "TPM lineage-adjusted mucinous coefficient for NXPE1 in the exact prior "
            "371-tumour cohort (42 mucinous, 329 conventional)"
        ),
        "units": unit_reports,
        "results": results.to_dict(orient="records"),
        "primary_nxpe1": primary.to_dict(),
        "fpkm_uq_sensitivity_nxpe1": sensitivity.to_dict(),
        "extended_current_gdc_nxpe1": extended.to_dict(),
        "nxpe1_direction_consistent_across_units": direction_consistent,
        "gdc_metadata": metadata,
        "provenance": {
            "clinical_sha256": sha256(args.clinical),
            "locked_samples_sha256": sha256(args.locked_samples),
            "script_sha256": sha256(Path(__file__)),
            "gdc_hub": HUB,
        },
        "claim_limit": (
            "Current GDC bulk-RNA context in the legacy histology-defined TCGA cohort. "
            "Expression does not establish NXPE1 protein abundance, O-acetyltransferase activity, "
            "free-Neu5Ac consumption, MUC2 glycoform abundance, cellular origin or flux."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
