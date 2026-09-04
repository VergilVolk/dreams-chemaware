#!/usr/bin/env python3
"""Post-result, patient-level epithelial composition diagnostic for GSE178341."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as student_t


ROOT = Path(__file__).resolve().parents[1]
TASKS = Path(__file__).resolve().parent
sys.path.insert(0, str(TASKS))
import audit_gse178341_nxpe1_mucinous_v1 as core  # noqa: E402


BASE = ROOT / "data/external/GSE178341_mucinous_secretory_audit"
PATIENT_MATRIX = BASE / "nxpe1_mucinous_patient_pseudobulk_v1/broad_epithelial_patient_matrix.csv"
MATCHES = BASE / "metadata_matches_v1.csv"
DEFAULT_OUT = BASE / "epithelial_composition_diagnostic_v1"
CONTRACT = ROOT / "docs/GSE178341_EPITHELIAL_COMPOSITION_DIAGNOSTIC_CONTRACT_20260831.md"
GOBLET = ("cE02", "cE06", "cE07", "cE08")
ENDPOINTS = (
    "SECRETORY_COMPOSITE",
    "SIALIC_BACKGROUND",
    "AGR2",
    "SLC35A1",
    "MUC2",
    "SPDEF",
    "NXPE1",
)
SEED = 20260831


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hc3(frame: pd.DataFrame, outcome: str, covariates: tuple[str, ...]) -> dict[str, float | int | list[str]]:
    work = frame.dropna(subset=[outcome, *covariates]).copy()
    columns = [np.ones(len(work)), work["histology"].eq(core.PURE_MUC).to_numpy(float)]
    names = ["intercept", "mucinous"]
    for covariate in covariates:
        value = work[covariate].to_numpy(float)
        if np.std(value) <= 1e-12:
            continue
        value = (value - np.mean(value)) / np.std(value)
        columns.append(value)
        names.append(covariate)
    x = np.column_stack(columns)
    y = work[outcome].to_numpy(float)
    inverse = np.linalg.pinv(x.T @ x)
    beta = inverse @ x.T @ y
    residual = y - x @ beta
    leverage = np.sum((x @ inverse) * x, axis=1)
    scaled = residual / np.maximum(1.0 - leverage, 1e-6)
    covariance = inverse @ (x.T @ ((scaled * scaled)[:, None] * x)) @ inverse
    se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    position = names.index("mucinous")
    df = max(len(work) - x.shape[1], 1)
    critical = float(student_t.ppf(0.975, df))
    statistic = float(beta[position] / max(se[position], 1e-12))
    return {
        "n_patients": int(len(work)),
        "parameters": names,
        "mucinous_beta": float(beta[position]),
        "HC3_se": float(se[position]),
        "ci_low": float(beta[position] - critical * se[position]),
        "ci_high": float(beta[position] + critical * se[position]),
        "p": float(2.0 * student_t.sf(abs(statistic), df)),
        "condition_number": float(np.linalg.cond(x)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    for path in (core.META, core.CLUSTERS, PATIENT_MATRIX, MATCHES, CONTRACT):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    meta = pd.read_csv(core.META)
    clusters = pd.read_csv(core.CLUSTERS)
    if not meta["cellID"].astype(str).equals(clusters["sampleID"].astype(str)):
        raise RuntimeError("official metadata and cluster tables are not row-aligned")
    frame = pd.concat(
        [meta.reset_index(drop=True), clusters.drop(columns=["sampleID"]).reset_index(drop=True)],
        axis=1,
    )
    tumour_epi = (
        frame["SPECIMEN_TYPE"].eq("T")
        & frame["HistologicTypeSimple"].isin((core.PURE_ADENO, core.PURE_MUC))
        & frame["clTopLevel"].eq("Epi")
    )
    rows: list[dict[str, object]] = []
    for patient, group in frame.loc[tumour_epi].groupby("PID", sort=True, observed=True):
        total = int(len(group))
        goblet = int(group["cl295v11SubShort"].isin(GOBLET).sum())
        mature = int(group["cl295v11SubShort"].eq("cE08").sum())
        first = group.sort_values("PatientTypeID").iloc[0]
        rows.append({
            "PID": str(patient),
            "histology": first["HistologicTypeSimple"],
            "MMRd": float(first["MMRStatus"] == "MMRd"),
            "right": float(first["TissueSiteSimple"] == "right"),
            "epithelial_cells": total,
            "goblet_cells": goblet,
            "mature_goblet_cells": mature,
            "goblet_fraction": goblet / total,
            "mature_goblet_fraction": mature / total,
            "logit_goblet_fraction": float(np.log((goblet + 0.5) / (total - goblet + 0.5))),
            "logit_mature_goblet_fraction": float(np.log((mature + 0.5) / (total - mature + 0.5))),
        })
    composition = pd.DataFrame(rows).sort_values("PID").reset_index(drop=True)
    observed = composition["histology"].value_counts().to_dict()
    if observed != {core.PURE_ADENO: 53, core.PURE_MUC: 6}:
        raise RuntimeError(f"patient counts changed: {observed}")

    expression = pd.read_csv(PATIENT_MATRIX)
    merged = expression.merge(composition, on=["PID", "histology"], how="inner", validate="one_to_one")
    if len(merged) != 59:
        raise RuntimeError(f"patient merge changed size: {len(merged)}")

    fraction_results = []
    for index, endpoint in enumerate(("goblet_fraction", "mature_goblet_fraction")):
        result = core.group_result(merged, endpoint, "all_pure_tumours", False, SEED + index)
        fraction_results.append({"endpoint": endpoint, **result})

    matches = pd.read_csv(MATCHES, dtype=str)
    lookup = merged.set_index("PID")
    matched_rows = []
    for endpoint in ("goblet_fraction", "mature_goblet_fraction"):
        contrasts = []
        for case, group in matches.groupby("case_PID", sort=True):
            controls = group["control_PID"].tolist()
            contrasts.append(float(lookup.loc[case, endpoint] - lookup.loc[controls, endpoint].mean()))
        matched_rows.append({
            "endpoint": endpoint,
            "n_cases": len(contrasts),
            "positive_cases": int(np.sum(np.asarray(contrasts) > 0)),
            "mean_contrast": float(np.mean(contrasts)),
            "exact_sign_flip_p": float(core.exact_sign_flip(np.asarray(contrasts))),
        })

    model_rows = []
    correlation_rows = []
    for endpoint in ENDPOINTS:
        unadjusted = hc3(merged, endpoint, ())
        compact = hc3(merged, endpoint, ("logit_goblet_fraction",))
        clinical = hc3(merged, endpoint, ("logit_goblet_fraction", "right", "MMRd"))
        retention = compact["mucinous_beta"] / unadjusted["mucinous_beta"] if abs(unadjusted["mucinous_beta"]) > 1e-12 else np.nan
        residual_state = bool(
            unadjusted["mucinous_beta"] > 0
            and compact["mucinous_beta"] > 0
            and retention >= 0.25
            and compact["ci_low"] > 0
        )
        for name, result in (("histology_only", unadjusted), ("plus_goblet_fraction", compact), ("plus_goblet_right_mmr", clinical)):
            model_rows.append({"endpoint": endpoint, "model": name, **result})
        rho, p_value = spearmanr(merged["goblet_fraction"], merged[endpoint])
        correlation_rows.append({
            "endpoint": endpoint,
            "spearman_rho_with_goblet_fraction": float(rho),
            "spearman_p": float(p_value),
            "compact_beta_retention_fraction": float(retention),
            "residual_state_diagnostic": residual_state,
        })

    args.output_dir.mkdir(parents=True, exist_ok=False)
    composition.to_csv(args.output_dir / "patient_epithelial_composition.csv", index=False)
    merged.to_csv(args.output_dir / "patient_composition_expression_matrix.csv", index=False)
    pd.DataFrame(fraction_results).to_csv(args.output_dir / "composition_group_results.csv", index=False)
    pd.DataFrame(matched_rows).to_csv(args.output_dir / "composition_matched_results.csv", index=False)
    pd.DataFrame(model_rows).to_csv(args.output_dir / "composition_adjusted_models.csv", index=False)
    pd.DataFrame(correlation_rows).to_csv(args.output_dir / "composition_expression_correlations.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = {core.PURE_ADENO: "conventional", core.PURE_MUC: "mucinous"}
    for position, endpoint in enumerate(("goblet_fraction", "mature_goblet_fraction")):
        ax = axes[position]
        for x, histology in enumerate((core.PURE_ADENO, core.PURE_MUC)):
            values = merged.loc[merged["histology"].eq(histology), endpoint].to_numpy(float)
            jitter = np.linspace(-0.12, 0.12, len(values))
            ax.scatter(np.full(len(values), x) + jitter, values, s=24, alpha=0.75)
            ax.plot([x - 0.2, x + 0.2], [values.mean(), values.mean()], color="black", lw=2)
        ax.set_xticks([0, 1], [labels[core.PURE_ADENO], labels[core.PURE_MUC]])
        ax.set_ylabel(endpoint.replace("_", " "))
        ax.set_title("Patient-level epithelial composition")
    figure.tight_layout()
    figure.savefig(args.output_dir / "epithelial_composition_diagnostic.png", dpi=220)
    plt.close(figure)

    residual = pd.DataFrame(correlation_rows).set_index("endpoint")["residual_state_diagnostic"].to_dict()
    report = {
        "status": "gse178341_epithelial_composition_diagnostic_complete",
        "formal": False,
        "analysis_type": "post-result patient-level diagnostic",
        "patients": {"mucinous": 6, "conventional": 53},
        "goblet_cluster_codes": list(GOBLET),
        "fraction_results": fraction_results,
        "matched_fraction_results": matched_rows,
        "residual_state_diagnostics": residual,
        "provenance": {
            "metadata_sha256": sha256(core.META),
            "clusters_sha256": sha256(core.CLUSTERS),
            "patient_matrix_sha256": sha256(PATIENT_MATRIX),
            "matches_sha256": sha256(MATCHES),
            "contract_sha256": sha256(CONTRACT),
        },
        "claim_limit": "Post-result dissociation-sensitive composition diagnostic; not confirmatory, causal mediation, biochemical source, flux, enzyme activity or glycan destination.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
