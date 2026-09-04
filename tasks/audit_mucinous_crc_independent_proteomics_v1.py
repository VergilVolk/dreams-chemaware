#!/usr/bin/env python3
"""Fixed-panel, patient-level audit of an independent mucinous CRC proteomics cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


SEED = 20260831
N_PERM = 200_000
N_BOOT = 10_000
FIXED_MODULES = {
    "secretory_mucin": ["AGR2", "MUC2", "TFF3", "FCGBP"],
    "sialic_biosynthesis_handling": ["GNE", "NANS", "CMAS", "SIAE"],
}
PRESPECIFIED_UNAVAILABLE = ["NXPE1", "SPDEF", "SLC35A1", "CASD1"]
MC_LABEL = "Mucinous adenocarcinoma"
AC_LABEL = "Adenocarcinoma not otherwise specified"
NORMAL_LABEL = "Normal colon"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def bh(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    q = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    q = np.minimum(q, 1.0)
    result = np.empty_like(q)
    result[order] = q
    return result.tolist()


def permutation_p(mc: np.ndarray, ac: np.ndarray, rng: np.random.Generator) -> float:
    observed = float(np.mean(mc) - np.mean(ac))
    pooled = np.concatenate([mc, ac])
    n_mc = len(mc)
    exceed = 0
    chunk = 2_000
    for start in range(0, N_PERM, chunk):
        size = min(chunk, N_PERM - start)
        keys = rng.random((size, len(pooled)))
        idx = np.argpartition(keys, n_mc - 1, axis=1)[:, :n_mc]
        mc_sums = pooled[idx].sum(axis=1)
        total = pooled.sum()
        diffs = mc_sums / n_mc - (total - mc_sums) / (len(pooled) - n_mc)
        exceed += int(np.sum(np.abs(diffs) >= abs(observed) - 1e-15))
    return float((exceed + 1) / (N_PERM + 1))


def bootstrap_ci(mc: np.ndarray, ac: np.ndarray, rng: np.random.Generator) -> list[float]:
    mc_idx = rng.integers(0, len(mc), size=(N_BOOT, len(mc)))
    ac_idx = rng.integers(0, len(ac), size=(N_BOOT, len(ac)))
    delta = mc[mc_idx].mean(axis=1) - ac[ac_idx].mean(axis=1)
    return [float(x) for x in np.quantile(delta, [0.025, 0.975])]


def hc3_adjusted(values: np.ndarray, metadata: pd.DataFrame) -> dict:
    sex = metadata["Gender"].astype(str).str.lower().eq("male").astype(float).to_numpy()
    age = pd.to_numeric(metadata["Age"], errors="raise").to_numpy(float)
    mc = metadata["Pathology Type"].eq(MC_LABEL).astype(float).to_numpy()
    design = np.column_stack([np.ones(len(values)), mc, age - age.mean(), sex])
    xtx_inv = np.linalg.inv(design.T @ design)
    beta = xtx_inv @ design.T @ values
    residual = values - design @ beta
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inv, design)
    scaled_sq = residual**2 / np.maximum(1.0 - leverage, 1e-12) ** 2
    meat = design.T @ (design * scaled_sq[:, None])
    covariance = xtx_inv @ meat @ xtx_inv
    se = float(np.sqrt(covariance[1, 1]))
    z_value = float(beta[1] / se)
    return {
        "beta_mucinous": float(beta[1]),
        "se_hc3": se,
        "z_hc3": z_value,
        "p_hc3_normal_reference": float(2.0 * stats.norm.sf(abs(z_value))),
        "condition_number": float(np.linalg.cond(design)),
    }


def summarize_vector(
    name: str,
    values: pd.Series,
    tumour_metadata: pd.DataFrame,
    rng: np.random.Generator,
    raw_values: pd.Series | None = None,
    raw_global_values: pd.Series | None = None,
) -> dict:
    mc_mask = tumour_metadata["Pathology Type"].eq(MC_LABEL).to_numpy()
    ac_mask = tumour_metadata["Pathology Type"].eq(AC_LABEL).to_numpy()
    arr = values.to_numpy(float)
    mc = arr[mc_mask]
    ac = arr[ac_mask]
    if len(mc) != 15 or len(ac) != 15:
        raise RuntimeError(f"{name}: expected 15 MC and 15 AC patients")
    effect = float(mc.mean() - ac.mean())
    loo = [float(np.delete(mc, i).mean() - ac.mean()) for i in range(len(mc))]
    report = {
        "name": name,
        "n_mc": int(len(mc)),
        "n_ac": int(len(ac)),
        "mc_mean": float(mc.mean()),
        "ac_mean": float(ac.mean()),
        "mc_minus_ac": effect,
        "bootstrap_95ci": bootstrap_ci(mc, ac, rng),
        "permutation_p": permutation_p(mc, ac, rng),
        "mannwhitney_p": float(stats.mannwhitneyu(mc, ac, alternative="two-sided").pvalue),
        "leave_one_mc_out_same_sign": int(sum(np.sign(x) == np.sign(effect) for x in loo)),
        "leave_one_mc_out_min_delta": float(min(loo)),
        "leave_one_mc_out_max_delta": float(max(loo)),
        "hc3_age_sex": hc3_adjusted(arr, tumour_metadata),
    }
    if raw_values is not None:
        raw = raw_values.to_numpy(float)
        if raw_global_values is None:
            raise RuntimeError(f"{name}: global raw values required for censoring audit")
        raw_global = raw_global_values.to_numpy(float)
        floor = float(np.min(raw_global))
        floor_mask = raw == floor
        report["left_censoring_audit"] = {
            "minimum_raw_value": floor,
            "floor_count_all_46": int(np.sum(raw_global == floor)),
            "floor_count_tumour_30": int(np.sum(floor_mask)),
            "floor_count_mc": int(np.sum(floor_mask[mc_mask])),
            "floor_count_ac": int(np.sum(floor_mask[ac_mask])),
            "floor_fraction_mc": float(np.mean(floor_mask[mc_mask])),
            "floor_fraction_ac": float(np.mean(floor_mask[ac_mask])),
        }
        keep_mc = mc[~floor_mask[mc_mask]]
        keep_ac = ac[~floor_mask[ac_mask]]
        if len(keep_mc) >= 5 and len(keep_ac) >= 5:
            report["nonfloor_descriptive_sensitivity"] = {
                "n_mc": int(len(keep_mc)),
                "n_ac": int(len(keep_ac)),
                "mc_minus_ac_log2": float(keep_mc.mean() - keep_ac.mean()),
            }
        else:
            report["nonfloor_descriptive_sensitivity"] = None
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, default=Path("data/external/GSE178341_mucinous_secretory_audit/PMC10114614_supplement/Table2.XLSX"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/external/GSE178341_mucinous_secretory_audit/independent_proteomics_fixed_panel_v1"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.workbook.is_file():
        raise FileNotFoundError(args.workbook)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    matrix = pd.read_excel(args.workbook, sheet_name="protein matrix", header=None)
    info = pd.read_excel(args.workbook, sheet_name="patient information")
    patient_ids = matrix.iloc[1, 3:].astype(str).tolist()
    if len(patient_ids) != 46 or len(set(patient_ids)) != 46:
        raise RuntimeError("protein matrix must contain 46 unique patient columns")
    pinfo = info[info["Data Type"].eq("proteomics")].copy()
    pinfo["Patient ID"] = pinfo["Patient ID"].astype(str)
    if set(patient_ids) != set(pinfo["Patient ID"]):
        raise RuntimeError("matrix patient columns do not exactly match proteomics patient metadata")
    counts = pinfo["Pathology Type"].value_counts().to_dict()
    expected = {NORMAL_LABEL: 16, AC_LABEL: 15, MC_LABEL: 15}
    if counts != expected:
        raise RuntimeError(f"unexpected cohort counts: {counts}")
    pinfo = pinfo.set_index("Patient ID").loc[patient_ids].reset_index()

    proteins = matrix.iloc[2:, :].copy()
    proteins.columns = ["uniprot", "protein_name", "gene"] + patient_ids
    proteins["gene"] = proteins["gene"].astype(str)
    requested = [g for genes in FIXED_MODULES.values() for g in genes]
    measured = {}
    for gene in requested:
        rows = proteins[proteins["gene"].eq(gene)]
        if len(rows) != 1:
            raise RuntimeError(f"fixed gene {gene} must have exactly one protein row, found {len(rows)}")
        values = pd.to_numeric(rows.iloc[0][patient_ids], errors="raise")
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise RuntimeError(f"{gene}: abundance values must be finite and positive")
        measured[gene] = values.astype(float)
    unexpectedly_present = [g for g in PRESPECIFIED_UNAVAILABLE if np.any(proteins["gene"].eq(g))]
    if unexpectedly_present:
        raise RuntimeError(f"prespecified unavailable genes unexpectedly present: {unexpectedly_present}")

    tumour_mask = pinfo["Pathology Type"].isin([MC_LABEL, AC_LABEL]).to_numpy()
    tumour_info = pinfo.loc[tumour_mask].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    protein_reports = []
    protein_log2 = {}
    for gene in requested:
        raw_all = measured[gene]
        log_all = np.log2(raw_all)
        raw_tumour = raw_all.iloc[np.flatnonzero(tumour_mask)].reset_index(drop=True)
        log_tumour = log_all.iloc[np.flatnonzero(tumour_mask)].reset_index(drop=True)
        protein_log2[gene] = log_tumour
        report = summarize_vector(gene, log_tumour, tumour_info, rng, raw_tumour, raw_all)
        report["scale"] = "log2 raw protein abundance"
        report["fold_change_mc_over_ac"] = float(2.0 ** report["mc_minus_ac"])
        mc_mask_tumour = tumour_info["Pathology Type"].eq(MC_LABEL).to_numpy()
        raw_mc = raw_tumour.to_numpy(float)[mc_mask_tumour]
        raw_ac = raw_tumour.to_numpy(float)[~mc_mask_tumour]
        report["published_raw_scale_context"] = {
            "arithmetic_mean_fold_change_mc_over_ac": float(raw_mc.mean() / raw_ac.mean()),
            "welch_t_p_on_untransformed_abundance": float(stats.ttest_ind(raw_mc, raw_ac, equal_var=False).pvalue),
            "interpretation": "contextual reproduction of the source article scale; not the preregistered primary inference",
        }
        normal_mask = pinfo["Pathology Type"].eq(NORMAL_LABEL).to_numpy()
        report["normal_context"] = {
            "normal_mean_log2": float(log_all.iloc[np.flatnonzero(normal_mask)].mean()),
            "mc_minus_normal_log2": float(log_all.iloc[np.flatnonzero(pinfo["Pathology Type"].eq(MC_LABEL).to_numpy())].mean() - log_all.iloc[np.flatnonzero(normal_mask)].mean()),
            "ac_minus_normal_log2": float(log_all.iloc[np.flatnonzero(pinfo["Pathology Type"].eq(AC_LABEL).to_numpy())].mean() - log_all.iloc[np.flatnonzero(normal_mask)].mean()),
        }
        protein_reports.append(report)
    qvals = bh([r["permutation_p"] for r in protein_reports])
    for report, q in zip(protein_reports, qvals):
        report["permutation_bh_q_across_8"] = float(q)
        same_direction = np.sign(report["mc_minus_ac"]) == np.sign(report["hc3_age_sex"]["beta_mucinous"])
        report["protein_specific_support"] = bool(
            q < 0.10
            and report["bootstrap_95ci"][0] * report["bootstrap_95ci"][1] > 0
            and report["leave_one_mc_out_same_sign"] >= 14
            and same_direction
        )

    module_reports = []
    for module, genes in FIXED_MODULES.items():
        block = np.column_stack([protein_log2[g].to_numpy(float) for g in genes])
        z = (block - block.mean(axis=0, keepdims=True)) / block.std(axis=0, ddof=1, keepdims=True)
        score = pd.Series(z.mean(axis=1))
        report = summarize_vector(module, score, tumour_info, rng)
        report["genes"] = genes
        report["scale"] = "mean of four within-tumour protein z-scores"
        module_reports.append(report)
    module_q = bh([r["permutation_p"] for r in module_reports])
    for report, q in zip(module_reports, module_q):
        report["permutation_bh_q_across_2_modules"] = float(q)
        same_direction = np.sign(report["mc_minus_ac"]) == np.sign(report["hc3_age_sex"]["beta_mucinous"])
        report["orthogonal_support"] = bool(q < 0.10 and report["bootstrap_95ci"][0] * report["bootstrap_95ci"][1] > 0 and same_direction)

    result = {
        "status": "independent_mucinous_crc_proteomics_fixed_panel_complete",
        "formal": True,
        "cohort": counts,
        "primary_comparison": "mucinous adenocarcinoma versus conventional adenocarcinoma NOS",
        "proteins": protein_reports,
        "modules": module_reports,
        "prespecified_unavailable": PRESPECIFIED_UNAVAILABLE,
        "parameters": {"seed": SEED, "permutations": N_PERM, "bootstraps": N_BOOT},
        "provenance": {"workbook": str(args.workbook.resolve()), "workbook_sha256": sha256(args.workbook)},
        "claim_limit": "Independent patient-level protein support only; not metabolite replication, flux, enzyme activity, or new-metabolite discovery.",
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(protein_reports).to_csv(args.output_dir / "protein_summary.csv", index=False)
    pd.DataFrame(module_reports).to_csv(args.output_dir / "module_summary.csv", index=False)
    pinfo.to_csv(args.output_dir / "patient_order_audit.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
