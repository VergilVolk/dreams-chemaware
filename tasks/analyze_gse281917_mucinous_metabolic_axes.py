#!/usr/bin/env python
"""Audit fixed metabolic axes in the 2025 mucinous CRC RNA-seq cohort.

The primary analysis is deliberately restricted to GSE281917 (mucinous CRC),
where platform and histology are not confounded.  It tests whether frozen
metabolic axes align with the published MuC23 risk score and tumour stage.

GSE281917 versus GSE281918 is reported only as a platform-confounded
sensitivity analysis because the two histologies were deposited on different
GEO platforms.  No batch adjustment can identify a histology effect when batch
and histology are perfectly collinear.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import re
import tarfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr, rankdata, spearmanr, t as student_t


AXES = {
    "modified_nucleoside_processing": [
        "METTL1", "WDR4", "RNMT", "CMTR1", "CMTR2", "TRMT1", "TRMT5",
        "TRMT10C", "TGS1", "THUMPD3", "NUDT16", "DCP2",
    ],
    "purine_synthesis_salvage": [
        "HPRT1", "PNP", "GMPS", "IMPDH1", "IMPDH2", "GDA", "APRT",
        "XDH", "ADA", "ADK",
    ],
    "carnitine_long_chain_fao": [
        "CPT1A", "CPT2", "SLC25A20", "ACADVL", "ACADM", "ACADS", "HADHA",
        "HADHB", "ETFA", "ETFB", "ETFDH", "CRAT", "CROT",
    ],
    "polyamine_acetylation_catabolism": ["SAT1", "PAOX", "SMOX"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bh(values: list[float]) -> list[float]:
    if not values:
        return []
    values_array = np.asarray(values, dtype=float)
    order = np.argsort(values_array)
    ranked = values_array[order] * len(values_array) / np.arange(1, len(values_array) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    output = np.empty_like(ranked)
    output[order] = np.minimum(ranked, 1.0)
    return output.tolist()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_series_matrix(path: Path, cohort: str) -> pd.DataFrame:
    wanted: dict[str, list[str]] = {}
    characteristics: list[list[str]] = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            if not raw.startswith("!Sample_"):
                continue
            fields = next(csv.reader([raw.rstrip("\n")], delimiter="\t", quotechar='"'))
            key, values = fields[0], fields[1:]
            if key == "!Sample_characteristics_ch1":
                characteristics.append(values)
            elif key in {"!Sample_geo_accession", "!Sample_title", "!Sample_platform_id"}:
                wanted[key] = values
    accessions = wanted["!Sample_geo_accession"]
    if not all(len(values) == len(accessions) for values in wanted.values()):
        raise RuntimeError(f"metadata width mismatch: {path}")
    rows = []
    for index, accession in enumerate(accessions):
        row = {
            "sample": accession,
            "title": wanted["!Sample_title"][index],
            "platform": wanted["!Sample_platform_id"][index],
            "cohort": cohort,
        }
        for values in characteristics:
            if len(values) != len(accessions):
                raise RuntimeError(f"characteristics width mismatch: {path}")
            value = values[index]
            if ":" in value:
                key, item = value.split(":", 1)
                row[re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")] = item.strip()
        rows.append(row)
    return pd.DataFrame(rows)


def load_tar_expression(path: Path, expected_samples: set[str]) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    with tarfile.open(path, "r") as archive:
        members = [member for member in archive.getmembers() if member.isfile() and member.name.endswith(".txt.gz")]
        for number, member in enumerate(members, 1):
            match = re.match(r"(GSM\d+)_", Path(member.name).name)
            if not match:
                raise RuntimeError(f"cannot recover GSM accession from {member.name}")
            sample = match.group(1)
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"cannot read {member.name}")
            payload = gzip.decompress(extracted.read()).decode("utf-8", errors="replace")
            frame = pd.read_csv(io.StringIO(payload), sep="\t")
            if frame.shape[1] != 2:
                raise RuntimeError(f"expected two columns in {member.name}, observed {frame.shape[1]}")
            genes = frame.iloc[:, 0].astype(str).str.strip()
            values = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
            series = pd.Series(values.to_numpy(float), index=genes)
            series = series[~series.index.duplicated(keep="first")]
            columns[sample] = series
            if number % 50 == 0 or number == len(members):
                print(f"[{path.stem}] {number:,}/{len(members):,} samples")
    observed = set(columns)
    if observed != expected_samples:
        missing = sorted(expected_samples - observed)
        extra = sorted(observed - expected_samples)
        raise RuntimeError(f"sample mismatch for {path}: missing={missing[:5]} extra={extra[:5]}")
    matrix = pd.DataFrame(columns).sort_index()
    if matrix.isna().mean().max() > 0.01:
        raise RuntimeError(f"unexpectedly sparse expression matrix: {path}")
    return matrix


def parse_stage(value: object) -> float:
    text = str(value or "").upper().strip()
    if text in {"", "NAN", "NA", "UNKNOWN"}:
        return np.nan
    if text.startswith("0"):
        return 0.0
    match = re.match(r"(IV|III|II|I)", text)
    return {"I": 1.0, "II": 2.0, "III": 3.0, "IV": 4.0}.get(match.group(1), np.nan) if match else np.nan


def load_risk_coefficients(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="Supplementary Table 5", header=None)
    header_rows = np.flatnonzero(raw.iloc[:, 0].astype(str).eq("Gene Symbol"))
    if len(header_rows) != 1:
        raise RuntimeError("could not locate MuC23 header")
    start = int(header_rows[0]) + 1
    rows = []
    for _, row in raw.iloc[start:].iterrows():
        gene = str(row.iloc[0]).strip()
        coefficient = pd.to_numeric(pd.Series([row.iloc[1]]), errors="coerce").iloc[0]
        if not gene or gene.lower() == "nan" or not np.isfinite(coefficient):
            break
        rows.append({"gene": gene, "coefficient": float(coefficient)})
    result = pd.DataFrame(rows)
    if len(result) != 23:
        raise RuntimeError(f"expected 23 risk genes, observed {len(result)}")
    return result


def load_published_degs(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="Supplementary Table 1", header=None)
    header_rows = np.flatnonzero(raw.iloc[:, 0].astype(str).eq("Gene name"))
    if len(header_rows) != 1:
        raise RuntimeError("could not locate DEG header")
    start = int(header_rows[0]) + 1
    result = raw.iloc[start:, :4].copy()
    result.columns = ["gene", "published_log2fc", "pvalue", "fdr"]
    result["gene"] = result["gene"].astype(str).str.strip()
    for column in ["published_log2fc", "pvalue", "fdr"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["published_log2fc"]).copy()
    if len(result) < 5000:
        raise RuntimeError(f"unexpectedly short DEG table: {len(result)}")
    return result


def axis_scores(expression: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    z = expression.sub(expression.mean(axis=1), axis=0).div(expression.std(axis=1, ddof=1).replace(0, np.nan), axis=0)
    scores = {}
    coverage = []
    for axis, genes in AXES.items():
        present = [gene for gene in genes if gene in expression.index]
        if len(present) < max(2, math.ceil(len(genes) * 0.6)):
            raise RuntimeError(f"insufficient gene coverage for {axis}: {len(present)}/{len(genes)}")
        scores[axis] = z.loc[present].mean(axis=0)
        coverage.append({
            "axis": axis,
            "genes_expected": len(genes),
            "genes_present": len(present),
            "coverage_fraction": len(present) / len(genes),
            "present_genes": ";".join(present),
            "missing_genes": ";".join(sorted(set(genes) - set(present))),
        })
    return pd.DataFrame(scores), coverage


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, seed: int, repeats: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    n = len(x)
    for _ in range(repeats):
        index = rng.integers(0, n, n)
        rho = spearmanr(x[index], y[index]).statistic
        if np.isfinite(rho):
            values.append(float(rho))
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())


def partial_rank_association(
    x: np.ndarray,
    y: np.ndarray,
    covariates: np.ndarray,
    seed: int,
    repeats: int,
) -> dict:
    """Residualized-rank association with row-bootstrap uncertainty."""
    mask = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(covariates), axis=1)
    x, y, covariates = x[mask], y[mask], covariates[mask]

    def estimate(index: np.ndarray) -> float:
        x_part = rankdata(x[index])
        y_part = rankdata(y[index])
        cov_part = covariates[index]
        design = np.column_stack([np.ones(len(index)), cov_part])
        x_residual = x_part - design @ np.linalg.lstsq(design, x_part, rcond=None)[0]
        y_residual = y_part - design @ np.linalg.lstsq(design, y_part, rcond=None)[0]
        return float(pearsonr(x_residual, y_residual).statistic)

    full_index = np.arange(len(x))
    coefficient = estimate(full_index)
    degrees = len(x) - covariates.shape[1] - 2
    statistic = coefficient * math.sqrt(degrees / max(1.0 - coefficient ** 2, 1e-12))
    pvalue = float(2 * student_t.sf(abs(statistic), degrees))
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(repeats):
        value = estimate(rng.integers(0, len(x), len(x)))
        if np.isfinite(value):
            bootstrap.append(value)
    leave_one_out = [estimate(np.delete(full_index, index)) for index in range(len(x))]
    return {
        "n": int(len(x)),
        "rho": coefficient,
        "pvalue": pvalue,
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "leave_one_out_min": float(np.min(leave_one_out)),
        "leave_one_out_max": float(np.max(leave_one_out)),
    }


def hc3_fit(y: np.ndarray, columns: dict[str, np.ndarray], target: str) -> dict:
    names = ["intercept", *columns]
    x = np.column_stack([np.ones(len(y)), *[np.asarray(columns[name], dtype=float) for name in columns]])
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    x, y = x[mask], y[mask]
    if len(y) <= x.shape[1] + 5:
        raise RuntimeError(f"insufficient complete cases for HC3: n={len(y)} p={x.shape[1]}")
    inverse = np.linalg.pinv(x.T @ x)
    beta = inverse @ x.T @ y
    residual = y - x @ beta
    leverage = np.einsum("ij,jk,ik->i", x, inverse, x)
    adjusted = residual / np.maximum(1.0 - leverage, 1e-8)
    meat = x.T @ ((adjusted ** 2)[:, None] * x)
    covariance = inverse @ meat @ inverse
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    position = names.index(target)
    statistic = beta[position] / se[position] if se[position] > 0 else np.nan
    degrees = len(y) - x.shape[1]
    pvalue = float(2 * student_t.sf(abs(statistic), degrees)) if np.isfinite(statistic) else np.nan
    return {
        "n": int(len(y)),
        "beta": float(beta[position]),
        "se_hc3": float(se[position]),
        "t": float(statistic),
        "pvalue": pvalue,
        "ci_low": float(beta[position] - student_t.ppf(0.975, degrees) * se[position]),
        "ci_high": float(beta[position] + student_t.ppf(0.975, degrees) * se[position]),
    }


def numeric_sex(values: pd.Series) -> np.ndarray:
    mapping = {"female": 0.0, "male": 1.0}
    return values.astype(str).str.lower().map(mapping).to_numpy(float)


def standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - np.nanmean(values)) / np.nanstd(values, ddof=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("data/external/GSE281917/source"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/external/GSE281917/mucinous_metabolic_axes_v1"))
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()

    source = args.source_dir
    paths = {
        "muc_matrix": source / "GSE281917_series_matrix.txt.gz",
        "nmuc_matrix": source / "GSE281918_series_matrix.txt.gz",
        "muc_tar": source / "GSE281917_RAW.tar",
        "nmuc_tar": source / "GSE281918_RAW.tar",
        "supplement": source / "41416_2025_3104_MOESM2_ESM.xlsx",
        "methods": source / "41416_2025_3104_MOESM3_ESM.docx",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    muc_meta = parse_series_matrix(paths["muc_matrix"], "MuC")
    nmuc_meta = parse_series_matrix(paths["nmuc_matrix"], "NMuC")
    muc_expression = load_tar_expression(paths["muc_tar"], set(muc_meta["sample"]))
    nmuc_expression = load_tar_expression(paths["nmuc_tar"], set(nmuc_meta["sample"]))
    if len(muc_meta) != 140 or len(nmuc_meta) != 119:
        raise RuntimeError(f"unexpected cohort sizes: MuC={len(muc_meta)} NMuC={len(nmuc_meta)}")

    risk = load_risk_coefficients(paths["supplement"])
    missing_risk = sorted(set(risk["gene"]) - set(muc_expression.index))
    if missing_risk:
        raise RuntimeError(f"MuC23 genes absent from expression matrix: {missing_risk}")
    risk_score = sum(
        float(row.coefficient) * muc_expression.loc[row.gene]
        for row in risk.itertuples(index=False)
    )
    muc_scores, coverage = axis_scores(muc_expression)
    muc_meta = muc_meta.set_index("sample").loc[muc_scores.index].copy()
    muc_meta["age_numeric"] = pd.to_numeric(muc_meta.get("age"), errors="coerce")
    muc_meta["stage_ordinal"] = muc_meta.get("tumor_stage", pd.Series(index=muc_meta.index, dtype=object)).map(parse_stage)
    muc_meta["sex_numeric"] = numeric_sex(muc_meta.get("sex", pd.Series(index=muc_meta.index, dtype=object)))
    muc_meta["muc23_risk_score"] = risk_score.loc[muc_meta.index]
    for column in muc_scores:
        muc_meta[column] = muc_scores[column]

    association_rows = []
    for number, axis in enumerate(AXES):
        x = muc_meta[axis].to_numpy(float)
        y = muc_meta["muc23_risk_score"].to_numpy(float)
        mask = np.isfinite(x) & np.isfinite(y)
        spearman = spearmanr(x[mask], y[mask])
        ci_low, ci_high = bootstrap_spearman(x[mask], y[mask], args.seed + number, args.bootstrap_resamples)
        adjusted = hc3_fit(
            standardize(y),
            {
                "axis": standardize(x),
                "stage": standardize(muc_meta["stage_ordinal"].to_numpy(float)),
                "age": standardize(muc_meta["age_numeric"].to_numpy(float)),
                "sex": muc_meta["sex_numeric"].to_numpy(float),
            },
            "axis",
        )
        stage_fit = hc3_fit(
            standardize(x),
            {
                "stage": standardize(muc_meta["stage_ordinal"].to_numpy(float)),
                "age": standardize(muc_meta["age_numeric"].to_numpy(float)),
                "sex": muc_meta["sex_numeric"].to_numpy(float),
            },
            "stage",
        )
        partial_rank = partial_rank_association(
            x,
            y,
            np.column_stack([
                muc_meta["stage_ordinal"].to_numpy(float),
                muc_meta["age_numeric"].to_numpy(float),
                muc_meta["sex_numeric"].to_numpy(float),
            ]),
            args.seed + 100 + number,
            args.bootstrap_resamples,
        )
        association_rows.append({
            "axis": axis,
            "n": int(mask.sum()),
            "risk_spearman_rho": float(spearman.statistic),
            "risk_spearman_p": float(spearman.pvalue),
            "risk_spearman_ci_low": ci_low,
            "risk_spearman_ci_high": ci_high,
            "risk_hc3_beta": adjusted["beta"],
            "risk_hc3_se": adjusted["se_hc3"],
            "risk_hc3_p": adjusted["pvalue"],
            "risk_hc3_ci_low": adjusted["ci_low"],
            "risk_hc3_ci_high": adjusted["ci_high"],
            "risk_partial_rank_rho": partial_rank["rho"],
            "risk_partial_rank_p": partial_rank["pvalue"],
            "risk_partial_rank_ci_low": partial_rank["ci_low"],
            "risk_partial_rank_ci_high": partial_rank["ci_high"],
            "risk_partial_rank_leave_one_out_min": partial_rank["leave_one_out_min"],
            "risk_partial_rank_leave_one_out_max": partial_rank["leave_one_out_max"],
            "stage_hc3_beta": stage_fit["beta"],
            "stage_hc3_se": stage_fit["se_hc3"],
            "stage_hc3_p": stage_fit["pvalue"],
            "stage_hc3_ci_low": stage_fit["ci_low"],
            "stage_hc3_ci_high": stage_fit["ci_high"],
        })
    risk_q = bh([row["risk_spearman_p"] for row in association_rows])
    risk_hc3_q = bh([row["risk_hc3_p"] for row in association_rows])
    risk_partial_rank_q = bh([row["risk_partial_rank_p"] for row in association_rows])
    stage_q = bh([row["stage_hc3_p"] for row in association_rows])
    for row, q1, q2, q3, q4 in zip(association_rows, risk_q, risk_hc3_q, risk_partial_rank_q, stage_q):
        row["risk_spearman_q"] = q1
        row["risk_hc3_q"] = q2
        row["risk_partial_rank_q"] = q3
        row["stage_hc3_q"] = q4

    degs = load_published_degs(paths["supplement"])
    deg_map = degs.set_index("gene")
    deg_rows = []
    for axis, genes in AXES.items():
        for gene in genes:
            if gene in deg_map.index:
                item = deg_map.loc[gene]
                deg_rows.append({
                    "axis": axis,
                    "gene": gene,
                    "present_in_published_significant_deg_table": True,
                    "published_log2fc": float(item["published_log2fc"]),
                    "published_fdr": float(item["fdr"]),
                    "inferred_muc_minus_nmuc_log2fc": float(-item["published_log2fc"]),
                })
            else:
                deg_rows.append({
                    "axis": axis,
                    "gene": gene,
                    "present_in_published_significant_deg_table": False,
                    "published_log2fc": np.nan,
                    "published_fdr": np.nan,
                    "inferred_muc_minus_nmuc_log2fc": np.nan,
                })

    common_genes = muc_expression.index.intersection(nmuc_expression.index)
    combined = pd.concat([muc_expression.loc[common_genes], nmuc_expression.loc[common_genes]], axis=1)
    percentile = combined.rank(axis=0, pct=True, method="average")
    platform_rows = []
    for axis, genes in AXES.items():
        present = [gene for gene in genes if gene in percentile.index]
        score = percentile.loc[present].mean(axis=0)
        muc_values = score[muc_meta.index].to_numpy(float)
        nmuc_values = score[nmuc_meta["sample"]].to_numpy(float)
        test = mannwhitneyu(muc_values, nmuc_values, alternative="two-sided")
        probability_superiority = float(test.statistic / (len(muc_values) * len(nmuc_values)))
        platform_rows.append({
            "axis": axis,
            "muc_n": len(muc_values),
            "nmuc_n": len(nmuc_values),
            "muc_mean_within_sample_percentile": float(np.mean(muc_values)),
            "nmuc_mean_within_sample_percentile": float(np.mean(nmuc_values)),
            "muc_minus_nmuc": float(np.mean(muc_values) - np.mean(nmuc_values)),
            "mannwhitney_p": float(test.pvalue),
            "probability_muc_gt_nmuc": probability_superiority,
            "platform_confounded": True,
        })
    for row, qvalue in zip(platform_rows, bh([row["mannwhitney_p"] for row in platform_rows])):
        row["mannwhitney_q"] = qvalue

    platform_table = pd.crosstab(
        pd.concat([muc_meta.reset_index(), nmuc_meta], ignore_index=True)["cohort"],
        pd.concat([muc_meta.reset_index(), nmuc_meta], ignore_index=True)["platform"],
    )
    platform_perfectly_confounded = bool((platform_table.gt(0).sum(axis=0) == 1).all())

    write_csv(args.output_dir / "axis_gene_coverage.csv", coverage)
    write_csv(args.output_dir / "mucinous_internal_associations.csv", association_rows)
    write_csv(args.output_dir / "published_deg_axis_audit.csv", deg_rows)
    write_csv(args.output_dir / "platform_confounded_histology_sensitivity.csv", platform_rows)
    muc_meta.reset_index().to_csv(args.output_dir / "mucinous_sample_scores.csv.gz", index=False, compression="gzip")

    figure, axes_plot = plt.subplots(1, 2, figsize=(12, 4.8))
    plot_frame = pd.DataFrame(association_rows)
    y_position = np.arange(len(plot_frame))
    axes_plot[0].errorbar(
        plot_frame["risk_spearman_rho"], y_position,
        xerr=np.vstack([
            plot_frame["risk_spearman_rho"] - plot_frame["risk_spearman_ci_low"],
            plot_frame["risk_spearman_ci_high"] - plot_frame["risk_spearman_rho"],
        ]), fmt="o", color="#3465a4", capsize=3,
    )
    axes_plot[0].axvline(0, color="black", linewidth=0.8)
    axes_plot[0].set_yticks(y_position, plot_frame["axis"])
    axes_plot[0].set_xlabel("Spearman correlation with MuC23 risk score")
    axes_plot[0].set_title("Within mucinous CRC (n=140)")
    sensitivity = pd.DataFrame(platform_rows)
    axes_plot[1].barh(np.arange(len(sensitivity)), sensitivity["muc_minus_nmuc"], color="#cc6677")
    axes_plot[1].axvline(0, color="black", linewidth=0.8)
    axes_plot[1].set_yticks(np.arange(len(sensitivity)), sensitivity["axis"])
    axes_plot[1].set_xlabel("MuC - NMuC mean within-sample percentile")
    axes_plot[1].set_title("Platform-confounded sensitivity only")
    figure.tight_layout()
    figure.savefig(args.output_dir / "mucinous_metabolic_axes.png", dpi=220, bbox_inches="tight")
    figure.savefig(args.output_dir / "mucinous_metabolic_axes.pdf", bbox_inches="tight")
    plt.close(figure)

    report = {
        "status": "gse281917_mucinous_metabolic_axes_complete",
        "formal": False,
        "primary_analysis": "within-GSE281917 mucinous CRC association of frozen axes with MuC23 risk score and stage",
        "cohort_sizes": {"MuC": len(muc_meta), "NMuC": len(nmuc_meta)},
        "platform_table": platform_table.to_dict(),
        "platform_perfectly_confounded_with_histology": platform_perfectly_confounded,
        "mucinous_internal_associations": association_rows,
        "platform_confounded_histology_sensitivity": platform_rows,
        "published_deg_axis_audit_summary": [
            {
                "axis": axis,
                "genes_in_significant_table": int(sum(row["present_in_published_significant_deg_table"] for row in deg_rows if row["axis"] == axis)),
                "genes_expected": len(genes),
            }
            for axis, genes in AXES.items()
        ],
        "muc23_axis_gene_overlap": {
            axis: sorted(set(genes) & set(risk["gene"])) for axis, genes in AXES.items()
        },
        "posthoc_robustness_note": (
            "Covariate-adjusted residualized-rank associations and leave-one-out ranges were added after the initial "
            "distribution audit showed disagreement between unadjusted Spearman and linear-scale HC3 estimates."
        ),
        "analysis_contract": {
            "fixed_axes": True,
            "within_muc_risk_and_stage_are_primary": True,
            "muc_vs_nmuc_is_sensitivity_only": True,
            "batch_correction_for_perfect_confounding_forbidden": True,
            "static_transcript_abundance_does_not_establish_metabolite_flux_or_enzyme_activity": True,
            "muc23_association_is_not_independent_survival_validation": True,
        },
        "claim_limit": (
            "This analysis can establish transcript-level alignment of frozen metabolic axes with internal MuC risk/stage structure. "
            "It cannot establish mucinous specificity from GSE281917/GSE281918 because histology and GEO platform are perfectly confounded; "
            "it cannot establish metabolite identity, flux, or enzyme causality."
        ),
        "parameters": {"bootstrap_resamples": args.bootstrap_resamples, "seed": args.seed},
        "provenance": {key: sha256(path) for key, path in paths.items()},
    }
    with (args.output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
