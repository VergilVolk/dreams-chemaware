#!/usr/bin/env python3
"""Reproduce the paired human CRC fatty-acid analysis from MTBLS7387.

The source workbook is Source Data Fig. 3 from Coleman, Sorbie et al.
(Nature Metabolism, 2025).  The primary analysis deliberately follows the
published protocol: paired t-tests on normalized intensities followed by a
Benjamini-Hochberg correction across all molecular features.  Additional
robustness statistics are reported separately and never replace that primary
reproduction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


TUMOUR = "Tumor Tissue"
ADJACENT = "Adjacent Tissue"
NORMALIZED_SHEET = "3c normalized data"
PUBLISHED_SHEET = "3c stats overlap mouse"
PAPER_URL = "https://www.nature.com/articles/s42255-025-01350-6"
PMC_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC12460170/"
CODE_URL = "https://github.com/adamsorbie/Coleman_Sorbie_et_al_2025"
DATA_URL = "https://www.ebi.ac.uk/metabolights/editor/MTBLS7387/descriptors"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return result
    order = finite[np.argsort(values[finite], kind="mergesort")]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result[order] = np.clip(ranked, 0.0, 1.0)
    return result


def chain_annotation(name: str) -> tuple[float, float, str]:
    match = re.match(r"^C(?P<carbon>\d+)\.(?P<double>\d+)_", name)
    if not match:
        return math.nan, math.nan, "unparsed"
    carbon = int(match.group("carbon"))
    double = int(match.group("double"))
    lipid_form = "hydroxy_free_fatty_acid" if "_Hydroxy." in name else "free_fatty_acid"
    return carbon, double, lipid_form


def safe_wilcoxon(tumour: np.ndarray, adjacent: np.ndarray) -> float:
    difference = tumour - adjacent
    if np.allclose(difference, 0.0):
        return 1.0
    return float(stats.wilcoxon(tumour, adjacent, alternative="two-sided").pvalue)


def paired_effect(feature: str, frame: pd.DataFrame) -> dict[str, object]:
    pivot = frame.pivot(index="ATTRIBUTE_Code", columns="ATTRIBUTE_Tissue", values=feature)
    pivot = pivot[[ADJACENT, TUMOUR]].dropna()
    adjacent = pivot[ADJACENT].to_numpy(float)
    tumour = pivot[TUMOUR].to_numpy(float)
    difference = tumour - adjacent
    t_result = stats.ttest_rel(tumour, adjacent)
    mean_difference = float(np.mean(difference))
    standard_deviation = float(np.std(difference, ddof=1))
    standard_error = standard_deviation / math.sqrt(len(difference))
    critical = float(stats.t.ppf(0.975, len(difference) - 1))
    mean_adjacent = float(np.mean(adjacent))
    mean_tumour = float(np.mean(tumour))
    carbon, double, lipid_form = chain_annotation(feature)
    standard_validated = bool(
        feature.startswith("C22.4_Docosatetraenoicacid_")
        or feature.startswith("C22.5_Docosapentanoicacid_")
        or feature.startswith("C22.6_Docosahexaenoicacid_")
    )
    return {
        "feature": feature,
        "carbon_count": carbon,
        "double_bonds": double,
        "lipid_form": lipid_form,
        "paper_standard_validated": standard_validated,
        "n_pairs": int(len(pivot)),
        "mean_adjacent": mean_adjacent,
        "mean_tumour": mean_tumour,
        "mean_difference": mean_difference,
        "mean_difference_ci_low": mean_difference - critical * standard_error,
        "mean_difference_ci_high": mean_difference + critical * standard_error,
        "tumour_to_adjacent_mean_ratio": mean_tumour / mean_adjacent,
        "log2_tumour_to_adjacent_mean_ratio": math.log2(mean_tumour / mean_adjacent),
        "paired_cohens_dz": mean_difference / standard_deviation if standard_deviation else 0.0,
        "tumour_greater_fraction": float(np.mean(tumour > adjacent)),
        "paired_t_statistic": float(t_result.statistic),
        "paired_t_p": float(t_result.pvalue),
        "paired_wilcoxon_p": safe_wilcoxon(tumour, adjacent),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()

    source = arguments.source.resolve()
    output_dir = arguments.output_dir.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized = pd.read_excel(source, sheet_name=NORMALIZED_SHEET)
    normalized = normalized.rename(columns={normalized.columns[0]: "sample"})
    published = pd.read_excel(source, sheet_name=PUBLISHED_SHEET)

    required = {"sample", "ATTRIBUTE_Code", "ATTRIBUTE_Tissue"}
    missing = sorted(required - set(normalized.columns))
    if missing:
        raise RuntimeError(f"missing required columns: {missing}")
    if normalized["sample"].duplicated().any():
        raise RuntimeError("sample identifiers are not unique")
    if set(normalized["ATTRIBUTE_Tissue"].dropna().astype(str)) != {TUMOUR, ADJACENT}:
        raise RuntimeError("unexpected tissue labels")

    pair_counts = pd.crosstab(normalized["ATTRIBUTE_Code"], normalized["ATTRIBUTE_Tissue"])
    complete = pair_counts[(pair_counts[TUMOUR] == 1) & (pair_counts[ADJACENT] == 1)].index
    if len(complete) != normalized["ATTRIBUTE_Code"].nunique():
        raise RuntimeError("human source matrix is not exactly one tumour/adjacent pair per code")
    normalized = normalized[normalized["ATTRIBUTE_Code"].isin(complete)].copy()

    metadata_columns = ["sample"] + [
        column for column in normalized.columns if str(column).startswith("ATTRIBUTE_")
    ]
    feature_columns = [column for column in normalized.columns if column not in metadata_columns]
    if len(feature_columns) < 150:
        raise RuntimeError(f"unexpectedly small fatty-acid panel: {len(feature_columns)}")

    results = pd.DataFrame([paired_effect(str(feature), normalized) for feature in feature_columns])
    results["paired_t_q_bh"] = benjamini_hochberg(results["paired_t_p"].to_numpy())
    results["paired_wilcoxon_q_bh"] = benjamini_hochberg(results["paired_wilcoxon_p"].to_numpy())
    results["paper_protocol_significant"] = results["paired_t_q_bh"] < 0.05
    results["robust_to_wilcoxon"] = results["paired_wilcoxon_q_bh"] < 0.05
    results["long_chain_c20_c24"] = results["carbon_count"].between(20, 24)
    results = results.sort_values(["paired_t_q_bh", "paired_t_p", "feature"], kind="mergesort")

    published_checks = []
    for row in published.itertuples(index=False):
        name = str(row.Metabolites)
        match = results[results["feature"] == name]
        if len(match) != 1:
            raise RuntimeError(f"published feature does not map uniquely: {name}")
        observed = match.iloc[0]
        published_p = float(row.p)
        published_q = float(row.p_fdr)
        p_relative_error = abs(float(observed["paired_t_p"]) - published_p) / published_p
        q_relative_error = abs(float(observed["paired_t_q_bh"]) - published_q) / published_q
        published_checks.append(
            {
                "feature": name,
                "published_p": published_p,
                "reproduced_p": float(observed["paired_t_p"]),
                "p_relative_error": p_relative_error,
                "published_q": published_q,
                "reproduced_q": float(observed["paired_t_q_bh"]),
                "q_relative_error": q_relative_error,
            }
        )
    reproduction = pd.DataFrame(published_checks)
    # The source-data table prints three to six significant digits.  A 0.5%
    # relative tolerance is strict enough to reject a protocol mismatch while
    # allowing the published decimal rounding.
    reproduction_tolerance = 5e-3
    if float(reproduction["p_relative_error"].max()) > reproduction_tolerance:
        raise RuntimeError("published paired t-test p values were not reproduced")
    if float(reproduction["q_relative_error"].max()) > reproduction_tolerance:
        raise RuntimeError("published BH-FDR values were not reproduced")

    results.to_csv(output_dir / "paired_fatty_acid_results.csv.gz", index=False)
    reproduction.to_csv(output_dir / "published_overlap_reproduction.csv", index=False)
    normalized[metadata_columns].to_csv(output_dir / "paired_sample_manifest.csv", index=False)

    subgroup_column = "ATTRIBUTE_crc_2groups"
    if subgroup_column not in normalized.columns:
        raise RuntimeError(f"missing subgroup column: {subgroup_column}")
    subgroup_counts = normalized.groupby("ATTRIBUTE_Code")[subgroup_column].nunique()
    if int(subgroup_counts.max()) != 1:
        raise RuntimeError("early/late CRC label differs within a patient pair")
    subgroup_results = []
    subgroup_summary = {}
    for subgroup in sorted(normalized[subgroup_column].dropna().astype(str).unique()):
        subset = normalized[normalized[subgroup_column].astype(str) == subgroup]
        subgroup_frame = pd.DataFrame(
            [paired_effect(str(feature), subset) for feature in feature_columns]
        )
        subgroup_frame["paired_t_q_bh"] = benjamini_hochberg(
            subgroup_frame["paired_t_p"].to_numpy()
        )
        subgroup_frame["paired_wilcoxon_q_bh"] = benjamini_hochberg(
            subgroup_frame["paired_wilcoxon_p"].to_numpy()
        )
        subgroup_frame["paper_protocol_significant"] = subgroup_frame["paired_t_q_bh"] < 0.05
        subgroup_frame["robust_to_wilcoxon"] = subgroup_frame["paired_wilcoxon_q_bh"] < 0.05
        subgroup_frame["long_chain_c20_c24"] = subgroup_frame["carbon_count"].between(20, 24)
        subgroup_frame.insert(0, "crc_age_group", subgroup)
        subgroup_results.append(subgroup_frame)
        subgroup_lcfa = subgroup_frame[
            subgroup_frame["long_chain_c20_c24"] & subgroup_frame["paper_protocol_significant"]
        ]
        subgroup_summary[subgroup] = {
            "patients": int(subset["ATTRIBUTE_Code"].nunique()),
            "fdr05_features": int(subgroup_frame["paper_protocol_significant"].sum()),
            "fdr05_long_chain_c20_c24": int(len(subgroup_lcfa)),
            "significant_long_chain_features": subgroup_lcfa[
                ["feature", "log2_tumour_to_adjacent_mean_ratio", "paired_t_q_bh"]
            ].sort_values("paired_t_q_bh").to_dict(orient="records"),
        }
    subgroup_table = pd.concat(subgroup_results, ignore_index=True)
    subgroup_table.to_csv(output_dir / "paired_fatty_acid_age_subgroups.csv.gz", index=False)

    significant = results[results["paper_protocol_significant"]]
    long_chain = results[results["long_chain_c20_c24"]]
    significant_long_chain = long_chain[long_chain["paper_protocol_significant"]]
    arachidonic = results[results["feature"].str.startswith("C20.4_")]
    report = {
        "status": "mtbls7387_paired_lcfa_replication_complete",
        "formal": True,
        "source": {
            "workbook": str(source),
            "sha256": sha256(source),
            "sheet": NORMALIZED_SHEET,
            "paper_url": PAPER_URL,
            "pmc_url": PMC_URL,
            "code_url": CODE_URL,
            "data_url": DATA_URL,
        },
        "cohort": {
            "rows": int(len(normalized)),
            "patients": int(normalized["ATTRIBUTE_Code"].nunique()),
            "tumour_samples": int((normalized["ATTRIBUTE_Tissue"] == TUMOUR).sum()),
            "adjacent_samples": int((normalized["ATTRIBUTE_Tissue"] == ADJACENT).sum()),
            "complete_pairs": int(len(complete)),
            "molecular_features": int(len(results)),
        },
        "published_protocol_reproduction": {
            "test": "paired t-test on normalized intensities; BH across all molecular features",
            "displayed_overlap_features": int(len(reproduction)),
            "maximum_p_relative_error": float(reproduction["p_relative_error"].max()),
            "maximum_q_relative_error": float(reproduction["q_relative_error"].max()),
            "relative_error_tolerance": reproduction_tolerance,
            "pass": True,
        },
        "whole_panel": {
            "fdr05_features": int(len(significant)),
            "fdr05_tumour_increased": int((significant["mean_difference"] > 0).sum()),
            "fdr05_tumour_decreased": int((significant["mean_difference"] < 0).sum()),
            "fdr05_also_wilcoxon_fdr05": int(significant["robust_to_wilcoxon"].sum()),
        },
        "long_chain_c20_c24": {
            "features": int(len(long_chain)),
            "fdr05_features": int(len(significant_long_chain)),
            "fdr05_tumour_increased": int((significant_long_chain["mean_difference"] > 0).sum()),
            "fdr05_tumour_decreased": int((significant_long_chain["mean_difference"] < 0).sum()),
            "fdr05_also_wilcoxon_fdr05": int(significant_long_chain["robust_to_wilcoxon"].sum()),
        },
        "early_late_subgroups": subgroup_summary,
        "c20_4_free_fatty_acid_context": arachidonic[
            [
                "feature",
                "lipid_form",
                "log2_tumour_to_adjacent_mean_ratio",
                "paired_t_p",
                "paired_t_q_bh",
                "paired_wilcoxon_p",
                "paired_wilcoxon_q_bh",
            ]
        ].to_dict(orient="records"),
        "identity_boundary": {
            "local_feature_3222": "long-chain acylcarnitine/C20:4-like; positional identity unresolved",
            "external_mtbls7387": "free and hydroxy fatty acids; C22:4, C22:5 and C22:6 were standard validated in the paper",
            "allowed_claim": "independent paired human CRC support for a long-chain-fatty-acid accumulation context",
            "forbidden_claim": "MTBLS7387 directly validates feature 3222 or acylcarnitine identity/flux",
        },
        "claim_limit": (
            "This reproduction validates a broad paired human CRC long-chain fatty-acid context. "
            "It does not validate the molecular identity of MTBLS13729 feature 3222, mucinous "
            "specificity, beta-oxidation flux, ATF6 causality in MTBLS13729, or enzyme activity."
        ),
    }
    with (output_dir / "mtbls7387_paired_lcfa_replication.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
