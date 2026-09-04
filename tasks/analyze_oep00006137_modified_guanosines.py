from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def paired_summary(normal: np.ndarray, tumor: np.ndarray) -> dict[str, float | int]:
    delta = np.log2(tumor + 1.0) - np.log2(normal + 1.0)
    nonzero = delta[np.abs(delta) > 1e-12]
    positive = int(np.sum(delta > 0))
    negative = int(np.sum(delta < 0))
    sign_p = float(stats.binomtest(positive, positive + negative, 0.5).pvalue) if len(nonzero) else 1.0
    ttest = stats.ttest_rel(np.log2(tumor + 1.0), np.log2(normal + 1.0))
    try:
        wilcoxon = stats.wilcoxon(delta, alternative="two-sided", zero_method="wilcox")
        wilcoxon_p = float(wilcoxon.pvalue)
    except ValueError:
        wilcoxon_p = 1.0
    return {
        "n_pairs": int(len(delta)),
        "mean_log2fc": float(np.mean(delta)),
        "median_log2fc": float(np.median(delta)),
        "geometric_fold_change": float(2 ** np.mean(delta)),
        "positive_pairs": positive,
        "negative_pairs": negative,
        "exact_sign_p": sign_p,
        "paired_t_p": float(ttest.pvalue),
        "wilcoxon_p": wilcoxon_p,
        "delta": delta.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/external/OEP00006137_support/modified_guanosine_level1_rows.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/external/OEP00006137_support/modified_guanosine_reanalysis"),
    )
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "status": "oep00006137_modified_guanosine_reanalysis_complete",
        "source": payload["source"],
        "protocol": "paired log2(tumor+1)-log2(normal+1) reanalysis of the published Level-1 supplementary abundance matrix",
        "metabolites": {},
    }
    flat_rows: list[dict[str, object]] = []
    source_by_name = {str(row["Compound name"]): row for row in payload["rows"]}

    for row in payload["rows"]:
        name = str(row["Compound name"])
        metabolite = {
            "source_row": int(row["source_row"]),
            "peak_name": row["Peak name"],
            "formula": row["Formula"],
            "msi_level": row["MSI level"],
            "polarity": row["Polarity"],
            "adduct": row["Adduct"],
            "mz_detected": row["mz_detected"],
            "rt_detected": row["RT_detected"],
            "ms2_match_score": row["MS2_match_score"],
        }
        deltas: dict[str, np.ndarray] = {}
        for subtype, normal_prefix, tumor_prefix in (
            ("MSI", "MSI_N", "MSI.H_T"),
            ("MSS", "MSS_N", "MSS_T"),
        ):
            normal = np.asarray([float(row[f"{normal_prefix}{i}"]) for i in range(1, 21)], dtype=float)
            tumor = np.asarray([float(row[f"{tumor_prefix}{i}"]) for i in range(1, 21)], dtype=float)
            summary = paired_summary(normal, tumor)
            deltas[subtype] = np.asarray(summary.pop("delta"), dtype=float)
            metabolite[subtype] = summary
            flat_rows.append({"compound": name, "subtype": subtype, **summary})

        welch = stats.ttest_ind(deltas["MSI"], deltas["MSS"], equal_var=False)
        mannwhitney = stats.mannwhitneyu(deltas["MSI"], deltas["MSS"], alternative="two-sided")
        metabolite["MSS_minus_MSI_interaction"] = {
            "mean_delta_log2fc": float(np.mean(deltas["MSS"]) - np.mean(deltas["MSI"])),
            "welch_p": float(welch.pvalue),
            "mannwhitney_p": float(mannwhitney.pvalue),
        }
        report["metabolites"][name] = metabolite

    module_members = [
        "1-Methylguanosine",  # same measured peak as 2'-O-Methylguanosine in the supplement
        "2-Methylguanosine",  # same measured peak as 3'-O-Methylguanosine in the supplement
        "N2,N2-Dimethylguanosine",
        "7-Methylguanosine",
    ]
    module_report: dict[str, object] = {"unique_measured_peaks": module_members}
    for subtype, normal_prefix, tumor_prefix in (
        ("MSI", "MSI_N", "MSI.H_T"),
        ("MSS", "MSS_N", "MSS_T"),
    ):
        normal_matrix = np.asarray(
            [[float(source_by_name[name][f"{normal_prefix}{i}"]) for i in range(1, 21)] for name in module_members]
        )
        tumor_matrix = np.asarray(
            [[float(source_by_name[name][f"{tumor_prefix}{i}"]) for i in range(1, 21)] for name in module_members]
        )
        normal_module = np.mean(np.log2(normal_matrix + 1.0), axis=0)
        tumor_module = np.mean(np.log2(tumor_matrix + 1.0), axis=0)
        module_report[subtype] = paired_summary(2**normal_module - 1.0, 2**tumor_module - 1.0)
    report["unique_peak_modified_guanosine_module"] = module_report

    report["identity_boundary"] = {
        "same_peak_duplicate_assignments": [
            {
                "peak_name_stem": "M296T181",
                "assignments": ["1-Methylguanosine", "2'-O-Methylguanosine"],
                "meaning": "the abundance vectors are identical; these are alternative annotations of one measured feature, not independent metabolites",
            }
        ],
        "claim_limit": "Cross-cohort abundance supports modified-guanosine biology. It does not transfer positional-isomer identity to MTBLS13729 without retention-time/standard matching.",
    }

    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(flat_rows).drop(columns=["delta"], errors="ignore").to_csv(
        args.output_dir / "paired_effects.csv", index=False
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
