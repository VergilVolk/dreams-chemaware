#!/usr/bin/env python
"""Post-selection convergence audit of MTBLS13729 metabolic modules.

The features in these modules were selected after phenotype-aware discovery.
Therefore all p-values and intervals are descriptive stability summaries, not
new confirmatory tests or full-feature-space FDR evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


FEATURES = {
    "myristoylcarnitine": ("full", 347),
    "palmitoylcarnitine": ("full", 150),
    "c20_4_acylcarnitine_like": ("closure", 3222),
    "free_carnitine": ("full", 398),
    "n1_acetylspermine": ("full", 457),
    "n1_n8_diacetylspermidine_like": ("closure", 1717),
    "methylthioadenosine": ("full", 494),
    "hypoxanthine": ("full", 73),
    "methylguanosine_family": ("closure", 1597),
    "dimethylguanosine_family": ("closure", 3019),
    "isoleucine": ("full", 83),
    "phenylalanine": ("full", 722),
    "tryptophan": ("full", 732),
    "sphingosine": ("sphingo", 9900175),
}


MODULES = {
    "long_chain_acylcarnitine_accumulation": [
        "myristoylcarnitine",
        "palmitoylcarnitine",
        "c20_4_acylcarnitine_like",
    ],
    "acetylated_polyamine_mta_turnover": [
        "n1_acetylspermine",
        "n1_n8_diacetylspermidine_like",
        "methylthioadenosine",
    ],
    "purine_modified_nucleoside_pool": [
        "hypoxanthine",
        "methylguanosine_family",
        "dimethylguanosine_family",
    ],
    "large_neutral_amino_acid_pool": [
        "isoleucine",
        "phenylalanine",
        "tryptophan",
    ],
}


def patient_pairs() -> list[tuple[str, str, str]]:
    return [(f"P{number:02d}", f"P{number:02d}-Rmu", f"P{number:02d}-RN") for number in range(21, 31)]


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, repeats: int) -> tuple[float, float]:
    draws = rng.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mtbls13729/convergent_metabolic_modules_v1"),
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    matrices = {
        "full": pd.read_csv("data/mtbls13729/full_space_eic_v1/pos_rp__eic_auc_matrix.csv.gz").set_index("feature_id"),
        "closure": pd.read_csv("data/mtbls13729/biology_closure_eic_v1/pos_rp__eic_auc_matrix.csv.gz").set_index("feature_id"),
        "sphingo": pd.read_csv("data/mtbls13729/author_sphingolipid_eic_v2/pos_hilic__eic_auc_matrix.csv.gz").set_index("feature_id"),
    }
    rows: list[dict[str, object]] = []
    for feature_name, (matrix_name, feature_id) in FEATURES.items():
        matrix = matrices[matrix_name]
        if feature_id not in matrix.index:
            raise RuntimeError(f"missing {feature_name}: {matrix_name}/{feature_id}")
        feature = matrix.loc[feature_id]
        for patient, tumor, normal in patient_pairs():
            tumor_value = pd.to_numeric(feature.get(tumor), errors="coerce")
            normal_value = pd.to_numeric(feature.get(normal), errors="coerce")
            if not np.isfinite(tumor_value) or not np.isfinite(normal_value):
                delta = np.nan
            else:
                delta = float(np.log2(float(tumor_value) + 1.0) - np.log2(float(normal_value) + 1.0))
            rows.append(
                {
                    "patient": patient,
                    "feature": feature_name,
                    "source_matrix": matrix_name,
                    "feature_id": feature_id,
                    "log2_tumor_normal": delta,
                }
            )
    feature_deltas = pd.DataFrame(rows)
    feature_deltas.to_csv(output / "feature_patient_deltas.csv", index=False)

    rng = np.random.default_rng(args.seed)
    feature_summary: list[dict[str, object]] = []
    for feature, group in feature_deltas.groupby("feature", sort=False):
        values = group.log2_tumor_normal.dropna().to_numpy(float)
        ci_low, ci_high = bootstrap_mean(values, rng, args.bootstrap)
        positive = int(np.sum(values > 0))
        feature_summary.append(
            {
                "feature": feature,
                "feature_id": int(group.feature_id.iloc[0]),
                "pairs": int(len(values)),
                "mean_log2fc": float(values.mean()),
                "median_log2fc": float(np.median(values)),
                "positive_pairs": positive,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "one_sided_sign_p": float(binomtest(positive, len(values), 0.5, alternative="greater").pvalue),
            }
        )
    feature_summary_table = pd.DataFrame(feature_summary)
    feature_summary_table.to_csv(output / "feature_summary.csv", index=False)

    module_patient_rows: list[dict[str, object]] = []
    module_summary: list[dict[str, object]] = []
    for module, feature_names in MODULES.items():
        pivot = feature_deltas[feature_deltas.feature.isin(feature_names)].pivot(
            index="patient", columns="feature", values="log2_tumor_normal"
        )
        pivot = pivot.reindex(columns=feature_names)
        module_delta = pivot.mean(axis=1, skipna=True)
        support = pivot.notna().sum(axis=1)
        eligible = support >= 2
        for patient in pivot.index:
            module_patient_rows.append(
                {
                    "module": module,
                    "patient": patient,
                    "module_mean_log2fc": float(module_delta.loc[patient]),
                    "features_observed": int(support.loc[patient]),
                    "eligible": bool(eligible.loc[patient]),
                }
            )
        values = module_delta[eligible].to_numpy(float)
        ci_low, ci_high = bootstrap_mean(values, rng, args.bootstrap)
        positive = int(np.sum(values > 0))
        loo_feature_means = {}
        for removed in feature_names:
            retained = [feature for feature in feature_names if feature != removed]
            loo_feature_means[removed] = float(pivot[retained].mean(axis=1, skipna=True).mean())
        module_summary.append(
            {
                "module": module,
                "features": feature_names,
                "patients": int(len(values)),
                "mean_module_log2fc": float(values.mean()),
                "median_module_log2fc": float(np.median(values)),
                "positive_patients": positive,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "one_sided_sign_p": float(binomtest(positive, len(values), 0.5, alternative="greater").pvalue),
                "leave_one_feature_out_mean_log2fc": loo_feature_means,
                "leave_one_feature_out_direction_stable": bool(all(value > 0 for value in loo_feature_means.values())),
            }
        )
    pd.DataFrame(module_patient_rows).to_csv(output / "module_patient_deltas.csv", index=False)
    pd.DataFrame(
        [
            {**row, "features": "|".join(row["features"]), "leave_one_feature_out_mean_log2fc": json.dumps(row["leave_one_feature_out_mean_log2fc"])}
            for row in module_summary
        ]
    ).to_csv(output / "module_summary.csv", index=False)

    report = {
        "status": "mtbls13729_convergent_metabolic_modules_complete",
        "formal": False,
        "feature_results": feature_summary,
        "module_results": module_summary,
        "single_node_context": {
            "free_carnitine": next(row for row in feature_summary if row["feature"] == "free_carnitine"),
            "sphingosine": next(row for row in feature_summary if row["feature"] == "sphingosine"),
        },
        "claim_limit": (
            "These modules were defined after phenotype-aware feature discovery. They are "
            "descriptive convergence and leave-one-feature-out stability analyses, not "
            "independent replication, full-feature-space FDR, metabolic flux, enzyme activity, "
            "or causal pathway validation."
        ),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
