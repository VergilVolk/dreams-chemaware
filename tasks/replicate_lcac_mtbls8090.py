"""External, phenotype-blind LCAC class-score replication in MTBLS8090.

MTBLS8090 is an independent 35-pair CRC tissue LC-MS cohort.  Its released
MAF is a targeted/annotated abundance matrix rather than raw DDA spectra, so
this script validates the *metabolic phenotype*, not DreaMS retrieval quality.
Feature inclusion is fixed before reading tumour/control abundances: an
annotated acylcarnitine with an explicit carbon count C>=12.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


CHAIN = re.compile(r"(?:^|\s)C(\d{1,2})(?=[:\-]|$)")


def signflip_monte_carlo(values: np.ndarray, seed: int = 8090, n_draws: int = 200_000) -> float:
    """Two-sided randomization p value; exact enumeration is infeasible at n=35."""
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    for start in range(0, n_draws, 10_000):
        signs = rng.choice(np.array([-1.0, 1.0]), size=(min(10_000, n_draws - start), len(values)))
        exceed += int((np.abs((signs * values).mean(axis=1)) >= observed).sum())
    return float((exceed + 1) / (n_draws + 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maf", type=Path, default=Path("data/external/MTBLS8090/reverse_phase.maf.tsv"))
    parser.add_argument("--samples", type=Path, default=Path("data/external/MTBLS8090/s_MTBLS8090.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/external/MTBLS8090/lcac_replication"))
    parser.add_argument("--min-carbon", type=int, default=12)
    args = parser.parse_args()

    matrix = pd.read_csv(args.maf, sep="\t")
    sample_meta = pd.read_csv(args.samples, sep="\t")
    tumour = sample_meta.loc[sample_meta["Factor Value[Disease]"].eq("colorectal carcinoma"), "Sample Name"].tolist()
    normal = sample_meta.loc[sample_meta["Factor Value[Disease]"].eq("control"), "Sample Name"].tolist()
    if len(tumour) != len(normal) or {x + "_1" for x in tumour} != set(normal):
        raise ValueError("MTBLS8090 does not expose one matched control per tumour sample.")
    required = set(tumour + normal)
    missing = required - set(matrix.columns)
    if missing:
        raise ValueError(f"MAF is missing sample abundance columns: {sorted(missing)[:5]}")

    labels = matrix.metabolite_identification.fillna("").astype(str)
    carbon = labels.str.extract(CHAIN, expand=False).astype("Float64")
    selected = matrix.loc[labels.str.contains("carnitine", case=False) & (carbon >= args.min_carbon)].copy()
    selected["carbon"] = carbon.loc[selected.index].astype(int)
    if selected.empty:
        raise ValueError("No annotated C>=12 acylcarnitines in supplied MAF.")

    # Avoid duplicate annotation rows mapping to the same stated metabolite name.
    selected = selected.sort_values(["metabolite_identification", "mass_to_charge"]).drop_duplicates("metabolite_identification")
    abundance = selected[tumour + normal].apply(pd.to_numeric, errors="coerce")
    positive = abundance.to_numpy(float)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    pseudocount = float(np.percentile(positive, 1) / 2)
    log_abundance = np.log2(abundance + pseudocount)
    pair_effects = pd.DataFrame(
        {
            "patient": tumour,
            "class_median_log2fc": [
                float(np.median(log_abundance[t] - log_abundance[t + "_1"])) for t in tumour
            ],
            "class_mean_log2fc": [
                float(np.mean(log_abundance[t] - log_abundance[t + "_1"])) for t in tumour
            ],
        }
    )
    individual = []
    for _, row in selected.iterrows():
        deltas = np.asarray([log_abundance.loc[row.name, t] - log_abundance.loc[row.name, t + "_1"] for t in tumour], dtype=float)
        individual.append(
            {
                "metabolite_identification": row.metabolite_identification,
                "mass_to_charge": row.mass_to_charge,
                "carbon": int(row.carbon),
                "mean_log2fc": float(deltas.mean()),
                "median_log2fc": float(np.median(deltas)),
                "wilcoxon_p": float(wilcoxon(deltas).pvalue),
                "direction_tumour_higher": bool(deltas.mean() > 0),
            }
        )
    values = pair_effects.class_median_log2fc.to_numpy(float)
    report = {
        "status": "complete",
        "study": "MTBLS8090",
        "study_role": "independent CRC tumour-vs-paired-normal metabolic phenotype replication only",
        "selection_rule": f"annotated acylcarnitine with explicit C >= {args.min_carbon}; independent of abundance and disease labels",
        "n_pairs": int(len(tumour)),
        "n_lcac_features": int(len(selected)),
        "pseudocount": pseudocount,
        "class_median_log2fc_mean": float(values.mean()),
        "class_median_log2fc_median": float(np.median(values)),
        "wilcoxon_p": float(wilcoxon(values).pvalue),
        "signflip_monte_carlo_p": signflip_monte_carlo(values),
        "interpretation_limit": (
            "This confirms or refutes a generic CRC LCAC steady-state pattern only. "
            "It lacks Rmu histology labels and raw DDA spectra, so it cannot validate "
            "Rmu specificity, DreaMS retrieval, fragmentation evidence, or metabolic flux."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected[["metabolite_identification", "mass_to_charge", "chemical_formula", "carbon"]].to_csv(
        args.output_dir / "pre_frozen_lcac_features.csv", index=False
    )
    pair_effects.to_csv(args.output_dir / "lcac_pair_effects.csv", index=False)
    pd.DataFrame(individual).to_csv(args.output_dir / "lcac_individual_effects.csv", index=False)
    (args.output_dir / "lcac_replication_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
