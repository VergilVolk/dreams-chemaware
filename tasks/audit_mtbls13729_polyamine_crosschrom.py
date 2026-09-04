#!/usr/bin/env python
"""Cross-chromatography audit of MTBLS13729 feature 1717.

The test asks whether the independently extracted RP feature tracks the source
study's positive-HILIC HMDB0041947 row across the same biological samples.  It
does not treat the source MAF name as an authentic-standard identification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


PROTON_MASS = 1.007276466621
NEUTRAL_EXACT_MASS = 229.179027


def tissue_label(sample: str) -> str:
    return sample.split("-", maxsplit=1)[1]


def patient_id(sample: str) -> str:
    return sample.split("-", maxsplit=1)[0]


def empirical_p(observed: float, null: np.ndarray) -> float:
    return float((1 + np.sum(null >= observed)) / (1 + len(null)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rp-matrix",
        type=Path,
        default=Path(
            "data/mtbls13729/biology_closure_eic_v1/pos_rp__eic_auc_matrix.csv.gz"
        ),
    )
    parser.add_argument(
        "--hilic-maf",
        type=Path,
        default=Path(
            "dreams-chemaware/_mtbls13729_meta/"
            "m_MTBLS13729_LC-MS_positive_hilic_metabolite_profiling_v2_maf.tsv"
        ),
    )
    parser.add_argument("--feature-id", type=int, default=1717)
    parser.add_argument("--hmdb-id", default="HMDB0041947")
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mtbls13729/polyamine_crosschrom_audit_v1"),
    )
    args = parser.parse_args()

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    rp_table = pd.read_csv(args.rp_matrix).set_index("feature_id")
    if args.feature_id not in rp_table.index:
        raise RuntimeError(f"RP feature {args.feature_id} is absent")
    rp = rp_table.loc[args.feature_id]
    maf = pd.read_csv(args.hilic_maf, sep="\t")
    target_rows = maf[maf["database_identifier"].eq(args.hmdb_id)]
    if len(target_rows) != 1:
        raise RuntimeError(f"expected one {args.hmdb_id} row, found {len(target_rows)}")
    target = target_rows.iloc[0]

    samples = [
        sample
        for sample in rp.index
        if sample in maf.columns
        and pd.notna(rp[sample])
        and pd.notna(pd.to_numeric(target[sample], errors="coerce"))
    ]
    if len(samples) < 50:
        raise RuntimeError(f"insufficient common samples: {len(samples)}")

    rp_log = pd.Series(
        np.log2(pd.to_numeric(rp[samples], errors="coerce").to_numpy(float) + 1.0),
        index=samples,
        name="rp_log2_auc",
    )
    hilic_log = pd.Series(
        np.log2(
            pd.to_numeric(target[samples], errors="coerce").to_numpy(float) + 1e-6
        ),
        index=samples,
        name="hilic_log2_abundance",
    )

    raw_spearman = float(spearmanr(rp_log, hilic_log).statistic)
    raw_pearson = float(pearsonr(rp_log, hilic_log).statistic)
    labels = pd.Series([tissue_label(sample) for sample in samples], index=samples)
    rp_residual = rp_log - rp_log.groupby(labels).transform("mean")
    hilic_residual = hilic_log - hilic_log.groupby(labels).transform("mean")
    residual_spearman = float(spearmanr(rp_residual, hilic_residual).statistic)

    patient_rows: list[dict[str, object]] = []
    for number in range(1, 31):
        patient = f"P{number:02d}"
        tumor = "Ltu" if number <= 10 else ("Rtu" if number <= 20 else "Rmu")
        normal = "LN" if number <= 10 else "RN"
        tumor_sample = f"{patient}-{tumor}"
        normal_sample = f"{patient}-{normal}"
        if tumor_sample not in samples or normal_sample not in samples:
            continue
        patient_rows.append(
            {
                "patient": patient,
                "subtype": tumor,
                "rp_tumor_minus_normal": float(
                    rp_log[tumor_sample] - rp_log[normal_sample]
                ),
                "hilic_tumor_minus_normal": float(
                    hilic_log[tumor_sample] - hilic_log[normal_sample]
                ),
            }
        )
    paired = pd.DataFrame(patient_rows)
    paired_spearman = float(
        spearmanr(
            paired.rp_tumor_minus_normal, paired.hilic_tumor_minus_normal
        ).statistic
    )

    feature_rows: list[dict[str, object]] = []
    for _, source in maf.iterrows():
        y = pd.to_numeric(source[samples], errors="coerce").to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(rp_log.to_numpy(float))
        if int(ok.sum()) < 50:
            continue
        rho = float(
            spearmanr(
                rp_log.to_numpy(float)[ok], np.log2(y[ok] + 1e-6)
            ).statistic
        )
        feature_rows.append(
            {
                "database_identifier": source["database_identifier"],
                "metabolite_identification": source["metabolite_identification"],
                "spearman_rho": rho,
                "n_common_samples": int(ok.sum()),
            }
        )
    correlations = pd.DataFrame(feature_rows).sort_values(
        "spearman_rho", ascending=False
    )
    correlations["rank"] = np.arange(1, len(correlations) + 1)
    target_rank = int(
        correlations.loc[
            correlations.database_identifier.eq(args.hmdb_id), "rank"
        ].iloc[0]
    )

    rng = np.random.default_rng(args.seed)
    tissue_null = np.empty(args.permutations, dtype=float)
    label_array = labels.to_numpy()
    y_array = hilic_log.to_numpy(float)
    x_array = rp_log.to_numpy(float)
    for iteration in range(args.permutations):
        shuffled = y_array.copy()
        for label in np.unique(label_array):
            index = np.flatnonzero(label_array == label)
            shuffled[index] = rng.permutation(shuffled[index])
        tissue_null[iteration] = float(spearmanr(x_array, shuffled).statistic)

    paired_null = np.empty(args.permutations, dtype=float)
    px = paired.rp_tumor_minus_normal.to_numpy(float)
    py = paired.hilic_tumor_minus_normal.to_numpy(float)
    for iteration in range(args.permutations):
        paired_null[iteration] = float(spearmanr(px, rng.permutation(py)).statistic)

    sample_table = pd.DataFrame(
        {
            "sample": samples,
            "patient": [patient_id(sample) for sample in samples],
            "tissue": [tissue_label(sample) for sample in samples],
            "rp_log2_auc": rp_log.to_numpy(float),
            "hilic_log2_abundance": hilic_log.to_numpy(float),
            "rp_within_tissue_residual": rp_residual.to_numpy(float),
            "hilic_within_tissue_residual": hilic_residual.to_numpy(float),
        }
    )
    sample_table.to_csv(output / "crosschrom_sample_values.csv", index=False)
    paired.to_csv(output / "paired_tumor_normal_deltas.csv", index=False)
    correlations.to_csv(output / "hilic_feature_correlation_rank.csv", index=False)

    rp_mz = 230.1859310563009
    hilic_mz = float(target["mass_to_charge"])
    theoretical_mh = NEUTRAL_EXACT_MASS + PROTON_MASS
    payload = {
        "status": "mtbls13729_polyamine_crosschrom_audit_complete",
        "formal": True,
        "rp_feature_id": args.feature_id,
        "hilic_database_identifier": args.hmdb_id,
        "hilic_name": str(target["metabolite_identification"]),
        "common_samples": int(len(samples)),
        "mass_consistency": {
            "rp_mz": rp_mz,
            "hilic_maf_mz": hilic_mz,
            "theoretical_mh": theoretical_mh,
            "rp_ppm_vs_theoretical": float((rp_mz - theoretical_mh) / theoretical_mh * 1e6),
            "hilic_ppm_vs_theoretical": float(
                (hilic_mz - theoretical_mh) / theoretical_mh * 1e6
            ),
            "crosspanel_mz_difference_da": float(rp_mz - hilic_mz),
        },
        "cross_sample_concordance": {
            "spearman_rho": raw_spearman,
            "pearson_log_r": raw_pearson,
            "within_exact_tissue_label_spearman_rho": residual_spearman,
            "tissue_stratified_permutation_p": empirical_p(
                raw_spearman, tissue_null
            ),
        },
        "paired_tumor_normal_concordance": {
            "pairs": int(len(paired)),
            "spearman_rho": paired_spearman,
            "permutation_p": empirical_p(paired_spearman, paired_null),
        },
        "source_hilic_specificity": {
            "eligible_hilic_features": int(len(correlations)),
            "target_rank_by_spearman": target_rank,
            "target_is_top_ranked": bool(target_rank == 1),
            "next_two_features": correlations.iloc[1:3][
                ["database_identifier", "metabolite_identification", "spearman_rho"]
            ].to_dict("records"),
        },
        "source_maf_annotation_fields": {
            "fragmentation": None
            if pd.isna(target.get("fragmentation"))
            else str(target.get("fragmentation")),
            "reliability": None
            if pd.isna(target.get("reliability"))
            else str(target.get("reliability")),
            "search_engine_score": None
            if pd.isna(target.get("search_engine_score"))
            else str(target.get("search_engine_score")),
        },
        "claim_limit": (
            "Cross-chromatography abundance concordance and raw RP DDA coverage strengthen "
            "a shared polyamine-like ion-family hypothesis. The source HILIC MAF lacks a "
            "reported fragmentation/reliability score, and the HILIC m/z differs from the "
            "theoretical [M+H]+ by more than 10 ppm; an authentic standard is still required "
            "for MSI Level 1 and the exact N1,N8-diacetylspermidine name."
        ),
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
