#!/usr/bin/env python
"""Orthogonal-panel abundance audit for named MTBLS13729 candidates.

Each candidate was discovered/re-quantified in an RP panel and is compared
against the source study's independently acquired opposite-polarity or HILIC
panel.  This is a same-cohort technical orthogonality test, not an independent
biological replication or an authentic-standard identity confirmation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


TARGETS = (
    {
        "feature_id": 73,
        "candidate": "Hypoxanthine",
        "discovery_panel": "pos_rp",
        "source_panel": "neg_rp",
        "hmdb_id": "HMDB0000157",
    },
    {
        "feature_id": 732,
        "candidate": "Tryptophan",
        "discovery_panel": "pos_rp",
        "source_panel": "neg_rp",
        "hmdb_id": "HMDB0000929",
    },
    {
        "feature_id": 398,
        "candidate": "Carnitine",
        "discovery_panel": "pos_rp",
        "source_panel": "pos_hilic",
        "hmdb_id": "HMDB0000062",
    },
    {
        "feature_id": 428,
        "candidate": "Taurine",
        "discovery_panel": "neg_rp",
        "source_panel": "neg_hilic",
        "hmdb_id": "HMDB0000251",
    },
)


MAF_NAMES = {
    "neg_rp": "m_MTBLS13729_LC-MS_negative_reverse-phase_metabolite_profiling_v2_maf.tsv",
    "pos_rp": "m_MTBLS13729_LC-MS_positive_reverse-phase_metabolite_profiling_v2_maf.tsv",
    "neg_hilic": "m_MTBLS13729_LC-MS_negative_hilic_metabolite_profiling_v2_maf.tsv",
    "pos_hilic": "m_MTBLS13729_LC-MS_positive_hilic_metabolite_profiling_v2_maf.tsv",
}


def patient_id(sample: str) -> str:
    return sample.split("-", maxsplit=1)[0]


def tissue_label(sample: str) -> str:
    return sample.split("-", maxsplit=1)[1]


def empirical_upper_p(observed: float, null: np.ndarray) -> float:
    if not np.isfinite(observed):
        return float("nan")
    return float((1 + np.sum(null >= observed)) / (1 + len(null)))


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 4 or np.unique(x[ok]).size < 2 or np.unique(y[ok]).size < 2:
        return float("nan")
    return float(spearmanr(x[ok], y[ok]).statistic)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eic-dir",
        type=Path,
        default=Path("data/mtbls13729/full_space_eic_v1"),
    )
    parser.add_argument(
        "--maf-dir",
        type=Path,
        default=Path("dreams-chemaware/_mtbls13729_meta"),
    )
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mtbls13729/named_candidate_crosspanel_audit_v1"),
    )
    args = parser.parse_args()

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    matrices: dict[str, pd.DataFrame] = {}
    mafs: dict[str, pd.DataFrame] = {}
    for panel in {str(target["discovery_panel"]) for target in TARGETS}:
        path = args.eic_dir / f"{panel}__eic_auc_matrix.csv.gz"
        matrices[panel] = pd.read_csv(path).set_index("feature_id")
    for panel in {str(target["source_panel"]) for target in TARGETS}:
        mafs[panel] = pd.read_csv(args.maf_dir / MAF_NAMES[panel], sep="\t")

    reports: list[dict[str, object]] = []
    sample_tables: list[pd.DataFrame] = []
    paired_tables: list[pd.DataFrame] = []
    rank_tables: list[pd.DataFrame] = []

    for target_spec in TARGETS:
        feature_id = int(target_spec["feature_id"])
        candidate = str(target_spec["candidate"])
        discovery_panel = str(target_spec["discovery_panel"])
        source_panel = str(target_spec["source_panel"])
        hmdb_id = str(target_spec["hmdb_id"])
        matrix = matrices[discovery_panel]
        if feature_id not in matrix.index:
            raise RuntimeError(f"{discovery_panel} feature {feature_id} is absent")
        maf = mafs[source_panel]
        hits = maf[maf.database_identifier.eq(hmdb_id)]
        if len(hits) != 1:
            raise RuntimeError(f"expected one {hmdb_id} row in {source_panel}; found {len(hits)}")
        source = hits.iloc[0]
        rp = matrix.loc[feature_id]
        samples = [
            column
            for column in matrix.columns
            if column in maf.columns
            and pd.notna(pd.to_numeric(rp[column], errors="coerce"))
            and pd.notna(pd.to_numeric(source[column], errors="coerce"))
        ]
        if len(samples) < 50:
            raise RuntimeError(f"{candidate}: only {len(samples)} common samples")

        x = np.log2(pd.to_numeric(rp[samples], errors="coerce").to_numpy(float) + 1.0)
        source_raw = pd.to_numeric(source[samples], errors="coerce").to_numpy(float)
        source_offset = max(1e-9, float(-np.nanmin(source_raw) + 1e-9))
        y = np.log2(source_raw + source_offset)
        labels = np.asarray([tissue_label(sample) for sample in samples], dtype=object)
        x_resid = x.copy()
        y_resid = y.copy()
        for label in np.unique(labels):
            index = np.flatnonzero(labels == label)
            x_resid[index] -= np.nanmean(x[index])
            y_resid[index] -= np.nanmean(y[index])

        raw_rho = safe_spearman(x, y)
        residual_rho = safe_spearman(x_resid, y_resid)
        raw_null = np.empty(args.permutations, dtype=float)
        for iteration in range(args.permutations):
            shuffled = y.copy()
            for label in np.unique(labels):
                index = np.flatnonzero(labels == label)
                shuffled[index] = rng.permutation(shuffled[index])
            raw_null[iteration] = safe_spearman(x, shuffled)

        paired_rows: list[dict[str, object]] = []
        sample_index = {sample: i for i, sample in enumerate(samples)}
        for number in range(1, 31):
            patient = f"P{number:02d}"
            tumor = "Ltu" if number <= 10 else ("Rtu" if number <= 20 else "Rmu")
            normal = "LN" if number <= 10 else "RN"
            tumor_sample = f"{patient}-{tumor}"
            normal_sample = f"{patient}-{normal}"
            if tumor_sample not in sample_index or normal_sample not in sample_index:
                continue
            ti, ni = sample_index[tumor_sample], sample_index[normal_sample]
            paired_rows.append(
                {
                    "candidate": candidate,
                    "feature_id": feature_id,
                    "patient": patient,
                    "subtype": tumor,
                    "discovery_tumor_minus_normal": float(x[ti] - x[ni]),
                    "source_tumor_minus_normal": float(y[ti] - y[ni]),
                }
            )
        paired = pd.DataFrame(paired_rows)
        paired_rho = safe_spearman(
            paired.discovery_tumor_minus_normal.to_numpy(float),
            paired.source_tumor_minus_normal.to_numpy(float),
        )
        paired_null = np.empty(args.permutations, dtype=float)
        py = paired.source_tumor_minus_normal.to_numpy(float)
        px = paired.discovery_tumor_minus_normal.to_numpy(float)
        for iteration in range(args.permutations):
            paired_null[iteration] = safe_spearman(px, rng.permutation(py))

        rank_rows: list[dict[str, object]] = []
        for _, other in maf.iterrows():
            other_raw = pd.to_numeric(other[samples], errors="coerce").to_numpy(float)
            ok = np.isfinite(other_raw) & np.isfinite(x)
            if int(ok.sum()) < 50:
                continue
            offset = max(1e-9, float(-np.nanmin(other_raw[ok]) + 1e-9))
            rho = safe_spearman(x[ok], np.log2(other_raw[ok] + offset))
            rank_rows.append(
                {
                    "candidate": candidate,
                    "feature_id": feature_id,
                    "source_panel": source_panel,
                    "database_identifier": other.database_identifier,
                    "metabolite_identification": other.metabolite_identification,
                    "spearman_rho": rho,
                    "n_common_samples": int(ok.sum()),
                }
            )
        ranks = pd.DataFrame(rank_rows).sort_values("spearman_rho", ascending=False)
        ranks["rank"] = np.arange(1, len(ranks) + 1)
        target_rank = int(ranks.loc[ranks.database_identifier.eq(hmdb_id), "rank"].iloc[0])

        samples_out = pd.DataFrame(
            {
                "candidate": candidate,
                "feature_id": feature_id,
                "sample": samples,
                "patient": [patient_id(sample) for sample in samples],
                "tissue": labels,
                "discovery_log2_auc": x,
                "source_log2_abundance": y,
                "discovery_within_tissue_residual": x_resid,
                "source_within_tissue_residual": y_resid,
            }
        )
        sample_tables.append(samples_out)
        paired_tables.append(paired)
        rank_tables.append(ranks)
        reports.append(
            {
                **target_spec,
                "source_name": str(source.metabolite_identification),
                "source_mz": float(source.mass_to_charge),
                "source_rt_min": float(source.retention_time),
                "common_samples": len(samples),
                "cross_sample_spearman": raw_rho,
                "within_tissue_spearman": residual_rho,
                "tissue_stratified_permutation_p": empirical_upper_p(raw_rho, raw_null),
                "paired_tumor_normal_pairs": int(len(paired)),
                "paired_tumor_normal_spearman": paired_rho,
                "paired_permutation_p": empirical_upper_p(paired_rho, paired_null),
                "source_panel_eligible_features": int(len(ranks)),
                "source_target_correlation_rank": target_rank,
                "source_target_top_ranked": target_rank == 1,
            }
        )

    pd.concat(sample_tables, ignore_index=True).to_csv(output / "crosspanel_sample_values.csv", index=False)
    pd.concat(paired_tables, ignore_index=True).to_csv(output / "paired_tumor_normal_deltas.csv", index=False)
    pd.concat(rank_tables, ignore_index=True).to_csv(output / "source_feature_correlation_ranks.csv", index=False)
    report_table = pd.DataFrame(reports)
    report_table.to_csv(output / "candidate_crosspanel_summary.csv", index=False)

    payload = {
        "status": "mtbls13729_named_candidate_crosspanel_audit_complete",
        "formal": True,
        "candidates": reports,
        "palmitoylcarnitine_crosspanel": {
            "status": "not_evaluable",
            "reason": "No exact palmitoylcarnitine row is present in the four source-study MAF tables; a related unsaturated C16 acylcarnitine is not an identity-equivalent substitute.",
        },
        "claim_limit": (
            "Same-cohort cross-panel abundance concordance is orthogonal technical evidence. "
            "It is not independent biological replication, authentic-standard confirmation, "
            "subtype specificity, metabolic flux, enzyme activity, or causality."
        ),
    }
    (output / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
