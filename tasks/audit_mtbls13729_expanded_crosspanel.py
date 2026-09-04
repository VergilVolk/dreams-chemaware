#!/usr/bin/env python
"""Cross-panel audit for newly recovered MTBLS13729 metabolite families.

The source-study panels use the same patients but orthogonal chromatography or
polarity.  Therefore concordance is technical/identity support, not independent
biological replication.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


TARGETS = (
    {"feature_id": 345, "candidate": "Proline", "source_panel": "neg_hilic", "hmdb_id": "HMDB0000162"},
    {"feature_id": 374, "candidate": "Glutamic acid", "source_panel": "pos_hilic", "hmdb_id": "HMDB0000148"},
    {"feature_id": 703, "candidate": "N-Acetylneuraminic acid", "source_panel": "neg_hilic", "hmdb_id": "HMDB0000230"},
    {"feature_id": 1695, "candidate": "Leucine-like resolved peak", "source_panel": "neg_hilic", "hmdb_id": "HMDB0000687"},
)

MAF = {
    "neg_hilic": "m_MTBLS13729_LC-MS_negative_hilic_metabolite_profiling_v2_maf.tsv",
    "pos_hilic": "m_MTBLS13729_LC-MS_positive_hilic_metabolite_profiling_v2_maf.tsv",
}


def rho(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4 or np.unique(x[ok]).size < 2 or np.unique(y[ok]).size < 2:
        return float("nan")
    return float(spearmanr(x[ok], y[ok]).statistic)


def p_upper(observed: float, null: np.ndarray) -> float:
    return float((1 + np.sum(null >= observed)) / (len(null) + 1))


def paired_delta(values: dict[str, float]) -> pd.DataFrame:
    rows = []
    for number in range(1, 31):
        patient = f"P{number:02d}"
        subtype = "Ltu" if number <= 10 else ("Rtu" if number <= 20 else "Rmu")
        normal = "LN" if number <= 10 else "RN"
        tumour_key, normal_key = f"{patient}-{subtype}", f"{patient}-{normal}"
        if tumour_key in values and normal_key in values:
            rows.append({"patient": patient, "subtype": subtype, "delta": values[tumour_key] - values[normal_key]})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eic", type=Path, default=Path("data/mtbls13729/full_space_eic_v1/pos_rp__eic_auc_matrix.csv.gz"))
    parser.add_argument("--maf-dir", type=Path, default=Path("dreams-chemaware/_mtbls13729_meta"))
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--output-dir", type=Path, default=Path("data/mtbls13729/expanded_crosspanel_audit_v1"))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    eic = np.log2(pd.read_csv(args.eic).set_index("feature_id").astype(float) + 1.0)
    mafs = {panel: pd.read_csv(args.maf_dir / filename, sep="\t") for panel, filename in MAF.items()}
    reports = []
    all_pairs = []
    all_ranks = []
    for spec in TARGETS:
        fid = int(spec["feature_id"])
        if fid not in eic.index:
            raise RuntimeError(f"feature {fid} missing from positive-RP EIC")
        maf = mafs[str(spec["source_panel"])]
        hit = maf[maf.database_identifier.eq(spec["hmdb_id"])]
        if len(hit) != 1:
            raise RuntimeError(f"{spec['candidate']}: source row count={len(hit)}")
        source = hit.iloc[0]
        common = [c for c in eic.columns if c in maf.columns]
        x = eic.loc[fid, common].to_numpy(float)
        raw_y = pd.to_numeric(source[common], errors="coerce").to_numpy(float)
        offset = max(1e-9, float(-np.nanmin(raw_y) + 1e-9))
        y = np.log2(raw_y + offset)
        labels = np.asarray([c.split("-", 1)[1] for c in common], dtype=object)
        xr, yr = x.copy(), y.copy()
        for label in np.unique(labels):
            idx = np.flatnonzero(labels == label)
            xr[idx] -= np.mean(x[idx])
            yr[idx] -= np.mean(y[idx])
        sample_rho = rho(x, y)
        residual_rho = rho(xr, yr)
        null = np.empty(args.permutations)
        for i in range(args.permutations):
            shuffled = y.copy()
            for label in np.unique(labels):
                idx = np.flatnonzero(labels == label)
                shuffled[idx] = rng.permutation(shuffled[idx])
            null[i] = rho(x, shuffled)

        dx = paired_delta(dict(zip(common, x)))
        dy = paired_delta(dict(zip(common, y)))
        pairs = dx.merge(dy, on=["patient", "subtype"], suffixes=("_positive_rp", "_source"), validate="one_to_one")
        paired_rho = rho(pairs.delta_positive_rp.to_numpy(float), pairs.delta_source.to_numpy(float))
        pair_null = np.asarray([
            rho(pairs.delta_positive_rp.to_numpy(float), rng.permutation(pairs.delta_source.to_numpy(float)))
            for _ in range(args.permutations)
        ])

        rank_rows = []
        for _, row in maf.iterrows():
            values = pd.to_numeric(row[common], errors="coerce").to_numpy(float)
            if np.isfinite(values).sum() < 50:
                continue
            row_offset = max(1e-9, float(-np.nanmin(values) + 1e-9))
            rank_rows.append({
                "feature_id": fid,
                "candidate": spec["candidate"],
                "database_identifier": row.database_identifier,
                "metabolite_identification": row.metabolite_identification,
                "spearman_rho": rho(x, np.log2(values + row_offset)),
            })
        ranks = pd.DataFrame(rank_rows).sort_values("spearman_rho", ascending=False).reset_index(drop=True)
        ranks["rank"] = np.arange(1, len(ranks) + 1)
        target_rank = int(ranks.loc[ranks.database_identifier.eq(spec["hmdb_id"]), "rank"].iloc[0])
        ranks["source_panel"] = spec["source_panel"]
        all_ranks.append(ranks)
        pairs["feature_id"] = fid
        pairs["candidate"] = spec["candidate"]
        all_pairs.append(pairs)
        reports.append({
            **spec,
            "source_name": str(source.metabolite_identification),
            "source_mz": float(source.mass_to_charge),
            "source_rt_min": float(source.retention_time),
            "common_samples": len(common),
            "sample_spearman": sample_rho,
            "within_tissue_spearman": residual_rho,
            "tissue_stratified_permutation_p": p_upper(sample_rho, null),
            "paired_delta_spearman": paired_rho,
            "paired_delta_permutation_p": p_upper(paired_rho, pair_null),
            "source_target_rank": target_rank,
            "source_features_rankable": int(len(ranks)),
        })

    pd.DataFrame(reports).to_csv(output / "expanded_crosspanel_summary.csv", index=False)
    pd.concat(all_pairs, ignore_index=True).to_csv(output / "paired_deltas.csv", index=False)
    pd.concat(all_ranks, ignore_index=True).to_csv(output / "source_identity_ranks.csv", index=False)
    payload = {
        "status": "mtbls13729_expanded_crosspanel_audit_complete",
        "formal": True,
        "candidates": reports,
        "claim_limit": "Same-cohort orthogonal panel concordance supports feature reconciliation, not independent replication, authentic-standard identity in positive RP, subtype specificity, flux, enzyme activity, or causality.",
    }
    (output / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
