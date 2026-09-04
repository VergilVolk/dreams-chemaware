"""Candidate-level paired abundance and subtype-sensitivity audit for MTBLS13729.

The 18 candidates are frozen before this analysis.  Tests remain descriptive
because the candidate panel was assembled from the same biological cohort.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from audit_mtbls13729_full_requantifiable_space import bh_adjust, exact_signflip_p, pqn, sample_pairs


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "data/mtbls13729/ms1_consensus/pos_rp__discovery_intensity_matrix.csv.gz"
TARGETS = ROOT / "data/mtbls13729/ms1_consensus/pos_rp__requantification_targets.csv.gz"
CANDIDATES = ROOT / "data/mtbls13729/manuscript_evidence_matrix_v2/candidate_manuscript_evidence_matrix_v2.csv"
OUT = ROOT / "data/mtbls13729/candidate_subtype_interactions_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_group_permutation_p(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return math.nan
    combined = np.concatenate([a, b])
    observed = abs(float(a.mean() - b.mean()))
    exceed = 0
    total = math.comb(len(combined), len(a))
    for chosen in itertools.combinations(range(len(combined)), len(a)):
        selected = np.zeros(len(combined), dtype=bool)
        selected[list(chosen)] = True
        effect = abs(float(combined[selected].mean() - combined[~selected].mean()))
        exceed += int(effect >= observed - 1e-12)
    return float(exceed / total)


def feature_deltas(frame: pd.DataFrame, pairs: list[tuple[str, str]], feature: int) -> pd.Series:
    left = [a for a, _ in pairs]
    right = [b for _, b in pairs]
    tumour = frame.loc[feature, left].copy()
    normal = frame.loc[feature, right].copy()
    tumour.index = [a.split("-")[0] for a, _ in pairs]
    normal.index = tumour.index
    return tumour - normal


def finite_summary(values: pd.Series) -> dict[str, float | int]:
    x = values.dropna().to_numpy(float)
    return {
        "n": int(len(x)),
        "mean": float(x.mean()) if len(x) else math.nan,
        "median": float(np.median(x)) if len(x) else math.nan,
        "positive_fraction": float((x > 0).mean()) if len(x) else math.nan,
        "exact_signflip_p": exact_signflip_p(x) if len(x) else math.nan,
    }


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)

    matrix = pd.read_csv(MATRIX).set_index("feature_id")
    targets = pd.read_csv(TARGETS).set_index("feature_id")
    candidates = pd.read_csv(CANDIDATES)
    candidates = candidates[candidates.feature_id.astype(str).str.fullmatch(r"\d+")].copy()
    candidates["feature_id"] = candidates.feature_id.astype(int)
    candidates = candidates[candidates.feature_id.isin(matrix.index.astype(int))].copy()
    if len(candidates) != 17:
        raise RuntimeError(f"expected 17 positive-RP frozen candidates, found {len(candidates)}")

    matrix = matrix.astype(float).where(lambda x: x > 0.0)
    pseudo = float(np.percentile(matrix.stack().to_numpy(float), 1) / 2.0)
    log_raw = np.log2(matrix + pseudo)
    pqn_matrix, factors = pqn(log_raw, targets["global_prevalence"])
    normalizations = {"log_raw": log_raw, "pqn_prev60": pqn_matrix}

    pairs = {
        "Rmu": sample_pairs(list(matrix.columns), "Rmu", "RN"),
        "Rtu": sample_pairs(list(matrix.columns), "Rtu", "RN"),
        "Ltu": sample_pairs(list(matrix.columns), "Ltu", "LN"),
    }
    sizes = {key: len(value) for key, value in pairs.items()}
    if sizes["Rmu"] != 10 or sizes["Rtu"] != 10 or sizes["Ltu"] < 8:
        raise RuntimeError(f"unexpected paired cohort sizes: {sizes}")

    rows: list[dict[str, object]] = []
    patient_rows: list[dict[str, object]] = []
    metadata = candidates.set_index("feature_id")
    for normalization, frame in normalizations.items():
        for feature in candidates.feature_id:
            values = {cohort: feature_deltas(frame, cohort_pairs, feature) for cohort, cohort_pairs in pairs.items()}
            summaries = {cohort: finite_summary(series) for cohort, series in values.items()}
            rmu = values["Rmu"].dropna().to_numpy(float)
            rtu = values["Rtu"].dropna().to_numpy(float)
            ltu = values["Ltu"].dropna().to_numpy(float)
            meta = metadata.loc[feature]
            row: dict[str, object] = {
                "normalization": normalization,
                "feature_id": int(feature),
                "label": meta["label"],
                "module": meta["module"],
                "manuscript_evidence_tier": meta["manuscript_evidence_tier"],
            }
            for cohort, summary in summaries.items():
                row.update({f"{cohort.lower()}_{key}": value for key, value in summary.items()})
            row.update(
                {
                    "rmu_minus_rtu": float(rmu.mean() - rtu.mean()) if len(rmu) and len(rtu) else math.nan,
                    "rmu_vs_rtu_exact_permutation_p": exact_group_permutation_p(rmu, rtu),
                    "rmu_minus_ltu": float(rmu.mean() - ltu.mean()) if len(rmu) and len(ltu) else math.nan,
                    "rmu_vs_ltu_exact_permutation_p": exact_group_permutation_p(rmu, ltu),
                }
            )
            rows.append(row)
            for cohort, series in values.items():
                for patient, value in series.items():
                    patient_rows.append(
                        {
                            "normalization": normalization,
                            "feature_id": int(feature),
                            "label": meta["label"],
                            "module": meta["module"],
                            "cohort": cohort,
                            "patient": patient,
                            "paired_tumour_normal_log2fc": value,
                        }
                    )

    summary = pd.DataFrame(rows)
    for normalization, index in summary.groupby("normalization").groups.items():
        loc = list(index)
        endpoints = {
            "rmu_exact_signflip_p": "rmu_exact_signflip_bh_q_17_candidates",
            "rmu_vs_rtu_exact_permutation_p": "rmu_vs_rtu_exact_permutation_bh_q_17_candidates",
            "rmu_vs_ltu_exact_permutation_p": "rmu_vs_ltu_exact_permutation_bh_q_17_candidates",
        }
        for endpoint, q_column in endpoints.items():
            summary.loc[loc, q_column] = bh_adjust(
                summary.loc[loc, endpoint].to_numpy(float)
            )
    summary.to_csv(OUT / "candidate_subtype_interaction_summary.csv", index=False)
    pd.DataFrame(patient_rows).to_csv(OUT / "candidate_patient_paired_effects.csv", index=False)
    pd.DataFrame({"sample": factors.index, "pqn_log2_factor": factors.to_numpy(float)}).to_csv(
        OUT / "pqn_factors.csv", index=False
    )

    decisions: list[dict[str, object]] = []
    for feature, group in summary.groupby("feature_id"):
        decisions.append(
            {
                "feature_id": int(feature),
                "label": group.label.iloc[0],
                "module": group.module.iloc[0],
                "rmu_n_min": int(group.rmu_n.min()),
                "rmu_primary_direction_consistent": bool((group.rmu_mean > 0).all()),
                "rmu_primary_bh_q_max": float(group.rmu_exact_signflip_bh_q_17_candidates.max()),
                "rmu_vs_rtu_direction_consistent": bool((group.rmu_minus_rtu > 0).all()),
                "rmu_vs_rtu_bh_q_max": float(group.rmu_vs_rtu_exact_permutation_bh_q_17_candidates.max()),
                "rmu_vs_ltu_bh_q_max": float(group.rmu_vs_ltu_exact_permutation_bh_q_17_candidates.max()),
            }
        )
    decision = pd.DataFrame(decisions).sort_values(
        ["rmu_vs_rtu_bh_q_max", "rmu_primary_bh_q_max", "feature_id"]
    )
    decision.to_csv(OUT / "cross_normalization_candidate_decisions.csv", index=False)

    report = {
        "status": "mtbls13729_candidate_subtype_interactions_v1_complete",
        "formal": False,
        "candidate_panel": "17 frozen positive-RP candidates; HILIC-only sphingosine excluded",
        "paired_cohorts": sizes,
        "candidate_results": decisions,
        "claim_limit": (
            "Candidate-panel BH values do not represent the full 13,155-target untargeted space and do not "
            "correct same-cohort candidate selection. Rmu-versus-Ltu is additionally confounded by tumour side."
        ),
        "provenance": {
            "matrix_sha256": sha256(MATRIX),
            "targets_sha256": sha256(TARGETS),
            "candidates_sha256": sha256(CANDIDATES),
            "summary_sha256": sha256(OUT / "candidate_subtype_interaction_summary.csv"),
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
