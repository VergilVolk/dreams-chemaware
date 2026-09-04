"""Separate paired tumour abundance from Rmu-versus-tubular interaction effects.

All modules are frozen but phenotype-selected in the same cohort.  The exact
tests are therefore descriptive subtype-sensitivity analyses, not independent
confirmation or selective-inference corrected discoveries.
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
OUT = ROOT / "data/mtbls13729/module_subtype_interactions_v1"

MODULES = {
    "acetylated_polyamine_mta": [457, 1717, 494],
    "purine_modified_guanosine": [73, 1597, 3019],
    "long_chain_acylcarnitine": [347, 150, 3222],
    "expanded_amino_acid": [83, 722, 732, 345, 374],
    "neu5ac": [703],
}


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
    total = math.comb(len(combined), len(a))
    exceed = 0
    for chosen in itertools.combinations(range(len(combined)), len(a)):
        mask = np.zeros(len(combined), dtype=bool)
        mask[list(chosen)] = True
        delta = abs(float(combined[mask].mean() - combined[~mask].mean()))
        exceed += int(delta >= observed - 1e-12)
    return float(exceed / total)


def module_deltas(
    log_matrix: pd.DataFrame,
    pairs: list[tuple[str, str]],
    features: list[int],
) -> pd.Series:
    tumour = log_matrix.loc[features, [left for left, _ in pairs]].copy()
    normal = log_matrix.loc[features, [right for _, right in pairs]].copy()
    tumour.columns = [left.split("-")[0] for left, _ in pairs]
    normal.columns = [left.split("-")[0] for left, _ in pairs]
    delta = tumour - normal
    minimum_features = max(1, math.ceil(len(features) * 0.60))
    return delta.mean(axis=0, skipna=True).where(delta.notna().sum(axis=0) >= minimum_features)


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    matrix = pd.read_csv(MATRIX).set_index("feature_id")
    targets = pd.read_csv(TARGETS).set_index("feature_id")
    expected = {value for features in MODULES.values() for value in features}
    if not expected.issubset(set(matrix.index.astype(int))):
        raise RuntimeError("frozen module feature absent from discovery matrix")

    matrix = matrix.astype(float).where(lambda frame: frame > 0.0)
    positive = matrix.stack().to_numpy(float)
    pseudo = float(np.percentile(positive, 1) / 2.0)
    raw = np.log2(matrix + pseudo)
    pqn_matrix, factors = pqn(raw, targets["global_prevalence"])
    normalizations = {"log_raw": raw, "pqn_prev60": pqn_matrix}

    columns = list(matrix.columns)
    cohort_pairs = {
        "Rmu": sample_pairs(columns, "Rmu", "RN"),
        "Rtu": sample_pairs(columns, "Rtu", "RN"),
        "Ltu": sample_pairs(columns, "Ltu", "LN"),
    }
    cohort_sizes = {key: len(value) for key, value in cohort_pairs.items()}
    # Rmu-vs-Rtu is the prespecified subtype comparison and must remain complete.
    # The public positive-RP deposition contains 59 rather than 60 samples, so the
    # side-confounded Ltu sensitivity analysis legitimately has one missing pair.
    if cohort_sizes["Rmu"] != 10 or cohort_sizes["Rtu"] != 10 or cohort_sizes["Ltu"] < 8:
        raise RuntimeError(f"unexpected paired cohort sizes: {cohort_sizes}")

    patient_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for normalization, log_matrix in normalizations.items():
        module_values: dict[tuple[str, str], np.ndarray] = {}
        for module, features in MODULES.items():
            for cohort, pairs in cohort_pairs.items():
                values = module_deltas(log_matrix, pairs, features)
                finite = values.dropna().to_numpy(float)
                module_values[(module, cohort)] = finite
                for patient, value in values.items():
                    patient_rows.append(
                        {
                            "normalization": normalization,
                            "module": module,
                            "cohort": cohort,
                            "patient": patient,
                            "paired_tumour_normal_log2fc": value,
                        }
                    )

            rmu = module_values[(module, "Rmu")]
            rtu = module_values[(module, "Rtu")]
            ltu = module_values[(module, "Ltu")]
            summary_rows.append(
                {
                    "normalization": normalization,
                    "module": module,
                    "n_features": len(features),
                    "rmu_n": len(rmu),
                    "rmu_mean": float(rmu.mean()),
                    "rmu_positive_fraction": float((rmu > 0).mean()),
                    "rmu_exact_signflip_p": exact_signflip_p(rmu),
                    "rtu_n": len(rtu),
                    "rtu_mean": float(rtu.mean()),
                    "rtu_positive_fraction": float((rtu > 0).mean()),
                    "rtu_exact_signflip_p": exact_signflip_p(rtu),
                    "ltu_n": len(ltu),
                    "ltu_mean": float(ltu.mean()),
                    "ltu_positive_fraction": float((ltu > 0).mean()),
                    "ltu_exact_signflip_p": exact_signflip_p(ltu),
                    "rmu_minus_rtu": float(rmu.mean() - rtu.mean()),
                    "rmu_vs_rtu_exact_permutation_p": exact_group_permutation_p(rmu, rtu),
                    "rmu_minus_ltu": float(rmu.mean() - ltu.mean()),
                    "rmu_vs_ltu_exact_permutation_p": exact_group_permutation_p(rmu, ltu),
                }
            )

    summary = pd.DataFrame(summary_rows)
    patient = pd.DataFrame(patient_rows)
    for normalization, index in summary.groupby("normalization").groups.items():
        positions = list(index)
        summary.loc[positions, "rmu_vs_rtu_bh_q_five_modules"] = bh_adjust(
            summary.loc[positions, "rmu_vs_rtu_exact_permutation_p"].to_numpy(float)
        )
        summary.loc[positions, "rmu_vs_ltu_bh_q_five_modules"] = bh_adjust(
            summary.loc[positions, "rmu_vs_ltu_exact_permutation_p"].to_numpy(float)
        )
    summary.to_csv(OUT / "module_subtype_interaction_summary.csv", index=False)
    patient.to_csv(OUT / "module_patient_paired_effects.csv", index=False)
    pd.DataFrame(
        {"sample": factors.index, "pqn_log2_factor": factors.to_numpy(float)}
    ).to_csv(OUT / "pqn_factors.csv", index=False)

    robust = []
    for module, group in summary.groupby("module"):
        robust.append(
            {
                "module": module,
                "rmu_primary_positive_both_normalizations": bool((group.rmu_mean > 0).all()),
                "rmu_vs_rtu_direction_consistent": bool(np.sign(group.rmu_minus_rtu).nunique() == 1),
                "rmu_vs_rtu_bh_q_max": float(group.rmu_vs_rtu_bh_q_five_modules.max()),
                "rmu_vs_ltu_bh_q_max": float(group.rmu_vs_ltu_bh_q_five_modules.max()),
            }
        )
    report = {
        "status": "mtbls13729_module_subtype_interactions_v1_complete",
        "formal": False,
        "paired_cohorts": {key: len(value) for key, value in cohort_pairs.items()},
        "primary_endpoint": "Rmu tumour versus its matched RN",
        "subtype_sensitivity": "difference of paired changes: (Rmu-RN) minus (Rtu-RN)",
        "left_side_sensitivity": "(Rmu-RN) minus (Ltu-LN); confounded by side and histology",
        "module_results": summary.to_dict(orient="records"),
        "cross_normalization_decision": robust,
        "claim_limit": (
            "Modules were phenotype-selected in the same cohort. Exact permutation and BH values describe "
            "subtype sensitivity but do not correct selection, establish independent replication, or prove a "
            "mucinous-specific mechanism. Rmu-versus-Ltu is additionally side-confounded."
        ),
        "provenance": {
            "matrix_sha256": sha256(MATRIX),
            "targets_sha256": sha256(TARGETS),
            "summary_sha256": sha256(OUT / "module_subtype_interaction_summary.csv"),
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
