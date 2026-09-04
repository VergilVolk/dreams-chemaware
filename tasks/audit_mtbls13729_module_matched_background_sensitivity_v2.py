"""Sensitivity audit for same-cohort, phenotype-blind matched module backgrounds.

The primary specification excludes DDA MS2 support because support count is
partly downstream of ion abundance and can create sparse, poor matches.  Two
outcome-blind alternatives are reported rather than choosing a specification
after seeing which gives the smallest tail area.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data/mtbls13729/full_requantifiable_space_audit_v1/pos_rp__full_feature_audit.csv.gz"
LEDGER = ROOT / "data/mtbls13729/integrated_biology_ledger_v2/integrated_candidate_ledger_v2.csv"
OUT = ROOT / "data/mtbls13729/module_matched_background_sensitivity_v2"
SEED = 20260830
REPEATS = 50_000
NEIGHBOURS = 100

MODULES = {
    "acetylated_polyamine_mta": [457, 1717, 494],
    "purine_modified_guanosine": [73, 1597, 3019],
    "long_chain_acylcarnitine": [347, 150, 3222],
    "expanded_amino_acid": [83, 722, 732, 345, 374],
    "neu5ac": [703],
}

SPECS = {
    "primary_acquisition_only": {
        "covariates": ["log_mz", "rt_sec", "global_prevalence", "log_family_size"],
        "prevalence_caliper": None,
    },
    "prevalence_caliper_0.10": {
        "covariates": ["log_mz", "rt_sec", "log_family_size"],
        "prevalence_caliper": 0.10,
    },
    "support_adjusted_sensitivity": {
        "covariates": ["log_mz", "rt_sec", "global_prevalence", "log_family_size", "log_support"],
        "prevalence_caliper": None,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def robust_z(array: np.ndarray) -> np.ndarray:
    median = np.nanmedian(array)
    scale = np.nanmedian(np.abs(array - median)) * 1.4826
    if not np.isfinite(scale) or scale <= 0:
        scale = np.nanstd(array)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return (array - median) / scale


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(AUDIT).copy()
    ledger = pd.read_csv(LEDGER)
    candidate_ids = {value for features in MODULES.values() for value in features}
    if not candidate_ids.issubset(set(audit.feature_id.astype(int))) or not candidate_ids.issubset(
        set(ledger.feature_id.astype(int))
    ):
        raise RuntimeError("frozen candidate mismatch")

    audit["effect"] = audit[["raw_mean_log2fc", "pqn_mean_log2fc"]].mean(axis=1)
    audit["log_mz"] = np.log(audit["mz"].clip(lower=1.0))
    audit["log_family_size"] = np.log1p(audit["ion_family_size"].fillna(1.0))
    audit["log_support"] = np.log1p(audit["n_support_spectra"].fillna(0.0))
    feature_to_idx = {int(value): idx for idx, value in enumerate(audit.feature_id)}
    family = audit["ion_family_id"].astype(int).to_numpy()
    candidate_families = {int(family[feature_to_idx[value]]) for value in candidate_ids}
    eligible_base = (
        audit["keep_for_requantification"].astype(bool)
        & audit["min_pairs"].ge(8)
        & np.isfinite(audit["effect"])
        & ~audit["ion_family_id"].astype(int).isin(candidate_families)
    ).to_numpy(bool)
    if eligible_base.sum() < 1000:
        raise RuntimeError("unexpectedly small background")
    effects = audit["effect"].to_numpy(float)
    prevalence = audit["global_prevalence"].to_numpy(float)

    result_rows: list[dict[str, object]] = []
    match_rows: list[dict[str, object]] = []
    for spec_index, (spec_name, spec) in enumerate(SPECS.items()):
        columns = list(spec["covariates"])
        z = np.column_stack([robust_z(audit[column].to_numpy(float)) for column in columns])
        neighbours: dict[int, np.ndarray] = {}
        for feature in sorted(candidate_ids):
            idx = feature_to_idx[feature]
            eligible = eligible_base.copy()
            caliper = spec["prevalence_caliper"]
            if caliper is not None:
                eligible &= np.abs(prevalence - prevalence[idx]) <= float(caliper)
            pool = np.flatnonzero(eligible)
            # Low-prevalence feature 3019 has only 31 honest neighbours inside
            # the frozen +/-0.10 caliper.  Keep the sparse local stratum rather
            # than silently widening it after looking at outcomes.
            if len(pool) < 20:
                raise RuntimeError(f"{spec_name}/{feature}: only {len(pool)} caliper matches")
            distance = np.sqrt(np.sum((z[pool] - z[idx]) ** 2, axis=1))
            order = np.argsort(distance, kind="stable")[: min(NEIGHBOURS, len(pool))]
            chosen = pool[order]
            neighbours[feature] = chosen
            match_rows.append(
                {
                    "specification": spec_name,
                    "target_feature_id": feature,
                    "matches": int(len(chosen)),
                    "median_distance": float(np.median(distance[order])),
                    "p95_distance": float(np.quantile(distance[order], 0.95)),
                    "target_prevalence": float(prevalence[idx]),
                    "match_prevalence_median": float(np.median(prevalence[chosen])),
                }
            )

        rng = np.random.default_rng(SEED + spec_index)
        for module, features in MODULES.items():
            observed = effects[np.array([feature_to_idx[value] for value in features])]
            null_mean = np.empty(REPEATS)
            null_min = np.empty(REPEATS)
            for repeat in range(REPEATS):
                sampled: list[int] = []
                used: set[int] = set()
                for feature in features:
                    pool = neighbours[feature]
                    available = pool[~np.isin(family[pool], list(used))]
                    if len(available) == 0:
                        raise RuntimeError(f"{spec_name}/{module}: unique-family sampling failed")
                    idx = int(rng.choice(available))
                    sampled.append(idx)
                    used.add(int(family[idx]))
                values = effects[np.asarray(sampled, dtype=int)]
                null_mean[repeat] = values.mean()
                null_min[repeat] = values.min()
            result_rows.append(
                {
                    "specification": spec_name,
                    "module": module,
                    "n_features": len(features),
                    "observed_mean_effect": float(observed.mean()),
                    "observed_min_effect": float(observed.min()),
                    "null_mean": float(null_mean.mean()),
                    "null_p95": float(np.quantile(null_mean, 0.95)),
                    "mean_effect_empirical_upper_tail": float(
                        (1 + np.sum(null_mean >= observed.mean())) / (REPEATS + 1)
                    ),
                    "minimum_effect_empirical_upper_tail": float(
                        (1 + np.sum(null_min >= observed.min())) / (REPEATS + 1)
                    ),
                }
            )

    results = pd.DataFrame(result_rows)
    matches = pd.DataFrame(match_rows)
    results.to_csv(OUT / "module_background_sensitivity.csv", index=False)
    matches.to_csv(OUT / "candidate_match_quality.csv", index=False)

    pivot = results.pivot(index="module", columns="specification", values="mean_effect_empirical_upper_tail")
    robust_below_005 = (pivot < 0.05).all(axis=1)
    robust_below_010 = (pivot < 0.10).all(axis=1)
    report = {
        "status": "mtbls13729_module_matched_background_sensitivity_v2_complete",
        "formal": False,
        "repeats_per_specification": REPEATS,
        "background_features": int(eligible_base.sum()),
        "specifications": SPECS,
        "results": result_rows,
        "modules_below_0_05_in_all_specs": sorted(robust_below_005[robust_below_005].index.tolist()),
        "modules_below_0_10_in_all_specs": sorted(robust_below_010[robust_below_010].index.tolist()),
        "decision": (
            "The primary matching excludes DDA support because support is partly downstream of abundance. "
            "A module is described as background-robust only when its direction and extremity persist across all "
            "three outcome-blind matching definitions."
        ),
        "claim_limit": (
            "These are same-cohort post-selection tail areas, not confirmatory p-values. Robustness only argues "
            "against the matched acquisition/background covariates as a complete explanation."
        ),
        "provenance": {
            "audit_sha256": sha256(AUDIT),
            "ledger_sha256": sha256(LEDGER),
            "results_sha256": sha256(OUT / "module_background_sensitivity.csv"),
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
