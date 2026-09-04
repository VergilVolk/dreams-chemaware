"""Place frozen MTBLS13729 abundance modules against a phenotype-blind matched background.

The candidates were selected in the same cohort, so the empirical tail areas
below quantify extremity and detect obvious technical-background explanations;
they are not confirmatory p-values or selective-inference correction.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data/mtbls13729/full_requantifiable_space_audit_v1/pos_rp__full_feature_audit.csv.gz"
LEDGER = ROOT / "data/mtbls13729/integrated_biology_ledger_v2/integrated_candidate_ledger_v2.csv"
OUT = ROOT / "data/mtbls13729/module_matched_background_v1"
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def robust_z(values: pd.Series) -> np.ndarray:
    array = values.to_numpy(float)
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

    audit = pd.read_csv(AUDIT)
    ledger = pd.read_csv(LEDGER)
    candidate_ids = {feature for values in MODULES.values() for feature in values}
    if not candidate_ids.issubset(set(audit.feature_id.astype(int))):
        raise RuntimeError("one or more frozen module features are absent from the positive-RP audit")
    if not candidate_ids.issubset(set(ledger.feature_id.astype(int))):
        raise RuntimeError("one or more frozen module features are absent from the integrated ledger")

    audit = audit.copy()
    audit["effect"] = audit[["raw_mean_log2fc", "pqn_mean_log2fc"]].mean(axis=1)
    audit["annotation_present"] = audit["best_name"].notna().astype(float)
    audit["log_mz"] = np.log(audit["mz"].clip(lower=1.0))
    audit["log_family_size"] = np.log1p(audit["ion_family_size"].fillna(1.0))
    audit["log_support"] = np.log1p(audit["n_support_spectra"].fillna(0.0))
    covariates = ["log_mz", "rt_sec", "global_prevalence", "log_family_size", "log_support"]
    z = np.column_stack([robust_z(audit[column]) for column in covariates])
    annotated = audit["annotation_present"].to_numpy(float)
    feature_to_index = {int(feature): idx for idx, feature in enumerate(audit.feature_id)}

    candidate_families = set(
        audit.loc[audit.feature_id.astype(int).isin(candidate_ids), "ion_family_id"].astype(int)
    )
    eligible = (
        audit["keep_for_requantification"].astype(bool)
        & audit["min_pairs"].ge(8)
        & np.isfinite(audit["effect"])
        & ~audit["ion_family_id"].astype(int).isin(candidate_families)
    ).to_numpy(bool)
    if eligible.sum() < 1000:
        raise RuntimeError(f"matched-background pool unexpectedly small: {eligible.sum()}")

    neighbours: dict[int, np.ndarray] = {}
    neighbour_rows = []
    eligible_indices = np.flatnonzero(eligible)
    for feature in sorted(candidate_ids):
        idx = feature_to_index[feature]
        distance = np.sqrt(np.sum((z[eligible_indices] - z[idx]) ** 2, axis=1))
        distance += 0.75 * np.abs(annotated[eligible_indices] - annotated[idx])
        order = np.argsort(distance, kind="stable")[:NEIGHBOURS]
        selected = eligible_indices[order]
        neighbours[feature] = selected
        for rank, (row_idx, dist) in enumerate(zip(selected, distance[order]), start=1):
            neighbour_rows.append(
                {
                    "target_feature_id": feature,
                    "match_rank": rank,
                    "matched_feature_id": int(audit.iloc[row_idx].feature_id),
                    "matched_ion_family_id": int(audit.iloc[row_idx].ion_family_id),
                    "distance": float(dist),
                }
            )
    pd.DataFrame(neighbour_rows).to_csv(OUT / "candidate_matched_neighbours.csv.gz", index=False)

    rng = np.random.default_rng(SEED)
    summary_rows = []
    null_rows = []
    effects = audit["effect"].to_numpy(float)
    family_ids = audit["ion_family_id"].astype(int).to_numpy()
    for module, features in MODULES.items():
        observed = np.array([effects[feature_to_index[feature]] for feature in features], dtype=float)
        null_mean = np.empty(REPEATS, dtype=float)
        null_min = np.empty(REPEATS, dtype=float)
        null_positive_fraction = np.empty(REPEATS, dtype=float)
        collision_fallbacks = 0
        for repeat in range(REPEATS):
            chosen: list[int] = []
            used_families: set[int] = set()
            for feature in features:
                pool = neighbours[feature]
                available = pool[~np.isin(family_ids[pool], list(used_families))]
                if len(available) == 0:
                    available = pool
                    collision_fallbacks += 1
                selected = int(rng.choice(available))
                chosen.append(selected)
                used_families.add(int(family_ids[selected]))
            values = effects[np.asarray(chosen, dtype=int)]
            null_mean[repeat] = float(values.mean())
            null_min[repeat] = float(values.min())
            null_positive_fraction[repeat] = float((values > 0).mean())

        observed_mean = float(observed.mean())
        observed_min = float(observed.min())
        observed_positive = float((observed > 0).mean())
        summary_rows.append(
            {
                "module": module,
                "features": ";".join(map(str, features)),
                "n_features": len(features),
                "observed_mean_effect": observed_mean,
                "observed_min_effect": observed_min,
                "observed_positive_fraction": observed_positive,
                "matched_null_mean_effect": float(null_mean.mean()),
                "matched_null_mean_effect_p95": float(np.quantile(null_mean, 0.95)),
                "empirical_upper_tail_mean": float((1 + np.sum(null_mean >= observed_mean)) / (REPEATS + 1)),
                "empirical_upper_tail_min": float((1 + np.sum(null_min >= observed_min)) / (REPEATS + 1)),
                "empirical_upper_tail_positive_fraction": float(
                    (1 + np.sum(null_positive_fraction >= observed_positive)) / (REPEATS + 1)
                ),
                "collision_fallbacks": int(collision_fallbacks),
            }
        )
        for repeat in range(REPEATS):
            null_rows.append(
                {
                    "module": module,
                    "repeat": repeat,
                    "mean_effect": null_mean[repeat],
                    "minimum_effect": null_min[repeat],
                    "positive_fraction": null_positive_fraction[repeat],
                }
            )

    summary = pd.DataFrame(summary_rows)
    null = pd.DataFrame(null_rows)
    summary.to_csv(OUT / "module_matched_background_summary.csv", index=False)
    null.to_csv(OUT / "module_matched_background_null.csv.gz", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    positions = np.arange(len(summary))
    axes[0].barh(positions, summary["observed_mean_effect"], color="#3b82f6", label="frozen module")
    axes[0].scatter(summary["matched_null_mean_effect"], positions, color="#111827", marker="|", s=140, label="matched-null mean")
    for position, item in summary.iterrows():
        axes[0].hlines(position, item.matched_null_mean_effect, item.matched_null_mean_effect_p95, color="#6b7280", lw=2)
    axes[0].set_yticks(positions, summary["module"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Mean Rmu-RN log2 effect (raw/PQN average)")
    axes[0].set_title("Frozen modules versus phenotype-blind matched background")
    axes[0].legend(frameon=False)

    axes[1].barh(positions, -np.log10(summary["empirical_upper_tail_mean"].clip(lower=1 / (REPEATS + 1))), color="#10b981")
    axes[1].axvline(-np.log10(0.05), color="#9ca3af", ls="--", lw=1)
    axes[1].set_yticks(positions, summary["module"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("-log10 empirical tail area")
    axes[1].set_title("Descriptive extremity only; same-cohort selected")
    figure_path = OUT / "module_matched_background_v1.png"
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)

    report = {
        "status": "mtbls13729_module_matched_background_v1_complete",
        "formal": False,
        "modules": int(len(summary)),
        "background_features": int(eligible.sum()),
        "repeats": REPEATS,
        "neighbours_per_candidate": NEIGHBOURS,
        "matching_covariates": covariates + ["annotation_presence_penalty"],
        "results": summary.to_dict(orient="records"),
        "interpretation": (
            "This audit asks whether frozen modules are extreme relative to features matched without phenotype labels. "
            "Because the modules were selected in this cohort, empirical tail areas are descriptive and cannot be "
            "reported as confirmatory p-values."
        ),
        "claim_limit": (
            "A positive result argues against simple m/z/RT/prevalence/MS2-support background as the entire explanation. "
            "It does not correct post-selection bias, confirm identity, replicate biology, establish subtype specificity, "
            "or identify flux and causality."
        ),
        "provenance": {
            "full_audit_sha256": sha256(AUDIT),
            "ledger_sha256": sha256(LEDGER),
            "summary_sha256": sha256(OUT / "module_matched_background_summary.csv"),
            "figure_sha256": sha256(figure_path),
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
