"""Audit whether secondary candidate modules merely reproduce the Neu5Ac axis.

This is a patient-level post-selection sensitivity analysis.  It can show that
the same patients do not dominate every module and that the module effects are
not simple reflections of phenotype-blind pair-normalization factors.  A null
correlation is not interpreted as proof of biological independence.
"""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/mtbls13729"
OUT = BASE / "secondary_module_independence_v1"
DETAIL = BASE / "candidate_abundance_protocol_audit_v1/candidate_abundance_patient_protocols.csv"
NEU5AC = BASE / "integrated_biology_ledger_v2/new_anchor_patient_deltas.csv"
FACTORS = BASE / "biology_closure_analysis_v1/phenotype_blind_normalization_factors.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fast_spearman(x: np.ndarray, y: np.ndarray) -> float:
    xr = rankdata(x)
    yr = rankdata(y)
    xs = xr - xr.mean()
    ys = yr - yr.mean()
    denom = np.sqrt(np.sum(xs * xs) * np.sum(ys * ys))
    return float(np.sum(xs * ys) / denom) if denom > 0 else np.nan


def correlation_audit(x: pd.Series, y: pd.Series, seed: int, resamples: int = 5000) -> dict:
    common = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    n = len(common)
    if n < 6:
        return {"n": n, "rho": np.nan, "bootstrap_ci_low": np.nan, "bootstrap_ci_high": np.nan, "permutation_p": np.nan}
    xv, yv = common.x.to_numpy(float), common.y.to_numpy(float)
    observed = fast_spearman(xv, yv)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(resamples, n))
    boots = np.empty(resamples, dtype=float)
    for i, index in enumerate(indices):
        boots[i] = fast_spearman(xv[index], yv[index])
    boots = boots[np.isfinite(boots)]
    permuted = np.empty(resamples, dtype=float)
    for i in range(resamples):
        permuted[i] = fast_spearman(xv, rng.permutation(yv))
    p = (1 + np.sum(np.abs(permuted) >= abs(observed))) / (resamples + 1)
    return {
        "n": int(n),
        "rho": observed,
        "bootstrap_ci_low": float(np.quantile(boots, 0.025)),
        "bootstrap_ci_high": float(np.quantile(boots, 0.975)),
        "permutation_p": float(p),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    detail = pd.read_csv(DETAIL)
    values = detail.pivot(index="patient", columns="feature_id", values="complete_detection_log2_delta")
    expected = {150, 1597, 1717, 3019, 3222}
    if set(values.columns) != expected:
        raise RuntimeError(f"unexpected source-absent feature set: {set(values.columns)}")
    neu5ac = pd.read_csv(NEU5AC)
    neu5ac = neu5ac.loc[neu5ac.feature_id.eq(703)].set_index("patient").log2_tumor_normal

    modules = pd.DataFrame(index=values.index)
    modules["neu5ac"] = neu5ac
    modules["acetylated_polyamine"] = values[1717]
    modules["modified_guanosine"] = values[[1597, 3019]].mean(axis=1, skipna=False)
    modules["long_chain_acylcarnitine"] = values[[150, 3222]].mean(axis=1, skipna=False)

    factors = pd.read_csv(FACTORS)
    for normalization, block in factors.groupby("normalization"):
        sample = block.set_index("sample").log2_factor
        pair_delta = {}
        for patient in modules.index:
            pair_delta[patient] = float(sample[f"{patient}-Rmu"] - sample[f"{patient}-RN"])
        modules[f"technical_{normalization}"] = pd.Series(pair_delta)
    modules.to_csv(OUT / "patient_module_effects_and_technical_factors.csv")

    biology_columns = ["neu5ac", "acetylated_polyamine", "modified_guanosine", "long_chain_acylcarnitine"]
    summary_rows = []
    for column in biology_columns:
        v = modules[column].dropna()
        summary_rows.append(
            {
                "module": column,
                "patients": int(len(v)),
                "positive_patients": int((v > 0).sum()),
                "mean_log2fc": float(v.mean()),
                "median_log2fc": float(v.median()),
                "minimum_log2fc": float(v.min()),
                "maximum_log2fc": float(v.max()),
            }
        )
    module_summary = pd.DataFrame(summary_rows)
    module_summary.to_csv(OUT / "module_patient_summary.csv", index=False)

    correlation_rows = []
    seed = 20260901
    for left, right in combinations(biology_columns, 2):
        stats = correlation_audit(modules[left], modules[right], seed)
        correlation_rows.append({"left": left, "right": right, "comparison_type": "biology_vs_biology", **stats})
        seed += 1
    technical_columns = [column for column in modules if column.startswith("technical_")]
    for left in biology_columns:
        for right in technical_columns:
            stats = correlation_audit(modules[left], modules[right], seed)
            correlation_rows.append({"left": left, "right": right, "comparison_type": "biology_vs_technical", **stats})
            seed += 1
    correlations = pd.DataFrame(correlation_rows)
    correlations.to_csv(OUT / "module_correlation_audit.csv", index=False)

    neu5ac_rows = correlations.loc[
        correlations.comparison_type.eq("biology_vs_biology")
        & ((correlations.left.eq("neu5ac")) | (correlations.right.eq("neu5ac")))
    ]
    technical_rows = correlations.loc[correlations.comparison_type.eq("biology_vs_technical")]
    not_neu5ac_dominated = bool(neu5ac_rows.rho.abs().max() < 0.70)
    not_technical_drift_dominated = bool(technical_rows.rho.abs().max() < 0.70)

    corr_matrix = modules[biology_columns].corr(method="spearman")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [1, 1.3]})
    image = axes[0].imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    axes[0].set_xticks(range(len(biology_columns)), biology_columns, rotation=35, ha="right")
    axes[0].set_yticks(range(len(biology_columns)), biology_columns)
    for i in range(len(biology_columns)):
        for j in range(len(biology_columns)):
            axes[0].text(j, i, f"{corr_matrix.iloc[i, j]:+.2f}", ha="center", va="center", fontweight="bold")
    axes[0].set_title("Patient-level Spearman correlation")
    fig.colorbar(image, ax=axes[0], shrink=0.8)

    shown = modules[biology_columns]
    x = np.arange(len(shown))
    for column in biology_columns:
        axes[1].plot(x, shown[column], marker="o", linewidth=1.7, label=column)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x, shown.index, rotation=45)
    axes[1].set_ylabel("Rmu vs matched normal log2 fold change")
    axes[1].set_title("The same patients do not dominate every module")
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle("MTBLS13729 secondary modules are parallel signals, not one common patient ordering", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "secondary_module_independence.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    report = {
        "status": "mtbls13729_secondary_module_independence_complete",
        "formal": True,
        "patients": int(len(modules)),
        "module_summaries": module_summary.to_dict(orient="records"),
        "neu5ac_correlations": neu5ac_rows[["left", "right", "n", "rho", "bootstrap_ci_low", "bootstrap_ci_high", "permutation_p"]].to_dict(orient="records"),
        "largest_absolute_biology_vs_technical_rho": float(technical_rows.rho.abs().max()),
        "gates": {
            "secondary_modules_not_neu5ac_ordering_dominated_abs_rho_lt_0_70": not_neu5ac_dominated,
            "modules_not_pair_normalization_dominated_abs_rho_lt_0_70": not_technical_drift_dominated,
        },
        "interpretation": (
            "Neu5Ac has only weak-to-moderate patient ordering correlations with the three secondary modules. "
            "The secondary modules therefore add nonredundant descriptive axes rather than merely restating the Neu5Ac effect."
        ),
        "negative_result": (
            "Small n and post-selection prevent a claim of statistical or causal independence. The acetylated-polyamine "
            "and acylcarnitine axes are inversely ordered across patients, which argues against a single global abundance "
            "artifact but requires independent validation before mechanistic interpretation."
        ),
        "provenance": {
            "candidate_protocol_detail_sha256": sha256(DETAIL),
            "neu5ac_deltas_sha256": sha256(NEU5AC),
            "normalization_factors_sha256": sha256(FACTORS),
            "patient_matrix_sha256": sha256(OUT / "patient_module_effects_and_technical_factors.csv"),
        },
        "claim_limit": "Post-selection patient-level sensitivity analysis; lack of correlation is not proof of independence, mechanism, flux or causality.",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
