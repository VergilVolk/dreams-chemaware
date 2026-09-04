"""Audit four frozen LCNEC abundance effects against source cotinine exposure.

This analysis is governed by the frozen JSON preregistration beside its output.
It treats cotinine as a smoking-exposure covariate, not as biological validation
of any candidate metabolite.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as student_t, ttest_ind


ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data/validation/lcnec_zenodo19005638_preflight/article_mmc7.xlsx"
EFFECTS = ROOT / "data/validation/lcnec_hsst3n_priority_patient_covariation_v1/patient_effect_matrix.csv"
PREREG = ROOT / "data/validation/lcnec_hsst3n_priority_smoking_confounding_preregistration_v1.json"
OUT = ROOT / "data/validation/lcnec_hsst3n_priority_smoking_confounding_v1"

PRIORITIES = [
    "adenosine_diphosphate_family",
    "adenosine_diphosphoribose_family",
    "quinolinate",
    "ascorbate",
]
LABELS = ["ADP family", "ADP-ribose family", "Quinolinate", "Ascorbate"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bh(values: list[float]) -> list[float]:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result.tolist()


def bootstrap_group_difference(
    y: np.ndarray, group: np.ndarray, rng: np.random.Generator, repeats: int = 5000
) -> tuple[float, float]:
    idx0 = np.flatnonzero(group == 0)
    idx1 = np.flatnonzero(group == 1)
    draws = np.empty(repeats, dtype=float)
    for i in range(repeats):
        a = rng.choice(idx1, size=len(idx1), replace=True)
        b = rng.choice(idx0, size=len(idx0), replace=True)
        draws[i] = y[a].mean() - y[b].mean()
    return tuple(float(v) for v in np.quantile(draws, [0.025, 0.975]))


def bootstrap_spearman(
    x: np.ndarray, y: np.ndarray, rng: np.random.Generator, repeats: int = 5000
) -> tuple[float, float]:
    draws: list[float] = []
    for _ in range(repeats):
        idx = rng.integers(0, len(x), len(x))
        rho = spearmanr(x[idx], y[idx]).statistic
        if np.isfinite(rho):
            draws.append(float(rho))
    if len(draws) < int(repeats * 0.9):
        raise RuntimeError("too few finite bootstrap Spearman estimates")
    return tuple(float(v) for v in np.quantile(draws, [0.025, 0.975]))


def hc3_cotinine_coefficient(frame: pd.DataFrame, endpoint: str) -> tuple[float, float, float]:
    columns = ["cotinine_smoker", "age", "sex_male", "bmi", "late_stage"]
    complete = frame[[endpoint, *columns]].dropna()
    if len(complete) < 30:
        raise RuntimeError(f"{endpoint}: only {len(complete)} complete adjusted cases")
    x = complete[columns].to_numpy(float)
    for j in (1, 3):
        sd = x[:, j].std(ddof=1)
        if not np.isfinite(sd) or sd == 0:
            raise RuntimeError(f"{endpoint}: zero-variance adjusted covariate")
        x[:, j] = (x[:, j] - x[:, j].mean()) / sd
    x = np.column_stack([np.ones(len(x)), x])
    y = complete[endpoint].to_numpy(float)
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residual = y - x @ beta
    leverage = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    scaled = residual / np.maximum(1.0 - leverage, 1e-10)
    meat = x.T @ ((scaled * scaled)[:, None] * x)
    covariance = xtx_inv @ meat @ xtx_inv
    se = float(np.sqrt(covariance[1, 1]))
    coefficient = float(beta[1])
    statistic = coefficient / se
    p = float(2 * student_t.sf(abs(statistic), df=len(y) - x.shape[1]))
    return coefficient, se, p


def main() -> None:
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if not prereg.get("frozen_before_outcome_computation"):
        raise RuntimeError("smoking-confounding preregistration is not frozen")

    effects = pd.read_csv(EFFECTS, index_col="patient_code")
    if effects.shape != (34, 4) or list(effects.columns) != PRIORITIES:
        raise RuntimeError(f"frozen effect matrix changed: {effects.shape}, {list(effects.columns)}")

    s1 = pd.read_excel(WORKBOOK, sheet_name="Table S1", header=None)
    sample_pairs = s1.iloc[3, 13:81].astype(str).tolist()
    groups = s1.iloc[4, 13:81].astype(str).tolist()
    if groups.count("TU") != 34 or groups.count("NG") != 34:
        raise RuntimeError("Table S1 does not contain 34 paired study samples")
    clinical_rows = {
        "age": 6,
        "sex": 7,
        "bmi": 8,
        "self_reported_smoking": 9,
        "stage": 10,
    }
    records: dict[str, dict[str, object]] = {}
    for column, (patient, group) in enumerate(zip(sample_pairs, groups), start=13):
        if group != "TU":
            continue
        records[patient] = {name: s1.iloc[row, column] for name, row in clinical_rows.items()}
    clinical = pd.DataFrame.from_dict(records, orient="index")
    clinical.index.name = "patient_code"
    clinical["sex_male"] = clinical["sex"].map({"M": 1, "F": 0})
    clinical["late_stage"] = clinical["stage"].map({"III&IV": 1, "I&II": 0})

    s4 = pd.read_excel(WORKBOOK, sheet_name="Table S4", header=None)
    cotinine = s4.iloc[5:, 1:7].copy()
    cotinine.columns = [
        "patient_code",
        "cotinine_ng_per_g",
        "cotinine_status",
        "self_report_detail",
        "self_report_general",
        "cotinine_self_report_agreement",
    ]
    cotinine = cotinine.dropna(subset=["patient_code"]).set_index("patient_code")
    # Table S4 ends with a source-defined cut-off note formatted like a data row.
    # Keep only study patient identifiers and fail later if any frozen patient is absent.
    cotinine = cotinine.loc[cotinine.index.astype(str).str.fullmatch(r"IT\d+")].copy()
    cotinine["cotinine_smoker"] = cotinine["cotinine_status"].map({"smoker": 1, "non-smoker": 0})
    cotinine["log2_cotinine_ng_per_g"] = np.log2(cotinine["cotinine_ng_per_g"].astype(float))

    if set(effects.index) != set(clinical.index) or set(effects.index) != set(cotinine.index):
        raise RuntimeError(
            "patient mismatch across frozen effects, Table S1 and Table S4: "
            f"effects={len(effects)}, clinical={len(clinical)}, cotinine={len(cotinine)}"
        )
    merged = effects.join(clinical).join(cotinine).sort_index()
    if merged[PRIORITIES + ["cotinine_smoker", "log2_cotinine_ng_per_g"]].isna().any().any():
        raise RuntimeError("primary smoking audit has missing data")

    rng = np.random.default_rng(int(prereg["stability"]["seed"]))
    rows: list[dict[str, object]] = []
    for endpoint in PRIORITIES:
        y = merged[endpoint].to_numpy(float)
        group = merged["cotinine_smoker"].to_numpy(int)
        smokers = y[group == 1]
        nonsmokers = y[group == 0]
        welch = ttest_ind(smokers, nonsmokers, equal_var=False)
        difference = float(smokers.mean() - nonsmokers.mean())
        primary_ci = bootstrap_group_difference(y, group, rng)
        primary_loo_values: list[float] = []
        for i in range(len(y)):
            y_loo = np.delete(y, i)
            group_loo = np.delete(group, i)
            primary_loo_values.append(float(y_loo[group_loo == 1].mean() - y_loo[group_loo == 0].mean()))
        primary_loo = np.asarray(primary_loo_values)

        x = merged["log2_cotinine_ng_per_g"].to_numpy(float)
        rho, spearman_p = spearmanr(x, y)
        rho_ci = bootstrap_spearman(x, y, rng)
        rho_loo = np.asarray([
            spearmanr(np.delete(x, i), np.delete(y, i)).statistic for i in range(len(y))
        ])
        adjusted_beta, adjusted_se, adjusted_p = hc3_cotinine_coefficient(merged, endpoint)
        rows.append({
            "priority": endpoint,
            "n_pairs": len(y),
            "cotinine_smokers": int(group.sum()),
            "cotinine_nonsmokers": int((group == 0).sum()),
            "smoker_minus_nonsmoker_mean_log2fc": difference,
            "welch_p": float(welch.pvalue),
            "primary_bootstrap_ci_low": primary_ci[0],
            "primary_bootstrap_ci_high": primary_ci[1],
            "primary_loo_min": float(primary_loo.min()),
            "primary_loo_max": float(primary_loo.max()),
            "primary_loo_sign_stable": bool(np.all(np.sign(primary_loo) == np.sign(difference))),
            "continuous_spearman_rho": float(rho),
            "continuous_spearman_p": float(spearman_p),
            "continuous_bootstrap_ci_low": rho_ci[0],
            "continuous_bootstrap_ci_high": rho_ci[1],
            "continuous_loo_min": float(np.nanmin(rho_loo)),
            "continuous_loo_max": float(np.nanmax(rho_loo)),
            "continuous_loo_sign_stable": bool(np.all(np.sign(rho_loo) == np.sign(rho))),
            "adjusted_cotinine_beta": adjusted_beta,
            "adjusted_cotinine_hc3_se": adjusted_se,
            "adjusted_cotinine_p": adjusted_p,
        })
    results = pd.DataFrame(rows)
    results["primary_bh_q4"] = bh(results["welch_p"].tolist())
    results["continuous_bh_q4"] = bh(results["continuous_spearman_p"].tolist())
    results["adjusted_bh_q4"] = bh(results["adjusted_cotinine_p"].tolist())
    direction_primary = np.sign(results["smoker_minus_nonsmoker_mean_log2fc"])
    direction_secondary = np.sign(results["continuous_spearman_rho"])
    direction_adjusted = np.sign(results["adjusted_cotinine_beta"])
    results["potential_smoking_sensitivity_gate"] = (
        (results["primary_bh_q4"] < 0.10)
        & (results["continuous_bh_q4"] < 0.10)
        & (results["adjusted_bh_q4"] < 0.10)
        & (direction_primary == direction_secondary)
        & (direction_primary == direction_adjusted)
    )

    OUT.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT / "patient_effects_cotinine_and_clinical_metadata.csv")
    results.to_csv(OUT / "priority_smoking_confounding_tests.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    positions = np.arange(len(PRIORITIES))
    differences = results["smoker_minus_nonsmoker_mean_log2fc"].to_numpy(float)
    low = results["primary_bootstrap_ci_low"].to_numpy(float)
    high = results["primary_bootstrap_ci_high"].to_numpy(float)
    axes[0].errorbar(positions, differences, yerr=[differences - low, high - differences], fmt="o", capsize=4)
    axes[0].axhline(0, color="#555555", linewidth=1)
    axes[0].set_xticks(positions, LABELS, rotation=25, ha="right")
    axes[0].set_ylabel("Smoker - non-smoker paired effect")
    axes[0].set_title("Cotinine-classified smoking contrast")
    rho = results["continuous_spearman_rho"].to_numpy(float)
    rho_low = results["continuous_bootstrap_ci_low"].to_numpy(float)
    rho_high = results["continuous_bootstrap_ci_high"].to_numpy(float)
    axes[1].errorbar(positions, rho, yerr=[rho - rho_low, rho_high - rho], fmt="o", capsize=4, color="#a34a28")
    axes[1].axhline(0, color="#555555", linewidth=1)
    axes[1].set_xticks(positions, LABELS, rotation=25, ha="right")
    axes[1].set_ylabel("Spearman rho with log2 cotinine")
    axes[1].set_title("Continuous tumor cotinine association")
    fig.suptitle("Frozen LCNEC priority effects: objective smoking-exposure audit", fontweight="bold")
    fig.text(0.5, 0.005, "34 paired effects; cotinine measured in tumor tissue. Null association does not exclude smoking confounding.", ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"priority_smoking_confounding.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)

    passed = results.loc[results["potential_smoking_sensitivity_gate"], "priority"].tolist()
    report = {
        "status": "lcnec_priority_smoking_confounding_complete",
        "formal": True,
        "patients": len(merged),
        "cotinine_smokers": int(merged["cotinine_smoker"].sum()),
        "cotinine_nonsmokers": int((merged["cotinine_smoker"] == 0).sum()),
        "cotinine_self_report_agreement_known": int(merged["cotinine_self_report_agreement"].notna().sum()),
        "cotinine_self_report_agreement": int(merged["cotinine_self_report_agreement"].fillna(False).astype(bool).sum()),
        "priorities": len(PRIORITIES),
        "potential_smoking_sensitive_priorities": passed,
        "tests_passing_gate": len(passed),
        "minimum_primary_q4": float(results["primary_bh_q4"].min()),
        "minimum_continuous_q4": float(results["continuous_bh_q4"].min()),
        "minimum_adjusted_q4": float(results["adjusted_bh_q4"].min()),
        "provenance": {
            "workbook_sha256": sha256(WORKBOOK),
            "effect_matrix_sha256": sha256(EFFECTS),
            "preregistration_sha256": sha256(PREREG),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "This is a covariate-sensitivity audit in the discovery cohort. Failure to pass the gate does not prove absence "
            "of smoking confounding. Tumor cotinine does not validate candidate identity, flux, enzyme activity or causality."
        ),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
