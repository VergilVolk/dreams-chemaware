"""Audit frozen LCNEC priority effects against recorded technical covariates.

The public LCNEC overview workbook contains no clinical-stage, smoking, sex or
purity variables.  It does contain tissue amount and injection order.  This
script therefore performs a fixed, diagnostic technical-confounding audit and
does not attempt clinical subgroup discovery.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data/validation/lcnec_zenodo19005638_preflight/MTB22_P073_overview_public_v1.xlsx"
EFFECTS = ROOT / "data/validation/lcnec_hsst3n_priority_patient_covariation_v1/patient_effect_matrix.csv"
OUT = ROOT / "data/validation/lcnec_hsst3n_priority_technical_confounding_v1"

PRIORITIES = [
    "adenosine_diphosphate_family",
    "adenosine_diphosphoribose_family",
    "quinolinate",
    "ascorbate",
]
PRIORITY_LABELS = ["ADP family", "ADP-ribose family", "Quinolinate", "Ascorbate"]
PREDICTORS = [
    "log2_tumor_to_normal_tissue_amount",
    "pair_mean_injection_number",
    "tumor_minus_normal_injection_number",
    "absolute_pair_injection_gap",
]
PREDICTOR_LABELS = [
    "log2 tissue\namount ratio",
    "mean injection\nposition",
    "tumor - normal\ninjection order",
    "absolute paired\ninjection gap",
]


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


def injection_number(value: object) -> int:
    match = re.fullmatch(r"Inj(\d+)", str(value))
    if match is None:
        raise RuntimeError(f"invalid injection identifier: {value!r}")
    return int(match.group(1))


def clustered_bootstrap(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values: list[float] = []
    for _ in range(5000):
        idx = rng.integers(0, len(x), len(x))
        value = spearmanr(x[idx], y[idx]).statistic
        if np.isfinite(value):
            values.append(float(value))
    if len(values) < 4500:
        raise RuntimeError(f"too few finite bootstrap correlations: {len(values)}")
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    overview = pd.read_excel(WORKBOOK, sheet_name="mzML_HSST3n")
    required = {"SAMPLE_CODE", "GROUP_CODE", "AMOUNT", "INJECTION_ID"}
    if not required.issubset(overview.columns):
        raise RuntimeError(f"HSST3n sheet lacks columns: {sorted(required - set(overview.columns))}")
    study = overview.loc[overview["GROUP_CODE"].isin(["TU", "NG"]), list(required)].copy()
    if len(study) != 68 or study["SAMPLE_CODE"].nunique() != 34:
        raise RuntimeError(f"expected 68 study injections from 34 pairs, got {len(study)} rows")
    if study.isna().any().any():
        raise RuntimeError("study technical metadata contain missing values")
    study["injection_number"] = study["INJECTION_ID"].map(injection_number)
    if study.duplicated(["SAMPLE_CODE", "GROUP_CODE"]).any():
        raise RuntimeError("duplicate patient/group study injection")

    amount = study.pivot(index="SAMPLE_CODE", columns="GROUP_CODE", values="AMOUNT")
    injection = study.pivot(index="SAMPLE_CODE", columns="GROUP_CODE", values="injection_number")
    metadata = pd.DataFrame(index=amount.index)
    metadata["tumor_tissue_amount_mg"] = amount["TU"].astype(float)
    metadata["normal_tissue_amount_mg"] = amount["NG"].astype(float)
    metadata["log2_tumor_to_normal_tissue_amount"] = np.log2(amount["TU"] / amount["NG"])
    metadata["tumor_injection_number"] = injection["TU"].astype(int)
    metadata["normal_injection_number"] = injection["NG"].astype(int)
    metadata["pair_mean_injection_number"] = (injection["TU"] + injection["NG"]) / 2.0
    metadata["tumor_minus_normal_injection_number"] = injection["TU"] - injection["NG"]
    metadata["absolute_pair_injection_gap"] = (injection["TU"] - injection["NG"]).abs()
    metadata["tumor_injected_after_normal"] = injection["TU"] > injection["NG"]
    metadata.index.name = "patient_code"

    effects = pd.read_csv(EFFECTS, index_col="patient_code")
    if list(effects.columns) != PRIORITIES or effects.shape != (34, 4):
        raise RuntimeError(f"frozen effect matrix changed: {effects.shape}, {list(effects.columns)}")
    if set(effects.index) != set(metadata.index):
        raise RuntimeError(
            f"patient mismatch: effects-only={sorted(set(effects.index)-set(metadata.index))}, "
            f"metadata-only={sorted(set(metadata.index)-set(effects.index))}"
        )
    merged = metadata.join(effects, how="inner").sort_index()
    merged.to_csv(OUT / "patient_effects_and_technical_metadata.csv")

    rng = np.random.default_rng(20260901)
    rows: list[dict[str, object]] = []
    for priority in PRIORITIES:
        y = merged[priority].to_numpy(float)
        for predictor in PREDICTORS:
            x = merged[predictor].to_numpy(float)
            rho, p = spearmanr(x, y)
            ci_low, ci_high = clustered_bootstrap(x, y, rng)
            loo = np.asarray([
                spearmanr(np.delete(x, i), np.delete(y, i)).statistic
                for i in range(len(x))
            ], dtype=float)
            rows.append({
                "priority": priority,
                "technical_predictor": predictor,
                "n_pairs": len(x),
                "spearman_rho": float(rho),
                "spearman_p": float(p),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "loo_rho_min": float(np.nanmin(loo)),
                "loo_rho_max": float(np.nanmax(loo)),
                "loo_sign_stable": bool(np.all(np.sign(loo) == np.sign(rho))),
            })
    tests = pd.DataFrame(rows)
    tests["bh_q_16"] = bh(tests["spearman_p"].tolist())
    tests["fixed_confounding_gate"] = (
        (tests["spearman_rho"].abs() >= 0.35)
        & (tests["bh_q_16"] < 0.10)
        & ((tests["bootstrap_ci_low"] > 0) | (tests["bootstrap_ci_high"] < 0))
        & tests["loo_sign_stable"]
    )
    tests.to_csv(OUT / "technical_confounding_tests.csv", index=False)

    rho = tests.pivot(index="priority", columns="technical_predictor", values="spearman_rho").loc[PRIORITIES, PREDICTORS]
    q = tests.pivot(index="priority", columns="technical_predictor", values="bh_q_16").loc[PRIORITIES, PREDICTORS]
    gate = tests.pivot(index="priority", columns="technical_predictor", values="fixed_confounding_gate").loc[PRIORITIES, PREDICTORS]
    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    image = ax.imshow(rho.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(PREDICTORS)), PREDICTOR_LABELS)
    ax.set_yticks(range(len(PRIORITIES)), PRIORITY_LABELS)
    for i in range(len(PRIORITIES)):
        for j in range(len(PREDICTORS)):
            star = " *" if bool(gate.iloc[i, j]) else ""
            value = float(rho.iloc[i, j])
            ax.text(j, i, f"{value:+.2f}\nq={float(q.iloc[i,j]):.3g}{star}", ha="center", va="center",
                    color="white" if abs(value) > 0.55 else "#222222", fontsize=9)
    ax.set_title("Frozen LCNEC priority effects versus recorded technical factors", fontweight="bold", pad=14)
    cbar = fig.colorbar(image, ax=ax, shrink=0.82)
    cbar.set_label("Spearman rho across 34 tumor/NAT pairs")
    fig.text(
        0.5,
        0.015,
        "BH across 16 fixed tests; * also requires |rho| >= 0.35, bootstrap CI excluding 0 and leave-one-out sign stability.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0.02, 0.05, 0.98, 0.98))
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"priority_technical_confounding.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)

    passed = tests.loc[tests["fixed_confounding_gate"]].copy()
    report = {
        "status": "lcnec_priority_technical_confounding_complete",
        "formal": True,
        "patients": 34,
        "study_injections": 68,
        "priorities": len(PRIORITIES),
        "technical_predictors": len(PREDICTORS),
        "fixed_tests": len(tests),
        "tests_passing_fixed_confounding_gate": int(len(passed)),
        "maximum_absolute_rho": float(tests["spearman_rho"].abs().max()),
        "minimum_bh_q_16": float(tests["bh_q_16"].min()),
        "tumor_injected_after_normal_pairs": int(metadata["tumor_injected_after_normal"].sum()),
        "tumor_injected_before_normal_pairs": int((~metadata["tumor_injected_after_normal"]).sum()),
        "clinical_metadata_available": False,
        "clinical_metadata_absent_fields": ["stage", "smoking", "sex", "tumor_purity"],
        "passed_relationships": passed[["priority", "technical_predictor", "spearman_rho", "bh_q_16"]].to_dict("records"),
        "provenance": {
            "workbook_sha256": sha256(WORKBOOK),
            "effect_matrix_sha256": sha256(EFFECTS),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "This is a technical-confounding diagnostic in the discovery cohort. A null result does not prove absence of "
            "all technical bias, and the public workbook lacks clinical covariates. It does not validate metabolite identity, "
            "replicate the biology, establish flux or establish causality."
        ),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if len(merged) != 34 or len(tests) != 16:
        raise RuntimeError("post-write validation failed")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
