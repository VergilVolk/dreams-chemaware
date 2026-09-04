"""Analyze patient-level covariation among four frozen LCNEC priorities."""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/validation/lcnec_hsst3n_manuscript_supplement/table_s6_priority_per_patient_effects.csv"
OUT = ROOT / "data/validation/lcnec_hsst3n_priority_patient_covariation_v1"
ORDER = [
    "adenosine_diphosphate_family",
    "adenosine_diphosphoribose_family",
    "quinolinate",
    "ascorbate",
]
LABELS = ["ADP family", "ADP-ribose family", "Quinolinate", "Ascorbate"]


def bh(values: list[float]) -> list[float]:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result.tolist()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(SOURCE)
    if set(data["priority_name"]) != set(ORDER):
        raise RuntimeError("priority set changed")
    matrix = data.pivot(index="patient_code", columns="priority_name", values="per_mg_log2fc_tumor_vs_normal")
    matrix = matrix[ORDER].sort_index()
    if matrix.shape != (34, 4) or matrix.isna().any().any():
        raise RuntimeError(f"expected complete 34x4 patient matrix, got {matrix.shape}")
    matrix.to_csv(OUT / "patient_effect_matrix.csv")

    rng = np.random.default_rng(20260901)
    rows = []
    for left, right in combinations(ORDER, 2):
        x = matrix[left].to_numpy(float)
        y = matrix[right].to_numpy(float)
        rho, p = spearmanr(x, y)
        bootstrap = []
        for _ in range(5000):
            idx = rng.integers(0, len(x), len(x))
            value = spearmanr(x[idx], y[idx]).statistic
            if np.isfinite(value):
                bootstrap.append(float(value))
        ci = np.quantile(bootstrap, [0.025, 0.975])
        loo = [spearmanr(np.delete(x, i), np.delete(y, i)).statistic for i in range(len(x))]
        rows.append({
            "left": left,
            "right": right,
            "spearman_rho": float(rho),
            "spearman_p": float(p),
            "bootstrap_ci_low": float(ci[0]),
            "bootstrap_ci_high": float(ci[1]),
            "loo_rho_min": float(np.min(loo)),
            "loo_rho_max": float(np.max(loo)),
            "loo_sign_stable": bool(np.all(np.sign(loo) == np.sign(rho))),
            "relationship_class": (
                "primary_nucleotide_family_pair"
                if {left, right} == {"adenosine_diphosphate_family", "adenosine_diphosphoribose_family"}
                else "secondary_nad_context_pair"
                if {left, right} == {"adenosine_diphosphoribose_family", "quinolinate"}
                else "exploratory_cross_axis_pair"
            ),
        })
    pairs = pd.DataFrame(rows)
    pairs["bh_q_6"] = bh(pairs["spearman_p"].tolist())
    pairs["fixed_covariation_gate"] = (
        (pairs["spearman_rho"].abs() >= 0.35)
        & (pairs["bh_q_6"] < 0.10)
        & ((pairs["bootstrap_ci_low"] > 0) | (pairs["bootstrap_ci_high"] < 0))
        & pairs["loo_sign_stable"]
    )
    pairs.to_csv(OUT / "pairwise_covariation.csv", index=False)

    rho_matrix = np.eye(4)
    annotations = np.full((4, 4), "", dtype=object)
    for i in range(4):
        annotations[i, i] = "1.00"
    for row in pairs.itertuples(index=False):
        i, j = ORDER.index(row.left), ORDER.index(row.right)
        rho_matrix[i, j] = rho_matrix[j, i] = row.spearman_rho
        label = f"{row.spearman_rho:+.2f}\nq={row.bh_q_6:.3g}"
        annotations[i, j] = annotations[j, i] = label
    fig, ax = plt.subplots(figsize=(8.1, 6.8))
    image = ax.imshow(rho_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(4), LABELS, rotation=25, ha="right")
    ax.set_yticks(range(4), LABELS)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, annotations[i, j], ha="center", va="center",
                    color="white" if abs(rho_matrix[i, j]) > 0.55 else "#222222", fontsize=10)
    ax.set_title("Patient-level covariation of frozen LCNEC priority effects", fontweight="bold", pad=14)
    cbar = fig.colorbar(image, ax=ax, shrink=0.82)
    cbar.set_label("Spearman rho across 34 paired tumor/NAT effects")
    fig.text(0.5, 0.015, "All six pairs are shown; q values use BH across six tests. Covariation is not flux or causality.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0.02, 0.05, 0.98, 0.98))
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"priority_patient_covariation.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)

    primary = pairs.loc[pairs["relationship_class"] == "primary_nucleotide_family_pair"].iloc[0]
    report = {
        "status": "lcnec_priority_patient_covariation_complete",
        "formal": True,
        "patients": 34,
        "priorities": 4,
        "pairwise_tests": 6,
        "pairs_passing_fixed_gate": int(pairs["fixed_covariation_gate"].sum()),
        "primary_adp_adpr": {
            "rho": float(primary["spearman_rho"]),
            "bh_q_6": float(primary["bh_q_6"]),
            "ci": [float(primary["bootstrap_ci_low"]), float(primary["bootstrap_ci_high"])],
            "gate": bool(primary["fixed_covariation_gate"]),
        },
        "provenance": {"source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest()},
        "claim_limit": "Patient-level effect covariation is an internal abundance-module diagnostic. It is not an independent cohort, identity validation, reaction direction, flux or causality.",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
