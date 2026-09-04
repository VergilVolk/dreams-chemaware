"""Describe patient-level coordination among post-selection Rmu modules.

The result is explicitly exploratory.  Module membership and the Rmu subgroup
were already selected, so permutation P values quantify coordination in these
ten patients only and are not confirmatory pathway tests.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data/mtbls13729/convergent_metabolic_modules_v1/module_patient_deltas.csv"
OUT = ROOT / "data/mtbls13729/module_coordination_v1"


def permutation_p(x: np.ndarray, y: np.ndarray, observed: float, rng: np.random.Generator, n: int = 20000) -> float:
    x_rank = rankdata(x).astype(float)
    y_rank = rankdata(y).astype(float)
    x_rank -= x_rank.mean()
    y_rank -= y_rank.mean()
    denominator = float(np.linalg.norm(x_rank) * np.linalg.norm(y_rank))
    exceed = 0
    for _ in range(n):
        value = float(np.dot(x_rank, rng.permutation(y_rank)) / denominator)
        exceed += int(abs(value) >= abs(observed) - 1e-12)
    return (exceed + 1) / (n + 1)


def bh_adjust(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    adjusted = np.empty_like(values, dtype=float)
    running = 1.0
    m = len(values)
    for rank_index in range(m - 1, -1, -1):
        original_index = order[rank_index]
        running = min(running, float(values[original_index]) * m / (rank_index + 1))
        adjusted[original_index] = running
    return adjusted


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    long = pd.read_csv(IN)
    long = long[long["eligible"].astype(bool)].copy()
    wide = long.pivot(index="patient", columns="module", values="module_mean_log2fc").sort_index()
    if len(wide) < 8:
        raise RuntimeError(f"too few complete Rmu patients: {len(wide)}")

    rng = np.random.default_rng(20260830)
    rows = []
    for left, right in itertools.combinations(wide.columns, 2):
        pair = wide[[left, right]].dropna()
        rho = float(spearmanr(pair[left], pair[right]).statistic)
        rows.append(
            {
                "module_left": left,
                "module_right": right,
                "patients": int(len(pair)),
                "spearman_rho": rho,
                "two_sided_permutation_p": permutation_p(
                    pair[left].to_numpy(float), pair[right].to_numpy(float), rho, rng
                ),
            }
        )

    result = pd.DataFrame(rows)
    result["bh_q_across_six_module_pairs"] = bh_adjust(result["two_sided_permutation_p"].to_numpy(float))
    result.to_csv(OUT / "module_pairwise_coordination.csv", index=False)
    wide.to_csv(OUT / "patient_module_matrix.csv")
    report = {
        "status": "mtbls13729_module_coordination_complete",
        "formal": False,
        "patients": int(len(wide)),
        "modules": list(wide.columns),
        "pairwise": result.to_dict(orient="records"),
        "claim_limit": (
            "Exploratory coordination among phenotype-selected modules in ten Rmu pairs. "
            "It cannot establish a common regulator, pathway flux, mediation or subtype specificity."
        ),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
