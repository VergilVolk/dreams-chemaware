"""Exploratory all-pairs covariation within the frozen independent protein axes."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/external/LCNEC_proteogenomic_2026/fixed_panel_patient_audit_v1/pure_lcnec_patient_pair_differences.csv"
OUT = ROOT / "data/external/LCNEC_proteogenomic_2026/protein_axis_covariation_exploratory_v1"
SEED = 20260901
RESAMPLES = 5000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def benjamini_hochberg(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    long = pd.read_csv(SOURCE)
    if long["patient_id"].nunique() != 80:
        raise RuntimeError("expected 80 pure-LCNEC patient pairs")
    rng = np.random.default_rng(SEED)
    rows = []
    for axis, block in long.groupby("axis", sort=True):
        matrix = block.pivot(index="patient_id", columns="gene", values="tumor_minus_normal").sort_index()
        for gene_a, gene_b in itertools.combinations(sorted(matrix.columns), 2):
            x = matrix[gene_a].to_numpy(float)
            y = matrix[gene_b].to_numpy(float)
            rho, p = spearmanr(x, y)
            boot = np.empty(RESAMPLES, float)
            for i in range(RESAMPLES):
                idx = rng.integers(0, len(x), len(x))
                boot[i] = spearmanr(x[idx], y[idx]).statistic
            loo = np.array([spearmanr(np.delete(x, i), np.delete(y, i)).statistic for i in range(len(x))])
            rows.append({
                "axis": axis,
                "gene_a": gene_a,
                "gene_b": gene_b,
                "patients": len(x),
                "rho": float(rho),
                "p": float(p),
                "bootstrap_ci_low": float(np.nanquantile(boot, 0.025)),
                "bootstrap_ci_high": float(np.nanquantile(boot, 0.975)),
                "loo_sign_stable": bool(np.all(np.sign(loo) == np.sign(rho))),
            })
    result = pd.DataFrame(rows)
    result["bh_q_46"] = benjamini_hochberg(result["p"].to_numpy())
    result["exploratory_gate"] = (
        (result["rho"].abs() >= 0.30)
        & (result["bh_q_46"] < 0.05)
        & ((result["bootstrap_ci_low"] > 0) | (result["bootstrap_ci_high"] < 0))
        & result["loo_sign_stable"]
    )
    result = result.sort_values(["exploratory_gate", "bh_q_46", "axis"], ascending=[False, True, True])
    result.to_csv(OUT / "all_within_axis_pairwise_covariation.csv", index=False)

    axis_summary = result.groupby("axis", as_index=False).agg(
        protein_pairs=("rho", "size"),
        passing_pairs=("exploratory_gate", "sum"),
        maximum_abs_rho=("rho", lambda s: float(s.abs().max())),
    )
    axis_summary.to_csv(OUT / "axis_covariation_summary.csv", index=False)

    def pair(a: str, b: str) -> dict:
        mask = ((result["gene_a"] == a) & (result["gene_b"] == b)) | ((result["gene_a"] == b) & (result["gene_b"] == a))
        row = result.loc[mask]
        return {} if row.empty else row.iloc[0].to_dict()

    report = {
        "status": "lcnec_independent_protein_axis_covariation_complete",
        "formal": False,
        "analysis_role": "exploratory all-within-axis structure audit after primary frozen panel testing",
        "patients": 80,
        "axes": int(result["axis"].nunique()),
        "pairwise_tests": int(len(result)),
        "pairs_passing_exploratory_gate": int(result["exploratory_gate"].sum()),
        "key_pairs": {
            "PARP1_PARP2": pair("PARP1", "PARP2"),
            "G6PD_TKT": pair("G6PD", "TKT"),
            "G6PD_TALDO1": pair("G6PD", "TALDO1"),
            "TKT_TALDO1": pair("TKT", "TALDO1"),
            "QPRT_HAAO": pair("QPRT", "HAAO"),
        },
        "provenance": {"source_sha256": sha256(SOURCE), "seed": SEED, "bootstrap_resamples": RESAMPLES},
        "claim_limit": "Post-primary exploratory patient covariation within the independent protein cohort. It is not a metabolite identity, metabolite replication, flux or causal test.",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=lambda x: x.item() if hasattr(x, "item") else x) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "key_pairs"}, indent=2))
    print(axis_summary.to_string(index=False))


if __name__ == "__main__":
    main()
