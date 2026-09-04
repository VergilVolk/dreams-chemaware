#!/usr/bin/env python
"""Patient-level coupling audit for frozen modified-guanosine and methyl/purine axes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


MODIFIED_RAW_TARGETS = ("M296T181", "M296T200", "M312T210")
HILIC_TARGETS = ("M150T308", "M282T290", "M385T405")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rplc-eic",
        type=Path,
        default=Path(
            "data/external/OEP00006137_raw/modified_guanosine_raw_reextraction_v1/target_eic.csv.gz"
        ),
    )
    parser.add_argument(
        "--hilic-eic",
        type=Path,
        default=Path(
            "data/external/OEP00006137_raw/hilic_methyl_purine_raw_reextraction_v1/target_eic.csv.gz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/external/OEP00006137_raw/methyl_purine_coupling_v1"
        ),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260830)
    return parser.parse_args()


def target_deltas(frame: pd.DataFrame, target_id: str) -> pd.DataFrame:
    subset = frame.loc[
        frame["target_id"].eq(target_id) & frame["subtype"].notna(),
        ["subtype", "patient", "tissue", "area"],
    ].copy()
    pivot = subset.pivot(index=["subtype", "patient"], columns="tissue", values="area")
    pivot = pivot.dropna().loc[lambda item: (item["N"] > 0) & (item["T"] > 0)]
    result = pivot.reset_index()[["subtype", "patient"]]
    result[target_id] = np.log2(pivot["T"].to_numpy() / pivot["N"].to_numpy())
    return result


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, resamples: int, rng) -> list[float]:
    if len(x) < 4:
        return [float("nan"), float("nan")]
    estimates = []
    for _ in range(resamples):
        index = rng.integers(0, len(x), len(x))
        if np.unique(x[index]).size < 2 or np.unique(y[index]).size < 2:
            continue
        estimates.append(stats.spearmanr(x[index], y[index]).statistic)
    if not estimates:
        return [float("nan"), float("nan")]
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def main() -> None:
    args = parse_args()
    rplc = pd.read_csv(args.rplc_eic)
    hilic = pd.read_csv(args.hilic_eic)
    merged = None
    for target_id in (*MODIFIED_RAW_TARGETS, *HILIC_TARGETS):
        source = rplc if target_id in MODIFIED_RAW_TARGETS else hilic
        delta = target_deltas(source, target_id)
        merged = delta if merged is None else merged.merge(
            delta, on=["subtype", "patient"], how="outer"
        )
    merged["modified_guanosine_3peak_mean"] = merged[
        list(MODIFIED_RAW_TARGETS)
    ].mean(axis=1, skipna=False)

    rng = np.random.default_rng(args.seed)
    results = {}
    for subtype in ("MSI-H", "MSS"):
        results[subtype] = {}
        group = merged.loc[merged["subtype"].eq(subtype)]
        for target_id in HILIC_TARGETS:
            pair = group[["modified_guanosine_3peak_mean", target_id]].dropna()
            if len(pair) >= 3:
                rho, p = stats.spearmanr(
                    pair["modified_guanosine_3peak_mean"], pair[target_id]
                )
                ci = bootstrap_spearman(
                    pair["modified_guanosine_3peak_mean"].to_numpy(),
                    pair[target_id].to_numpy(),
                    args.bootstrap_resamples,
                    rng,
                )
            else:
                rho = p = float("nan")
                ci = [float("nan"), float("nan")]
            results[subtype][target_id] = {
                "n": int(len(pair)),
                "spearman_rho": float(rho),
                "spearman_p": float(p),
                "patient_bootstrap_95ci": ci,
            }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.output_dir / "patient_deltas.csv"
    merged.to_csv(table_path, index=False)
    report = {
        "status": "OEP00006137_methyl_purine_coupling_complete",
        "formal": False,
        "modified_module": {
            "definition": "mean patient log2FC across three raw-reproducible RPLC peaks",
            "targets": list(MODIFIED_RAW_TARGETS),
            "excluded": "M298T55 excluded because its primary extraction is RT-censored",
        },
        "coupling": results,
        "claim_limit": (
            "Exploratory patient-level correlations in n<=20 pairs. They test co-variation, "
            "not causal coupling, methylation flux, or enzyme activity."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
