"""Paired molecule-cluster comparison of targeted and matched-control adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def ci(values: np.ndarray, seed: int, n: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, (n, len(values)), replace=True).mean(1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def molecule_effects(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    official = frame[frame["seed"] == -1].copy()
    adapted = frame[frame["seed"] >= 0].copy()
    keys = ["split", "ik14", "view"]
    baseline = official[keys + ["phospholipid_like", "top1", "pairwise", "margin"]].rename(columns={
        "top1": "official_top1", "pairwise": "official_pairwise", "margin": "official_margin"
    })
    merged = adapted.merge(baseline, on=keys, suffixes=("", "_baseline"))
    for metric in ("top1", "pairwise", "margin"):
        merged[f"{metric}_effect"] = merged[metric].astype(float) - merged[f"official_{metric}"].astype(float)
    return merged.groupby(["seed", "split", "phospholipid_like", "ik14"])[
        ["top1_effect", "pairwise_effect", "margin_effect"]
    ].mean().reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, default=Path("data/validation/lipid_projection_adapter_external/query_results.csv"))
    parser.add_argument("--control", type=Path, default=Path("data/validation/lipid_projection_shared_selection_control_external/query_results.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/lipid_projection_target_vs_control"))
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    target, control = molecule_effects(args.target), molecule_effects(args.control)
    keys = ["seed", "split", "phospholipid_like", "ik14"]
    paired = target.merge(control, on=keys, suffixes=("_target", "_control"))
    rows = []
    for (seed, split, phospho), group in paired.groupby(["seed", "split", "phospholipid_like"]):
        for metric in ("top1", "pairwise", "margin"):
            delta = group[f"{metric}_effect_target"] - group[f"{metric}_effect_control"]
            low, high = ci(delta.to_numpy(float), int(seed) * 100 + len(rows))
            rows.append({
                "seed": int(seed), "split": split,
                "domain": "phospholipid_like" if phospho else "non_phospholipid",
                "metric": metric, "molecules": len(group),
                "target_minus_control": float(delta.mean()), "ci_low": low, "ci_high": high,
                "target_better_molecules": int((delta > 0).sum()),
                "control_better_molecules": int((delta < 0).sum()),
            })
    result = pd.DataFrame(rows); result.to_csv(args.output_dir / "paired_comparison.csv", index=False)
    phospho = result[result["domain"] == "phospholipid_like"]
    report = {
        "status": "targeted_vs_drift_matched_general_identity_adapter",
        "unit": "molecule; paired within seed/split",
        "target_supervision": "same-formula phospholipid negatives",
        "control_supervision": "different-formula embedding-hard lipid negatives",
        "common_selection_task": "same-formula formula-disjoint internal validation",
        "target_consistently_better": {
            split: {metric: bool((phospho[(phospho["split"] == split) & (phospho["metric"] == metric)]["target_minus_control"] > 0).all())
                    for metric in ("top1", "pairwise", "margin")}
            for split in ("discovery", "confirmation")
        },
        "claim_limit": "Small external molecule counts; confidence intervals are descriptive paired cluster bootstrap intervals.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(result.to_string(index=False)); print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
