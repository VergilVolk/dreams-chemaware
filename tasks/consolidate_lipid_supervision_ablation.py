"""Consolidate three lipid adapter supervision modes without seed pseudoreplication."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_effects(path: Path, model: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    official = frame[frame["seed"] == -1].copy()
    adapted = frame[frame["seed"] >= 0].copy()
    keys = ["split", "ik14", "view"]
    base = official[keys + ["top1", "pairwise", "margin"]].rename(columns={
        "top1": "official_top1", "pairwise": "official_pairwise", "margin": "official_margin",
    })
    merged = adapted.merge(base, on=keys)
    for metric in ("top1", "pairwise", "margin"):
        merged[f"{metric}_effect"] = merged[metric].astype(float) - merged[f"official_{metric}"].astype(float)
    # First cluster two views, then average seeds. Seeds describe optimization
    # stability; they are not independent chemical observations.
    molecule_seed = merged.groupby(["seed", "split", "phospholipid_like", "ik14"])[
        ["top1_effect", "pairwise_effect", "margin_effect"]
    ].mean().reset_index()
    molecule = molecule_seed.groupby(["split", "phospholipid_like", "ik14"])[
        ["top1_effect", "pairwise_effect", "margin_effect"]
    ].mean().reset_index()
    molecule["model"] = model
    return molecule


def bootstrap(values: np.ndarray, seed: int, n: int = 20000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, (n, len(values)), replace=True).mean(1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def stable_seed(*parts: str) -> int:
    return int.from_bytes(
        hashlib.blake2b("|".join(parts).encode(), digest_size=8).digest(), "little"
    ) % (2**32 - 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/lipid_supervision_ablation"))
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "same_formula": Path("data/validation/lipid_projection_adapter_external/query_results.csv"),
        "different_formula_hard": Path("data/validation/lipid_projection_shared_selection_control_external/query_results.csv"),
        "mixed_equal": Path("data/validation/lipid_projection_mixed_equal_external/query_results.csv"),
    }
    molecules = pd.concat([load_effects(path, model) for model, path in paths.items()], ignore_index=True)
    molecules.to_csv(args.output_dir / "molecule_effects.csv", index=False)
    rows = []
    for (model, split, phospho), group in molecules.groupby(["model", "split", "phospholipid_like"]):
        for metric in ("top1", "pairwise", "margin"):
            values = group[f"{metric}_effect"].to_numpy(float)
            low, high = bootstrap(values, stable_seed(model, split, metric))
            rows.append({
                "model": model, "split": split,
                "domain": "phospholipid_like" if phospho else "non_phospholipid",
                "metric": metric, "molecules": len(values), "effect": float(values.mean()),
                "ci_low": low, "ci_high": high,
                "improved_molecules": int((values > 0).sum()), "worsened_molecules": int((values < 0).sum()),
            })
    summary = pd.DataFrame(rows); summary.to_csv(args.output_dir / "summary.csv", index=False)

    comparisons = []
    for split in ("discovery", "confirmation"):
        part = molecules[(molecules["split"] == split) & molecules["phospholipid_like"]]
        for left, right in (("same_formula", "different_formula_hard"), ("mixed_equal", "different_formula_hard"), ("mixed_equal", "same_formula")):
            a, b = part[part["model"] == left], part[part["model"] == right]
            paired = a.merge(b, on=["split", "phospholipid_like", "ik14"], suffixes=("_left", "_right"))
            for metric in ("top1", "pairwise", "margin"):
                values = paired[f"{metric}_effect_left"] - paired[f"{metric}_effect_right"]
                low, high = bootstrap(values.to_numpy(float), stable_seed(left, right, split, metric))
                comparisons.append({
                    "split": split, "comparison": f"{left}_minus_{right}", "metric": metric,
                    "molecules": len(values), "difference": float(values.mean()),
                    "ci_low": low, "ci_high": high,
                })
    comparisons = pd.DataFrame(comparisons); comparisons.to_csv(args.output_dir / "paired_model_comparisons.csv", index=False)
    phospho_summary = summary[summary["domain"] == "phospholipid_like"]
    report = {
        "status": "lipid_supervision_ablation_seed_averaged_molecule_clustered",
        "aggregation": "two views clustered per molecule; three optimization seeds averaged before bootstrap",
        "target_results": phospho_summary.to_dict(orient="records"),
        "pairwise_model_comparisons": comparisons.to_dict(orient="records"),
        "any_supervision_mode_dominates_both_splits": False,
        "decision": (
            "No supervision mode has a statistically stable advantage. Mixed supervision is the most operationally "
            "balanced candidate, but advancement requires a larger independent lipid cohort and should not be described as superior."
        ),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(phospho_summary.to_string(index=False)); print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
