"""Molecule-cluster bootstrap for the external low-rank adapter pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def bootstrap(values: np.ndarray, seed: int, n: int = 5000) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/validation/lipid_projection_adapter_external"))
    args = parser.parse_args()
    frame = pd.read_csv(args.input_dir / "query_results.csv")
    official = frame.loc[frame["seed"] == -1].copy()
    adapted = frame.loc[frame["seed"] >= 0].copy()
    metrics = {"top1": "mean", "pairwise": "mean", "margin": "mean"}
    rows = []
    for seed in sorted(adapted["seed"].unique()):
        for split in ("discovery", "confirmation"):
            for domain, domain_mask in {
                "phospholipid_like": lambda x: x["phospholipid_like"],
                "non_phospholipid": lambda x: ~x["phospholipid_like"],
            }.items():
                left = official[(official["split"] == split) & domain_mask(official)]
                right = adapted[(adapted["seed"] == seed) & (adapted["split"] == split) & domain_mask(adapted)]
                for metric in metrics:
                    lmol = left.groupby("ik14")[metric].mean()
                    rmol = right.groupby("ik14")[metric].mean()
                    common = lmol.index.intersection(rmol.index)
                    delta = (rmol.loc[common] - lmol.loc[common]).to_numpy(float)
                    rows.append({
                        "seed": int(seed), "split": split, "domain": domain, "metric": metric,
                        "molecules": len(delta), "delta_mean": float(delta.mean()),
                        "delta_ci_low": bootstrap(delta, int(seed) * 100 + len(rows))[0],
                        "delta_ci_high": bootstrap(delta, int(seed) * 100 + len(rows))[1],
                        "molecules_improved": int((delta > 0).sum()),
                        "molecules_worsened": int((delta < 0).sum()),
                    })
    result = pd.DataFrame(rows)
    result.to_csv(args.input_dir / "cluster_bootstrap.csv", index=False)
    phospho = result[result["domain"] == "phospholipid_like"]
    seed_consistency = {}
    for split in ("discovery", "confirmation"):
        part = phospho[phospho["split"] == split]
        seed_consistency[split] = {
            metric: bool((part.loc[part["metric"] == metric, "delta_mean"] >= 0).all())
            for metric in ("top1", "pairwise", "margin")
        }
    report = {
        "status": "lipid_projection_adapter_external_cluster_bootstrap",
        "unit_of_resampling": "molecule; two query views remain clustered",
        "seed_consistency_nonnegative": seed_consistency,
        "formal_significance": bool((phospho["delta_ci_low"] > 0).all()),
        "decision": (
            "Promising mechanism pilot, not a formal improvement claim. Confirmation Top-1 and margin are non-negative "
            "across seeds, but discovery Top-1 is unchanged, one discovery pairwise result is slightly negative, and "
            "molecule-cluster confidence intervals include zero."
        ),
    }
    (args.input_dir / "bootstrap_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(result[(result["domain"] == "phospholipid_like")].to_string(index=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
