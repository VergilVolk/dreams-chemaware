"""Audit how strongly KPGT bond tokens depend on global knowledge nodes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def rowwise_comparison(a: np.ndarray, b: np.ndarray) -> dict[str, np.ndarray]:
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)
    denominator = np.maximum(a_norm * b_norm, 1e-12)
    cosine = np.sum(a * b, axis=1) / denominator
    relative_delta = np.linalg.norm(a - b, axis=1) / np.maximum(a_norm, 1e-12)
    return {"cosine": cosine, "relative_delta": relative_delta}


def bootstrap_summary(
    values: np.ndarray, n_bootstrap: int, rng: np.random.Generator
) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    samples = rng.choice(values, size=(n_bootstrap, len(values)), replace=True)
    means = samples.mean(axis=1)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
        "bootstrap_mean_ci95_low": float(np.quantile(means, 0.025)),
        "bootstrap_mean_ci95_high": float(np.quantile(means, 0.975)),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.token_dir / "manifest.json").read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []

    comparisons = {
        "normal_vs_zero_global_priors": (
            "contextual_bond_tokens",
            "contextual_bond_tokens_zero_global_priors",
        ),
        "normal_vs_no_virtual_nodes": (
            "contextual_bond_tokens",
            "contextual_bond_tokens_no_virtual_nodes",
        ),
        "zero_global_priors_vs_no_virtual_nodes": (
            "contextual_bond_tokens_zero_global_priors",
            "contextual_bond_tokens_no_virtual_nodes",
        ),
        "input_local_vs_normal": (
            "input_bond_tokens",
            "contextual_bond_tokens",
        ),
        "input_local_vs_no_virtual_nodes": (
            "input_bond_tokens",
            "contextual_bond_tokens_no_virtual_nodes",
        ),
    }

    for item in manifest:
        if item.get("status") != "ok":
            continue
        archive = np.load(args.token_dir / f"{item['file_stem']}.npz")
        row: dict[str, object] = {
            "id": item["id"],
            "smiles": item["smiles"],
            "n_bonds": int(item["n_bonds"]),
        }
        for name, (left_key, right_key) in comparisons.items():
            comparison = rowwise_comparison(archive[left_key], archive[right_key])
            row[f"{name}__cosine_mean"] = float(comparison["cosine"].mean())
            row[f"{name}__cosine_min"] = float(comparison["cosine"].min())
            row[f"{name}__relative_delta_mean"] = float(
                comparison["relative_delta"].mean()
            )
            row[f"{name}__relative_delta_max"] = float(
                comparison["relative_delta"].max()
            )
        rows.append(row)

    if not rows:
        raise RuntimeError("No successful token files found")

    with (args.output_dir / "per_molecule.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    rng = np.random.default_rng(args.seed)
    metric_columns = [key for key in rows[0] if "__" in key]
    summary = {
        "status": "kpgt_bond_prior_dependence_audit",
        "statistical_unit": "molecule",
        "n_molecules": len(rows),
        "n_bonds_total": int(sum(int(row["n_bonds"]) for row in rows)),
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "metrics": {
            column: bootstrap_summary(
                np.asarray([float(row[column]) for row in rows]),
                args.bootstrap,
                rng,
            )
            for column in metric_columns
        },
        "interpretation_guardrail": (
            "This audit measures representation dependence, not fragmentation "
            "rule discovery. Removing virtual nodes is out-of-distribution for "
            "the published checkpoint."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
